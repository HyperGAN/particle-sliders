#!/usr/bin/env python3
"""Train a YuE2 AR slider using the winning UNI16 adversarial formulation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conceptmod.textsliders.yue2_backend import (
    DEFAULT_MODEL, YuE2Backend, YuE2Slider, aligned_suffix, normalized_mse, sound_only,
)

DATA = Path(__file__).resolve().parent / "data"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", type=Path)
    parser.add_argument("--recipe", choices=["uni16", "hidden"], default="uni16")
    parser.add_argument("--adv_batch", type=int, default=4)
    parser.add_argument("--resume_state", type=Path)
    parser.add_argument("--name", default="breath-yue2-ar")
    parser.add_argument("--model_id", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--cache_dir")
    parser.add_argument("--prompts_file", type=Path, default=DATA / "prompts-yue2.yaml")
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--train_tokens", type=int, default=250)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--hold_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_hub", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--no_checkpointing", action="store_true")
    initial, _ = parser.parse_known_args(argv)
    if initial.config_file:
        config = yaml.safe_load(initial.config_file.read_text())
        allowed = {a.dest for a in parser._actions} - {"help", "config_file"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("config_file must contain supported command-line option names")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    args.prompts_file = Path(args.prompts_file)
    args.save_dir = Path(args.save_dir) if args.save_dir else ROOT / "models" / args.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("name must be filename-safe")
    if min(args.rank, args.steps, args.save_every, args.train_tokens, args.max_seq_len) < 1:
        parser.error("rank, steps, save_every, train_tokens and max_seq_len must be positive")
    if not all(math.isfinite(v) for v in (args.alpha, args.lr, args.hold_weight)) or min(args.alpha, args.lr) <= 0 or args.hold_weight < 0:
        parser.error("alpha/lr must be positive and hold_weight non-negative; all must be finite")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    if args.adv_batch < 1:
        parser.error("adv_batch must be positive")
    if args.recipe == "uni16" and (args.hold_weight != 0 or args.steps > 600):
        parser.error("UNI16 warm-up uses no lyric hold and at most 600 fixed-history updates")
    if args.recipe != "uni16" and args.resume_state:
        parser.error("resume_state is supported by the UNI16 formulation")
    return args


def load_rows(path):
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Prompts must contain a nonempty rows list")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each prompt row must be a mapping")
        row = {}
        for key in ("neutral", "positive", "lyrics"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Each row needs nonempty {key}")
            row[key] = sound_only(value.strip())
        if row["neutral"] == row["positive"]:
            raise ValueError("Neutral and positive styles must differ")
        attrs = item.get("attributes") or [""]
        if not isinstance(attrs, list) or not all(isinstance(a, str) for a in attrs):
            raise ValueError("attributes must be a list of sound descriptions")
        for attr in attrs:
            sound_only(attr)
            rows.append(dict(row, neutral=f"{attr} {row['neutral']}".strip(),
                             positive=f"{attr} {row['positive']}".strip()))
    return rows


def prepare_pairs(backend, rows, train_tokens, seed, max_seq_len):
    pairs = []
    for i, row in enumerate(rows):
        print(f"Preparing YuE2 teacher pair {i + 1}/{len(rows)}", flush=True)
        neu = backend.prefix(row["neutral"], row["lyrics"])
        pos = backend.prefix(row["positive"], row["lyrics"])
        shared = aligned_suffix(neu, pos)
        if max(len(neu), len(pos)) + train_tokens > min(max_seq_len, backend.model.config.max_position_embeddings):
            raise ValueError("Prompt plus continuation exceeds max_seq_len; shorten the sheet or increase the limit")
        tail = backend.continuation(neu, train_tokens, seed + i)
        with torch.no_grad():
            plus = backend.hidden(pos + tail)
            neutral = backend.hidden(neu + tail)
        # MUSIC_START and the continuation are the concept targets. The shared
        # lyric suffix before MUSIC_START is a separate neutral preservation loss.
        pairs.append({"ids": neu + tail, "start": len(neu) - 1,
            "hold_start": len(neu) - shared, "hold_end": len(neu) - 3,
            "teacher": plus[:, len(pos) - 1:].float().cpu(),
            "hold": neutral[:, len(neu) - shared:len(neu) - 3].float().cpu()})
    return pairs


def train(args):
    rows = load_rows(args.prompts_file)
    if args.dummy:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    print("Loading YuE2 composition model", flush=True)
    backend = YuE2Backend(args.model_id, device=args.device, allow_hub=args.allow_hub,
                         revision=args.revision, cache_dir=args.cache_dir, dummy=args.dummy)
    if args.recipe == "uni16":
        from conceptmod.textsliders.yue2_uni import train_uni16
        return train_uni16(args, backend, rows)
    pairs = prepare_pairs(backend, rows, args.train_tokens, args.seed, args.max_seq_len)
    network = YuE2Slider(backend.model, rank=args.rank, alpha=args.alpha)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=0)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"backend": "yue2", "recipe": "ar-positive-hidden-with-lyric-hold",
        "model_id": args.model_id, "model_identity": backend.identity, "revision": args.revision,
        "dummy": args.dummy, "rows": rows, "seed": args.seed, "lr": args.lr,
        "hold_weight": args.hold_weight, "train_tokens": args.train_tokens, "cot": "off",
        "recommended_range": [0, 1], "validation_status": "unvalidated"}
    device = next(backend.model.parameters()).device
    with (args.save_dir / f"{args.name}_train.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            pair = pairs[(step - 1) % len(pairs)]
            optimizer.zero_grad(set_to_none=True)
            with network.scaled(1):
                hidden = backend.hidden(pair["ids"], checkpointing=not args.no_checkpointing)
                concept = normalized_mse(hidden[:, pair["start"]:], pair["teacher"].to(device))
                hold = normalized_mse(hidden[:, pair["hold_start"]:pair["hold_end"]], pair["hold"].to(device))
                loss = concept + args.hold_weight * hold
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite YuE2 slider loss; checkpoint not saved")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            metrics = {"step": step, "loss": loss.item(), "concept": concept.item(), "lyric_hold": hold.item()}
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(metrics), flush=True)
            if step % args.save_every == 0:
                network.save(args.save_dir / f"{args.name}_{step}.safetensors", dict(metadata, **metrics))
        network.save(args.save_dir / f"{args.name}_last.safetensors", dict(metadata, **metrics))
    return metadata


if __name__ == "__main__":
    train(parse_args())
