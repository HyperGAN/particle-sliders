#!/usr/bin/env python3
"""One unchanged UNI16 game with fresh histories from the first update."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conceptmod.textsliders.train_lora_yue2 import load_rows
from conceptmod.textsliders.yue2_backend import YuE2Backend, YuE2Slider, file_digest, DEFAULT_MODEL
from conceptmod.textsliders.yue2_uni import RECIPE, _cpu, lyric_positions, end_margins, update
from conceptmod.textsliders.lm_adv import SpanTransformerD
from conceptmod.textsliders.gan_v2.critic import pad_sequences

MILESTONES = (600, 1000, 2000, 3000, 3400)
FRESH_RECIPE = dict(RECIPE, name="uni16-rpgan-bcap-always-fresh",
    history_policy="fresh_base_for_every_prompt_draw_from_step_1", batch=1,
    row_policy="balanced_shuffled_passes", phase_changes=[], parameter_step_limit=None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class RowSampler:
    """Same balanced permutation schedule as gan_v2.state, without legacy imports."""
    def __init__(self, count, seed=7):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor = [], 0

    def next(self):
        if self.cursor >= len(self.order):
            self.order = torch.randperm(self.count, generator=self.generator).tolist()
            self.cursor = 0
        row = self.order[self.cursor]
        self.cursor += 1
        return row

    def state_dict(self):
        return dict(count=self.count, order=self.order, cursor=self.cursor, rng=self.generator.get_state())

    def load_state_dict(self, value):
        if value["count"] != self.count or (value["order"] and sorted(value["order"]) != list(range(self.count))):
            raise ValueError("Invalid saved row permutation")
        if not 0 <= value["cursor"] <= len(value["order"]):
            raise ValueError("Invalid row cursor")
        self.order, self.cursor = value["order"], value["cursor"]
        self.generator.set_state(value["rng"])


def prepare(backend, rows, frames, max_seq_len):
    prepared = []
    with torch.no_grad():
        for row in rows:
            prefix, positions = lyric_positions(backend, row["neutral"], row["lyrics"])
            positive, target_positions = lyric_positions(backend, row["positive"], row["lyrics"])
            if [prefix[i] for i in positions] != [positive[i] for i in target_positions]:
                raise ValueError("Teacher/student lyric spans differ")
            if max(len(prefix), len(positive)) + frames > min(max_seq_len, backend.model.config.max_position_embeddings):
                raise ValueError("Prompt and history exceed context budget")
            neutral = backend.hidden(prefix)[:, positions].float()
            real = backend.hidden(positive)[:, target_positions].float() - neutral
            prepared.append(dict(prefix=prefix, prefix_len=len(prefix), positions=positions,
                neutral=neutral.cpu(), real=real.cpu()))
    return prepared


def fresh_history(backend, network, row, fixed, frames, seed, graph):
    from yue2.protocol import SongRequest, Sampling, negative_prefix
    from yue2.sampling import generate_tokens
    with network.scaled(0), torch.no_grad():
        if backend.dummy:
            tokens = backend.continuation(fixed["prefix"], frames, seed)
            ended, timing = False, {"execution": "dummy"}
        else:
            request = SongRequest(style=row["neutral"], lyrics=row["lyrics"], cot="off", seed=seed)
            tokens, timing, truncated = generate_tokens(backend.model, fixed["prefix"],
                Sampling(min_tokens=min(200, frames), max_tokens=frames), seed, "semantic",
                negative=negative_prefix(request, backend.tokenizer), cfg_scale=request.guidance,
                legacy_off=True, use_cuda_graph=graph)
            ended = not truncated
        if not tokens:
            raise ValueError("Fresh history ended without content; seed is retained for investigation")
        ids = fixed["prefix"] + tokens
        margins = end_margins(backend.model, backend.hidden(ids)[:, fixed["prefix_len"] - 1:]).cpu()
    digest = hashlib.sha256(torch.tensor(tokens, dtype=torch.int64).numpy().tobytes()).hexdigest()
    current = dict(fixed, ids=ids, end_teacher=margins)
    audit = dict(seed=seed, tokens=len(tokens), tokens_sha256=digest, ended_naturally=ended,
        timing=timing, fixed_style_target_change=0.)
    return current, tokens, audit


def source_hashes():
    paths = [Path(__file__), *(ROOT / "conceptmod/textsliders" / name for name in (
        "yue2_uni.py", "yue2_backend.py", "train_lora_yue2.py", "lora.py", "lm_adv.py", "lm_gan.py", "gan_v2/critic.py"))]
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths}


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rows = load_rows(args.prompts_file)
    backend = YuE2Backend(args.model_id, device=args.device, dummy=args.dummy)
    signature = dict(recipe=FRESH_RECIPE, model=backend.identity, rows=rows, rank=args.rank,
        alpha=args.alpha, g_lr=args.lr, d_lr=args.lr*1.5, seed=args.seed,
        seed_start=args.seed_start, frames=args.train_tokens, horizon=args.steps,
        history_backend=args.history_backend, dummy=args.dummy, sources=source_hashes())
    run = args.save_dir
    run.mkdir(parents=True, exist_ok=True)
    path = run / "state.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
    if saved and saved["signature"] != signature:
        raise ValueError("Resume recipe, model, prompt, horizon or code differs")
    if not saved and (run / "manifest.json").exists():
        raise FileExistsError("Existing run has no recovery state; use a fresh save_dir")
    fixed = saved["prepared"] if saved else prepare(backend, rows, args.train_tokens, args.max_seq_len)
    network = YuE2Slider(backend.model, rank=args.rank, alpha=args.alpha)
    device = next(backend.model.parameters()).device
    critic = SpanTransformerD(backend.model.config.hidden_size, **RECIPE["critic"]).to(device)
    real, mask = pad_sequences([r["real"].to(device) for r in fixed])
    critic.calibrate_input_scale(real, mask)
    g = torch.optim.AdamW(network.parameters(), lr=args.lr, betas=(0., .999), weight_decay=1e-6)
    d = torch.optim.Adam(critic.parameters(), lr=args.lr*1.5, betas=(0., .999))
    sampler = RowSampler(len(rows), args.seed)
    completed, history = 0, []
    if saved:
        network.load_state_dict(saved["network"], strict=True)
        critic.load_state_dict(saved["critic"], strict=True)
        g.load_state_dict(saved["g_optimizer"])
        d.load_state_dict(saved["d_optimizer"])
        sampler.load_state_dict(saved["sampler"])
        completed, history = saved["completed"], saved["history"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if [h["step"] for h in history] != list(range(1, completed+1)):
        raise ValueError("Saved history is not contiguous")
    seen = {h["history"]["tokens_sha256"] for h in history}
    seeds = {h["history"]["seed"] for h in history}
    if len(seen) != completed or len(seeds) != completed:
        raise ValueError("Saved histories are not unique")
    metadata = dict(backend="yue2", recipe=FRESH_RECIPE["name"], recipe_settings=signature,
        model_id=args.model_id, model_identity=backend.identity, dummy=args.dummy, rows=rows,
        seed=args.seed, lr=args.lr, hold_weight=0., train_tokens=args.train_tokens, cot="off",
        recommended_range=[0,1], validation_status="experimental", teacher_rms=float(critic.input_scale))
    write_json(run / "manifest.json", signature)
    started, start_step = time.monotonic(), completed
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    old_handlers = {sig:signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    def status(phase, **extra):
        write_json(run / "status.json", dict(status=phase, completed=completed, total=args.steps,
            until=args.until, elapsed_sec=time.monotonic()-started, start_step=start_step,
            gpu=os.getenv("CUDA_VISIBLE_DEVICES"), **extra))
    def save():
        if not all(torch.isfinite(p).all() for p in [*network.state_dict().values(), *critic.state_dict().values()]):
            raise FloatingPointError("Non-finite checkpoint")
        blob = dict(signature=signature, prepared=fixed, completed=completed, history=history,
            network=_cpu(network.state_dict()), critic=_cpu(critic.state_dict()),
            g_optimizer=_cpu(g.state_dict()), d_optimizer=_cpu(d.state_dict()),
            sampler=sampler.state_dict(), rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        temporary = path.with_suffix(".pt.tmp")
        torch.save(blob, temporary)
        temporary.replace(path)
        record = dict(metadata, step=completed, fresh_histories=len(history),
            row_counts=dict(Counter(h["row"] for h in history)))
        network.save(run / "female-yue2-fresh_last.safetensors", record)
        if completed in MILESTONES or completed == args.steps:
            pinned = run / f"state-step{completed}.pt"
            if pinned.exists():
                raise FileExistsError("Milestone is immutable")
            os.link(path, pinned)
            network.save(run / f"female-yue2-fresh_step{completed}.safetensors", record)
        status("complete" if completed == args.steps else "milestone_complete" if completed == args.until
               else "paused" if stop else "training", fresh_histories=len(seen), row_counts=record["row_counts"])
    try:
        if completed >= args.until:
            status("complete" if completed == args.steps else "milestone_complete")
            return
        if not saved:
            save()
        with (run / f"updates-from-{completed}-{time.time_ns()}.jsonl").open("w") as log:
            while completed < args.until and not stop:
                step = completed + 1
                row_index = sampler.next()
                seed = args.seed_start + step - 1
                status("sampling", next_step=step, row=row_index, next_seed=seed)
                begin = time.monotonic()
                current, tokens, audit = fresh_history(backend, network, rows[row_index], fixed[row_index],
                    args.train_tokens, seed, args.history_backend == "cuda_graph")
                if seed in seeds or audit["tokens_sha256"] in seen:
                    raise ValueError("Duplicate fresh continuation; no replacement seed selected")
                history_path = run / "histories" / f"step{step}.pt"
                history_path.parent.mkdir(exist_ok=True)
                history_blob = dict(row=row_index, audit=audit, tokens=tokens, end_teacher=current["end_teacher"])
                temporary = history_path.with_suffix(".pt.tmp")
                torch.save(history_blob, temporary)
                temporary.replace(history_path)
                sampling_seconds = time.monotonic()-begin
                metrics = update(backend, network, critic, g, d, [current], checkpointing=not args.no_checkpointing)
                completed = step
                record = dict(metrics, step=step, row=row_index, history=audit,
                    sampling_seconds=sampling_seconds, step_seconds=time.monotonic()-begin)
                history.append(record)
                seen.add(audit["tokens_sha256"]); seeds.add(seed)
                log.write(json.dumps(record, allow_nan=False)+"\n"); log.flush()
                write_json(run / "progress.json", record)
                print(json.dumps(record), flush=True)
                if completed % args.save_every == 0 and completed != args.until:
                    save()
            save()
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts_file", type=Path, default=ROOT/"conceptmod/textsliders/data/prompts-yue2-female.yaml")
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3400)
    p.add_argument("--until", type=int)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.)
    p.add_argument("--lr", type=float, default=.0005)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seed_start", type=int, default=3000001)
    p.add_argument("--train_tokens", type=int, default=250)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--history_backend", choices=("eager","cuda_graph"), default="cuda_graph")
    p.add_argument("--model_id", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--no_checkpointing", action="store_true")
    args = p.parse_args(argv)
    args.until = args.until or args.steps
    if not 1 <= args.until <= args.steps or min(args.rank,args.save_every,args.train_tokens,args.max_seq_len) < 1:
        p.error("Invalid positive budget or checkpoint interval")
    if not all(__import__('math').isfinite(v) and v > 0 for v in (args.alpha,args.lr)):
        p.error("alpha and lr must be finite and positive")
    if not 0 <= args.seed < 2**63 or not 0 <= args.seed_start <= 2**63 - args.steps:
        p.error("Sampling seeds must fit signed int64")
    return args


if __name__ == "__main__":
    train(parse_args())
