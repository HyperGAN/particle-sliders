#!/usr/bin/env python3
"""Opt-in tiny-LM concept-slider trainer (UNI on last-hidden states).

Test target for slider ideas (LoRA / hidden-delta / UNI polarity) without
loading Music 3, YuE2, H3 or a big LM. Default Music 3 trainers are
unchanged (``--lm_target v9`` / ``--pole_mode hidden``).

Pinned model: ``Qwen/Qwen3-0.6B-Base`` (596M params, 2025, Apache-2.0).
``--dummy`` is the CI / CPU path: tiny random stand-in of the same config
shape, no Hub, no GPU. Live runs need ``transformers>=4.51`` and pass
``--allow_hub`` once to download the weights.

UNI: student scale +1 fits the + caption hidden states, student scale 0
fits the neutral caption hidden states (full-sequence last-hidden MSE).
No minus teacher (unconditional row is a canary only, like H3).
After train (or ``--steps 0 --load_tiny_lora``) writes a hidden-delta
report under ``save_dir/report/`` at scales 0 / 0.5 / 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.tiny_llm_backend import (
    ATTN_CLASS_NAMES,
    DEFAULT_ALPHA,
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_MODEL,
    DEFAULT_RANK,
    FORMAT,
    LORA_LINEAR_NAMES,
    MODEL_LICENSE,
    MODEL_PARAMS_TOTAL,
    MODEL_RELEASE,
    TinyLLMBackend,
    hidden_delta_metrics,
    tiny_uni_loss,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-tiny-llm.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-tiny-llm.yaml"
DEFAULT_SAMPLE_SCALES = (0.0, 0.5, 1.0)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Opt-in tiny-LM UNI slider trainer")
    p.add_argument("--name", type=str, default="tiny-llm-smoke")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument(
        "--lora_up_init_std",
        type=float,
        default=DEFAULT_LORA_UP_INIT_STD,
        help=(
            "N(0, std) on LoRA-up (default 0.02). Zero-init is UNI identity: "
            "scale-1 vs scale-0 gap is 0 and the loss never moves. "
            "Pass 0 to restore zeros for ablation."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--allow_hub",
        action="store_true",
        help="Allow downloading the pinned weights (live path only)",
    )
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_tiny_lora",
        type=str,
        default=None,
        help="Dir or .safetensors with custom lora_tiny-* keys (not PEFT)",
    )
    p.add_argument(
        "--no_report",
        action="store_true",
        help="Skip the hidden-delta report after train / load",
    )
    p.add_argument(
        "--report_scales",
        type=str,
        default="0,0.5,1",
        help="LoRA scales for the hidden-delta report, comma-separated",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU mock causal LM; no Hub, no GPU, no Qwen weights",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _make_parser()
    initial, _ = parser.parse_known_args(argv)
    if initial.config_file:
        config = yaml.safe_load(Path(initial.config_file).read_text())
        if not isinstance(config, dict):
            raise ValueError("config_file must be a mapping")
        flat = _flatten_config(config)
        allowed = {a.dest for a in parser._actions} - {"help", "config_file"}
        unknown = set(flat) - allowed
        if unknown:
            parser.error(f"config_file has unsupported keys: {sorted(unknown)}")
        parser.set_defaults(**flat)
    return parser.parse_args(argv)


def load_slider_rows(prompts_file: str) -> list[dict]:
    raw = yaml.safe_load(Path(prompts_file).read_text()) or []
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("rows") or [raw]
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos = str(item.get("positive") or "")
        neu = str(item.get("neutral") or "")
        if not pos or not neu or pos == neu:
            raise ValueError(f"row needs distinct positive/neutral: {item!r}")
        rows.append({
            "positive": pos,
            "neutral": neu,
            "unconditional": str(item.get("unconditional") or ""),
            "target": str(item.get("target") or neu),
        })
    if not rows:
        raise ValueError(f"no slider rows in {prompts_file}")
    return rows


def parse_report_scales(text: str) -> list[float]:
    scales = [float(x) for x in str(text).split(",") if str(x).strip()]
    if not scales:
        raise ValueError("--report_scales is empty")
    return scales


def build_backend(args: argparse.Namespace) -> TinyLLMBackend:
    if args.dummy:
        return TinyLLMBackend(
            device="cpu",
            model_id=args.model_id,
            rank=args.rank,
            alpha=args.alpha,
            lora_up_init_std=float(args.lora_up_init_std),
            dummy=True,
        )
    return TinyLLMBackend(
        device=args.device,
        model_id=args.model_id,
        rank=args.rank,
        alpha=args.alpha,
        lora_up_init_std=float(args.lora_up_init_std),
        allow_hub=bool(args.allow_hub),
        dummy=False,
    )


def _shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def emit_report(
    backend: TinyLLMBackend,
    args: argparse.Namespace,
    save_dir: Path,
    rows: list[dict],
) -> list[dict]:
    """Hidden-delta report at each scale: does +1 move along h+ - h0?"""
    scales = parse_report_scales(getattr(args, "report_scales", "0,0.5,1"))
    out_dir = Path(save_dir) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            plus_ids = backend.encode(row["positive"]).ids
            neu_ids = backend.encode(row["neutral"]).ids
            teacher_plus = backend.teacher_hidden(plus_ids)
            teacher_zero = backend.teacher_hidden(neu_ids)
            base_zero = backend.hidden(neu_ids, scale=0.0)
            assert torch.allclose(base_zero, teacher_zero), "scale 0 must be exact base"
            for scale in scales:
                student_plus = backend.hidden(plus_ids, scale=float(scale))
                student_zero = backend.hidden(neu_ids, scale=float(scale))
                metrics = hidden_delta_metrics(
                    student_plus, student_zero, teacher_plus, teacher_zero
                )
                records.append({
                    "row": i,
                    "positive": row["positive"],
                    "neutral": row["neutral"],
                    "scale": float(scale),
                    "shared_prefix_tokens": _shared_prefix_len(plus_ids, neu_ids),
                    **metrics,
                })
    (out_dir / "hidden_delta.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} hidden-delta rows under {out_dir}")
    return records


def train(args: argparse.Namespace, backend: TinyLLMBackend | None = None) -> dict:
    rows = load_slider_rows(args.prompts_file)
    backend = backend or build_backend(args)
    loaded_lora = None
    if getattr(args, "load_tiny_lora", None):
        loaded_lora = backend.load_trained(args.load_tiny_lora)
        print(f"loaded lora_tiny weights from {loaded_lora}")
    params = backend.trainable_parameters()
    opt = None
    if int(args.steps) > 0:
        opt = torch.optim.Adam(params, lr=float(args.lr))
    torch.manual_seed(int(args.seed))

    history = []
    for step in range(int(args.steps)):
        row = rows[step % len(rows)]
        plus_ids = backend.encode(row["positive"]).ids
        neu_ids = backend.encode(row["neutral"]).ids
        with torch.no_grad():
            tgt_plus = backend.teacher_hidden(plus_ids)
            tgt_zero = backend.teacher_hidden(neu_ids)
        pred_plus = backend.hidden(plus_ids, scale=1.0)
        pred_zero = backend.hidden(neu_ids, scale=0.0)
        loss = tiny_uni_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rec = {
            "step": step,
            "loss": float(loss.detach().item()),
            "positive": row["positive"],
            "neutral": row["neutral"],
        }
        history.append(rec)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == int(args.steps):
            print(f"tiny-llm uni step {step}: loss={rec['loss']:.6f}")

    sidecar = {
        "name": args.name,
        "backend": "tiny_llm",
        "model_id": args.model_id,
        "resolved_model_id": DEFAULT_MODEL,
        "model_params_total": MODEL_PARAMS_TOTAL,
        "model_release": MODEL_RELEASE,
        "model_license": MODEL_LICENSE,
        "format": FORMAT,
        "stack": "causal_lm_hidden",
        "recipe": "tiny_llm_uni_hidden",
        "plus_neu": True,
        "minus_teacher": False,
        "lora_only": True,
        "lora_host": list(ATTN_CLASS_NAMES),
        "lora_linears": list(LORA_LINEAR_NAMES),
        "train_mlp": False,
        "train_norm": False,
        "train_embed": False,
        "train_lm_head": False,
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_up_init_std": float(
            getattr(args, "lora_up_init_std", backend.lora_up_init_std)
        ),
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device if not args.dummy else "cpu",
        "load_tiny_lora": loaded_lora,
        "dummy": bool(args.dummy),
        "first_loss": history[0]["loss"] if history else None,
        "last_loss": history[-1]["loss"] if history else None,
        "history": history[-8:],
    }
    save_dir = Path(args.save_dir or f"models/{args.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = save_dir / f"{args.name}_last.json"
    backend.save_trained(str(save_dir / f"{args.name}_lora"))
    report_records: list[dict] = []
    if not getattr(args, "no_report", False):
        report_records = emit_report(backend, args, save_dir, rows)
        sidecar["report"] = {
            "method": "hidden_delta",
            "scales": parse_report_scales(getattr(args, "report_scales", "0,0.5,1")),
            "n": len(report_records),
        }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {sidecar_path}")
    return sidecar


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return train(args)


def _flatten_config(config: dict) -> dict:
    """Map the YAML card onto CLI names (only fills unset CLI options)."""
    flat: dict = {}
    model = config.get("pretrained_model") or {}
    if isinstance(model, dict) and model.get("name_or_path"):
        flat["model_id"] = model["name_or_path"]
    network = config.get("network") or {}
    if isinstance(network, dict):
        for key in ("rank", "alpha", "lora_up_init_std"):
            if network.get(key) is not None:
                flat[key] = network[key]
    train_cfg = config.get("train") or {}
    if isinstance(train_cfg, dict):
        for key in ("lr", "steps", "seed"):
            if train_cfg.get(key) is not None:
                flat[key] = train_cfg[key]
        if train_cfg.get("iterations") is not None:
            flat.setdefault("steps", train_cfg["iterations"])
    save = config.get("save") or {}
    if isinstance(save, dict) and save.get("name"):
        flat["name"] = save["name"]
    if config.get("prompts_file"):
        flat["prompts_file"] = config["prompts_file"]
    return flat


if __name__ == "__main__":
    main()
