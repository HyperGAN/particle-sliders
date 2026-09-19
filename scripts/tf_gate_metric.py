#!/usr/bin/env python3
"""Score a transformer slider checkpoint with the closed-loop render gate.

Runs the shipped sampler from the cached neutral condition at fixed seed for
each scale and reports RMS ratios vs scale 0 plus ln-space spread scores.
This is the metric that actually tracks render quality (the open-loop probe
is blind to pole explode/collapse); see docs/tf-leak.md.

  python scripts/tf_gate_metric.py --weights models/distortion-tf-v7/distortion-tf-v7_last.safetensors \
      --prompts_file conceptmod/textsliders/data/prompts-distortion-tf-v7.yaml \
      --cache_dir cache/distortion-tf-v7

Pass --no-vocoder to get the cheap latent-energy variant (used to measure
whether latent energy tracks audio before trusting it in-training).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork  # noqa: E402
from conceptmod.textsliders.render_gate import load_vocoder, run_gate  # noqa: E402
from conceptmod.textsliders.train_lora_music3 import (  # noqa: E402
    _load_transformer,
    _pick_device,
    build_conditions,
    load_prompts,
)

DEFAULT_MODEL_DIR = Path("/ml2/music/models/MiniMax-Music3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--scales", type=str, default="-2,-1,0,1,2")
    parser.add_argument("--steps", type=int, default=30, help="Euler steps per rollout (shipped default)")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--cond_seeds", type=str, default="7",
                        help="AR seeds the conditions were CACHED under (v7 campaign used 7)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--targets", choices=["attn", "full"], default=None,
                        help="default: sidecar targets, else full")
    parser.add_argument("--no-vocoder", dest="vocoder", action="store_false",
                        help="skip audio decode; report latent-energy spread only")
    args = parser.parse_args()

    weights = Path(args.weights)
    meta = {}
    for candidate in (weights.with_suffix(".json"), Path(str(weights) + ".json")):
        if candidate.exists():
            meta = json.loads(candidate.read_text(encoding="utf-8"))
            break
    targets = args.targets or str(meta.get("targets") or "full")
    target_replace = (
        ["MiniMaxMusic3Attention", "MiniMaxMusic3TransformerBlock", "MiniMaxMusic3Transformer1DModel"]
        if targets == "full"
        else ["MiniMaxMusic3Attention"]
    )

    device = _pick_device(int(args.device), dummy=False)
    model_dir = Path(args.model_dir)
    prompts, prompts_meta = load_prompts(Path(args.prompts_file))
    cond_seeds = [int(part) for part in str(args.cond_seeds).split(",") if part.strip()]
    print(f"building conditions ({len(prompts)} rows x {len(cond_seeds)} seeds) from {args.cache_dir}", flush=True)
    entries = build_conditions(
        prompts,
        cache_dir=Path(args.cache_dir),
        duration=float(args.duration),
        seeds=cond_seeds,
        device=device,
        model_dir=model_dir,
        skip_ar=True,
        dummy=False,
    )
    transformer = _load_transformer(model_dir, device)

    rank = int(meta.get("rank") or 8)
    alpha = float(meta.get("alpha") or 8.0)
    network = LoRANetwork(
        transformer,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=target_replace,
        train_method=str(meta.get("train_method") or "full"),
        delimiter=str(meta.get("delimiter") or "-"),
        prefix=str(meta.get("prefix") or "lora_unet"),
    ).to(device)
    from safetensors.torch import load_file

    state = load_file(str(weights), device="cpu")
    missing, unexpected = network.load_state_dict(state, strict=False)
    if len(network.unet_loras) == 0:
        raise RuntimeError(f"LoRA wrapped 0 modules for targets={targets}")
    if missing:
        raise RuntimeError(f"LoRA load missed {len(missing)} keys (first={missing[0]})")

    scales = [float(part) for part in str(args.scales).split(",") if part.strip()]
    vocoder_box = [None] if args.vocoder else None
    # Gate on the first entry's neutral condition; extra entries are reported too.
    reports = []
    for index, (_prompt, conds) in enumerate(entries):
        report = run_gate(
            transformer, network, conds["neutral"], model_dir, device,
            seed=int(args.seed), scales=scales, num_steps=int(args.steps),
            amp=device.type == "cuda", vocoder_box=vocoder_box,
        )
        report["entry"] = index
        reports.append(report)
        print(
            f"entry {index}: measure={report['measure']} "
            + " ".join(f"{r['scale']:+g}:{r['ratio_vs_zero']:g}" for r in report["rows"])
            + f" spread_1={report['spread_1']} spread_all={report['spread_all']} {report['verdict']}",
            flush=True,
        )
    worst = max(reports, key=lambda r: r["spread_all"])
    out = {
        "weights": str(weights),
        "scales": scales,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "measure": reports[0]["measure"],
        "entries": reports,
        "worst_entry_spread_all": worst["spread_all"],
        "verdict": (
            "OK"
            if all(r["verdict"] == "OK" for r in reports)
            else f"FAIL ({worst['verdict']})"
        ),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
