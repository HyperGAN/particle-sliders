#!/usr/bin/env python3
"""Compare effective LoRA weight updates; these are not audio-quality metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def inner(left, right):
    """Frobenius inner product of U1 D1 and U2 D2 without dense products."""
    up1, down1 = left
    up2, down2 = right
    return float(((up1.T @ up2) * (down1 @ down2.T)).sum())


def effective_factors(path):
    state = load_file(str(path), device="cpu")
    result = {}
    for key, value in state.items():
        if key.endswith(".lora_down.weight"):
            prefix = key.removesuffix(".lora_down.weight")
            down = value.double()
            up = state[prefix + ".lora_up.weight"].double()
            if down.ndim != 2 or up.ndim != 2:
                raise ValueError("This audit supports linear LoRA modules only")
            alpha = float(state[prefix + ".alpha"])
            result[prefix] = (up * (alpha / down.shape[0]), down)
    if not result:
        raise ValueError(f"No linear LoRA modules in {path}")
    return result


def compare(left, right):
    if left.keys() != right.keys():
        raise ValueError("LoRA module sets differ")
    a2 = sum(inner(v, v) for v in left.values())
    b2 = sum(inner(v, v) for v in right.values())
    dot = sum(inner(left[k], right[k]) for k in left)
    if a2 <= 0 or b2 <= 0:
        raise ValueError("Effective adapter norm is zero")
    return dict(modules=len(left), earlier_norm=math.sqrt(a2), later_norm=math.sqrt(b2),
                norm_ratio=math.sqrt(b2 / a2), cosine=dot / math.sqrt(a2 * b2),
                best_scalar_on_earlier=dot / a2,
                relative_residual_after_best_scalar=math.sqrt(max(0., 1. - dot * dot / (a2 * b2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.weights) < 2:
        parser.error("Provide at least two ordered checkpoints")
    torch.set_num_threads(2)
    records = []
    previous = None
    for path in args.weights:
        factors = effective_factors(path)
        if previous is not None:
            prev_path, prev_factors = previous
            records.append(dict(earlier=str(prev_path.resolve()), later=str(path.resolve()),
                                **compare(prev_factors, factors)))
        previous = path, factors
    report = dict(definition="Global Frobenius geometry of effective delta W = (alpha/rank) up @ down, at slider +1",
                  limitations=["Weight geometry does not establish perceptual strength or quality.",
                               "Global weighting favors larger matrices; inputs and downstream nonlinearities are not modeled.",
                               "The best scalar is a weight-space fit, not a validated equivalent listening scale."],
                  checkpoints=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.weights],
                  comparisons=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in records:
        print(f"{Path(row['earlier']).stem} -> {Path(row['later']).stem}: norm x{row['norm_ratio']:.4f}, cosine {row['cosine']:.6f}, residual {row['relative_residual_after_best_scalar']:.4f}")


if __name__ == "__main__":
    main()
