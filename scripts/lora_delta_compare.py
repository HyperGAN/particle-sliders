#!/usr/bin/env python3
"""Compare two LoRA safetensors per module: cosine of direction + norm ratio.

Usage: python scripts/lora_delta_compare.py A.safetensors B.safetensors
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load_file(str(a_path), device="cpu"), load_file(str(b_path), device="cpu")
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no shared keys")
    rows = []
    for k in keys:
        va, vb = a[k].float().flatten(), b[k].float().flatten()
        na, nb = va.norm(), vb.norm()
        cos = torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        rows.append((k, cos, (na / nb.clamp_min(1e-12)).item()))
    # Aggregate: mean cos and median norm-ratio weighted by A's energy share.
    total_energy = sum(a[k].float().pow(2).sum().item() for k in keys)
    cos_w = norm_w = 0.0
    for k, cos, ratio in rows:
        share = a[k].float().pow(2).sum().item() / max(total_energy, 1e-12)
        cos_w += share * cos
        norm_w += share * ratio
    print(f"modules={len(rows)}")
    print(f"energy-weighted mean cos(A,B)     = {cos_w:.4f}")
    print(f"energy-weighted mean ||A||/||B||  = {norm_w:.4f}")
    lows = sorted(rows, key=lambda r: r[1])[:5]
    print("lowest-cos modules:")
    for k, cos, ratio in lows:
        print(f"  cos={cos:+.3f} normratio={ratio:.2f}  {k}")


if __name__ == "__main__":
    main()
