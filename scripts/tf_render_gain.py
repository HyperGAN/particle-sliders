#!/usr/bin/env python3
"""Render-based gain check for transformer slider checkpoints.

H1 proved velocity-projection unit_scale does not survive 50-step renders
(energy-tf-v6: +/-1 oversaturated, -2 silent, +1 piercing). This tool measures
what the shipped path ACTUALLY does, either from an existing generate_listen
ladder directory (CPU-only, instant) or by rendering a fresh one (--render):

  python scripts/tf_render_gain.py --ladder_dir eval/listen/v7-h4/distortion-tf-v7

Reports per-scale RMS ratio vs the 0 clip and suggests a sidecar derate
multiplier so the hottest pole lands at --cap (default 1.5x base RMS).
Energy-style axes may legitimately exceed it -- the number informs, the ear
decides. Apply a derate by scaling alpha (and sidecar alpha) or ratio; never
by touching the safetensors weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _rms(path: Path) -> float:
    wav, _sr = sf.read(str(path))
    mono = wav.mean(axis=1) if wav.ndim > 1 else wav
    return float(np.sqrt(np.mean(mono**2)))


_SCALE_RE = re.compile(r"(minus|plus|_n)?(\d)", re.IGNORECASE)


def parse_ladder(folder: Path) -> dict[float, Path]:
    """Map |scale| -> clip path from a generate_listen out_dir."""
    out: dict[float, Path] = {}
    for p in sorted(folder.glob("*.wav")):
        name = p.stem.lower()
        if "ref" in name:
            continue
        m = re.search(r"_(minus|plus)(_)?(\d)", name) or re.search(r"(minus|plus)(\d)", name)
        if m:
            groups = [g for g in m.groups() if g]
            sign = -1.0 if groups[0] == "minus" else 1.0
            out[sign * float(groups[-1])] = p
            continue
        if "zero" in name:
            out[0.0] = p
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder_dir", type=Path, help="existing generate_listen output folder")
    parser.add_argument("--render", action="store_true", help="render a fresh ladder first")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--prompts_file", type=Path)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cap", type=float, default=1.5, help="acceptable max RMS ratio vs 0 clip")
    args = parser.parse_args()

    folder = args.ladder_dir
    if args.render:
        import os
        import subprocess

        ROOT = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable, str(ROOT / "conceptmod/textsliders/generate_listen.py"),
            "--weights", str(args.weights), "--prompts_file", str(args.prompts_file),
            "--name", args.weights.stem if args.weights else "gain",
            "--out_dir", str(folder), "--scales", "-2,-1,0,1,2",
            "--duration", str(args.duration), "--seed", str(args.seed), "--device", str(args.device),
        ]
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
                   PYTHONPATH=str(ROOT), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)

    if folder is None or not folder.is_dir():
        raise SystemExit("need --ladder_dir (with --render to create one)")

    ladder = parse_ladder(folder)
    if 0.0 not in ladder:
        raise SystemExit(f"no 0 clip in {folder}")

    base = _rms(ladder[0.0])
    rows = []
    hottest = 1.0
    for scale in sorted(ladder):
        if scale == 0.0:
            continue
        ratio = _rms(ladder[scale]) / max(base, 1e-9)
        rows.append({"scale": scale, "rms": round(_rms(ladder[scale]), 5),
                     "ratio_vs_zero": round(ratio, 3)})
        hottest = max(hottest, abs(ratio))

    derate = args.cap / hottest if hottest > args.cap else 1.0
    verdict = "OK" if derate == 1.0 else f"DERATE x{derate:.3f} (alpha *= {derate:.3f})"

    print(json.dumps({"folder": str(folder), "base_rms": round(base, 5),
                      "clips": rows, "hottest_abs_ratio": round(hottest, 3),
                      "cap": args.cap, "suggested_derate": round(derate, 4),
                      "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
