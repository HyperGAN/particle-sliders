#!/usr/bin/env python3
"""Bake sidecar unit_scale into LoRA `.alpha` tensors so the file drops in at 1.

    python scripts/bake_unit_scale.py --catalog
    python scripts/bake_unit_scale.py models/gender-tf-v6/gender-tf-v6_alpha8.0_rank8_full_last.safetensors

The trainer (`train_lora_music3.py`) and `calibrate_weights` already call this
after calibration. This CLI is for files that were trained before that hook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_APP_REGISTRY = Path("/ml2/music/app/sliders.json")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from conceptmod.textsliders.bake_unit import bake_weights  # noqa: E402


def _catalog_weights() -> list[Path]:
    raw = json.loads(_APP_REGISTRY.read_text(encoding="utf-8"))
    root = Path(str(raw.get("root") or _REPO / "models"))
    paths = []
    for slider in raw.get("sliders") or []:
        for comp in slider.get("components") or []:
            weights = Path(str(comp.get("weights") or ""))
            if not weights.is_absolute():
                weights = root / weights
            paths.append(weights)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", nargs="*", type=Path)
    parser.add_argument("--catalog", action="store_true", help=f"bake every component in {_APP_REGISTRY}")
    parser.add_argument("--force", action="store_true", help="bake even when unit_scale is already 1")
    args = parser.parse_args(argv)
    paths = list(args.weights)
    if args.catalog:
        paths.extend(_catalog_weights())
    if not paths:
        parser.error("pass weights paths or --catalog")
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        out = bake_weights(path, force=args.force)
        if out is None:
            print(f"{path.name}: already unit 1 — skip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
