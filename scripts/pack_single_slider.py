#!/usr/bin/env python3
"""Pack each catalog slider into one ComfyUI safetensors.

A dual-host slider is two LoRAs. Shipping them as-is means Comfy strength 1
applies the transformer at its baked unit (energy 2.4–3.8) and the language
model at energy 1 — the mix goes underwater. This script concatenates the
already-converted Comfy files and rescales each host's `.alpha` tensors so
**one strength on MODEL and CLIP** is the whole slider. A/B weights are
copied unchanged.

    Comfy ±2  →  transformer energy 1.0, language-model energy 3.1
    Comfy ±1  →  half of that (TF 0.5 / LM 1.55)
    0         →  off

Do not set MODEL and CLIP to different strengths — the ratio is already in
alpha. Space is transformer-only; the same TF scale still applies.
Original host files are not touched.

    python scripts/pack_single_slider.py
    python scripts/pack_single_slider.py --out_dir /tmp/packed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_REPO = Path(__file__).resolve().parents[1]
_APP_REGISTRY = Path("/ml2/music/app/sliders.json")
# LM hotter than TF at the ±2 pole. TF 2.0 is the last measured underwater
# point; LM 3.1 is a working desk pot (max 4). Distortion poles at LM 2 did
# not move instrumentation far enough (acoustic still not acoustic).
TF_ENERGY_AT_2 = 1.0
LM_ENERGY_AT_2 = 3.1


def _comfy_path(weights: Path) -> Path:
    return weights.with_name(weights.stem + "_comfyui.safetensors")


def _load(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _scale_alphas(tensors: dict[str, torch.Tensor], factor: float) -> int:
    n = 0
    for key, value in list(tensors.items()):
        if not key.endswith(".alpha"):
            continue
        tensors[key] = (value.float() * factor).contiguous()
        n += 1
    return n


def _host_energy(sidecar: dict) -> float:
    rank = float(sidecar.get("rank") or 8) or 8.0
    alpha = float(sidecar.get("alpha") or 8.0)
    return alpha / rank


def pack_slider(entry: dict, root: Path, out_dir: Path) -> Path | None:
    slider_id = str(entry.get("id") or "").strip()
    components = list(entry.get("components") or [])
    if not slider_id or not components:
        return None

    merged: dict[str, torch.Tensor] = {}
    hosts: list[dict] = []
    for comp in components:
        weights = Path(str(comp.get("weights") or ""))
        if not weights.is_absolute():
            weights = root / weights
        comfy = _comfy_path(weights)
        side = weights.with_suffix(".json")
        if not comfy.exists():
            print(f"skip {slider_id}: missing {comfy.name}", file=sys.stderr)
            return None
        sidecar = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        kind = str(sidecar.get("kind") or "transformer")
        current = _host_energy(sidecar)
        target_at_1 = (TF_ENERGY_AT_2 if kind == "transformer" else LM_ENERGY_AT_2) / 2.0
        factor = target_at_1 / current if current else 1.0
        tensors = _load(comfy)
        overlap = set(merged) & set(tensors)
        if overlap:
            raise SystemExit(f"{slider_id}: key collision {sorted(overlap)[:3]}")
        n_alpha = _scale_alphas(tensors, factor)
        merged.update(tensors)
        hosts.append(
            {
                "kind": kind,
                "source": str(comfy),
                "rank": int(sidecar.get("rank") or 8),
                "alpha_before": float(sidecar.get("alpha") or 8),
                "energy_before": round(current, 6),
                "alpha_factor": round(factor, 6),
                "energy_at_strength_1": round(target_at_1, 6),
                "energy_at_strength_2": round(target_at_1 * 2.0, 6),
                "n_alpha": n_alpha,
                "n_keys": len(tensors),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{slider_id}-v6.safetensors"
    save_file(merged, str(dest), metadata={"format": "pt"})
    payload = {
        "slider": slider_id,
        "version": "v6",
        "label_minus": entry.get("label_minus"),
        "label_plus": entry.get("label_plus"),
        "description": entry.get("description"),
        "n_keys": len(merged),
        "hosts": hosts,
        "strength": {
            "0": "off",
            "1": {"transformer": TF_ENERGY_AT_2 / 2.0, "language_model": LM_ENERGY_AT_2 / 2.0},
            "2": {"transformer": TF_ENERGY_AT_2, "language_model": LM_ENERGY_AT_2},
        },
        "recommended_range": [-2.0, 2.0],
        "same_strength_on_model_and_clip": True,
        "note": (
            "One Load LoRA on MiniMax Music 3. Set MODEL strength and CLIP "
            "strength to the SAME number. The TF/LM mix is baked in; splitting "
            "them unbalances the slider. 0 is off, ±1 a mild push, ±2 the poles "
            "(transformer energy 1 / language-model energy 3.1)."
        ),
    }
    dest.with_suffix(".safetensors.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {dest.name}  {len(merged)} keys  "
        + ", ".join(f"{h['kind']} x{h['alpha_factor']:g}" for h in hosts)
    )
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_REPO / "models" / "packed-v6",
    )
    args = parser.parse_args()
    raw = json.loads(_APP_REGISTRY.read_text(encoding="utf-8"))
    root = Path(str(raw.get("root") or _REPO / "models"))
    written = 0
    for entry in raw.get("sliders") or []:
        if pack_slider(entry, root, args.out_dir) is not None:
            written += 1
    print(f"{written} sliders packed into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
