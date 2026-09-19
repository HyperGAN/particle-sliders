"""Bake sidecar unit_scale into LoRA `.alpha` tensors so a file drops in at 1.

A dual-host slider shares one user fader across transformer and language-model
halves. If only the TF sidecar carries unit_scale 2–4, that fader secretly runs
the transformer hotter than the LM. Fold the factor into `.alpha` (tensors and
sidecar) and set `unit_scale: 1.0` so both halves see the same multiplier.

Writes a sibling `*_unit_last.safetensors` and leaves the original alone.
Applied strength is unchanged: multiplier /= B, alpha *= B.

Do not bake when `axis_tracking_low` is set — that unit_scale is not meaningful.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

UNIT_TOLERANCE = 1e-3


def sidecar_path(weights: Path) -> Path:
    for candidate in (weights.with_suffix(".json"), Path(str(weights) + ".json")):
        if candidate.exists():
            return candidate
    return weights.with_suffix(".json")


def unit_out_path(weights: Path) -> Path:
    name = weights.name
    for tag in ("_alpha8.0_rank8_full_last.safetensors", "_last.safetensors"):
        if name.endswith(tag):
            return weights.with_name(name[: -len(tag)] + "_unit_last.safetensors")
    return weights.with_name(weights.stem + "_unit_last.safetensors")


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def bake_factor(meta: dict[str, Any]) -> float | None:
    """B to fold into `.alpha`, or None if the file is already a drop-in at 1.

    Returns None for axis_tracking_low / missing / non-positive unit_scale
    (those stay at raw multiplier 1). Raises ValueError if a real unit_scale
    is present but unusable.
    """
    if meta.get("axis_tracking_low"):
        return None
    if not (meta.get("calibrated") or meta.get("unit_scale") is not None):
        return None
    unit = _finite_positive(meta.get("unit_scale") or 1.0)
    if unit is None:
        return None
    if abs(unit - 1.0) <= UNIT_TOLERANCE:
        return None
    return unit


def bake_weights(weights: Path, *, force: bool = False) -> Path | None:
    """Write `*_unit_last.safetensors` next to `weights`. None if already unit 1."""
    weights = Path(weights)
    side = sidecar_path(weights)
    if not side.exists():
        raise FileNotFoundError(f"no sidecar next to {weights}")
    meta = json.loads(side.read_text(encoding="utf-8"))
    unit = bake_factor(meta)
    if unit is None and not force:
        return None
    if force:
        unit = _finite_positive(meta.get("unit_scale") or 1.0)
        if unit is None:
            raise ValueError(f"{weights.name}: unit_scale={meta.get('unit_scale')!r} is not a positive finite number")

    tensors = load_file(str(weights))
    baked = {}
    n_alpha = 0
    alpha_values: list[float] = []
    for key, value in tensors.items():
        if key.endswith(".alpha"):
            baked[key] = (value.float() * unit).to(value.dtype)
            alpha_values.append(float(baked[key].reshape(-1)[0]))
            n_alpha += 1
        else:
            baked[key] = value
    if n_alpha == 0:
        raise ValueError(f"{weights.name}: no `.alpha` tensors to bake")

    out = unit_out_path(weights)
    save_file(baked, str(out))

    new_alpha = float(meta.get("alpha") or 8.0) * unit
    out_meta = dict(meta)
    out_meta["alpha"] = new_alpha
    out_meta["unit_scale"] = 1.0
    out_meta["unit_scale_before"] = float(meta.get("unit_scale") or unit)
    out_meta["bake_factor"] = unit
    out_meta["baked_from"] = str(weights)
    out_meta["weights"] = str(out)
    if "checkpoint" in out_meta:
        out_meta["checkpoint"] = str(out)
    out.with_suffix(".json").write_text(json.dumps(out_meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"{weights.name}: B={unit:g}  {n_alpha} alphas "
        f"{min(alpha_values):g}..{max(alpha_values):g}  "
        f"sidecar alpha {meta.get('alpha')} -> {new_alpha:g}  -> {out.name}",
        flush=True,
    )
    return out
