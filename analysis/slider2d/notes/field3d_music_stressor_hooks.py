#!/usr/bin/env python3
"""Scaffold hooks: Music-derived Field3D ctors to merge into analysis/slider2d/field3d.py.

Idempotent applicator — run on pop-os inside the repo. Does not touch servers.
Adds lyric_span_entangle / close_live_noise / dual_arm cells if missing.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # sliders-conceptmod root
FIELD3D = REPO / "analysis/slider2d/field3d.py"

CTORS = r'''
def lyric_span_entangle_field3d(**kwargs) -> Field3D:
    """Music lyric-span entangle: leftover ê shares energy with content across staggered spans.

    Cross-axis lyric analogue (M1): heterogeneous per-row axis mix + e_on_content>0.
    """
    amps = (
        (1.05, 0.55, 0.35),
        (0.60, 1.10, 0.45),
        (0.75, 0.50, 0.90),
        (1.10, 0.85, 0.55),
        (0.90, 0.70, 0.65),
    )
    base = {
        "kind": "lyric_span_entangle",
        "rows": 5,
        "row_scales": (0.75, 0.95, 1.05, 1.2, 1.35),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.7,
        "leak": 0.55,
        "e_on_u": 0.05,
        "e_on_content": 0.45,
        "e_unused": 0.7,
    }
    base.update(kwargs)
    return Field3D(**base)


def close_live_noise_field3d(seed: int = 0, amp_noise: float = 0.04, span_noise: float = 0.08, **kwargs) -> Field3D:
    """Close-pair + live-like amp/span noise (M3/M11 Music caption jitter)."""
    # Deterministic jitter from seed (no global RNG coupling).
    j = ((seed * 37) % 11) - 5
    k = ((seed * 53) % 9) - 4
    content = 0.90 + amp_noise * (j / 5.0)
    leak = max(0.02, 0.10 + amp_noise * (k / 4.0) * 0.5)
    scale = 1.0 + span_noise * (j / 10.0)
    base = {
        "kind": "close_live_noise",
        "rows": 1,
        "row_scales": (scale,),
        "slider": 1.0,
        "content": content,
        "leak": leak,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    # Prefer cloning close_field3d geometry if available.
    try:
        close = close_field3d(seed=seed)
        for attr in ("slider", "e_on_u", "e_on_content", "e_unused", "rows"):
            if hasattr(close, attr):
                base[attr] = getattr(close, attr)
        base["rows"] = int(getattr(close, "rows", 1))
        base["row_scales"] = tuple(
            float(s) * scale for s in getattr(close, "row_scales", (1.0,))
        ) or (scale,)
        base["content"] = content
        base["leak"] = leak
        base["kind"] = "close_live_noise"
    except Exception:
        pass
    base.update(kwargs)
    return Field3D(**base)


def dual_arm_leftover_geom_field3d(**kwargs) -> Field3D:
    """Shared leftover geometry for dual-arm leftover vs listen exam cells (M4/M7/M8)."""
    base = {
        "kind": "dual_arm_leftover_geom",
        "rows": 3,
        "row_scales": (1.0, 1.1, 0.9),
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)

'''


def apply() -> None:
    text = FIELD3D.read_text()
    changed = False
    if "def lyric_span_entangle_field3d" not in text:
        marker = "CELLS_3D = {"
        idx = text.find(marker)
        if idx < 0:
            # append before __all__ if CELLS_3D already expanded
            idx = text.find("__all__ = [")
        assert idx >= 0, "no insert point in field3d.py"
        text = text[:idx] + CTORS + "\n" + text[idx:]
        changed = True
        print("inserted Music ctors")
    # Register in CELLS_3D if present as a simple dict ending unused_e
    if '"lyric_span_entangle"' not in text and "CELLS_3D = {" in text:
        needle = '"unused_e": unused_e_field3d,\n'
        if needle in text and '"lyric_span_entangle"' not in text:
            text = text.replace(
                needle,
                needle
                + '    "lyric_span_entangle": lyric_span_entangle_field3d,\n'
                + '    "close_live_noise": close_live_noise_field3d,\n'
                + '    "dual_arm_leftover_geom": dual_arm_leftover_geom_field3d,\n',
                1,
            )
            changed = True
            print("patched CELLS_3D")
    if '"lyric_span_entangle_field3d"' not in text and "__all__ = [" in text:
        needle = '    "unused_e_field3d",\n'
        if needle in text:
            text = text.replace(
                needle,
                needle
                + '    "lyric_span_entangle_field3d",\n'
                + '    "close_live_noise_field3d",\n'
                + '    "dual_arm_leftover_geom_field3d",\n',
                1,
            )
            changed = True
            print("patched __all__")
    if changed:
        FIELD3D.write_text(text)
        print("wrote", FIELD3D)
    else:
        print("field3d.py already has Music hooks (or nothing to do)")


if __name__ == "__main__":
    apply()
