"""Music Arm B single-source winning shape (Strategy E gates).

Arm B = ``#94`` ParticleGAN-faithful RpGAN + ``b_cap``, leftover-gated —
the KEEP / DR✓ recipe from the Slider CPU / Field3D selection work
(``locked_shared`` / hold_ablation). This module is the Music-trainer
side of that handoff: one dict, one fail-closed checker, no GPU, no
torch.

Music transfer (differs from the Field3D demo pin in exactly one place:
cover 1.0 here vs 1.5 on the demo)::

    lm_target=faithful_guard_e · adv_arch=mlp · fm_weight=0
    b_cap=1 · adv_reg_kappa=1 · parts=0
    pole_weight=1 · cover_weight=1 · vicreg_weight=0

The live trainer default stays ``v9`` / ``hidden``; Arm B is explicit
(``--lm_target faithful_guard_e``) and checked by ``check_music_arm_b``.
``--require_arm_b`` turns a mismatch report into a refusal at parse
time, before any GPU work.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for the Music Arm B argv shape. Literals are also
# repeated in tests/test_music_arm_b_gates.py so drift in either fails.
MUSIC_ARM_B: dict[str, Any] = {
    "lm_target": "faithful_guard_e",
    "adv_arch": "mlp",
    "fm_weight": 0.0,
    "b_cap": 1.0,
    "adv_reg_kappa": 1.0,
    "parts": 0,
    "pole_weight": 1.0,
    "cover_weight": 1.0,
    "vicreg_weight": 0.0,
}

_ADV_KNOBS = (
    "adv_arch",
    "fm_weight",
    "b_cap",
    "adv_reg_kappa",
    "parts",
    "pole_weight",
    "cover_weight",
    "vicreg_weight",
)


def _approx_eq(a: Any, b: Any, *, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def check_music_arm_b(args: Any) -> list[str]:
    """Return human-readable Arm B mismatch strings (empty = Arm B).

    Fail-closed: anything that is not the winning shape is reported —
    the caller decides whether to stop (``--require_arm_b``) or just
    log the drift before spend. Never raises, never mutates.
    """
    bad: list[str] = []
    get = getattr(args, "lm_target", None)
    if get != MUSIC_ARM_B["lm_target"]:
        bad.append(f"lm_target={get!r} want {MUSIC_ARM_B['lm_target']!r} (Arm B teacher)")
    arch = getattr(args, "adv_arch", None)
    if arch != MUSIC_ARM_B["adv_arch"]:
        bad.append(f"adv_arch={arch!r} want {MUSIC_ARM_B['adv_arch']!r} (Arm B critic)")
    if arch == "tx" and get == "faithful_guard_e":
        bad.append("adv_arch=tx with lm_target=faithful_guard_e is dual-arm incompatible")
    fm = getattr(args, "fm_weight", None)
    if not _approx_eq(fm, MUSIC_ARM_B["fm_weight"]):
        bad.append(f"fm_weight={fm} want 0 (raw FM under b_cap is the false path)")
    b_cap = getattr(args, "b_cap", None)
    if not _approx_eq(b_cap, MUSIC_ARM_B["b_cap"]):
        bad.append(f"b_cap={b_cap} want 1")
    kappa = getattr(args, "adv_reg_kappa", None)
    if not _approx_eq(kappa, MUSIC_ARM_B["adv_reg_kappa"]):
        bad.append(f"adv_reg_kappa={kappa} want 1")
    parts = getattr(args, "parts", None)
    try:
        n_parts = int(parts)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        n_parts = -1
    if n_parts != MUSIC_ARM_B["parts"]:
        bad.append(f"parts={parts} want 0 (Music Arm B)")
    for knob in ("pole_weight", "cover_weight"):
        val = getattr(args, knob, None)
        if not _approx_eq(val, MUSIC_ARM_B[knob]):
            bad.append(f"{knob}={val} want 1 (Music transfer)")
    vicreg = getattr(args, "vicreg_weight", None)
    if not _approx_eq(vicreg, MUSIC_ARM_B["vicreg_weight"]):
        scope = "parts<=1" if n_parts <= 1 else f"parts={parts}"
        bad.append(f"vicreg_weight={vicreg} want 0 ({scope})")
    return bad


def assert_music_arm_b(args: Any) -> None:
    """Raise AssertionError unless ``args`` is exactly the Arm B shape."""
    bad = check_music_arm_b(args)
    if bad:
        raise AssertionError(
            "Music Arm B gate failed (drifted recipe — stop before spend): "
            + "; ".join(bad)
            + f" | argv: {format_arm_b_argv(args)}"
        )


def format_arm_b_argv(args: Any) -> str:
    """One-line adv-related argv for the handoff's verify-before-train step."""
    bits = [f"lm_target={getattr(args, 'lm_target', None)}"]
    for knob in _ADV_KNOBS:
        bits.append(f"{knob}={getattr(args, knob, None)}")
    shown = " ".join(bits)
    return shown.replace("adv_reg_kappa=", "kappa=")


__all__ = [
    "MUSIC_ARM_B",
    "assert_music_arm_b",
    "check_music_arm_b",
    "format_arm_b_argv",
]
