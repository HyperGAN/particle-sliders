"""Exact shared teacher geometry extracted from slider_targets.py.

Functions below retain their original source. See locked_provenance.json.
"""
from __future__ import annotations
import torch
HOLD_DIR_EPS = 1e-6

def lm_unit(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.flatten()
    return flat / flat.norm().clamp_min(1e-8)

def lm_hold_dir(
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor | None = None,
    odd_dir: torch.Tensor | None = None,
    mode: str = "raw",
) -> torch.Tensor | None:
    """Direction the hold actually penalizes.

    ``raw``: declared ê.
    ``slider``: ê_⊥ = ê − (ê·û)û. Hold cannot punch the slider name.
    ``odd``: ê_⊥ = ê − (ê·â)â. Hold cannot punch the pair-odd teacher.
    Near-zero leftover returns ``None`` (hold is off). Missing ``slider_dir``
    / ``odd_dir`` for that mode falls back to raw ê — do not invent an axis.
    """
    kind = str(mode).strip().lower()
    axis = leak_dir.flatten()
    if kind == "raw":
        out = axis
    elif kind == "slider":
        if slider_dir is None:
            out = axis
        else:
            unit = lm_unit(slider_dir)
            out = axis - (axis @ unit) * unit
    elif kind in ("odd", "pair_odd", "teacher"):
        if odd_dir is None:
            out = axis
        else:
            unit = lm_unit(odd_dir)
            out = axis - (axis @ unit) * unit
    else:
        raise ValueError(f"hold dir mode must be raw/slider/odd, got {mode!r}")
    if float(out.norm()) <= HOLD_DIR_EPS:
        return None
    return out.reshape_as(leak_dir) if out.numel() == leak_dir.numel() else out

def lm_pair_odd_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair-odd teacher with the hold axis removed.

        a = (pos − neg) / 2
        ê_⊥ = ê − (ê·û)û          # same axis ``lm_axis_hold`` uses
        â = a − (a · ê̂_⊥) ê̂_⊥
        tgt(±1) = neu ± â · target_scale

    This is the λ→∞ hold equilibrium in one step: no ``λ·D/2``
    stiffness. Subtract ``ê_⊥``, not raw ê — raw ê takes û with it.
    If ê is already parallel to û, ``ê_⊥`` vanishes and the teacher
    stays full pair-odd (hold would have been off).
    """
    axis = (pos - neg) / 2.0 * float(target_scale)
    held = lm_hold_dir(leak_dir, slider_dir=slider_dir, mode="slider")
    if held is not None:
        unit = lm_unit(held)
        axis = axis - ((axis.flatten() @ unit) * unit).view_as(axis)
    return neu + axis, neu - axis

def lm_faithful_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real caption poles with leftover ê removed from the odd part only.

        a = (pos − neg) / 2
        ê_⊥ = ê − (ê·û)û          # same leftover geometry as pair_odd_sub_e
        â = a − (a · ê̂_⊥) ê̂_⊥
        mid = ½(pos + neg)
        tgt(±1) = mid ± â · target_scale

    Midpoint stays ½(h++h−). Teacher is a real caption minus leftover
    unused, not ``t± = h0 ± a``. ``pair_odd_sub_e`` is the
    midpoint-minus-ê teacher (v15) and is not this function — it is
    this leftover odd plus ``h0`` instead of ``mid``.
    """
    plus, minus = lm_pair_odd_sub_e(
        pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
    )
    common = (pos + neg) / 2.0 - neu
    return plus + common, minus + common

def lm_blend_guard(
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> dict[str, float | bool]:
    """Is a target still nearer its own caption than the pair's midpoint?

        mid = ½(pos + neg)
        to_pole = max(‖t₊ − pos‖, ‖t₋ − neg‖)
        to_mid  = min(‖t₊ − mid‖, ‖t₋ − mid‖)
        admissible ⟺ to_pole < to_mid

    No threshold is chosen here. ``mid`` is the one point on the segment
    that is not either caption, and on a divergent pair it is a state no
    caption occupies at all — half of each song. A teacher that has drifted
    closer to ``mid`` than to the pole it claims to be is a *blend* teacher,
    and the ±1 ends of a blend sing both songs at once.

    This is a test on the target **point**, so it can be run at setup from
    the four declared captions, before a single step. It says nothing about
    the loss.
    """
    mid = 0.5 * (pos + neg)
    to_pole = max(
        float((tgt_plus - pos).norm()), float((tgt_minus - neg).norm())
    )
    to_mid = min(float((tgt_plus - mid).norm()), float((tgt_minus - mid).norm()))
    return {
        "to_pole": to_pole,
        "to_mid": to_mid,
        "admissible": bool(to_pole < to_mid),
    }

def lm_faithful_guard_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lm_faithful_sub_e`` when the blend guard admits it, else raw poles.

    ``faithful_sub_e`` is right when ``ê`` names a leftover the captions
    left unpinned and wrong when ``ê`` restates the pole difference: on
    energy-v4 the declared pair (``"Pop-punk mix, BPM 168."`` /
    ``"Ambient lullaby mix, BPM 52."``) is the same genre and BPM the poles
    move, so subtracting ``ê_⊥`` deletes the slider and the target lands
    nearer ``½(h₊+h₋)`` than the caption. :func:`lm_blend_guard` is exactly
    that condition, so this teacher subtracts ``ê_⊥`` only while what is
    left of the axis is longer than what was taken, and otherwise keeps the
    caption it was already aiming at.

    One declaration, both pair types: a yaml may keep its ``leak_*`` pair
    without that pair being able to eat the axis. The guard is per row, so
    ``lm_teachers_mixed`` can report a yaml whose rows disagree.
    """
    plus, minus = lm_faithful_sub_e(
        pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
    )
    if lm_blend_guard(plus, minus, pos, neg)["admissible"]:
        return plus, minus
    if float(target_scale) != 1.0:
        raise ValueError("--target_scale is defined for symmetric targets only")
    return pos, neg
