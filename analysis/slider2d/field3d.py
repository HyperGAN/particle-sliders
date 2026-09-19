"""R³ leftover field + PairField-style multi-row exam cells (2D→3D).

Pedagogy that should transfer from the locked 2D recipe (Fire #8):
``b_cap``, ``cover_weight``, leftover gating (``faithful_guard_e``),
particle vs residual. No 3D-only hacks.

Orthonormal axes (ambient dim = 3 + rows)::

    0              û    concept (intended slider)
    1              ĉ    content / intended-off-û (moves with the poles)
    2              ê    leftover unused attribute
    3 .. 3+rows-1  l̂_r  written lyric per row

Poles (per row with scale ``k``)::

    h0 = lyric · l̂_r
    a  = k · (slider·û + content·ĉ + leak·ê)
    h± = h0 ± a

i.e. pure ``neu ± a`` (no common ŝ). Sheet readout is optional; primary
smoke is leftover/residual metrics via ``score_adv_field3d``, which reuses
``fit_adv`` by duck-typing like ``SheetField``.

PairField-style exam cells (Fire #12) add multi-row ``row_scales`` and a
``declared_e`` composition (``e_on_u`` / ``e_on_content`` / ``e_unused``)
mirroring divergent / close / unused_e — so ``faithful_guard_e`` can refuse
when ê restates the axis (divergent) and subtract when ê is unused.

CPU only. Does not change the live trainer default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.adv import AdvConfig
from analysis.slider2d.field import cosine
from conceptmod.textsliders.slider_targets import (
    leftover_bipolar,
    lm_faithful_guard_e,
    lm_faithful_sub_e_if_unused,
    lm_hold_dir,
)


@dataclass(frozen=True)
class Field3D:
    """R³ leftover toy (+ lyric rows). SheetField-compatible duck type."""

    kind: str = "leftover"
    rows: int = 1
    slider: float = 1.0
    content: float = 0.55
    leak: float = 0.45
    lyric: float = 1.0
    row_scales: tuple[float, ...] = (1.0,)
    # Optional per-row (slider, content, leak) amplitudes. When set, odd(row)
    # uses these instead of scalar*row_scales — true cross-axis multipair
    # (different rows live on different R3 axes). Fire #13 transfer stress.
    row_amps: tuple[tuple[float, float, float], ...] | None = None
    # PairField-style declared ê composition (teacher leak_dir). Defaults
    # are all-zero so legacy leftover cells keep ``declared_e = ê if leak>0``.
    e_on_u: float = 0.0
    e_on_content: float = 0.0
    e_unused: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.rows) < 1:
            raise ValueError("rows must be ≥ 1, got %r" % (self.rows,))
        if len(self.row_scales) < int(self.rows):
            raise ValueError("row_scales must cover every row")
        if float(self.slider) <= 0.0:
            raise ValueError("slider must be > 0")
        if self.row_amps is not None and len(self.row_amps) < int(self.rows):
            raise ValueError("row_amps must cover every row")

    @property
    def dim(self) -> int:
        return 3 + int(self.rows)

    def _basis(self, index: int) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[index] = 1.0
        return out

    def short_u(self) -> torch.Tensor:
        return self._basis(0)

    def content_dir(self) -> torch.Tensor:
        return self._basis(1)

    def leak_e(self) -> torch.Tensor:
        return self._basis(2)

    def lyric_dir(self, row: int = 0) -> torch.Tensor:
        return self._basis(3 + int(row))

    def odd(self, row: int = 0) -> torch.Tensor:
        scale = float(self.row_scales[int(row)])
        if self.row_amps is not None:
            su, sc, se = self.row_amps[int(row)]
            return scale * (
                float(su) * self.short_u()
                + float(sc) * self.content_dir()
                + float(se) * self.leak_e()
            )
        return scale * (
            float(self.slider) * self.short_u()
            + float(self.content) * self.content_dir()
            + float(self.leak) * self.leak_e()
        )

    def poles(self, row: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.lyric_dir(row)
        a = self.odd(row)
        return neu + a, neu - a, neu

    def probe_cos(self, row: int = 0) -> float:
        pos, neg, neu = self.poles(row)
        return cosine(pos - neu, neg - neu)

    def declared_e(self) -> torch.Tensor | None:
        """Yaml ``leak_*`` stand-in, or legacy leftover ê when ``leak>0``.

        PairField cells set ``e_on_*`` / ``e_unused`` explicitly. Legacy
        single-row leftover (Fire #10/#11) leaves them at 0 and relies on
        ``leak`` amplitude → ``ê``.
        """
        composed = (
            float(self.e_on_u) * self.short_u()
            + float(self.e_on_content) * self.content_dir()
            + float(self.e_unused) * self.leak_e()
        )
        if float(composed.norm()) > 1e-8:
            return composed
        if float(self.leak) <= 1e-8:
            return None
        return self.leak_e()



# Music parts0 close-family kinds that knife under n_particles==1 (Fire #21/#22).
MUSIC_CLOSE_POSTURE_KINDS = frozenset(
    {"close", "close_live_noise", "tiny_slider_dom", "close_with_leak"}
)


def music_close_posture_warn(
    cfg_or_n_particles: "AdvConfig | int | None",
    cell_kind: str | None = None,
    *,
    kind: str | None = None,
) -> str | None:
    """Return a warning when Music parts0 proxy (n_particles==1) hits a close knife cell.

    Fire #21 / close_seed0_deep: f3d_close / close_live_noise under n=1 seed-flake
    (û undershoot). Harden without changing locked defaults: prefer
    ``n_particles>=2`` OR a multi-seed gate; optional candidate
    ``vicreg_weight=0`` @ n=1 (do not silent-flip defaults).

    Pure helper — does not mutate cfg / train loss. Returns ``None`` when
    posture is safe or cell is outside the close family.
    """
    kind_s = str(cell_kind if cell_kind is not None else (kind or "")).strip()
    if kind_s not in MUSIC_CLOSE_POSTURE_KINDS:
        return None
    if cfg_or_n_particles is None:
        return None
    if isinstance(cfg_or_n_particles, int):
        n = int(cfg_or_n_particles)
    else:
        n = int(getattr(cfg_or_n_particles, "n_particles", -1))
    if n != 1:
        return None
    return (
        "Music-posture warn: cell=%s with n_particles=1 (Music --parts 0 proxy) "
        "is knife-prone (Fire #21 close seed flake / û undershoot). "
        "Harden: prefer n_particles>=2 OR multi-seed gate; "
        "optional candidate vicreg_weight=0@n=1 (do not silent-flip locked defaults)."
        % (kind_s,)
    )


def leftover_field3d(**kwargs) -> Field3D:
    """Default leftover R³ geometry (û + content + ê)."""
    return Field3D(**kwargs)


def divergent_field3d(**kwargs) -> Field3D:
    """PairField divergent analogue: ê restates content (energy-v4 lesson).

    Large content (track stand-in), no unused ê in ``a``, declared ê points
    along content (+ small û). ``faithful_guard_e`` should refuse subtract.
    """
    base = {
        "kind": "divergent",
        "rows": 3,
        "row_scales": (1.0, 0.92, 1.08),
        "slider": 0.90,
        "content": 2.00,
        "leak": 0.0,
        "e_on_u": 0.30,
        "e_on_content": 2.00,
        "e_unused": 0.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def close_field3d(**kwargs) -> Field3D:
    """PairField close analogue: one song, small slider, content = delivery.

    Music parts0 proxy note (Fire #21 / harden bites 2026-09-09): under
    ``n_particles=1`` seed0 undershoots û (5/6 knife). Harden postures
    (do not change locked defaults): ``n_particles>=2``, or multi-seed gate,
    or ``vicreg_weight=0`` at n=1 (candidate). particle_l2/steps do not fix.
    """
    base = {
        "kind": "close",
        "rows": 3,
        "row_scales": (1.0, 0.92, 1.08),
        "slider": 0.12,
        "content": 1.00,
        "leak": 0.0,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 0.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def unused_e_field3d(**kwargs) -> Field3D:
    """PairField unused_e analogue: unpinned ê inside ``a``, declared ê names it."""
    base = {
        "kind": "unused_e",
        "rows": 3,
        "row_scales": (1.0, 0.92, 1.08),
        "slider": 1.00,
        "content": 0.35,
        "leak": 0.45,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)



def cross_axis_rows_field3d(**kwargs) -> Field3D:
    """True cross-axis multipair: each lyric row lives on a different R3 mix."""
    amps = ((1.2, 0.15, 0.1), (0.35, 1.4, 0.1), (0.4, 0.2, 1.1), (0.9, 0.7, 0.65))
    base = {
        "kind": 'cross_axis_rows',
        "rows": 4,
        "row_scales": (1.0, 1.0, 1.0, 1.0),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def axis_u_primary_field3d(**kwargs) -> Field3D:
    """Control: all rows u-primary (same axis family)."""
    amps = ((1.1, 0.2, 0.1), (1.0, 0.25, 0.12), (1.15, 0.18, 0.08), (0.95, 0.22, 0.15))
    base = {
        "kind": 'axis_u_primary',
        "rows": 4,
        "row_scales": (1.0, 1.0, 1.0, 1.0),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.2,
        "leak": 0.1,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def axis_content_primary_field3d(**kwargs) -> Field3D:
    """Pairs mostly along content; declared e unused."""
    amps = ((0.4, 1.5, 0.08), (0.35, 1.3, 0.1), (0.45, 1.6, 0.05), (0.3, 1.4, 0.12))
    base = {
        "kind": 'axis_content_primary',
        "rows": 4,
        "row_scales": (1.0, 1.0, 1.0, 1.0),
        "row_amps": amps,
        "slider": 0.4,
        "content": 1.5,
        "leak": 0.08,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def axis_leak_primary_field3d(**kwargs) -> Field3D:
    """Pairs mostly along unused e; leftover gate must suppress leak."""
    amps = ((0.55, 0.15, 1.2), (0.5, 0.18, 1.1), (0.6, 0.12, 1.3), (0.45, 0.2, 1.0))
    base = {
        "kind": 'axis_leak_primary',
        "rows": 4,
        "row_scales": (1.0, 1.0, 1.0, 1.0),
        "row_amps": amps,
        "slider": 0.55,
        "content": 0.15,
        "leak": 1.2,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def cross_axis_span_sample_field3d(**kwargs) -> Field3D:
    """Live-like span sampling: staggered scales + heterogeneous axis mix."""
    amps = ((1.0, 0.4, 0.25), (0.55, 1.1, 0.35), (0.7, 0.45, 0.95), (1.15, 0.8, 0.55), (0.85, 0.65, 0.7))
    base = {
        "kind": 'cross_axis_span_sample',
        "rows": 5,
        "row_scales": (0.7, 0.9, 1.05, 1.2, 1.4),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def cross_axis_mismatch_declare_field3d(**kwargs) -> Field3D:
    """Cross-axis rows + declared_e points at content (YAML lie)."""
    amps = ((1.1, 0.2, 0.3), (0.4, 1.2, 0.25), (0.5, 0.3, 1.0), (0.9, 0.7, 0.6))
    base = {
        "kind": 'cross_axis_mismatch_declare',
        "rows": 4,
        "row_scales": (1.0, 1.0, 1.0, 1.0),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.15,
        "e_on_content": 1.6,
        "e_unused": 0.1,
    }
    base.update(kwargs)
    return Field3D(**base)


def lyric_span_entangle_field3d(**kwargs) -> Field3D:
    """Music lyric-span entangle: leftover ê shares energy with content across staggered spans.

    Cross-axis lyric analogue (M1): heterogeneous per-row axis mix + e_on_content>0.

    Fire #20 hard boundary under locked shared AdvResidual: multi-row coverage
    stays 0/5 (exam_pass False via pass_multi_row) even when row0 u_kept/exam
    look strong. Heterogeneous row_amps alone is enough; e_on_content worsens
    leak. cover/n/steps probes did not recover — document, do not soften recipe.
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



def tiny_slider_dom_field3d(**kwargs) -> Field3D:
    """M13: harder close — tinier û, content-dominant delivery (Music close flake proxy)."""
    base = {
        "kind": "tiny_slider_dom",
        "rows": 3,
        "row_scales": (1.0, 0.92, 1.08),
        "slider": 0.05,
        "content": 1.20,
        "leak": 0.0,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 0.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def e_on_u_declare_lie_field3d(**kwargs) -> Field3D:
    """M14: declared ê restates û (YAML amplitude lie on concept axis)."""
    base = {
        "kind": "e_on_u_declare_lie",
        "rows": 3,
        "row_scales": (1.0, 1.05, 0.95),
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.35,
        "e_on_u": 1.50,
        "e_on_content": 0.10,
        "e_unused": 0.10,
    }
    base.update(kwargs)
    return Field3D(**base)


def prefix_shared_proxy_field3d(**kwargs) -> Field3D:
    """M15: shared strong neu/lyric + small odd — whole-prefix hold proxy (grit risk)."""
    base = {
        "kind": "prefix_shared_proxy",
        "rows": 4,
        "row_scales": (0.9, 1.0, 1.1, 1.2),
        "lyric": 1.4,
        "slider": 0.35,
        "content": 0.85,
        "leak": 0.25,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def scale_stagger_homo_field3d(**kwargs) -> Field3D:
    """M16: homogeneous axis mix + staggered scales only (mild multi-row Music span)."""
    amps = tuple((1.0, 0.55, 0.40) for _ in range(5))
    base = {
        "kind": "scale_stagger_homo",
        "rows": 5,
        "row_scales": (0.7, 0.9, 1.05, 1.25, 1.45),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.40,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def roles_split_proxy_field3d(**kwargs) -> Field3D:
    """M17: half rows û-primary, half content-primary (role-split UNI proxy; milder cross-axis)."""
    amps = (
        (1.15, 0.25, 0.10),
        (1.05, 0.30, 0.12),
        (0.35, 1.35, 0.10),
        (0.40, 1.25, 0.15),
    )
    base = {
        "kind": "roles_split_proxy",
        "rows": 4,
        "row_scales": (1.0, 1.0, 1.0, 1.0),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.20,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def close_with_leak_field3d(**kwargs) -> Field3D:
    """M18: close delivery + small unused ê (Music caption with weak unused attribute)."""
    base = {
        "kind": "close_with_leak",
        "rows": 3,
        "row_scales": (1.0, 0.92, 1.08),
        "slider": 0.12,
        "content": 1.00,
        "leak": 0.18,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def grit_content_dom_field3d(**kwargs) -> Field3D:
    """M19: grit/distortion proxy — content dominates, modest slider, unused ê present."""
    base = {
        "kind": "grit_content_dom",
        "rows": 3,
        "row_scales": (1.0, 0.95, 1.05),
        "slider": 0.40,
        "content": 1.80,
        "leak": 0.30,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)



def amp_lie_leftover_declare_field3d(**kwargs) -> Field3D:
    """M20: homogeneous leftover + declared YAML content-axis amplitude lie.

    Distinct from ``e_on_u_declare_lie`` (û restatement) and from
    ``cross_axis_mismatch_declare`` (hetero row_amps). Bad leak YAML points
    declared ê at content while geometry stays leftover-family.
    """
    base = {
        "kind": "amp_lie_leftover_declare",
        "rows": 3,
        "row_scales": (1.0, 1.1, 0.9),
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.10,
        "e_on_content": 1.25,
        "e_unused": 0.15,
    }
    base.update(kwargs)
    return Field3D(**base)


def hold_e_lyric_mix_field3d(**kwargs) -> Field3D:
    """M21: content↔leftover mix ratios mimicking hold-ê lyric pool.

    Homogeneous leftover family with elevated content:leak and mild
    e_on_content so declared ê shares lyric-pool energy with content —
    no hetero row_amps (distinct from lyric_span_entangle).
    """
    base = {
        "kind": "hold_e_lyric_mix",
        "rows": 4,
        "row_scales": (0.9, 1.0, 1.1, 1.2),
        "slider": 1.0,
        "content": 0.85,
        "leak": 0.65,
        "e_on_u": 0.0,
        "e_on_content": 0.35,
        "e_unused": 0.85,
    }
    base.update(kwargs)
    return Field3D(**base)


def stagger_mild_cross_field3d(**kwargs) -> Field3D:
    """M22: staggered scales + mild cross-axis (between homo and full cross_axis).

    Soft Music multipair: û-leaning mixes with modest ĉ/ê drift — not
    scale_stagger_homo (identical amps) and not roles_split / cross_axis hard.
    """
    amps = (
        (1.05, 0.30, 0.20),
        (0.95, 0.45, 0.25),
        (1.00, 0.35, 0.40),
        (0.90, 0.50, 0.30),
    )
    base = {
        "kind": "stagger_mild_cross",
        "rows": 4,
        "row_scales": (0.80, 1.00, 1.15, 1.30),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.40,
        "leak": 0.30,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def multipair_corr_seed_field3d(seed: int = 0, **kwargs) -> Field3D:
    """M23: multi-pair R³ with seed-correlated row_amps (Music multi-caption seeds).

    Correlated drift across pairs from one seed basin — distinct from
    close_live_noise (close jitter) and full cross_axis_rows hard fail.
    """
    j = ((int(seed) * 41) % 13) - 6
    k = ((int(seed) * 17) % 11) - 5
    du = 0.08 * (j / 6.0)
    dc = 0.10 * (k / 5.0)
    de = 0.06 * ((j - k) / 11.0)
    amps = (
        (1.00 + du, 0.40 + dc, 0.35 + de),
        (0.95 + du, 0.50 + dc, 0.30 + de),
        (1.05 + du, 0.35 + dc, 0.40 + de),
        (0.90 + du, 0.45 + dc, 0.38 + de),
    )
    base = {
        "kind": "multipair_corr_seed",
        "rows": 4,
        "row_scales": (0.85, 1.0, 1.1, 1.25),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.45,
        "leak": 0.35,
        "e_on_u": 0.0,
        "e_on_content": 0.05,
        "e_unused": 0.95,
        "seed": int(seed),
    }
    base.update(kwargs)
    return Field3D(**base)



def content_leak_flip_rows_field3d(**kwargs) -> Field3D:
    """M24: û-primary rows with content↔leak dominance flip (Music mid-caption attr swap).

    Concept axis stays primary; content and leak swap which attribute dominates
    across rows — distinct from roles_split (û vs content primary), full
    cross_axis_rows, and stagger_mild_cross (soft drift without flip).
    """
    amps = (
        (1.05, 0.70, 0.18),
        (1.00, 0.18, 0.70),
        (1.05, 0.65, 0.22),
        (0.95, 0.22, 0.65),
    )
    base = {
        "kind": "content_leak_flip_rows",
        "rows": 4,
        "row_scales": (0.95, 1.0, 1.05, 1.1),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.45,
        "leak": 0.45,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def lyric_neu_heavy_gate_field3d(**kwargs) -> Field3D:
    """M25: heavy shared lyric neu + modest û + unused ê (lyric-hold vs leftover gate).

    Music lyric-token hold competing with leftover gate — distinct from
    prefix_shared_proxy (grit/content-heavy prefix) and hold_e_lyric_mix
    (content↔leak pool mix with e_on_content).
    """
    base = {
        "kind": "lyric_neu_heavy_gate",
        "rows": 3,
        "row_scales": (0.95, 1.0, 1.08),
        "lyric": 2.2,
        "slider": 0.55,
        "content": 0.30,
        "leak": 0.55,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def declare_split_three_field3d(**kwargs) -> Field3D:
    """M26: declared ê split across û/content/unused (ambiguous Music YAML attrs).

    Balanced e_on_u / e_on_content / e_unused — not a hard amp lie (M14/M20)
    and not cross_axis_mismatch_declare (hetero row_amps + e_on_content blowup).
    """
    base = {
        "kind": "declare_split_three",
        "rows": 3,
        "row_scales": (1.0, 1.05, 0.95),
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.45,
        "e_on_content": 0.45,
        "e_unused": 0.45,
    }
    base.update(kwargs)
    return Field3D(**base)


def scale_descent_homo_field3d(**kwargs) -> Field3D:
    """M27: homogeneous axis mix + descending row_scales (Music outro/traj reverse).

    Same homo amps as scale_stagger_homo but scales descend 1.45→0.7 — proxies
    traj_frac / outro-weighted lyric spans. Distinct from M16 ascending stagger.
    """
    amps = tuple((1.0, 0.55, 0.40) for _ in range(5))
    base = {
        "kind": "scale_descent_homo",
        "rows": 5,
        "row_scales": (1.45, 1.25, 1.05, 0.90, 0.70),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.40,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)




def guard_refuse_hot_eoc_field3d(**kwargs) -> Field3D:
    """M28: hot e_on_content so blend guard refuses subtract → raw-pole leak.

    Same content/leak pool as hold_e_lyric_mix (M21) but eoc elevated into the
    refuse band (M21 mechanistic dig: eoc≥~0.7 admits=False). Distinct from
    M21 (guard admits, ê-floor lr≈0.205) and M20 amp_lie (YAML content blowup).
    """
    base = {
        "kind": "guard_refuse_hot_eoc",
        "rows": 4,
        "row_scales": (0.9, 1.0, 1.1, 1.2),
        "slider": 1.0,
        "content": 0.85,
        "leak": 0.65,
        "e_on_u": 0.0,
        "e_on_content": 0.85,
        "e_unused": 0.85,
    }
    base.update(kwargs)
    return Field3D(**base)


def content_cascade_rows_field3d(**kwargs) -> Field3D:
    """M29: û-stable rows with ascending content dominance (Music verse→chorus).

    Slider stays primary; content grows 0.25→1.15 across rows with leak pinned.
    Distinct from content_leak_flip (swap), cross_axis_rows (full hetero), and
    roles_split_proxy (û vs content primary flip).
    """
    amps = (
        (1.05, 0.25, 0.35),
        (1.00, 0.55, 0.35),
        (1.00, 0.85, 0.35),
        (0.95, 1.15, 0.40),
    )
    base = {
        "kind": "content_cascade_rows",
        "rows": 4,
        "row_scales": (0.90, 1.00, 1.10, 1.20),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.35,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def eoc_threshold_edge_field3d(**kwargs) -> Field3D:
    """M30: M21 pool with eoc just over analytic ê-floor (lr≈0.202).

    Knife/edge stressor: eoc=0.33 sits in the fail band start (M21 dig:
    eoc=0.32 pass / 0.33 fail analytically). Distinct from M21 default 0.35
    and M28 guard-refuse hot.
    """
    base = {
        "kind": "eoc_threshold_edge",
        "rows": 4,
        "row_scales": (0.9, 1.0, 1.1, 1.2),
        "slider": 1.0,
        "content": 0.85,
        "leak": 0.65,
        "e_on_u": 0.0,
        "e_on_content": 0.33,
        "e_unused": 0.85,
    }
    base.update(kwargs)
    return Field3D(**base)


def leftover_hot_eoc_declare_field3d(**kwargs) -> Field3D:
    """M31: leftover-family amps + hot eoc that still admits (ê-floor bite).

    content/leak like leftover (0.55/0.45) with eoc=1.15, e_unused=0.45 —
    guard admits but teacher ê floor >0.20 (M21 dig leftover-hot probe).
    Distinct from M20 amp_lie (eoc=1.25, e_unused=0.15 YAML blowup) and M28
    (guard refuse).
    """
    base = {
        "kind": "leftover_hot_eoc_declare",
        "rows": 3,
        "row_scales": (0.95, 1.0, 1.08),
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.05,
        "e_on_content": 1.15,
        "e_unused": 0.45,
    }
    base.update(kwargs)
    return Field3D(**base)


CELLS_3D = {
    "divergent": divergent_field3d,
    "close": close_field3d,
    "unused_e": unused_e_field3d,
    "cross_axis_rows": cross_axis_rows_field3d,
    "axis_u_primary": axis_u_primary_field3d,
    "axis_content_primary": axis_content_primary_field3d,
    "axis_leak_primary": axis_leak_primary_field3d,
    "cross_axis_span_sample": cross_axis_span_sample_field3d,
    "cross_axis_mismatch_declare": cross_axis_mismatch_declare_field3d,
    "lyric_span_entangle": lyric_span_entangle_field3d,
    "close_live_noise": close_live_noise_field3d,
    "dual_arm_leftover_geom": dual_arm_leftover_geom_field3d,
    "tiny_slider_dom": tiny_slider_dom_field3d,
    "e_on_u_declare_lie": e_on_u_declare_lie_field3d,
    "prefix_shared_proxy": prefix_shared_proxy_field3d,
    "scale_stagger_homo": scale_stagger_homo_field3d,
    "roles_split_proxy": roles_split_proxy_field3d,
    "close_with_leak": close_with_leak_field3d,
    "grit_content_dom": grit_content_dom_field3d,
    "amp_lie_leftover_declare": amp_lie_leftover_declare_field3d,
    "hold_e_lyric_mix": hold_e_lyric_mix_field3d,
    "stagger_mild_cross": stagger_mild_cross_field3d,
    "multipair_corr_seed": multipair_corr_seed_field3d,
    "content_leak_flip_rows": content_leak_flip_rows_field3d,
    "lyric_neu_heavy_gate": lyric_neu_heavy_gate_field3d,
    "declare_split_three": declare_split_three_field3d,
    "scale_descent_homo": scale_descent_homo_field3d,
    "guard_refuse_hot_eoc": guard_refuse_hot_eoc_field3d,
    "content_cascade_rows": content_cascade_rows_field3d,
    "eoc_threshold_edge": eoc_threshold_edge_field3d,
    "leftover_hot_eoc_declare": leftover_hot_eoc_declare_field3d,
}


def field3d_teacher_points(
    field: Field3D,
    row: int,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same leftover-gate teachers the sheet/exam paths use."""
    pos, neg, neu = field.poles(row)
    mode = str(teacher).strip().lower()
    if mode == "faithful":
        return pos, neg
    if mode == "faithful_guard_e":
        if leak_dir is None:
            return pos, neg
        return lm_faithful_guard_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful_sub_e_if_unused":
        return lm_faithful_sub_e_if_unused(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
    if mode == "pair_odd":
        a = field.odd(row)
        return neu + a, neu - a
    raise ValueError("unsupported Field3D teacher %r" % (teacher,))


def _row_coverage(
    residual,
    field: Field3D,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None,
) -> list[dict]:
    """Per-row pole coverage after fit (PairField multi-row exam analogue)."""
    out = []
    for r in range(int(field.rows)):
        t_plus, t_minus = field3d_teacher_points(
            field, r, teacher=teacher, leak_dir=leak_dir
        )
        neu = field.poles(r)[2]
        pred_p = neu + residual.delta(1.0)
        pred_m = neu + residual.delta(-1.0)
        err_p = float(
            (pred_p - t_plus).norm() / t_plus.norm().clamp_min(1e-8)
        )
        err_m = float(
            (pred_m - t_minus).norm() / t_minus.norm().clamp_min(1e-8)
        )
        out.append(
            {
                "row": int(r),
                "scale": float(field.row_scales[r]),
                "pole_rel_err_plus": err_p,
                "pole_rel_err_minus": err_m,
                "covered": bool(err_p <= 0.20 and err_m <= 0.20),
                "pole_cos_plus": cosine(pred_p - neu, t_plus - neu),
                "pole_cos_minus": cosine(pred_m - neu, t_minus - neu),
            }
        )
    return out


def score_adv_field3d(
    field: Field3D | None = None,
    *,
    teacher: str = "faithful_guard_e",
    leak_dir: torch.Tensor | None = None,
    cfg: AdvConfig | None = None,
    name: str = "field3d_leftover",
) -> dict:
    """Fit locked-recipe residual on Field3D; report leftover + residual metrics.

    Reuses ``fit_adv`` (gan.py registers Field3D like SheetField). No sheet
    readout — this cell has no ŝ; the transfer question is leftover gating
    + cover/b_cap/particles on R³ geometry.
    """
    # Local import avoids a circular import at module load (gan imports us).
    from analysis.slider2d.gan import DEFAULT_TEACHER, fit_adv

    field = field if field is not None else leftover_field3d()
    cfg = cfg or AdvConfig()
    posture_warn = music_close_posture_warn(cfg, cell_kind=field.kind)
    if leak_dir is None:
        leak_dir = field.declared_e()
    teacher = teacher or DEFAULT_TEACHER
    residual, stats = fit_adv(field, teacher=teacher, leak_dir=leak_dir, cfg=cfg)
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    leftover = leftover_bipolar(d_plus, d_minus)
    on_u = float(d_plus @ field.short_u())
    on_c = float(d_plus @ field.content_dir())
    on_e = float(d_plus @ field.leak_e())
    a_u = float(a @ field.short_u())
    a_c = float(a @ field.content_dir())
    a_e = float(a @ field.leak_e())
    u_kept = on_u / (a_u + 1e-8)
    content_kept = on_c / (a_c + 1e-8) if abs(a_c) > 1e-8 else 0.0
    leak_ratio = abs(on_e) / (abs(on_u) + 1e-8)
    hold = (
        lm_hold_dir(leak_dir, slider_dir=field.short_u(), mode="slider")
        if leak_dir is not None
        else None
    )
    pass_u = u_kept >= 0.85
    pass_content = content_kept >= 0.75 if abs(a_c) > 1e-8 else True
    pass_leak = leak_ratio <= 0.20
    row_cov = _row_coverage(residual, field, teacher=teacher, leak_dir=leak_dir)
    rows_covered = sum(1 for rc in row_cov if rc["covered"])
    out = {
        "name": name,
        "cell": field.kind,
        "dim": int(field.dim),
        "rows": int(field.rows),
        "teacher": teacher,
        "seed": int(cfg.seed),
        "steps": int(cfg.steps),
        "cover_weight": float(cfg.cover_weight),
        "b_cap": float(cfg.b_cap),
        "n_particles": int(cfg.n_particles),
        "particle_l2": float(cfg.particle_l2),
        "fm_weight": float(cfg.fm_weight),
        "probe_cos": field.probe_cos(0),
        "pair_odd_cos": cosine(d_plus, a),
        "collapse": cosine(d_plus, d_minus),
        "pole_cos": cosine(d_plus, pos - neu),
        "u_kept": float(u_kept),
        "content_kept": float(content_kept),
        "leak_ratio": float(leak_ratio),
        "on_u": float(on_u),
        "on_content": float(on_c),
        "on_e": float(on_e),
        "a_u": float(a_u),
        "a_content": float(a_c),
        "a_e": float(a_e),
        "residual_norm": float(d_plus.norm()),
        "odd_norm": float(residual.w_odd.norm()),
        "even_norm": float(residual.w_even.norm()),
        "leak_frac": leftover["leak_frac"],
        "same_dir": leftover["same_dir"],
        "hold_norm": float(hold.norm()) if hold is not None else 0.0,
        "pass_u": bool(pass_u),
        "pass_content": bool(pass_content),
        "pass_leak": bool(pass_leak),
        "pass": bool(pass_u and pass_content and pass_leak),
        "rows_covered": int(rows_covered),
        "rows_total": int(field.rows),
        "all_rows_covered": bool(rows_covered == int(field.rows)),
        "row_coverage": row_cov,
        "music_close_posture_warn": posture_warn,
        **{k: v for k, v in stats.items() if k != "log"},
    }
    return out


def score_adv_field3d_exam(
    field: Field3D | None = None,
    *,
    teacher: str = "faithful_guard_e",
    leak_dir: torch.Tensor | None = None,
    cfg: AdvConfig | None = None,
    name: str = "field3d_pairfield_exam",
) -> dict:
    """PairField-style exam listen on multi-row Field3D (no token rollout).

    Transfers the exam *gates* (continuation / swing / leftover) onto R³
    residual metrics + multi-row coverage:

    - continuation ≈ ``u_kept`` (and ``content_kept`` when content>0)
    - swing ≈ ``(-collapse)`` clipped with ``pair_odd_cos``
    - leftover ≈ ``leak_ratio`` (unused_e) or content survival (divergent)

    ``exam_score`` = min(u_kept, swing_proxy) over the cell, mirroring
    scoreboard ``min(overlap, swing)``.
    """
    row = score_adv_field3d(
        field, teacher=teacher, leak_dir=leak_dir, cfg=cfg, name=name
    )
    kind = row.get("cell") or (field.kind if field is not None else "leftover")
    u_kept = float(row["u_kept"])
    content_kept = float(row["content_kept"])
    leak_ratio = float(row["leak_ratio"])
    collapse = float(row["collapse"])
    pair_odd = float(row["pair_odd_cos"])
    # Swing proxy: anti-collapse × alignment with odd (PairField roll_swing_kept).
    swing_proxy = max(0.0, min(1.0, 0.5 * (1.0 - collapse) * max(0.0, pair_odd)))
    # Continuation proxy: û kept; fold content when present in a.
    a_c = float(row.get("a_content") or 0.0)
    if abs(a_c) > 1e-8:
        cont_proxy = min(u_kept, content_kept)
    else:
        cont_proxy = u_kept
    exam_score_val = min(cont_proxy, swing_proxy) if swing_proxy > 0 else cont_proxy

    # Cell-specific leftover gate (PairField exam lesson + Fire #13 cross-axis).
    if kind in ("divergent", "axis_content_primary", "cross_axis_mismatch_declare", "amp_lie_leftover_declare", "e_on_u_declare_lie", "declare_split_three"):
        # content-primary / declare-lie — must NOT delete content / u.
        leftover_ok = bool(row["pass_u"] and row["pass_content"])
    elif kind in (
        "unused_e",
        "axis_leak_primary",
        "cross_axis_rows",
        "cross_axis_span_sample",
        "axis_u_primary",
        "lyric_span_entangle",
        "scale_stagger_homo",
        "roles_split_proxy",
        "prefix_shared_proxy",
        "close_with_leak",
        "hold_e_lyric_mix",
        "stagger_mild_cross",
        "multipair_corr_seed",
        "content_leak_flip_rows",
        "lyric_neu_heavy_gate",
        "scale_descent_homo",
        "guard_refuse_hot_eoc",
        "content_cascade_rows",
        "eoc_threshold_edge",
        "leftover_hot_eoc_declare",
    ):
        # M1 / Fire #20: treat like cross-axis leftover — require leak floor too.
        leftover_ok = bool(row["pass_leak"] and row["pass_u"])
    else:  # close / leftover
        leftover_ok = bool(row["pass_u"] and row["pass_content"])

    multi_ok = bool(row["all_rows_covered"])
    # Exam pass mirrors scoreboard: continuation + swing floors + leftover.
    pass_cont = cont_proxy >= 0.85
    pass_swing = swing_proxy >= 0.60
    exam_pass = bool(pass_cont and pass_swing and leftover_ok and multi_ok)

    row.update(
        {
            "exam_cont": float(cont_proxy),
            "exam_swing": float(swing_proxy),
            "exam_score": float(exam_score_val),
            "pass_cont": bool(pass_cont),
            "pass_swing": bool(pass_swing),
            "pass_leftover_gate": bool(leftover_ok),
            "pass_multi_row": bool(multi_ok),
            "exam_pass": bool(exam_pass),
            # Prefer exam_pass as the named gate for this dive.
            "pass": bool(exam_pass),
        }
    )
    return row


def summarize_music_close_posture(rows: list[dict] | tuple[dict, ...]) -> dict:
    """Aggregate ``music_close_posture_warn`` keys from dig / score rows.

    Dig runners should call this on their result list so Music parts0 close
    posture is explicit in JSON/md (Fire #22 helper → Fire #23 wire-up).
    Does not train or mutate cfg.
    """
    warns: list[dict] = []
    clear = 0
    missing = 0
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            missing += 1
            continue
        if "music_close_posture_warn" not in row:
            missing += 1
            continue
        w = row.get("music_close_posture_warn")
        meta = {
            "i": i,
            "name": row.get("name") or row.get("probe") or row.get("cell"),
            "cell": row.get("cell"),
            "n_particles": row.get("n_particles"),
            "seed": row.get("seed"),
            "pass": row.get("pass"),
        }
        if w:
            warns.append({**meta, "warn": str(w)})
        else:
            clear += 1
    return {
        "n_rows": len(rows or []),
        "n_warn": len(warns),
        "n_clear": clear,
        "n_missing_key": missing,
        "any_warn": bool(warns),
        "warns": warns,
    }




__all__ = [
    "Field3D",
    "CELLS_3D",
    "MUSIC_CLOSE_POSTURE_KINDS",
    "music_close_posture_warn",
    "summarize_music_close_posture",
    "leftover_field3d",
    "divergent_field3d",
    "close_field3d",
    "unused_e_field3d",
    "lyric_span_entangle_field3d",
    "close_live_noise_field3d",
    "dual_arm_leftover_geom_field3d",
    "tiny_slider_dom_field3d",
    "e_on_u_declare_lie_field3d",
    "prefix_shared_proxy_field3d",
    "scale_stagger_homo_field3d",
    "roles_split_proxy_field3d",
    "close_with_leak_field3d",
    "grit_content_dom_field3d",
    "amp_lie_leftover_declare_field3d",
    "hold_e_lyric_mix_field3d",
    "stagger_mild_cross_field3d",
    "multipair_corr_seed_field3d",
    "content_leak_flip_rows_field3d",
    "lyric_neu_heavy_gate_field3d",
    "declare_split_three_field3d",
    "scale_descent_homo_field3d",
    "cross_axis_rows_field3d",
    "axis_u_primary_field3d",
    "axis_content_primary_field3d",
    "axis_leak_primary_field3d",
    "cross_axis_span_sample_field3d",
    "cross_axis_mismatch_declare_field3d",
    "field3d_teacher_points",
    "score_adv_field3d",
    "score_adv_field3d_exam",
]
