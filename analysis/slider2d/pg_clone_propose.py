"""Consolidated ParticleGAN clone proposals (propose-only, CPU toys).

One thin module salvaging the five HOLD clone arms (#111 #112 #113 #114
#115) without dumping all five full trees:

- #111 big particles: single-sourced in `locked_baseline_defaults.py`;
  this module carries no n-particles living recipe (HOLD tie, dropped).
- #112 2x LR: `pg_2x_lr_cfg()` (lr 1e-2, d 1.5x, prior 1.0 HOLD) over the
  `toy_lr_triplet` / per-party wiring in `adv.py` / `gan.py`.
- #113 VicReg: `vicreg_faithful_loss` (upstream linear-hinge var + cov,
  no sim) + `pg_vicreg_faithful_cfg()` (weight 1.0); thread via
  `fit_adv` / `train_lm_adv(vicreg_fn=...)`, None = locked byte-identical.
- #115 anneal/g_interp: no new harness here — the math already lives in
  `GradRegularizer` (`g_interp_cap`, linear/delayed anneal); this module
  only documents the HOLD tie. CLI opt-in lives in `run_lm_adv.py`.
- #114 full clone: `pg_full_clone_cfg()` — closest-ParticleGAN recipe
  expressible with existing `AdvConfig` knobs only (thin surface; the
  remaining deltas are a documented gap checklist, not new config fields).

``PROPOSE_ONLY = True`` and ``MERGE_TO_TRAINER = False``: nothing here
changes a locked default, the Music trainer row, or the live default.
CPU only. No Music weights, no audio, no GPU.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch
import torch.nn.functional as F

from analysis.slider2d.adv import AdvConfig, toy_lr_triplet

PROPOSE_ONLY = True
MERGE_TO_TRAINER = False

# Upstream ParticleGAN recipe values (particlegan 0.2.0: Recipe.prior_reg=1.0,
# ParticleRegularizer(target_std=1.0, eps=1e-4, weight=1.0)).
PG_VICREG_WEIGHT = 1.0
PG_STD_TARGET = 1.0
PG_EPS = 1e-4

UPSTREAM_REF = (
    "particlegan 0.2.0 particlegan/vicreg_loss.py "
    "(VICRegLikeLoss + ParticleRegularizer)"
)

# Clone arm 3/5 values (ParticleGAN-champion 2xLR on the CPU 2-D toy).
ARM_2X_LR = 1.0e-2
ARM_2X_BASE_LR = 5.0e-3
ARM_2X_D_MULT = 1.5
ARM_2X_PRIOR_MULT = 1.0  # HOLD: PG 10x prior not cloned (particles stay jitter prior)


def vicreg_faithful_loss(
    z: torch.Tensor,
    *,
    target_std: float = PG_STD_TARGET,
    eps: float = PG_EPS,
) -> torch.Tensor:
    """Upstream ParticleGAN VICReg math: linear-hinge var + decorrelating cov.

    Variance: ``mean(relu(target_std - sqrt(var_unbiased + eps)))`` — a
    *linear* hinge that only punishes collapse below ``target_std`` and lets
    the particles expand. Covariance: squared off-diagonal mass of the
    ``(N-1)``-normalized covariance, divided by dim. No invariance/sim term:
    upstream never augments the particles.

    Intentional drift vs upstream (fail-closed toy convention, BUG HUNT B):
    fewer than 2 rows raises instead of returning a silent 0.
    """
    if z.ndim != 2:
        raise ValueError(f"vicreg_faithful_loss needs a 2-D particle batch, got ndim={z.ndim}")
    if z.shape[0] < 2:
        raise ValueError(
            f"VICReg needs >= 2 particles, got {z.shape[0]}: a single particle "
            "has no variance/covariance, so the regularizer would silently be 0"
        )
    if z.shape[1] == 0:
        raise ValueError("vicreg_faithful_loss needs a positive latent dimension")
    std_z = torch.sqrt(z.var(dim=0) + float(eps))
    std_loss = F.relu(float(target_std) - std_z).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / float(z.shape[0] - 1)
    off = cov.pow(2).sum() - cov.diag().pow(2).sum()
    cov_loss = off / float(z.shape[1])
    return std_loss + cov_loss


def pg_2x_lr_cfg(**overrides) -> AdvConfig:
    """Champion 2xLR config: locked recipe, LR-only deltas.

    Extra ``overrides`` are for CPU-budget smoke runs (e.g. ``steps=250``);
    the LR deltas themselves cannot be overridden here — build an explicit
    ``AdvConfig`` instead so a quiet kwarg cannot un-clone the arm.
    """
    for key in ("lr", "d_lr_mult", "prior_lr_mult"):
        if key in overrides:
            raise ValueError(f"{key} is fixed by the pg_2x_lr arm; build AdvConfig directly")
    cfg = AdvConfig(lr=ARM_2X_LR, d_lr_mult=ARM_2X_D_MULT, prior_lr_mult=ARM_2X_PRIOR_MULT)
    return replace(cfg, **overrides) if overrides else cfg


def pg_vicreg_faithful_cfg(base: AdvConfig | None = None) -> AdvConfig:
    """Locked recipe with only the VICReg weight moved to upstream (1.0).

    Pair with ``vicreg_fn=vicreg_faithful_loss`` on ``fit_adv`` /
    ``train_lm_adv`` for the full formulation, not just the weight.
    """
    base = base if base is not None else AdvConfig()
    return replace(base, vicreg_weight=float(PG_VICREG_WEIGHT))


# Closest-clone recipe expressible with existing AdvConfig knobs (thin
# surface — no new config fields). Full upstream deltas that need new
# machinery (60% LR hold, EMA on G+particles, VICReg inner-scale, 20k
# particles, lyric/mode teacher) are documented gaps in the salvage note,
# not wired here.
PG_FULL_CLONE: dict[str, object] = {
    "beta2": 0.999,
    "n_particles": 32,
    "vicreg_weight": 1.0,
    "particle_l2": 0.0,
    "lr": 5.0e-3,
    "d_lr_mult": 1.0,
    "prior_lr_mult": 1.0,
}

PG_FULL_CLONE_LEDGER: tuple[tuple[str, str, str, str], ...] = (
    ("GAN loss", "RpGAN pair logistic", "RpGAN pair logistic", "MATCH"),
    ("b_cap / kappa / norm", "1 / 1 / L2", "1 / 1 / L2", "MATCH"),
    ("particles", "20k prior", "32/side (thin; full 20k infeasible on CPU)", "PARTIAL"),
    ("VICReg weight/form", "1, var+cov", "1 + vicreg_faithful_loss (sim dropped)", "MATCH"),
    ("VICReg inner scale", "4-D particle scale", "toy std_target=1.0 via vicreg_fn; fixture scale gap", "HOLD"),
    ("Adam beta2", "0.999", "0.999", "MATCH"),
    ("LR hold", "60% hold + cosine", "absolute delay 80 (delay_frac not wired; gap)", "HOLD"),
    ("LR values", "6e-4 G, x1.5 D, x10 prior", "shared 5e-3 (2xLR is a separate arm)", "HOLD"),
    ("EMA", "G+particles", "residual-only (ema_scope not wired; gap)", "HOLD"),
    ("particle L2", "none", "0.0", "MATCH"),
    ("teacher / span-end / cover", "modes/spans", "faithful_guard_e + span/end + cover 1.5 (Music HOLD)", "HOLD"),
)


def pg_full_clone_cfg(**overrides) -> AdvConfig:
    """Closest ParticleGAN clone toy config (propose-only preset).

    Thin preset: only existing `AdvConfig` knobs. Pair with
    ``vicreg_fn=vicreg_faithful_loss`` for the var+cov form. Never mutates
    the ``AdvConfig()`` defaults — pass overrides explicitly for sweeps.
    """
    cfg = AdvConfig(**{k: v for k, v in PG_FULL_CLONE.items()})  # type: ignore[arg-type]
    return replace(cfg, **overrides) if overrides else cfg


def arm_lr_triplet(cfg: AdvConfig | None = None) -> tuple[float, float, float]:
    """``(g_lr, d_lr, prior_lr)`` for the 2xLR arm (default) or any config."""
    return toy_lr_triplet(cfg if cfg is not None else pg_2x_lr_cfg())


def disposition() -> dict[str, dict[str, str]]:
    """KEEP/HOLD/DROP verdicts for the consolidated salvage."""
    return {
        "locked": {"verdict": "KEEP", "note": "b_cap k=1 coeff=1 + cover/teacher/FM0/LR extras; production + live untouched"},
        "pg_big_particles": {"verdict": "HOLD", "note": "n 12->64 ties locked; dropped as living recipe"},
        "pg_2x_lr": {"verdict": "HOLD", "note": "2x + D 1.5x tracks/ahead on toy smoke; needs 1200-step multi-seed + prior question"},
        "pg_vicreg_faithful": {"verdict": "HOLD", "note": "weight/form/std upstream-faithful; needs multi-seed GPU + exam rollout"},
        "anneal_ginterp": {"verdict": "HOLD", "note": "delayed/linear + g_interp_cap tie locked at 1200 steps seed 0; high-dim transfer unproven"},
        "pg_full_clone": {"verdict": "HOLD", "note": "thin preset reaches parity on toy cells; trainer transfer + unwired gaps unproven"},
    }


VicregFn = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "PROPOSE_ONLY",
    "MERGE_TO_TRAINER",
    "PG_VICREG_WEIGHT",
    "PG_STD_TARGET",
    "PG_EPS",
    "UPSTREAM_REF",
    "ARM_2X_LR",
    "ARM_2X_BASE_LR",
    "ARM_2X_D_MULT",
    "ARM_2X_PRIOR_MULT",
    "PG_FULL_CLONE",
    "PG_FULL_CLONE_LEDGER",
    "VicregFn",
    "arm_lr_triplet",
    "disposition",
    "pg_2x_lr_cfg",
    "pg_vicreg_faithful_cfg",
    "pg_full_clone_cfg",
    "vicreg_faithful_loss",
]
