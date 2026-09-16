"""CLONE ARM 2/5 — VICReg-faithful propose-only arm (``pg_vicreg_faithful``).

ParticleGAN-faithful VICReg for the 2-D toy: upstream ``ParticleRegularizer``
(``particlegan 0.2.0`` ``particlegan/vicreg_loss.py``) is weight 1,
``target_std=1``, var+cov only — no invariance (sim) term. The demo's
``vicreg_loss`` (``analysis/slider2d/adv.py``) is ``vicreg_weight=0.05``,
``std_target=0.05``, sim+var+cov with a *squared* variance hinge.

``PROPOSE_ONLY = True`` — this arm never flips a locked default and never
touches Music trainer argv. The locked KEEP (``b_cap`` kappa=1 + Music
extras: cover/pole weights, span/end cloud, ``particle_l2``) is shared with
the default recipe; only the VICReg formulation/weight/std moves. The
remaining drift vs upstream is documented in ``docs/pg-vicreg-faithful-arm.md``
and in :func:`disposition`.

CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch
import torch.nn.functional as F

from analysis.slider2d.adv import AdvConfig

PROPOSE_ONLY = True

# Upstream ParticleGAN recipe values (particlegan 0.2.0: Recipe.prior_reg=1.0,
# ParticleRegularizer(target_std=1.0, eps=1e-4, weight=1.0)).
PG_VICREG_WEIGHT = 1.0
PG_STD_TARGET = 1.0
PG_EPS = 1e-4

UPSTREAM_REF = (
    "particlegan 0.2.0 particlegan/vicreg_loss.py "
    "(VICRegLikeLoss + ParticleRegularizer)"
)


def vicreg_faithful_loss(
    z: torch.Tensor,
    *,
    target_std: float = PG_STD_TARGET,
    eps: float = PG_EPS,
) -> torch.Tensor:
    """Upstream ParticleGAN VICReg math: linear-hinge var + decorrelating cov.

    Variance: ``mean(relu(target_std - sqrt(var_unbiased + eps)))`` — a
    *linear* hinge that only punishes collapse below ``target_std`` and lets
    the particles expand (holes/clusters allowed). Covariance: squared
    off-diagonal mass of the ``(N-1)``-normalized covariance, divided by dim.
    No invariance/sim term: upstream never augments the particles.

    Intentional drift vs upstream (fail-closed toy convention, BUG HUNT B):
    fewer than 2 rows raises instead of returning a silent 0 — a 1-particle
    prior has no variance/covariance and a 0 would disable the spread
    pressure without a trace. Upstream's ``len(z) < 2`` early-return is the
    only formulation line not carried over.
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


def pg_vicreg_faithful_cfg(base: AdvConfig | None = None) -> AdvConfig:
    """Locked recipe with only the VICReg weight moved to upstream (1.0).

    Everything else — ``b_cap``/``kappa``, cover, ``particle_l2``, span/end
    cloud, LR/schedule — stays exactly as passed (default: the locked
    ``AdvConfig()``). Pair with ``vicreg_fn=vicreg_faithful_loss`` on
    ``fit_adv`` / ``train_lm_adv`` for the full formulation, not just the
    weight.
    """
    base = base if base is not None else AdvConfig()
    return replace(base, vicreg_weight=float(PG_VICREG_WEIGHT))


def locked_cfg() -> AdvConfig:
    """The locked_shared recipe: plain ``AdvConfig()`` defaults (demo VICReg)."""
    return AdvConfig()


def disposition() -> dict[str, dict[str, str]]:
    """KEEP/HOLD/DROP verdicts for every knob this arm touches or refuses."""
    return {
        "b_cap": {"verdict": "KEEP", "note": "locked at coeff 1, kappa 1 (GradRegularizer path)"},
        "kappa": {"verdict": "KEEP", "note": "locked at 1.0; arm reuses make_grad_regularizer"},
        "music_extras": {
            "verdict": "KEEP",
            "note": "cover_weight, span/end cloud, particle_l2 stay locked",
        },
        "vicreg_weight": {
            "verdict": "HOLD",
            "note": "0.05 (locked demo) vs 1.0 (this arm, upstream-faithful); needs multi-seed GPU before any flip",
        },
        "vicreg_formulation": {
            "verdict": "HOLD",
            "note": "sim+var+cov squared-hinge (locked demo) vs var+cov linear-hinge (this arm); same gate as weight",
        },
        "vicreg_std_target": {
            "verdict": "HOLD",
            "note": "0.05 (toy poles O(1)) vs 1.0 (upstream); toy-scale fit unproven",
        },
        "music_trainer_argv": {
            "verdict": "DROP",
            "note": "ARM_B keeps vicreg_weight=0 at --parts 0; weight 1.0 is refused under --adv_preset arm_b by design",
        },
        "live_defaults": {
            "verdict": "DROP",
            "note": "no change to --lm_target v9 / --pole_mode hidden / --adv_preset none",
        },
    }


VicregFn = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "PROPOSE_ONLY",
    "PG_VICREG_WEIGHT",
    "PG_STD_TARGET",
    "PG_EPS",
    "UPSTREAM_REF",
    "VicregFn",
    "disposition",
    "locked_cfg",
    "pg_vicreg_faithful_cfg",
    "vicreg_faithful_loss",
]