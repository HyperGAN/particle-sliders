"""Clone arm 3/5 (``pg_2x_lr``): ParticleGAN-champion 2xLR on the 2-D slider toy.

``PROPOSE_ONLY = True`` — this module proposes an experiment. It changes no
default: ``AdvConfig`` stays shared-LR ``5e-3``, the Music trainer ``ARM_B``
row and its argv gates are untouched, and the live default stays
``--lm_target v9``. Nothing here trains on Music weights or renders audio;
CPU-only 2-D fixtures.

Champion mapping (gap note; ``docs/music-arm-b-gates.md`` "LR / schedule"):
logistic RpGAN track, ``b_cap`` L2 coeff 1, ``kappa`` 1, at 2x LR, with the
ParticleGAN per-party ratios G / 1.5x D / 10x prior (Music side already
uses ``D_LR_MULT = 1.5`` in ``conceptmod/textsliders/lm_adv.py`` and
``d_lr = g_lr * 1.5`` in ``yue2_uni.py`` / ``train_lora_yue2_fresh.py`` /
``yue2_arm_b.py``).

Toy expression of the arm (``pg_2x_lr_cfg``):

- ``lr``: ``5e-3`` -> ``1e-2`` (champion 2x on the toy's shared base).
- ``d_lr_mult``: ``1.0`` -> ``1.5`` (PG-like D ratio; zero extra CPU — same
  steps, same batches — via the ``opt_d`` LR in ``analysis/slider2d/gan.py``).
- ``prior_lr_mult``: stays ``1.0`` (HOLD). ParticleGAN's 10x prior drives a
  20k-particle movable prior in 4-D; the toy scores the *shared residual*
  while its 12 particles must stay a jitter prior (see "Particle L2 +
  small VICReg" in ``analysis/slider2d/gan_bcap_findings.md``). A 10x prior
  LR is a second variable — prior structure, not LR magnitude — and risks
  particles stealing the modes under the ``cover_weight`` pin. It belongs
  in a follow-up arm, not in this single-variable LR clone.
- Everything else (``b_cap``, ``kappa``, ``fm_weight``, ``cover_weight``,
  ``vicreg_weight``, ``particle_l2``, betas, delay/schedule, teacher, steps,
  seed) is locked to the champion recipe.
"""

from __future__ import annotations

from dataclasses import replace

from analysis.slider2d.adv import AdvConfig, toy_lr_triplet

PROPOSE_ONLY = True

#: Clone arm 3/5 name.
ARM_NAME = "pg_2x_lr"

#: Locked toy base this arm clones (``AdvConfig`` defaults: shared 5e-3).
BASE_LR = 5.0e-3

#: Champion 2x magnitude on the toy base.
ARM_LR = 1.0e-2

#: PG-like critic ratio (matches Music ``D_LR_MULT = 1.5``).
ARM_D_LR_MULT = 1.5

#: Prior ratio deliberately locked (see module docstring: HOLD, not DROP).
ARM_PRIOR_LR_MULT = 1.0


def pg_2x_lr_cfg(**overrides) -> AdvConfig:
    """Champion 2xLR config: locked recipe, LR-only deltas.

    Extra ``overrides`` are for CPU-budget smoke runs (e.g. ``steps=250``);
    the LR deltas themselves cannot be overridden here — pass an explicit
    ``AdvConfig`` instead so a quiet kwarg cannot un-clone the arm.
    """
    for key in ("lr", "d_lr_mult", "prior_lr_mult"):
        if key in overrides:
            raise ValueError(
                f"{key} is fixed by the {ARM_NAME} arm; build AdvConfig directly"
            )
    cfg = AdvConfig(lr=ARM_LR, d_lr_mult=ARM_D_LR_MULT, prior_lr_mult=ARM_PRIOR_LR_MULT)
    return replace(cfg, **overrides) if overrides else cfg


def locked_cfg(**overrides) -> AdvConfig:
    """Locked baseline for arm-vs-locked comparison (same overrides allowed)."""
    cfg = AdvConfig()
    return replace(cfg, **overrides) if overrides else cfg


def arm_lr_triplet(cfg: AdvConfig | None = None) -> tuple[float, float, float]:
    """``(g_lr, d_lr, prior_lr)`` for the arm (default) or any config."""
    return toy_lr_triplet(cfg if cfg is not None else pg_2x_lr_cfg())


__all__ = [
    "ARM_NAME",
    "ARM_D_LR_MULT",
    "ARM_LR",
    "ARM_PRIOR_LR_MULT",
    "BASE_LR",
    "PROPOSE_ONLY",
    "arm_lr_triplet",
    "locked_cfg",
    "pg_2x_lr_cfg",
]
