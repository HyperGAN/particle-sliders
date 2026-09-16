"""Unipolar PairField toy exam for GAN-only ParticleGAN training (UniPG-B).

The spine is ParticleGAN: relativistic-pair logistic (``rp_g_loss`` /
``rp_d_loss``) plus the vendored ``GradRegularizer`` champion
(``b_cap``, ``kappa=1``, ``coeff=1``). The generator objective is
adversarial only — there is intentionally no positive MSE, cover MSE,
ending, feature matching, hold, plan, or zero-point knob on the
config at all (there is nothing to set to zero by accident).

Unipolar contract (same gates as the unipolar formulation board):

- ``cover`` at scale +1 ``>= 0.85``
- ``leak`` (off-caption on the + continuation) at scale +1 ``<= 0.05``
- ``neu_hold`` at scale 0 ``>= 0.85``

Eval scales are ``0 / 0.5 / 1`` (gates read 0 and 1; 0.5 is a logged
interpolation diagnostic). Scale ``-1`` is an unscored canary.
Antipodal ``cos(+1, -1)`` is never consulted. The teacher is the raw
positive caption (the ``lm_faithful_plus_neu`` contract for what ``+``
is). Scale 0 is the exact base by construction: the scored student is
the shared ``SharedResidual`` with ``delta(0) == 0``, so ``neu_hold``
is architectural, not learned.

ARM IDENTITY UniPG-B: this module closes the intentional ParticlePrior
+ ParticleRegularizer drifts around the locked toy:

- the learnable prior mean absorbs the pole shift, so the scored
  residual undershoots (``|d+| ~ 1.9`` vs teacher ``~2.6``) and cover
  stalls at ``~0.65``;
- freezing the prior (prior LR ``0``) keeps a zero-mean jitter bank and
  the residual carries the full shift (cover ``~0.91``);
- a wider frozen bank (``n_particles`` 12 ``->`` 32 ``->`` 64, toy-small
  toward ParticleGAN-like larger) stabilises the seed-1 divergent cell;
- the VICReg ladder (0 / toy 0.05 / ParticleGAN-like 1.0),
  var+cov-only vs sim+var+cov, and ``std_target`` are first-class knobs;
- ``particle_l2`` on/off is first-class.

``GradRegularizer`` stays at the champion (``b_cap``, ``kappa=1``,
``coeff=1``) on every arm.

``PROPOSE_ONLY = True``: nothing here changes a locked default, the
Music trainer row (bipolar ``ARM_B`` / ``locked_shared``), or the live
``--lm_target`` default. CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field as _field
from pathlib import Path

import torch

from analysis.slider2d.adv import (
    AdvConfig,
    EMA,
    Fourier2MLP,
    ParticlePrior,
    delayed_cosine,
    make_grad_regularizer,
    rp_d_loss,
    rp_g_loss,
    sample_real_cloud,
    vicreg_loss,
)
from analysis.slider2d.exam import (
    SharedResidual,
    close_field,
    divergent_field,
)
from analysis.slider2d.pg_clone_propose import (
    PG_EPS,
    PG_STD_TARGET,
    vicreg_faithful_loss,
)
from analysis.slider2d.plus_exam import (
    PLUS_COVER_MIN,
    PLUS_OFF_MAX,
    _continue,
    _off_share,
    _token_share,
    blend_toward_mid,
    nearest_pole,
    plus_bags,
    plus_cover,
)
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_HOLD_MIN,
    drift_from_neu,
    neu_bags,
    neu_hold,
)

PROPOSE_ONLY = True
MERGE_TO_TRAINER = False

# Gate contract, shared with the unipolar formulation board.
COVER_MIN = PLUS_COVER_MIN  # 0.85
LEAK_MAX = PLUS_OFF_MAX  # 0.05
NEU_HOLD_MIN = PLUS_NEU_HOLD_MIN  # 0.85

# Eval scales. Gates read 0 and 1; 0.5 is a logged diagnostic; -1 is an
# unscored canary and never a gate.
EVAL_SCALES = (0.0, 0.5, 1.0)

REQUIRED_CELLS = ("divergent", "close")

CELL_CTORS = {
    "divergent": divergent_field,
    "close": close_field,
}

VICREG_FORMS = ("locked", "faithful")


@dataclass
class UniGanConfig:
    """GAN-only unipolar config. There is deliberately no MSE knob.

    ``freeze_prior`` (prior LR 0) keeps the ParticlePrior a zero-mean
    jitter bank; ``center_particles`` re-centers each fake sample on the
    bank mean so a learnable prior provides spread without absorbing the
    shift. ``vicreg_form`` selects the locked toy VICReg
    (sim+var+cov, ``std_target=0.05``) or the ParticleGAN-faithful
    var+cov-only loss (``std_target`` below, default upstream 1.0).
    """

    steps: int = 1200
    seed: int = 0
    lr: float = 5.0e-3
    beta1: float = 0.0
    beta2: float = 0.99
    d_lr_mult: float = 1.0
    # GradRegularizer champion. Pinned; arms do not move these.
    b_cap: float = 1.0
    kappa: float = 1.0
    grad_arm: str = "b_cap"
    grad_norm: str = "l2"
    grad_lazy: int = 1
    target_anneal: str = "none"
    # ParticlePrior (UniPG-B axes).
    n_particles: int = 12
    init_std: float = 0.05
    freeze_prior: bool = False
    center_particles: bool = False
    particle_jitter: float = 0.01
    # ParticleRegularizer / VICReg (UniPG-B axes).
    vicreg_weight: float = 0.0
    vicreg_form: str = "locked"
    vicreg_std_target: float = 1.0
    particle_l2: float = 0.0
    # Game / cloud.
    batch: int = 32
    cloud_std: float = 0.03
    span_frac: float = 0.40
    end_margin: float = 0.60
    d_steps: int = 1
    ema: float = 0.995
    delay: int = 80
    min_lr_ratio: float = 0.05
    critic_hidden: int = 64
    critic_n_rand: int = 16

    def vicreg_fn(self):
        """The particle regularizer for this config."""
        if self.vicreg_form == "faithful":
            target = float(self.vicreg_std_target)
            eps = float(PG_EPS)

            def _faithful(z: torch.Tensor) -> torch.Tensor:
                return vicreg_faithful_loss(z, target_std=target, eps=eps)

            return _faithful
        if self.vicreg_form == "locked":

            def _locked(z: torch.Tensor) -> torch.Tensor:
                return vicreg_loss(z)

            return _locked
        raise ValueError(f"vicreg_form must be one of {VICREG_FORMS}, got {self.vicreg_form!r}")

    def adv_config(self) -> AdvConfig:
        """Mirror onto ``AdvConfig`` for the shared GradRegularizer path."""
        return AdvConfig(
            steps=int(self.steps),
            lr=float(self.lr),
            beta1=float(self.beta1),
            beta2=float(self.beta2),
            d_lr_mult=float(self.d_lr_mult),
            prior_lr_mult=1.0,
            b_cap=float(self.b_cap),
            kappa=float(self.kappa),
            grad_arm=str(self.grad_arm),
            grad_norm=str(self.grad_norm),
            grad_lazy=int(self.grad_lazy),
            target_anneal=str(self.target_anneal),
            n_particles=int(self.n_particles),
            batch=int(self.batch),
            cloud_std=float(self.cloud_std),
            particle_jitter=float(self.particle_jitter),
            span_frac=float(self.span_frac),
            end_margin=float(self.end_margin),
            fm_weight=0.0,
            vicreg_weight=float(self.vicreg_weight),
            particle_l2=float(self.particle_l2),
            cover_weight=0.0,
            d_steps=int(self.d_steps),
            ema=float(self.ema),
            delay=int(self.delay),
            min_lr_ratio=float(self.min_lr_ratio),
            critic_hidden=int(self.critic_hidden),
            critic_n_rand=int(self.critic_n_rand),
            seed=int(self.seed),
        )


def assert_gan_only(cfg: UniGanConfig) -> None:
    """Fail closed: the champion regularizer is pinned and G has no MSE."""
    assert cfg.grad_arm == "b_cap", cfg.grad_arm
    assert float(cfg.b_cap) == 1.0, cfg.b_cap
    assert float(cfg.kappa) == 1.0, cfg.kappa
    assert cfg.grad_norm == "l2", cfg.grad_norm
    assert int(cfg.grad_lazy) == 1, cfg.grad_lazy
    assert cfg.target_anneal == "none", cfg.target_anneal
    # There is no cover/FM/MSE field to check: the dataclass has none.


def fit_uni_gan(field, *, cfg: UniGanConfig | None = None) -> SharedResidual:
    """Fit one shared residual with the GAN-only unipolar game.

    Real cloud: raw-positive poles (+ span/end analogue). Fakes: neutral
    + shared ``delta(1.0)`` + ParticlePrior sample. G loss is
    ``rp_g_loss`` plus (optionally) VICReg / particle-L2 on the bank.
    No supervised pole loss touches G.
    """
    cfg = cfg or UniGanConfig()
    assert_gan_only(cfg)
    torch.manual_seed(int(cfg.seed))
    dim = int(field.dim)
    residual = SharedResidual.create(field)
    prior = ParticlePrior(int(cfg.n_particles), dim, init_std=float(cfg.init_std))
    if cfg.freeze_prior:
        for param in prior.parameters():
            param.requires_grad_(False)
    critic = Fourier2MLP(
        dim,
        n_rand=int(cfg.critic_n_rand),
        hidden=int(cfg.critic_hidden),
        seed=int(cfg.seed),
    )
    g_lr = float(cfg.lr)
    d_lr = float(cfg.lr) * float(cfg.d_lr_mult)
    opt_params: list[dict] = [{"params": residual.parameters(), "lr": g_lr}]
    if not cfg.freeze_prior:
        opt_params.append({"params": list(prior.parameters()), "lr": g_lr})
    opt_g = torch.optim.Adam(opt_params, lr=g_lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=d_lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(residual.parameters(), decay=float(cfg.ema))

    poles_p = torch.stack(
        [field.poles(row)[0].flatten() for row in range(int(field.rows))]
    )
    neus = torch.stack(
        [field.poles(row)[2].flatten() for row in range(int(field.rows))]
    )
    half = max(1, int(cfg.batch) // 2)
    reg = make_grad_regularizer(cfg.adv_config())
    vfn = cfg.vicreg_fn()

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step,
            total=int(cfg.steps),
            delay=int(cfg.delay),
            min_ratio=float(cfg.min_lr_ratio),
        )
        opt_g.param_groups[0]["lr"] = g_lr * scale
        if len(opt_g.param_groups) > 1:
            opt_g.param_groups[1]["lr"] = g_lr * scale
        for group in opt_d.param_groups:
            group["lr"] = d_lr * scale

    def fake_batch() -> torch.Tensor:
        idx = torch.randint(0, neus.shape[0], (half,))
        deltas = torch.stack([residual.delta(1.0)] * half)
        samp = prior.sample(half, jitter=float(cfg.particle_jitter))
        if cfg.center_particles and not cfg.freeze_prior:
            samp = samp - prior.particles.mean(0, keepdim=True)
        return neus[idx] + deltas + samp

    for step in range(int(cfg.steps)):
        set_lr(step)
        real = sample_real_cloud(
            poles_p,
            neus,
            n=half,
            cloud_std=float(cfg.cloud_std),
            span_frac=float(cfg.span_frac),
            end_margin=float(cfg.end_margin),
        )
        for _ in range(int(cfg.d_steps)):
            fake_det = fake_batch().detach()
            real_det = real.detach()
            cap, _ = reg.penalty(critic, real_det, fake_det, step=step + 1)
            d_loss = rp_d_loss(critic(real_det), critic(fake_det)) + cap
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
        fake = fake_batch()
        g_adv = rp_g_loss(critic(real.detach()), critic(fake))
        g_extra = residual.w.new_zeros(())
        if not cfg.freeze_prior and float(cfg.vicreg_weight) > 0.0:
            g_extra = g_extra + float(cfg.vicreg_weight) * vfn(prior.particles)
        if not cfg.freeze_prior and float(cfg.particle_l2) > 0.0:
            g_extra = g_extra + float(cfg.particle_l2) * prior.particles.pow(2).mean()
        g_loss = g_adv + g_extra
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema.update(residual.parameters())

    ema.copy_to(residual.parameters())
    return residual.snapshot()


def score_uni_gan(field, residual: SharedResidual, *, name: str = "uni_gan") -> dict:
    """Score a fitted residual on the unipolar gate contract.

    ``cover`` / ``leak`` read scale +1; ``neu_hold`` reads scale 0
    (exact base by construction); ``half_cover`` reads scale 0.5 as a
    logged diagnostic; the ``-1`` canary is unscored.
    """
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    head = field.readout()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    d_zero = residual.delta(0.0)
    d_half = residual.delta(0.5)
    cover_rows: list[float] = []
    off_rows: list[float] = []
    hold_rows: list[float] = []
    half_rows: list[float] = []
    canary_landed: list[str] = []
    canary_off: list[float] = []
    sings_plus: list[str] = []
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        student_plus = neu + d_plus
        student_zero = neu + d_zero
        student_half = neu + d_half
        student_minus = neu + d_minus
        plus_seqs = _continue(field, student_plus, row=row, sign=1.0)
        zero_seqs = _continue(field, student_zero, row=row, sign=0.0)
        half_seqs = _continue(field, student_half, row=row, sign=1.0)
        minus_seqs = _continue(field, student_minus, row=row, sign=-1.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        blend = blend_toward_mid(student_plus, pos, mid, neg)
        cover_rows.append(plus_cover(overlap, blend))
        off_rows.append(off)
        half_rows.append(
            plus_cover(
                _token_share(half_seqs, bags["pos"]),
                blend_toward_mid(student_half, pos, mid, neg),
            )
        )
        hold_rows.append(
            neu_hold(
                _token_share(zero_seqs, neu_bag),
                drift_from_neu(student_zero, neu, pos, mid),
            )
        )
        canary_landed.append(nearest_pole(student_minus, pos, neu, neg))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
    cover = sum(cover_rows) / len(cover_rows)
    leak = sum(off_rows) / len(off_rows)
    hold = sum(hold_rows) / len(hold_rows)
    half_cover = sum(half_rows) / len(half_rows)
    landed = max(set(canary_landed), key=canary_landed.count)
    canary_off_mean = sum(canary_off) / len(canary_off)
    hit = bool(cover >= COVER_MIN and leak <= LEAK_MAX and hold >= NEU_HOLD_MIN)
    return {
        "name": name,
        "cell": field.kind,
        "polarity": "uni",
        "train": "gan-only",
        "cover": cover,
        "leak": leak,
        "off_caption": leak,
        "neu_hold": hold,
        "half_cover": half_cover,
        "eval_scales": list(EVAL_SCALES),
        "hit": hit,
        "pass": hit,
        "delta_norm": float(d_plus.norm()),
        "sings_plus": " | ".join(sings_plus),
        "canary": {
            "scored": False,
            "minus_landed": landed,
            "minus_off_caption": canary_off_mean,
            "dangerous": bool(landed == "pos" or canary_off_mean > LEAK_MAX),
        },
        "antipodal_consulted": False,
    }


# -- propose_only arms (UniPG-B) -------------------------------------------
#
# Each arm is (one-line identity, config delta, status). The baseline is
# the learned-prior toy (the drift); UniPG-B arms move exactly one
# particle/VICReg axis at a time toward ParticleGAN.


def _base_cfg(*, steps: int, seed: int) -> UniGanConfig:
    return UniGanConfig(steps=int(steps), seed=int(seed))


ARMS: dict[str, dict] = {
    "baseline_learned12": {
        "identity": "learned prior n=12 + locked VICReg 0.05 (sim+var+cov, std 0.05) + L2 0.02 — the drift; expect FAIL cover~0.65",
        "status": "baseline",
        "match": "DRIFT",
        "delta": {
            "n_particles": 12,
            "init_std": 0.05,
            "freeze_prior": False,
            "center_particles": False,
            "vicreg_weight": 0.05,
            "vicreg_form": "locked",
            "particle_l2": 0.02,
        },
    },
    "unipg_b_frozen12": {
        "identity": "frozen jitter prior n=12 (prior LR 0) — residual carries the full shift",
        "status": "propose_only",
        "match": "MATCH",
        "delta": {
            "n_particles": 12,
            "init_std": 0.05,
            "freeze_prior": True,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
        },
    },
    "unipg_b_frozen32": {
        "identity": "frozen jitter prior n=32 — wider bank, same zero-mean prior",
        "status": "propose_only",
        "match": "MATCH",
        "delta": {
            "n_particles": 32,
            "init_std": 0.05,
            "freeze_prior": True,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
        },
    },
    "unipg_b_frozen64": {
        "identity": "CHAMPION: frozen jitter prior n=64 — ParticleGAN-like larger bank; ALL-PASS at 1200",
        "status": "propose_only",
        "match": "MATCH",
        "delta": {
            "n_particles": 64,
            "init_std": 0.05,
            "freeze_prior": True,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
        },
    },
    "unipg_b_frozen32_init001": {
        "identity": "frozen tight prior n=32 init_std 0.01 — init_std axis; ALL-PASS at 1200",
        "status": "propose_only",
        "match": "MATCH",
        "delta": {
            "n_particles": 32,
            "init_std": 0.01,
            "freeze_prior": True,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
        },
    },
    "unipg_b_center_vicreg0": {
        "identity": "learned + centered sampling, VICReg 0 — ablation: centering alone without spread pressure",
        "status": "propose_only",
        "match": "PARTIAL",
        "delta": {
            "n_particles": 32,
            "init_std": 0.05,
            "freeze_prior": False,
            "center_particles": True,
            "vicreg_weight": 0.0,
            "vicreg_form": "locked",
            "particle_l2": 0.0,
        },
    },
    "unipg_b_center_vicregfaithful1": {
        "identity": "learned + centered + faithful VICReg 1.0 (var+cov-only, std_target 1.0) — closest ParticleRegularizer; rescues divergent seed 1",
        "status": "propose_only",
        "match": "MATCH",
        "delta": {
            "n_particles": 32,
            "init_std": 0.05,
            "freeze_prior": False,
            "center_particles": True,
            "vicreg_weight": 1.0,
            "vicreg_form": "faithful",
            "vicreg_std_target": 1.0,
            "particle_l2": 0.0,
        },
    },
    "unipg_b_frozen32_2xlr": {
        "identity": "frozen n=32 + 2x LR with D 1.5x (ParticleGAN LR axis, G 1e-2)",
        "status": "propose_only",
        "match": "PARTIAL",
        "delta": {
            "n_particles": 32,
            "init_std": 0.05,
            "freeze_prior": True,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
            "lr": 1.0e-2,
            "d_lr_mult": 1.5,
        },
    },
}

ARM_ORDER = (
    "baseline_learned12",
    "unipg_b_frozen12",
    "unipg_b_frozen32",
    "unipg_b_frozen64",
    "unipg_b_frozen32_init001",
    "unipg_b_center_vicreg0",
    "unipg_b_center_vicregfaithful1",
    "unipg_b_frozen32_2xlr",
)


def arm_cfg(name: str, *, steps: int, seed: int) -> UniGanConfig:
    """Build the config for a named arm at a budget/seed."""
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r} (expected one of {sorted(ARMS)})")
    cfg = _base_cfg(steps=steps, seed=seed)
    for key, value in ARMS[name]["delta"].items():
        setattr(cfg, key, value)
    return cfg


def score_arm_cell(name: str, cell: str, *, steps: int, seed: int) -> dict:
    """Fit + score one arm on one cell. GAN-only; no supervised fit."""
    if cell not in CELL_CTORS:
        raise ValueError(f"cell must be one of {sorted(CELL_CTORS)}, got {cell!r}")
    cfg = arm_cfg(name, steps=steps, seed=seed)
    field = CELL_CTORS[cell](seed=seed)
    residual = fit_uni_gan(field, cfg=cfg)
    row = score_uni_gan(field, residual, name=name)
    row.update(
        {
            "arm": name,
            "identity": ARMS[name]["identity"],
            "status": ARMS[name]["status"],
            "particlegan": ARMS[name]["match"],
            "steps": int(steps),
            "seed": int(seed),
            "vicreg_weight": float(cfg.vicreg_weight),
            "vicreg_form": str(cfg.vicreg_form),
            "n_particles": int(cfg.n_particles),
            "init_std": float(cfg.init_std),
            "freeze_prior": bool(cfg.freeze_prior),
            "particle_l2": float(cfg.particle_l2),
        }
    )
    return row


def exam_table(
    *,
    arms: tuple[str, ...] = ARM_ORDER,
    steps: tuple[int, ...] = (600, 1200, 3400),
    seeds: tuple[int, ...] = (0, 1, 7),
    cells: tuple[str, ...] = REQUIRED_CELLS,
) -> list[dict]:
    """Score every arm on the required cells over the budget ladder."""
    rows: list[dict] = []
    for arm in arms:
        for n_steps in steps:
            for seed in seeds:
                for cell in cells:
                    rows.append(
                        score_arm_cell(arm, cell, steps=int(n_steps), seed=int(seed))
                    )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Collapse per-seed rows into per-(arm, steps) gate numbers."""
    out: list[dict] = []
    arms = sorted({r["arm"] for r in rows})
    steps = sorted({r["steps"] for r in rows})
    for arm in arms:
        for n_steps in steps:
            sel = [r for r in rows if r["arm"] == arm and r["steps"] == n_steps]
            if not sel:
                continue
            per_cell: dict[str, dict] = {}
            for cell in sorted({r["cell"] for r in sel}):
                crows = [r for r in sel if r["cell"] == cell]
                per_cell[cell] = {
                    "cover": sum(r["cover"] for r in crows) / len(crows),
                    "leak": sum(r["leak"] for r in crows) / len(crows),
                    "neu_hold": sum(r["neu_hold"] for r in crows) / len(crows),
                    "hits": sum(1 for r in crows if r["hit"]),
                    "n": len(crows),
                    "all_hit": all(r["hit"] for r in crows),
                }
            required = [per_cell[c] for c in REQUIRED_CELLS if c in per_cell]
            out.append(
                {
                    "arm": arm,
                    "identity": ARMS[arm]["identity"],
                    "status": ARMS[arm]["status"],
                    "particlegan": ARMS[arm]["match"],
                    "steps": n_steps,
                    "cells": per_cell,
                    "pass": bool(required) and all(c["all_hit"] for c in required),
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[600, 1200, 3400])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7])
    parser.add_argument("--cells", nargs="+", default=list(REQUIRED_CELLS))
    parser.add_argument("--arms", nargs="+", default=list(ARM_ORDER))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    for cell in args.cells:
        if cell not in CELL_CTORS:
            raise ValueError(f"cell must be one of {sorted(CELL_CTORS)}, got {cell!r}")
    for arm in args.arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r} (expected one of {sorted(ARMS)})")

    rows = exam_table(
        arms=tuple(args.arms),
        steps=tuple(int(s) for s in args.steps),
        seeds=tuple(int(s) for s in args.seeds),
        cells=tuple(args.cells),
    )
    summary = summarize(rows)
    for entry in summary:
        bits = []
        for cell in REQUIRED_CELLS:
            if cell in entry["cells"]:
                c = entry["cells"][cell]
                bits.append(
                    f"{cell}: cover {c['cover']:.3f} leak {c['leak']:.3f} "
                    f"hold {c['neu_hold']:.3f} {c['hits']}/{c['n']}"
                )
        print(
            f"{entry['arm']} @ {entry['steps']}: "
            f"{'PASS' if entry['pass'] else 'FAIL'} | " + " | ".join(bits),
            flush=True,
        )
    blob = {
        "gates": {"cover_min": COVER_MIN, "leak_max": LEAK_MAX, "neu_hold_min": NEU_HOLD_MIN},
        "eval_scales": list(EVAL_SCALES),
        "arms": {name: ARMS[name] for name in args.arms},
        "summary": summary,
        "rows": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    # Nonzero unless every requested (arm, budget) passes on both cells.
    return 0 if all(e["pass"] for e in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
