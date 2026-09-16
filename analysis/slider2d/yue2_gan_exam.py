"""Unipolar YuE2-style toy GAN exam: RpGAN + b_cap, +1 only, GAN-only G.

CPU fixture analogue of the production YuE2 unipolar loop
(``conceptmod/textsliders/yue2_arm_b.py``: ``unipolar-rpgan-bcap-yue2-v3``)
scored under the unipolar toy gates (divergent + close):

- cover @ scale +1 >= 0.85
- off-caption / leak @ scale +1 <= 0.05
- neu_hold @ scale 0 >= 0.85

Eval scales are 0 / 0.5 / 1. Scale -1 is a canary only (never a gate).
Antipodal cos(+1, -1) is never consulted.

HARD CONSTRAINTS (every arm, enforced in code):

- GAN-only G: paired Rp logistic (``rp_g_loss`` / ``rp_d_loss``). Any
  ``cover_weight > 0`` or ``fm_weight > 0`` raises instead of training --
  no positive MSE, cover MSE, ending, FM, lyric-hold, plan, or
  zero-anchor on G. VICReg / particle L2 apply to the *particles* only
  (ParticleGAN spread pressure, not supervision of the residual).
- Unipolar: +1 student vs raw-positive teacher only
  (``lm_faithful_plus_neu`` contract: the teacher IS ``pos``). No
  negative captions, bipolar ranges, leak-axis metadata, or -1 branch.
- Scale 0 is the exact base by construction: ``delta(0) == 0``, so
  ``neu_hold`` is measured honestly (and is 1.0 up to rollout sampling).
- Spine always includes ParticleGAN Rp + a GradRegularizer arm
  (default ``b_cap``). Non-Rp losses are never wired here.
- Every arm is propose_only: nothing here changes ``AdvConfig()``
  defaults, the Music trainer row, or the live ``--lm_target`` default.

Sweep arms wire each #116 salvage hook into a distinct unipolar recipe:

- ``locked_baseline_defaults`` -> ``uni_locked``
- LR multipliers (``toy_lr_triplet``) -> ``uni_pg_2x_lr``,
  ``uni_pg_d15``, ``uni_pg_prior10``
- VicReg fn (``vicreg_fn``) -> ``uni_pg_vicreg_faithful``,
  ``uni_pg_vicreg_off``
- anneal CLI (``target_anneal`` / ``grad_arm``) -> ``uni_anneal_linear``,
  ``uni_anneal_delayed``, ``uni_g_interp``
- propose_only cards (``pg_clone_propose``) -> ``uni_pg_full_clone``
- big particles (``arm_cfg("pg_big_particles")``) -> ``uni_big_particles``

plus ParticleGAN formulation axes (kappa, critic width, beta2, EMA).

CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.adv import (  # noqa: E402
    AdvConfig,
    EMA,
    Fourier2MLP,
    ParticlePrior,
    delayed_cosine,
    make_grad_regularizer,
    rp_d_loss,
    rp_g_loss,
    sample_real_cloud,
    toy_lr_triplet,
    vicreg_loss,
)
from analysis.slider2d.exam import (  # noqa: E402
    PairField,
    close_field,
    divergent_field,
)
from analysis.slider2d.locked_baseline_defaults import LOCKED  # noqa: E402
from analysis.slider2d.pg_clone_propose import (  # noqa: E402
    PROPOSE_ONLY,
    pg_full_clone_cfg,
    pg_vicreg_faithful_cfg,
    vicreg_faithful_loss,
)
from analysis.slider2d.plus_exam import (  # noqa: E402
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
from analysis.slider2d.plus_neu_exam import (  # noqa: E402
    PLUS_NEU_HOLD_MIN,
    drift_from_neu,
    neu_bags,
    neu_hold,
)
from analysis.slider2d.rng import isolated_seed  # noqa: E402
from conceptmod.textsliders.slider_targets import lm_faithful_plus_neu  # noqa: E402

assert PROPOSE_ONLY is True

# Gates (unipolar toys, divergent + close).
UNI_COVER_MIN = PLUS_COVER_MIN  # 0.85
UNI_OFF_MAX = PLUS_OFF_MAX  # 0.05
UNI_HOLD_MIN = PLUS_NEU_HOLD_MIN  # 0.85

UNI_CELLS = {
    "divergent": divergent_field,
    "close": close_field,
}

# Budget ladder and seeds for the sweep (production baseline: 600 FAIL,
# 1200 FAIL divergent cover, 3400 PASS on seeds 0/1/7).
BUDGETS = (600, 1200, 2400, 3400)
SEEDS = (0, 1, 7)


class UniResidual:
    """Unipolar residual: ``delta(s) = s*w_odd + |s|*w_even``.

    Scale 0 is the exact base by construction (``delta(0) == 0``) -- no
    learned neutral anchor, matching the YuE2 adapter behavior.
    """

    def __init__(self, w_odd: torch.Tensor, w_even: torch.Tensor):
        self.w_odd = w_odd
        self.w_even = w_even

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd + abs(float(scale)) * self.w_even

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_odd, self.w_even]

    def snapshot(self) -> "UniResidual":
        return UniResidual(self.w_odd.detach().clone(), self.w_even.detach().clone())


def uni_teacher_points(field: PairField) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw-positive teacher per row (``lm_faithful_plus_neu`` contract).

    The teacher IS ``pos``: the shared UNI helper ignores its legacy
    negative argument and never leftover-gates. Asserted row by row so a
    future helper change fails loudly instead of silently re-teaching.
    """
    plus, neus = [], []
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        taught = lm_faithful_plus_neu(
            pos, neu, neu, None, slider_dir=field.short_u()
        )
        if not torch.equal(taught, pos):
            raise AssertionError(
                "lm_faithful_plus_neu contract broke: teacher is not raw pos"
            )
        plus.append(pos.flatten())
        neus.append(neu.flatten())
    return torch.stack(plus), torch.stack(neus)


@isolated_seed("cfg.seed", default=0)
def fit_uni_gan(
    field: PairField,
    *,
    cfg: AdvConfig | None = None,
    vicreg_fn=None,
) -> tuple[UniResidual, dict]:
    """Fit one unipolar residual with RpGAN + GradRegularizer. GAN-only G.

    ``vicreg_fn`` overrides the particle regularizer (default: the locked
    ``vicreg_loss``). ``None`` keeps the locked path byte-identical.

    Raises on any supervised knob: ``cover_weight`` / ``fm_weight`` must
    be exactly 0. VICReg / particle L2 apply to the particles only.
    """
    cfg = cfg or AdvConfig()
    if float(cfg.cover_weight) != 0.0:
        raise ValueError(
            f"fit_uni_gan is GAN-only: cover_weight must be 0, got {cfg.cover_weight!r}"
        )
    if float(cfg.fm_weight) != 0.0:
        raise ValueError(
            f"fit_uni_gan is GAN-only: fm_weight must be 0, got {cfg.fm_weight!r}"
        )
    vfn = vicreg_fn if vicreg_fn is not None else vicreg_loss
    dim = int(field.dim)
    residual = UniResidual(
        torch.zeros(dim, requires_grad=True),
        torch.zeros(dim, requires_grad=True),
    )
    prior = ParticlePrior(cfg.n_particles, dim)
    critic = Fourier2MLP(
        dim, n_rand=cfg.critic_n_rand, hidden=cfg.critic_hidden, seed=cfg.seed
    )
    g_lr, d_lr, prior_lr = toy_lr_triplet(cfg)
    opt_g = torch.optim.Adam(
        [
            {"params": residual.parameters(), "lr": g_lr},
            {"params": list(prior.parameters()), "lr": prior_lr},
        ],
        lr=g_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    opt_d = torch.optim.Adam(critic.parameters(), lr=d_lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(residual.parameters(), decay=cfg.ema)

    poles_p, neus = uni_teacher_points(field)
    half = max(1, int(cfg.batch) // 2)
    logs: dict[str, list[float]] = {"d": [], "g": [], "cap": []}
    reg = make_grad_regularizer(cfg)

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step, total=cfg.steps, delay=cfg.delay, min_ratio=cfg.min_lr_ratio
        )
        opt_g.param_groups[0]["lr"] = g_lr * scale
        opt_g.param_groups[1]["lr"] = prior_lr * scale
        for group in opt_d.param_groups:
            group["lr"] = d_lr * scale

    def fake_batch() -> torch.Tensor:
        idx = torch.randint(0, neus.shape[0], (half,))
        return (
            neus[idx] + residual.delta(1.0) + prior.sample(half, jitter=cfg.particle_jitter)
        )

    for step in range(int(cfg.steps)):
        set_lr(step)
        real = sample_real_cloud(
            poles_p,
            neus,
            n=half,
            cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac,
            end_margin=cfg.end_margin,
        )
        for _ in range(int(cfg.d_steps)):
            fake = fake_batch().detach()
            real_g = real.detach()
            cap, _ = reg.penalty(critic, real_g, fake, step=step + 1)
            d_loss = rp_d_loss(critic(real_g), critic(fake)) + cap
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        fake = fake_batch()
        d_real = critic(real.detach())
        d_fake = critic(fake)
        g_loss = rp_g_loss(d_real, d_fake)
        parts = prior.particles
        if float(cfg.vicreg_weight) > 0.0:
            g_loss = g_loss + float(cfg.vicreg_weight) * vfn(parts)
        if float(cfg.particle_l2) > 0.0:
            g_loss = g_loss + float(cfg.particle_l2) * parts.pow(2).mean()
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema.update(residual.parameters())

        if step == 0 or (step + 1) % 100 == 0 or step + 1 == cfg.steps:
            logs["d"].append(float(d_loss.detach()))
            logs["g"].append(float(g_loss.detach()))
            logs["cap"].append(float(cap.detach()))

    ema.copy_to(residual.parameters())
    snap = residual.snapshot()
    stats = {
        "d_loss": logs["d"][-1] if logs["d"] else None,
        "g_loss": logs["g"][-1] if logs["g"] else None,
        "cap": logs["cap"][-1] if logs["cap"] else None,
        "teacher": "faithful_plus_neu(raw pos)",
        "steps": int(cfg.steps),
        "seed": int(cfg.seed),
        "grad_arm": cfg.grad_arm,
        "target_anneal": cfg.target_anneal,
        "gan_only": True,
        "cover_weight": 0.0,
        "fm_weight": 0.0,
    }
    return snap, stats


def score_uni_cell(name: str, field: PairField, residual: UniResidual) -> dict:
    """Score cover/off @ +1, half_cover @ 0.5, neu_hold @ 0, -1 canary."""
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    head = field.readout()
    d_plus = residual.delta(1.0)
    d_half = residual.delta(0.5)
    d_zero = residual.delta(0.0)
    d_minus = residual.delta(-1.0)
    cover_rows, off_rows, half_rows = [], [], []
    hold_rows, overlap_rows, drift_rows = [], [], []
    canary_off, canary_landed, sings_plus, sings_zero = [], [], [], []
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        s_plus = neu + d_plus
        s_half = neu + d_half
        s_zero = neu + d_zero
        s_minus = neu + d_minus
        plus_seqs = _continue(field, s_plus, row=row, sign=1.0)
        half_seqs = _continue(field, s_half, row=row, sign=1.0)
        zero_seqs = _continue(field, s_zero, row=row, sign=0.0)
        minus_seqs = _continue(field, s_minus, row=row, sign=-1.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        blend = blend_toward_mid(s_plus, pos, mid, neg)
        cover_rows.append(plus_cover(overlap, blend))
        off_rows.append(off)
        half_rows.append(
            plus_cover(
                _token_share(half_seqs, bags["pos"]),
                blend_toward_mid(s_half, pos, mid, neg),
            )
        )
        neu_ov = _token_share(zero_seqs, neu_bag)
        drift = drift_from_neu(s_zero, neu, pos, mid)
        overlap_rows.append(neu_ov)
        drift_rows.append(drift)
        hold_rows.append(neu_hold(neu_ov, drift))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(s_minus, pos, neu, neg))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_zero.append(" ".join(head.tokens[t] for t in zero_seqs[0]))
    cover = sum(cover_rows) / len(cover_rows)
    off = sum(off_rows) / len(off_rows)
    hold = sum(hold_rows) / len(hold_rows)
    hit = bool(cover >= UNI_COVER_MIN and off <= UNI_OFF_MAX and hold >= UNI_HOLD_MIN)
    landed = max(set(canary_landed), key=canary_landed.count)
    canary_off_mean = sum(canary_off) / len(canary_off)
    return {
        "name": name,
        "cell": field.kind,
        "train": "unipolar GAN-only (Rp + %s)" % "b_cap",
        "cover": cover,
        "off_caption": off,
        "neu_hold": hold,
        "half_cover": sum(half_rows) / len(half_rows),
        "overlap_neu": sum(overlap_rows) / len(overlap_rows),
        "drift_from_neu": sum(drift_rows) / len(drift_rows),
        "hit": hit,
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
        "canary": {
            "scored": False,
            "minus_off_caption": canary_off_mean,
            "minus_landed": landed,
            "dangerous": bool(landed == "pos" or canary_off_mean > UNI_OFF_MAX),
        },
    }


def score_uni_arm(
    arm: str, cell: str, *, steps: int, seed: int
) -> dict:
    """Fit one arm on one cell at one budget/seed, then score uni gates."""
    if cell not in UNI_CELLS:
        raise ValueError(f"cell must be one of {sorted(UNI_CELLS)}, got {cell!r}")
    cfg, vfn_key = arm_cfg(arm, steps=steps, seed=seed)
    vfn = {"locked": None, "faithful": vicreg_faithful_loss}[vfn_key]
    field = UNI_CELLS[cell](seed=seed)
    residual, stats = fit_uni_gan(field, cfg=cfg, vicreg_fn=vfn)
    row = score_uni_cell(arm, field, residual)
    row.update(
        {
            "arm": arm,
            "steps": int(steps),
            "seed": int(seed),
            "vicreg_fn": vfn_key,
            "g_loss": stats.get("g_loss"),
            "d_loss": stats.get("d_loss"),
            "cap": stats.get("cap"),
        }
    )
    return row


# -- sweep arms (propose_only) -------------------------------------------
#
# Each arm = locked toy knobs + exactly its named delta, with
# cover_weight forced to 0 (GAN-only). The baseline is the production
# YuE2-unipolar toy analogue. Salvage arms each wire one #116 hook.
def _base(**overrides) -> AdvConfig:
    kw = {"cover_weight": 0.0, "fm_weight": 0.0}
    kw.update(overrides)
    return replace(AdvConfig(), **kw)


def _locked_base(**overrides) -> AdvConfig:
    kw = {**LOCKED, "cover_weight": 0.0, "fm_weight": 0.0}
    kw.update(overrides)
    return AdvConfig(**kw)


ARMS: dict[str, dict] = {
    # Production baseline analogue (not a #116 hook; the comparator).
    "uni_baseline": {
        "hook": "production yue2 unipolar (v3 analogue)",
        "card": "Rp + b_cap k=1 c=1, shared LR 5e-3, n=12, vicreg .05, L2 .02, GAN-only",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed), "locked"),
    },
    # #116 hook: locked_baseline_defaults single-source mirror.
    "uni_locked": {
        "hook": "locked_baseline_defaults (#116: LOCKED mirror)",
        "card": "LOCKED knobs + cover forced 0 for GAN-only (delta: cover 1.5->0)",
        "match": "MATCH",
        "make": lambda steps, seed: (_locked_base(steps=steps, seed=seed), "locked"),
    },
    # #116 hook: LR multipliers (#112 toy_lr_triplet wiring).
    "uni_pg_2x_lr": {
        "hook": "LR multipliers #112 (pg_2x_lr_cfg: lr 1e-2, D x1.5)",
        "card": "champion 2xLR: lr 1e-2, d_lr_mult 1.5, prior x1.0, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            replace(AdvConfig(),
                    lr=1.0e-2, d_lr_mult=1.5, prior_lr_mult=1.0,
                    cover_weight=0.0, fm_weight=0.0,
                    steps=steps, seed=seed),
            "locked",
        ),
    },
    "uni_pg_d15": {
        "hook": "LR multipliers #112 (ParticleGAN x1.5 D)",
        "card": "D LR x1.5 only: d_lr_mult 1.5, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, d_lr_mult=1.5), "locked"),
    },
    "uni_pg_prior10": {
        "hook": "LR multipliers #112 (ParticleGAN x10 prior; HOLD question)",
        "card": "prior LR x10: prior_lr_mult 10.0, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=10.0),
            "locked",
        ),
    },
    # #116 hook: VicReg fn (#113 vicreg_fn threading).
    "uni_pg_vicreg_faithful": {
        "hook": "VicReg fn #113 (vicreg_faithful_loss var+cov, weight 1.0)",
        "card": "upstream-faithful VICReg: weight 1.0 + faithful fn, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, vicreg_weight=1.0),
            "faithful",
        ),
    },
    "uni_pg_vicreg_off": {
        "hook": "VICReg weight ladder (0 / toy .05 / PG-like 1)",
        "card": "VICReg off: weight 0, particles held by L2 only, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, vicreg_weight=0.0),
            "locked",
        ),
    },
    # #116 hook: full-clone card (#114 thin preset, existing knobs only).
    "uni_pg_full_clone": {
        "hook": "full clone #114 (pg_full_clone_cfg thin preset)",
        "card": "closest-PG: beta2 .999, n=32, vicreg 1 faithful, L2 0, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            replace(pg_full_clone_cfg(), cover_weight=0.0, fm_weight=0.0,
                    steps=steps, seed=seed),
            "faithful",
        ),
    },
    # #116 hook: anneal CLI (#115 GradRegularizer math, opt-in flags).
    "uni_anneal_linear": {
        "hook": "anneal CLI #115 (target_anneal linear)",
        "card": "linear center anneal to R1/R2-like flatness, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, target_anneal="linear"),
            "locked",
        ),
    },
    "uni_anneal_delayed": {
        "hook": "anneal CLI #115 (target_anneal delayed, 60% hold)",
        "card": "delayed anneal: hold k=1 to 60% then ramp to 0, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, target_anneal="delayed"),
            "locked",
        ),
    },
    "uni_g_interp": {
        "hook": "anneal/g_interp #115 (grad_arm g_interp_cap)",
        "card": "interp-path cap: penalty on real/fake interpolates, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, grad_arm="g_interp_cap"),
            "locked",
        ),
    },
    # Formulation axes: kappa / coeff / particles / critic / Adam / EMA.
    "uni_kappa05": {
        "hook": "GradRegularizer kappa sweep (0.5)",
        "card": "tighter cap: kappa 0.5, freer D below, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, kappa=0.5), "locked"),
    },
    "uni_kappa20": {
        "hook": "GradRegularizer kappa sweep (2.0)",
        "card": "looser cap: kappa 2.0, steeper D allowed, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, kappa=2.0), "locked"),
    },
    "uni_big_particles": {
        "hook": 'big particles #111 (arm_cfg "pg_big_particles", n=64)',
        "card": "toward 20k prior: n_particles 64, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, n_particles=64),
            "locked",
        ),
    },
    "uni_thick_critic": {
        "hook": "critic/game axis (thicker Fourier critic)",
        "card": "thicker critic: hidden 128, n_rand 32, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, critic_hidden=128, critic_n_rand=32),
            "locked",
        ),
    },
    "uni_beta2_0999": {
        "hook": "critic/game axis (Adam beta2 0.999, PG value)",
        "card": "PG Adam: beta2 0.999, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, beta2=0.999), "locked"),
    },
    "uni_no_ema": {
        "hook": "critic/game axis (EMA off)",
        "card": "no EMA smoothing: ema 0.0 (raw last step), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, ema=0.0), "locked"),
    },
    # Schedule / data-geometry axes (all GAN-legitimate: no G supervision).
    "uni_hold_long": {
        "hook": "schedule axis (ParticleGAN-like 60% LR hold via long delay)",
        "card": "long hold: delay 2000 (constant LR inside our budgets), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, delay=2000), "locked"),
    },
    "uni_const_lr": {
        "hook": "schedule axis (constant LR, no cosine decay)",
        "card": "constant LR: delay 0 + min_lr_ratio 1.0, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, delay=0, min_lr_ratio=1.0),
            "locked",
        ),
    },
    "uni_prior01": {
        "hook": "particles axis (slow prior so the residual does the work)",
        "card": "slow prior: prior_lr_mult 0.1, particles stay jitter, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1),
            "locked",
        ),
    },
    "uni_dsteps2": {
        "hook": "critic/game axis (2 D steps per G step)",
        "card": "sharper D: d_steps 2, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, d_steps=2), "locked"),
    },
    "uni_batch64": {
        "hook": "critic/game axis (batch 64)",
        "card": "bigger batch: batch 64, less noisy Rp pairs, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, batch=64), "locked"),
    },
    "uni_nol2": {
        "hook": "particles axis (particle L2 off)",
        "card": "free particles: particle_l2 0.0, VICReg holds spread, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, particle_l2=0.0),
            "locked",
        ),
    },
    "uni_cloud01": {
        "hook": "data axis (tighter real cloud, Music span/end analogue)",
        "card": "tight modes: cloud_std 0.01, D sees poles not blur, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, cloud_std=0.01),
            "locked",
        ),
    },
    # Follow-ups on the prior01 breakthrough (screen winner at 1200/seed0).
    "uni_prior001": {
        "hook": "particles axis (slower prior than the 0.1 breakthrough)",
        "card": "slower prior: prior_lr_mult 0.01, residual does the work, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.01),
            "locked",
        ),
    },
    "uni_prior01_big": {
        "hook": "particles axis (slow prior + toward-20k particle count)",
        "card": "slow prior 0.1 + n=64, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, n_particles=64),
            "locked",
        ),
    },
    "uni_prior01_2xlr": {
        "hook": "LR multipliers #112 + slow prior (combo)",
        "card": "slow prior 0.1 + 2xLR (lr 1e-2, D x1.5), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            replace(AdvConfig(),
                    lr=1.0e-2, d_lr_mult=1.5, prior_lr_mult=0.1,
                    cover_weight=0.0, fm_weight=0.0,
                    steps=steps, seed=seed),
            "locked",
        ),
    },
    "uni_prior01_hold": {
        "hook": "schedule axis + slow prior (combo)",
        "card": "slow prior 0.1 + long LR hold (delay 2000), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, delay=2000),
            "locked",
        ),
    },
    # Robustness follow-ups: train-seed 1 collapses while 0/7 pass, so the
    # game is seed-fragile. All variants below stay GAN-only (no G MSE).
    "uni_d05": {
        "hook": "critic/game axis (weaker D: x0.5, anti lock-in)",
        "card": "gentle D: d_lr_mult 0.5, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, d_lr_mult=0.5), "locked"),
    },
    "uni_prior01_d05": {
        "hook": "critic/game axis + slow prior (combo)",
        "card": "slow prior 0.1 + gentle D x0.5, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, d_lr_mult=0.5),
            "locked",
        ),
    },
    "uni_prior01_lr2e3": {
        "hook": "schedule axis + slow prior (gentler LR combo)",
        "card": "slow prior 0.1 + lr 2e-3, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, lr=2.0e-3),
            "locked",
        ),
    },
    "uni_prior01_batch64": {
        "hook": "critic/game axis + slow prior (bigger batch combo)",
        "card": "slow prior 0.1 + batch 64, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, batch=64),
            "locked",
        ),
    },
    "uni_prior01_ema999": {
        "hook": "critic/game axis + slow prior (long EMA combo)",
        "card": "slow prior 0.1 + ema 0.999, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, ema=0.999),
            "locked",
        ),
    },
    "uni_prior01_n32": {
        "hook": "particles axis + slow prior (more particles combo)",
        "card": "slow prior 0.1 + n=32, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, n_particles=32),
            "locked",
        ),
    },
    "uni_prior01_g2x": {
        "hook": "LR multipliers #112 + slow prior (G:D 2:1 combo)",
        "card": "G 1e-2 vs D 5e-3 (lr 1e-2, d x0.5) + slow prior 0.1, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            replace(AdvConfig(),
                    lr=1.0e-2, d_lr_mult=0.5, prior_lr_mult=0.1,
                    cover_weight=0.0, fm_weight=0.0,
                    steps=steps, seed=seed),
            "locked",
        ),
    },
    "uni_lr2e3": {
        "hook": "schedule axis (gentler shared LR)",
        "card": "gentle game: lr 2e-3 shared, rest locked, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (_base(steps=steps, seed=seed, lr=2.0e-3), "locked"),
    },
    # Seed-1 dead-game follow-ups: keep D's slope informative on the
    # real->fake path G must climb + slow prior base. All GAN-only.
    "uni_prior01_const": {
        "hook": "schedule axis + slow prior (constant LR combo)",
        "card": "slow prior 0.1 + constant LR (delay 0, floor 1.0), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, delay=0, min_lr_ratio=1.0),
            "locked",
        ),
    },
    "uni_prior01_ginterp": {
        "hook": "anneal/g_interp #115 + slow prior (combo)",
        "card": "slow prior 0.1 + interp-path cap (slope bounded where G climbs), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, grad_arm="g_interp_cap"),
            "locked",
        ),
    },
    "uni_prior01_jit005": {
        "hook": "particles axis + slow prior (harder D combo)",
        "card": "slow prior 0.1 + jitter 0.05 (diffuse fakes, D cannot memorize), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, particle_jitter=0.05),
            "locked",
        ),
    },
    "uni_prior01_cloud006": {
        "hook": "data axis + slow prior (blurrier reals combo)",
        "card": "slow prior 0.1 + cloud_std 0.06, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, cloud_std=0.06),
            "locked",
        ),
    },
    "uni_prior01_coeff5": {
        "hook": "GradRegularizer coeff sweep + slow prior (flatter D combo)",
        "card": "slow prior 0.1 + b_cap coeff 5.0 (D stays flat, grads stay live), cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1, b_cap=5.0),
            "locked",
        ),
    },
    "uni_prior01_thick": {
        "hook": "critic/game axis + slow prior (thicker critic combo)",
        "card": "slow prior 0.1 + critic hidden 128 n_rand 32, cover 0",
        "match": "MATCH",
        "make": lambda steps, seed: (
            _base(steps=steps, seed=seed, prior_lr_mult=0.1,
                  critic_hidden=128, critic_n_rand=32),
            "locked",
        ),
    },
}

ARM_NAMES = tuple(ARMS)


def arm_cfg(arm: str, *, steps: int, seed: int) -> tuple[AdvConfig, str]:
    """Build ``(cfg, vicreg_fn_key)`` for a propose_only sweep arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown uni arm {arm!r} (expected one of {ARM_NAMES})")
    cfg, vfn_key = ARMS[arm]["make"](int(steps), int(seed))
    if vfn_key not in ("locked", "faithful"):
        raise ValueError(f"arm {arm!r} vicreg_fn key must be locked/faithful")
    return cfg, vfn_key


def run_sweep(
    *,
    arms: list[str] | None = None,
    budgets: tuple[int, ...] = BUDGETS,
    seeds: tuple[int, ...] = SEEDS,
    cells: tuple[str, ...] = ("divergent", "close"),
    jobs: int = 1,
) -> list[dict]:
    """Score every arm x budget x seed x cell. Sequential unless jobs > 1."""
    arms = list(arms) if arms else list(ARM_NAMES)
    tasks = [
        (arm, cell, budget, seed)
        for arm in arms
        for budget in budgets
        for seed in seeds
        for cell in cells
    ]
    if int(jobs) <= 1:
        return [score_uni_arm(a, c, steps=b, seed=s) for a, c, b, s in tasks]
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(jobs)) as pool:
        futures = [
            pool.submit(_score_task, a, c, b, s) for a, c, b, s in tasks
        ]
        return [f.result() for f in futures]


def _score_task(arm: str, cell: str, steps: int, seed: int) -> dict:
    import torch as _torch

    _torch.set_num_threads(1)
    return score_uni_arm(arm, cell, steps=int(steps), seed=int(seed))


def summarize(rows: list[dict]) -> list[dict]:
    """Per arm x budget: mean cover/off/hold + PASS iff all seeds+cells hit."""
    out = []
    arms = sorted({r["arm"] for r in rows})
    budgets = sorted({r["steps"] for r in rows})
    for arm in arms:
        for budget in budgets:
            sub = [r for r in rows if r["arm"] == arm and r["steps"] == budget]
            if not sub:
                continue
            cells = sorted({r["cell"] for r in rows})
            per_cell = {}
            for cell in cells:
                cs = [r for r in sub if r["cell"] == cell]
                per_cell[cell] = {
                    "cover": sum(r["cover"] for r in cs) / len(cs),
                    "off_caption": sum(r["off_caption"] for r in cs) / len(cs),
                    "neu_hold": sum(r["neu_hold"] for r in cs) / len(cs),
                    "hit_all_seeds": all(r["hit"] for r in cs),
                    "seeds": {str(r["seed"]): bool(r["hit"]) for r in cs},
                }
            out.append(
                {
                    "arm": arm,
                    "steps": budget,
                    "pass": all(v["hit_all_seeds"] for v in per_cell.values()),
                    "hook": ARMS[arm]["hook"],
                    "card": ARMS[arm]["card"],
                    "match": ARMS[arm]["match"],
                    "cells": per_cell,
                }
            )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", type=str, nargs="+", default=list(ARM_NAMES))
    parser.add_argument(
        "--cells", type=str, nargs="+", default=["divergent", "close"]
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    rows = run_sweep(
        arms=list(args.arms),
        budgets=tuple(int(s) for s in args.steps),
        seeds=tuple(int(s) for s in args.seeds),
        cells=tuple(args.cells),
        jobs=int(args.jobs),
    )
    summary = summarize(rows)
    blob = {
        "propose_only": True,
        "gan_only": True,
        "no_supervised_mse": True,
        "music_untouched": True,
        "gates": {
            "cover_min": UNI_COVER_MIN,
            "off_caption_max": UNI_OFF_MAX,
            "neu_hold_min": UNI_HOLD_MIN,
            "cells": ["divergent", "close"],
            "eval_scales": [0, 0.5, 1],
            "canary": -1,
        },
        "arms": {
            name: {
                "hook": ARMS[name]["hook"],
                "card": ARMS[name]["card"],
                "match": ARMS[name]["match"],
            }
            for name in args.arms
        },
        "summary": summary,
        "rows": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=2, default=str) + "\n")
    for entry in summary:
        cells = entry["cells"]
        print(
            f"{entry['arm']:24s} steps={entry['steps']:5d} "
            f"{'PASS' if entry['pass'] else 'FAIL':4s} "
            + " ".join(
                f"{c}:cov={v['cover']:.3f}/off={v['off_caption']:.3f}/hold={v['neu_hold']:.3f}"
                for c, v in cells.items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
