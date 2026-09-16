"""Unipolar PairField toy gates with GAN-only ParticleGAN training (UniPG-A).

Missing-piece harness for the unipolar + ParticleGAN sweep: the repo has a
bipolar RpGAN + ``b_cap`` toy (``analysis.slider2d.gan``) and a supervised
UNI teacher (``faithful_plus_neu`` MSE), but no GAN-only unipolar exam. This
module is that exam, scoped to the UniPG-A arm identity: **GradRegularizer /
``b_cap`` sweep**, keeping the Rp logistic + unipolar raw-positive teacher
fixed.

Hard constraints (fail-closed, see ``uni_gan_cfg`` / ``fit_uni_gan``):

- GAN-only G: paired Rp logistic (``rp_g_loss`` / ``rp_d_loss``) plus the
  ParticleGAN particle terms (VICReg, particle L2) only. No positive MSE,
  cover MSE, ending, FM, lyric-hold, plan, or zero-anchor on G.
- Unipolar: +1 student vs raw-positive teacher only
  (``lm_faithful_plus_neu`` contract: the teacher *is* ``pos``). No
  negative captions, bipolar ranges, leak-axis metadata, or -1 train
  branch. ``leak_dir`` is accepted and ignored so the call site matches
  the other exams; passing one never changes the teacher.
- Scale 0 is the base / exact zero by construction: the student is a
  single delta with no ``w0`` (``delta(0) == 0`` exactly, mirroring the
  YuE2 adapter construction). ``neu_hold`` is still *measured*, not
  asserted.
- Spine always includes ParticleGAN Rp + a GradRegularizer arm (default
  ``b_cap``). No WGAN/hinge/LSGAN primary loss.
- ``PROPOSE_ONLY``: nothing here changes ``AdvConfig()`` defaults, the
  Music bipolar ``ARM_B`` row, ``locked_shared``, or the live
  ``--lm_target`` default.

Gates (both required cells: divergent, close; eval scales 0 / 0.5 / 1):

- cover @ +1 >= 0.85, off-caption / leak @ +1 <= 0.05, neu_hold @ 0 >= 0.85.
- Scale 0.5 is a logged interpolation diagnostic (``half_cover``).
- Scale -1 is an unscored canary. Antipodal ``cos(+1, -1)`` is never a gate.

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

from analysis.slider2d.adv import (
    AdvConfig,
    EMA,
    Fourier2MLP,
    ParticlePrior,
    _l2_norm,
    delayed_cosine,
    input_grad,
    make_grad_regularizer,
    rp_d_loss,
    rp_g_loss,
    sample_real_cloud,
    toy_lr_triplet,
    vicreg_loss,
)
from analysis.slider2d.exam import (
    PairField,
    close_field,
    divergent_field,
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
from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders.slider_targets import lm_faithful_plus_neu

PROPOSE_ONLY = True
MERGE_TO_TRAINER = False

REQUIRED_CELLS = ("divergent", "close")
CELL_CTORS = {
    "divergent": divergent_field,
    "close": close_field,
}

# UniPG-A may only move these knobs (GradRegularizer axes + compute
# budget). Particles / VICReg / LR stay at production defaults; any other
# key raises instead of silently forking the recipe.
UNIPG_A_KNOBS = frozenset(
    {"steps", "seed", "grad_arm", "b_cap", "kappa", "grad_lazy", "target_anneal"}
)

# GradRegularizer sweep: (arm name, knob delta, one-line identity card).
# Every arm carries the single companion knob ``prior``: "jitter" freezes
# the ParticlePrior to a pure jitter prior (production YuE2 unipolar has
# no learned particle branch); "learned" keeps the bipolar-rig wiring as a
# DRIFT control documenting particle-mode theft. Formulation tags come
# from formulation_tag (MATCH = Rp + b_cap-family + GAN-only + unipolar
# raw-positive + jitter prior).
UNIPG_A_ARMS: dict[str, dict] = {
    "unipg_a_baseline": {
        "delta": {},
        "prior": "jitter",
        "blurb": "locked b_cap k=1 coeff=1, anneal none, lazy 1, jitter prior (GAN-only unipolar baseline)",
    },
    "unipg_a_kappa05": {
        "delta": {"kappa": 0.5},
        "prior": "jitter",
        "blurb": "tighter cap: b_cap k=0.5, coeff=1 (D flatter, G gradients smaller)",
    },
    "unipg_a_kappa20": {
        "delta": {"kappa": 2.0},
        "prior": "jitter",
        "blurb": "looser cap: b_cap k=2, coeff=1 (D steeper before penalty bites)",
    },
    "unipg_a_coeff05": {
        "delta": {"b_cap": 0.5},
        "prior": "jitter",
        "blurb": "softer cap: b_cap coeff=0.5, k=1 (half pressure above kappa)",
    },
    "unipg_a_coeff20": {
        "delta": {"b_cap": 2.0},
        "prior": "jitter",
        "blurb": "harder cap: b_cap coeff=2, k=1 (double pressure above kappa)",
    },
    "unipg_a_ginterp": {
        "delta": {"grad_arm": "g_interp_cap"},
        "prior": "jitter",
        "blurb": "interp-path cap: g_interp_cap k=1 coeff=1 (cap between samples, not at them)",
    },
    "unipg_a_delayed": {
        "delta": {"target_anneal": "delayed"},
        "prior": "jitter",
        "blurb": "delayed anneal: b_cap center holds k=1 to 60% then ramps to 0 (flat-D handover)",
    },
    "unipg_a_linear": {
        "delta": {"target_anneal": "linear"},
        "prior": "jitter",
        "blurb": "linear anneal: b_cap center ramps k=1 to 0 over the run (early no-flat-D, late R1/R2-like)",
    },
    "unipg_a_lazy2": {
        "delta": {"grad_lazy": 2},
        "prior": "jitter",
        "blurb": "lazy reg_every=2: b_cap applied every 2nd step at 2x coeff (StyleGAN2 lazy)",
    },
    "unipg_a_learned_prior": {
        "delta": {},
        "prior": "learned",
        "blurb": "DRIFT control: locked cap but learned 12-particle prior under G (mode theft demo, expect FAIL)",
    },
}


def uni_gan_cfg(**overrides) -> AdvConfig:
    """GAN-only unipolar config: locked defaults, cover/FM pinned to 0.

    Only ``UNIPG_A_KNOBS`` may be overridden (GradRegularizer axes +
    budget). Anything else — LR, particles, VICReg, cover, FM, critic,
    schedule — raises: UniPG-A keeps those at production defaults.
    """
    bad = [k for k in overrides if k not in UNIPG_A_KNOBS]
    if bad:
        raise ValueError(
            f"uni_gan_cfg refuses non-UniPG-A overrides {bad}: "
            f"UniPG-A moves only {sorted(UNIPG_A_KNOBS)}"
        )
    cfg = replace(AdvConfig(), cover_weight=0.0, fm_weight=0.0, **overrides)
    _assert_gan_only(cfg)
    return cfg


def arm_cfg(name: str, **budget) -> AdvConfig:
    """A propose_only UniPG-A arm: baseline plus exactly its named delta."""
    if name not in UNIPG_A_ARMS:
        raise ValueError(
            f"unknown UniPG-A arm {name!r} (expected one of {sorted(UNIPG_A_ARMS)})"
        )
    arm = UNIPG_A_ARMS[name]
    if arm.get("status", "propose_only") != "propose_only":
        raise ValueError(f"arm {name!r} is not propose_only")
    bad = [k for k in budget if k not in ("steps", "seed")]
    if bad:
        raise ValueError(f"arm_cfg budget keys are steps/seed, got {bad}")
    merged = {**arm["delta"], **budget}
    return uni_gan_cfg(**merged)


def _assert_gan_only(cfg: AdvConfig) -> None:
    """Fail-closed: no supervised / helper loss on G."""
    violations = []
    if float(cfg.cover_weight) != 0.0:
        violations.append(f"cover_weight={cfg.cover_weight!r} (supervised mode pin)")
    if float(cfg.fm_weight) != 0.0:
        violations.append(f"fm_weight={cfg.fm_weight!r} (feature matching)")
    if violations:
        raise ValueError(
            "yue2_gan_exam is GAN-only; refusing G losses: " + "; ".join(violations)
        )


def formulation_tag(cfg: AdvConfig, *, freeze_prior: bool = False) -> str:
    """ParticleGAN MATCH vs DRIFT tag for a scored config.

    MATCH: Rp logistic game + b_cap-family GradRegularizer + GAN-only G +
    unipolar raw-positive teacher + jitter (frozen) prior — i.e. the
    production YuE2 unipolar loop in toy form (that recipe has no learned
    particle branch and no VICReg). DRIFT: any deviation — a learned
    particle branch under G (bipolar-rig leftover that absorbs the mode),
    supervised G terms, or a non-cap arm.
    """
    try:
        _assert_gan_only(cfg)
    except ValueError:
        return "DRIFT"
    if str(cfg.grad_arm) not in ("b_cap", "g_interp_cap"):
        return "DRIFT"
    if not freeze_prior:
        return "DRIFT"
    return "MATCH"


class UniResidual:
    """Single-delta unipolar student: ``delta(s) = s * w``.

    No ``w0``: scale 0 is the base / exact zero by construction (the YuE2
    adapter property). Scale -1 is the deterministic mirror (canary only).
    """

    def __init__(self, w: torch.Tensor):
        self.w = w

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w

    def parameters(self) -> list[torch.Tensor]:
        return [self.w]

    def snapshot(self) -> "UniResidual":
        return UniResidual(self.w.detach().clone())


def uni_teacher_plus(
    field: PairField,
    row: int,
) -> torch.Tensor:
    """Raw-positive teacher (the ``lm_faithful_plus_neu`` contract)."""
    pos, neg, neu = field.poles(row)
    return lm_faithful_plus_neu(pos, neg, neu, None)


@isolated_seed("cfg.seed", default=0)
def fit_uni_gan(
    field: PairField,
    *,
    cfg: AdvConfig | None = None,
    leak_dir: torch.Tensor | None = None,
    vicreg_fn=None,
    freeze_prior: bool = False,
) -> tuple[UniResidual, dict]:
    """Fit the unipolar delta with RpGAN + GradRegularizer cap. GAN-only.

    ``leak_dir`` is accepted and ignored (call-site parity with the other
    exams): the teacher is raw ``pos`` with or without it. ``vicreg_fn``
    overrides the particle regularizer (default locked ``vicreg_loss``).
    ``freeze_prior`` keeps the ParticlePrior a pure jitter prior (its
    parameters leave G's optimizer; still sampled): the production YuE2
    unipolar loop has no learned particle branch, and a learned 12-particle
    prior under G absorbs the mode while the scored residual stalls.
    """
    cfg = cfg or uni_gan_cfg()
    _assert_gan_only(cfg)
    vfn = vicreg_fn if vicreg_fn is not None else vicreg_loss
    dim = int(field.dim)
    residual = UniResidual(torch.zeros(dim, requires_grad=True))
    prior = ParticlePrior(cfg.n_particles, dim)
    critic = Fourier2MLP(
        dim,
        n_rand=cfg.critic_n_rand,
        hidden=cfg.critic_hidden,
        seed=cfg.seed,
    )
    g_lr, d_lr, prior_lr = toy_lr_triplet(cfg)
    g_groups: list[dict] = [{"params": residual.parameters(), "lr": g_lr}]
    if not freeze_prior:
        g_groups.append({"params": list(prior.parameters()), "lr": prior_lr})
    opt_g = torch.optim.Adam(
        g_groups,
        lr=g_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    opt_d = torch.optim.Adam(critic.parameters(), lr=d_lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(residual.parameters(), decay=cfg.ema)

    poles_p, neus = [], []
    for row in range(int(field.rows)):
        poles_p.append(uni_teacher_plus(field, row).flatten())
        neus.append(field.poles(row)[2].flatten())
    poles_p = torch.stack(poles_p)
    neus_t = torch.stack(neus)

    logs = {"d": [], "g": [], "cap": [], "grad_real": [], "grad_fake": []}
    reg = make_grad_regularizer(cfg)

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step, total=cfg.steps, delay=cfg.delay, min_ratio=cfg.min_lr_ratio
        )
        opt_g.param_groups[0]["lr"] = g_lr * scale
        if len(opt_g.param_groups) > 1:
            # Learned-prior arm only: the jitter-prior build has no group 1.
            opt_g.param_groups[1]["lr"] = prior_lr * scale
        for group in opt_d.param_groups:
            group["lr"] = d_lr * scale

    def fake_batch() -> torch.Tensor:
        idx = torch.randint(0, neus_t.shape[0], (int(cfg.batch),))
        return (
            neus_t[idx]
            + residual.delta(1.0)
            + prior.sample(int(cfg.batch), jitter=cfg.particle_jitter)
        )

    for step in range(int(cfg.steps)):
        set_lr(step)
        real = sample_real_cloud(
            poles_p,
            neus_t,
            n=int(cfg.batch),
            cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac,
            end_margin=cfg.end_margin,
        )
        for _ in range(int(cfg.d_steps)):
            fake = fake_batch().detach()
            real_g = real.detach()
            d_real = critic(real_g)
            d_fake = critic(fake)
            cap, _cap_stats = reg.penalty(critic, real_g, fake, step=step + 1)
            d_loss = rp_d_loss(d_real, d_fake) + cap
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

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == cfg.steps:
            probe_r = real.detach().requires_grad_(True)
            probe_f = fake.detach().requires_grad_(True)
            gn_r = _l2_norm(input_grad(critic, probe_r)).mean()
            gn_f = _l2_norm(input_grad(critic, probe_f)).mean()
            logs["d"].append(float(d_loss.detach()))
            logs["g"].append(float(g_loss.detach()))
            logs["cap"].append(float(cap.detach()))
            logs["grad_real"].append(float(gn_r.detach()))
            logs["grad_fake"].append(float(gn_f.detach()))

    ema.copy_to(residual.parameters())
    snap = residual.snapshot()
    stats = {
        "d_loss": logs["d"][-1] if logs["d"] else None,
        "g_loss": logs["g"][-1] if logs["g"] else None,
        "cap": logs["cap"][-1] if logs["cap"] else None,
        "grad_real": logs["grad_real"][-1] if logs["grad_real"] else None,
        "grad_fake": logs["grad_fake"][-1] if logs["grad_fake"] else None,
        "teacher": "faithful_plus_neu(raw-positive)",
        "grad_arm": str(cfg.grad_arm),
        "b_cap": float(cfg.b_cap),
        "kappa": float(cfg.kappa),
        "grad_lazy": int(cfg.grad_lazy),
        "target_anneal": str(cfg.target_anneal),
        "steps": int(cfg.steps),
        "seed": int(cfg.seed),
        "prior": "jitter" if freeze_prior else "learned",
        "tag": formulation_tag(cfg, freeze_prior=freeze_prior),
    }
    return snap, stats


def score_uni_residual(
    name: str,
    field: PairField,
    residual: UniResidual,
) -> dict:
    """Score a fitted unipolar delta: cover / leak / neu_hold + diagnostics."""
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    d_zero = residual.delta(0.0)
    d_half = residual.delta(0.5)
    cover_rows, off_rows = [], []
    neu_hold_rows = []
    half_rows = []
    canary_landed: list[str] = []
    canary_off: list[float] = []
    sings_plus, sings_zero, sings_minus = [], [], []
    head = field.readout()
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        plus_seqs = _continue(field, neu + d_plus, row=row, sign=1.0)
        zero_seqs = _continue(field, neu + d_zero, row=row, sign=0.0)
        minus_seqs = _continue(field, neu + d_minus, row=row, sign=-1.0)
        half_seqs = _continue(field, neu + d_half, row=row, sign=1.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        blend = blend_toward_mid(neu + d_plus, pos, mid, neg)
        cover_rows.append(plus_cover(overlap, blend))
        off_rows.append(off)
        neu_ov = _token_share(zero_seqs, neu_bag)
        drift = drift_from_neu(neu + d_zero, neu, pos, mid)
        neu_hold_rows.append(neu_hold(neu_ov, drift))
        half_rows.append(
            plus_cover(
                _token_share(half_seqs, bags["pos"]),
                blend_toward_mid(neu + d_half, pos, mid, neg),
            )
        )
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(neu + d_minus, pos, neu, neg))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_zero.append(" ".join(head.tokens[t] for t in zero_seqs[0]))
        sings_minus.append(" ".join(head.tokens[t] for t in minus_seqs[0]))
    cover = sum(cover_rows) / len(cover_rows)
    off_caption = sum(off_rows) / len(off_rows)
    hold = sum(neu_hold_rows) / len(neu_hold_rows)
    hit = bool(
        cover >= PLUS_COVER_MIN
        and off_caption <= PLUS_OFF_MAX
        and hold >= PLUS_NEU_HOLD_MIN
    )
    landed = max(set(canary_landed), key=canary_landed.count)
    canary_off_mean = sum(canary_off) / len(canary_off)
    return {
        "name": name,
        "cell": field.kind,
        "teacher": "faithful_plus_neu(raw-positive)",
        "train": "unipolar-gan-only",
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold,
        "hit": hit,
        "half_scale": {"scored": False, "scale": 0.5, "half_cover": sum(half_rows) / len(half_rows)},
        "canary": {
            "scored": False,
            "minus_landed": landed,
            "minus_off_caption": canary_off_mean,
            "minus_sings": " | ".join(sings_minus),
            "dangerous": bool(landed == "pos" or canary_off_mean > PLUS_OFF_MAX),
        },
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
    }


def score_uni_gan(
    name: str,
    field: PairField,
    *,
    cfg: AdvConfig | None = None,
    freeze_prior: bool = False,
) -> dict:
    """Fit GAN-only unipolar, then score the uni gates."""
    cfg = cfg or uni_gan_cfg()
    residual, stats = fit_uni_gan(field, cfg=cfg, freeze_prior=freeze_prior)
    out = score_uni_residual(name, field, residual)
    out.update({k: v for k, v in stats.items()})
    out["gates"] = {
        "cover_min": PLUS_COVER_MIN,
        "off_caption_max": PLUS_OFF_MAX,
        "neu_hold_min": PLUS_NEU_HOLD_MIN,
    }
    return out


def run_exam(
    *,
    arms: list[str],
    steps_list: list[int],
    seeds: list[int],
) -> dict:
    """Score every arm x budget x seed on the required cells."""
    blob: dict = {
        "arms": {},
        "required_cells": list(REQUIRED_CELLS),
        "eval_scales": [0.0, 0.5, 1.0],
        "gates": {
            "cover_min": PLUS_COVER_MIN,
            "off_caption_max": PLUS_OFF_MAX,
            "neu_hold_min": PLUS_NEU_HOLD_MIN,
        },
        "gan_only": True,
        "propose_only": PROPOSE_ONLY,
    }
    for arm in arms:
        delta = dict(UNIPG_A_ARMS[arm]["delta"])
        freeze_prior = UNIPG_A_ARMS[arm].get("prior", "jitter") == "jitter"
        entry: dict = {
            "blurb": UNIPG_A_ARMS[arm]["blurb"],
            "knobs": {"grad_arm": "b_cap", "b_cap": 1.0, "kappa": 1.0,
                       "grad_lazy": 1, "target_anneal": "none",
                       "prior": "jitter" if freeze_prior else "learned",
                       **delta},
            "budgets": {},
        }
        for steps in steps_list:
            budget: dict = {"seeds": {}, "pass": True}
            for seed in seeds:
                cfg = arm_cfg(arm, steps=steps, seed=seed)
                per_cell = {}
                seed_pass = True
                for cell in REQUIRED_CELLS:
                    field = CELL_CTORS[cell](seed=seed)
                    row = score_uni_gan(arm, field, cfg=cfg,
                                        freeze_prior=freeze_prior)
                    row.pop("sings_plus", None)
                    row.pop("sings_zero", None)
                    row["canary"].pop("minus_sings", None)
                    per_cell[cell] = row
                    seed_pass = seed_pass and bool(row["hit"])
                budget["seeds"][str(seed)] = {"cells": per_cell, "pass": seed_pass}
                budget["pass"] = budget["pass"] and seed_pass
            entry["budgets"][str(steps)] = budget
        blob["arms"][arm] = entry
    blob["pass"] = all(
        b["pass"] for arm in blob["arms"].values() for b in arm["budgets"].values()
    )
    return blob


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=str, default="all",
                        help="'all' or comma-separated UniPG-A arm names")
    parser.add_argument("--steps", type=int, nargs="+", default=[1200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.arms.strip() == "all":
        arms = sorted(UNIPG_A_ARMS)
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in arms if a not in UNIPG_A_ARMS]
        if unknown:
            raise SystemExit(f"unknown arms: {unknown} (known: {sorted(UNIPG_A_ARMS)})")
    blob = run_exam(arms=arms, steps_list=list(args.steps), seeds=list(args.seeds))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=2, default=str) + "\n",
                            encoding="utf-8")
    for arm, entry in blob["arms"].items():
        for steps, budget in entry["budgets"].items():
            cells = []
            for seed, s in budget["seeds"].items():
                det = ",".join(
                    f"{c}={'HIT' if r['hit'] else 'miss'}"
                    f"(cov={r['cover']:.3f},off={r['off_caption']:.3f},hold={r['neu_hold']:.3f})"
                    for c, r in s["cells"].items()
                )
                cells.append(f"seed{seed}:[{det}]")
            print(f"{arm} steps={steps} {'PASS' if budget['pass'] else 'FAIL'} " + " ".join(cells))
    # Exits 1 unless every requested budget passes — expected at 600/1200.
    return 0 if blob["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
