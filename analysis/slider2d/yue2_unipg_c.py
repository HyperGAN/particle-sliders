"""UniPG-C: ParticleGAN schedule / optimizer clone arms on the YuE2 toy port.

Propose-only formulation search over the schedule/optimizer DRIFTs listed in
the ParticleGAN formulation gap (vs the production YuE2 unipolar GAN-only
loop in ``conceptmod.textsliders.yue2_arm_b``):

- D:G LR multiplier (production x1.5 D; ParticleGAN x1.5 D / x10 prior --
  the prior term is N/A here: the YuE2 port has no ParticlePrior, so no
  prior-LR arm is proposed, and that gap is tagged HOLD).
- LR schedule: production ``constant`` vs absolute-delay cosine (locked-toy
  ``delay=80``) vs ParticleGAN 60%-hold cosine, floor 0.05.
- Adam beta2 0.999 (production) vs 0.99 (ParticleGAN / locked toy).
- EMA 0.995 residual-only on/off (production: off; ParticleGAN proper keeps
  EMA over G+particles -- residual-only is the faithful analogue with no
  particle store, tagged PARTIAL).
- Fourier-2 critic (locked toy, width 64) vs the production thicker MLP
  critic (width 256 x 2 layers), if needed for cover at 600-1200 steps.

Hard constraints (every arm):

- GAN-only G: paired Rp logistic (``rp_g_loss`` / ``rp_d_loss``) plus the
  ``b_cap`` GradRegularizer arm only. No positive MSE, cover MSE, ending,
  FM, lyric-hold, plan, or zero-anchor on G.
- Unipolar: +1 student vs raw-positive teacher only
  (``faithful_plus_neu`` contract for what + is). Scale 0 is exact by
  construction. No negative captions, bipolar ranges, or -1 train branch.
- Spine ``b_cap`` kappa=1 coeff=1 on every arm; no VICReg / particles /
  cover companions except the single Fourier-critic arm (c8), which still
  adds no G-side loss.
- ``PROPOSE_ONLY = True``, ``MERGE_TO_TRAINER = False``: nothing here
  changes ``yue2_arm_b.RECIPE``, the Music bipolar ARM_B row,
  ``locked_shared``, or the live ``--lm_target`` default.

The ``c0_production`` entry is the reference row (production knobs through
this harness, used for parity against ``yue2_gan_exam``); ``c1``-``c8`` are
the propose-only schedule/optimizer arms.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from analysis.slider2d.adv import delayed_cosine, make_grad_regularizer, rp_d_loss, rp_g_loss
from analysis.slider2d.adv import Fourier2MLP
from analysis.slider2d.formulation_leaderboard import REQUIRED_CELLS, unipolar_hit
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_CELLS,
    score_plus_neu_exam,
    score_plus_neu_residual,
)
from analysis.slider2d.rng import isolated_seed
from analysis.slider2d.yue2_gan_exam import ToyBackend, ToySlider, accepted
from conceptmod.textsliders import yue2_arm_b as game
from conceptmod.textsliders.lm_adv import LMDiscriminator, param_grad_norm

PROPOSE_ONLY = True
MERGE_TO_TRAINER = False

AUDIT_SEEDS = (0, 1, 7)
BUDGETS = (600, 1200, 3400)

# b_cap spine pinned on every arm (ParticleGAN FINDINGS champion).
B_CAP = 1.0
KAPPA = 1.0

# Production optimizer/schedule pins (mirrored, never mutated).
PROD_G_LR = 0.0005
PROD_D_LR = 0.00075
PROD_BETAS = (0.0, 0.999)


class FourierScaledCritic(nn.Module):
    """Fourier-2 critic in the fixed teacher-RMS coordinate system.

    Same calibration contract as ``LMDiscriminator`` in ``scaled`` mode:
    ``forward(delta) = net(delta / input_scale)`` and the ``b_cap`` penalty
    is measured on ``net`` at ``real/scale`` / ``fake/scale``. The inner
    ``net`` is the locked-toy ``Fourier2MLP`` (width 64).
    """

    def __init__(self, dim, *, hidden=64, n_rand=16, seed=0):
        super().__init__()
        self.net = Fourier2MLP(dim, n_rand=n_rand, hidden=hidden, seed=seed)
        self.register_buffer("input_scale", torch.tensor(1.0))

    @torch.no_grad()
    def calibrate_input_scale(self, real):
        values = real.detach().float()
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("calibration requires nonempty finite teacher deltas")
        scale = values.square().mean().sqrt()
        if not bool(scale > 0):
            raise ValueError("calibration requires a nonzero teacher delta")
        self.input_scale.copy_(scale.to(self.input_scale))
        return float(self.input_scale)

    def forward(self, delta):
        return self.net(delta.float() / self.input_scale)


ARMS: dict[str, dict] = {
    "c0_production": {
        "identity": "reference: production YuE2 knobs (constant LR, beta2 0.999, no EMA, thick critic)",
        "propose_only": False,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.999,
        "schedule": "constant",
        "ema": 0.0,
        "critic": "thick256",
        "tags": {"schedule": "REFERENCE", "beta2": "REFERENCE", "ema": "REFERENCE", "critic": "REFERENCE"},
    },
    "c1_beta2_099": {
        "identity": "beta2 0.99 (ParticleGAN / locked-toy value); else production",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.99,
        "schedule": "constant",
        "ema": 0.0,
        "critic": "thick256",
        "tags": {"schedule": "MATCH(constant=production drift; beta2 is the clone)", "beta2": "MATCH", "ema": "DRIFT(off)", "critic": "DRIFT(thick)"},
    },
    "c2_ema_on": {
        "identity": "EMA 0.995 residual-only on; else production",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.999,
        "schedule": "constant",
        "ema": 0.995,
        "critic": "thick256",
        "tags": {"schedule": "DRIFT(constant)", "beta2": "DRIFT(0.999)", "ema": "PARTIAL(residual-only; proper is G+particles)", "critic": "DRIFT(thick)"},
    },
    "c3_delay80": {
        "identity": "delayed cosine, absolute delay 80, floor 0.05; else production",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.999,
        "schedule": "delayed_abs",
        "ema": 0.0,
        "critic": "thick256",
        "tags": {"schedule": "MATCH(locked-toy delay80)", "beta2": "DRIFT(0.999)", "ema": "DRIFT(off)", "critic": "DRIFT(thick)"},
    },
    "c4_hold60": {
        "identity": "delayed cosine, 60% hold (delay=0.6*budget), floor 0.05; else production",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.999,
        "schedule": "hold60",
        "ema": 0.0,
        "critic": "thick256",
        "tags": {"schedule": "MATCH(ParticleGAN 60% hold)", "beta2": "DRIFT(0.999)", "ema": "DRIFT(off)", "critic": "DRIFT(thick)"},
    },
    "c5_d1x": {
        "identity": "shared LR (D 1.0x: G 5e-4 / D 5e-4); else production",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_G_LR,
        "beta2": 0.999,
        "schedule": "constant",
        "ema": 0.0,
        "critic": "thick256",
        "tags": {"schedule": "DRIFT(1.0x D; proper is 1.5x)", "beta2": "DRIFT(0.999)", "ema": "DRIFT(off)", "critic": "DRIFT(thick)"},
    },
    "c6_g2x": {
        "identity": "2x G LR keeping 1.5x D (G 1e-3 / D 1.5e-3); else production",
        "propose_only": True,
        "g_lr": 0.001,
        "d_lr": 0.0015,
        "beta2": 0.999,
        "schedule": "constant",
        "ema": 0.0,
        "critic": "thick256",
        "tags": {"schedule": "DRIFT(2x LR hunt for early cover)", "beta2": "DRIFT(0.999)", "ema": "DRIFT(off)", "critic": "DRIFT(thick)"},
    },
    "c7_sched_clone": {
        "identity": "schedule clone: beta2 0.99 + EMA on + 60% hold cosine + D 1.5x, thick critic",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.99,
        "schedule": "hold60",
        "ema": 0.995,
        "critic": "thick256",
        "tags": {"schedule": "MATCH(60% hold)", "beta2": "MATCH", "ema": "PARTIAL(residual-only)", "critic": "DRIFT(thick)"},
    },
    "c8_fourier_sched": {
        "identity": "c7 schedule clone + Fourier-2 critic w64 (full locked-toy optimizer/critic/schedule clone; Rp-only G)",
        "propose_only": True,
        "g_lr": PROD_G_LR,
        "d_lr": PROD_D_LR,
        "beta2": 0.99,
        "schedule": "hold60",
        "ema": 0.995,
        "critic": "fourier64",
        "tags": {"schedule": "MATCH(60% hold)", "beta2": "MATCH", "ema": "PARTIAL(residual-only)", "critic": "MATCH(Fourier-2)"},
    },
}

# Shared gap ledger for every arm: no particles / VICReg / cover on G.
SHARED_TAGS = {
    "gan_loss": "MATCH(Rp pair logistic)",
    "b_cap": "MATCH(kappa=1 coeff=1)",
    "particles": "HOLD(no ParticlePrior in YuE2 port; prior x10 N/A)",
    "vicreg": "HOLD(none; Rp-only G)",
    "cover_weight": "HOLD(0; Rp-only G, no MSE)",
    "teacher": "HOLD(raw positive faithful_plus_neu; not modes/spans)",
}


def lr_scale(step: int, arm: dict, total: int) -> float:
    sched = arm["schedule"]
    if sched == "constant":
        return 1.0
    if sched == "delayed_abs":
        return delayed_cosine(step, total=total, delay=80, min_ratio=0.05)
    if sched == "hold60":
        return delayed_cosine(step, total=total, delay=int(0.6 * total), min_ratio=0.05)
    raise ValueError(f"unknown schedule {sched!r}")


def build_game_variant(backend, network, rows, arm: dict):
    device = next(backend.model.parameters()).device
    dim = int(rows[0]["targets"].numel())
    if arm["critic"] == "thick256":
        critic = LMDiscriminator(dim, hidden_dim=256, n_hidden=2, in_mode="scaled").to(device)
    elif arm["critic"] == "fourier64":
        critic = FourierScaledCritic(dim, hidden=64, n_rand=16, seed=0).to(device)
    else:
        raise ValueError(f"unknown critic {arm['critic']!r}")
    real = torch.cat([r["targets"] - r["neutral"] for r in rows]).to(device)
    critic.calibrate_input_scale(real)
    betas = (0.0, float(arm["beta2"]))
    g = torch.optim.AdamW(network.parameters(), lr=float(arm["g_lr"]), betas=betas, weight_decay=1e-6)
    d = torch.optim.Adam(critic.parameters(), lr=float(arm["d_lr"]), betas=betas)
    return critic, g, d


def make_regularizer():
    return make_grad_regularizer(arm="b_cap", coeff=B_CAP, kappa=KAPPA, norm="l2", lazy_k=1, target_anneal="none")


class ResidualEMA:
    """EMA over the toy residual (odd/even), decay 0.995, residual-only scope."""

    def __init__(self, network: ToySlider, decay: float):
        self.decay = float(decay)
        self.shadow = [network.odd.detach().clone(), network.even.detach().clone()]

    def update(self, network: ToySlider) -> None:
        for s, p in zip(self.shadow, (network.odd, network.even)):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def snapshot(self):
        from analysis.slider2d.gan import AdvResidual

        return AdvResidual(self.shadow[0].clone(), self.shadow[1].clone())


def update_variant(backend, network, critic, g, d, rows, *, step, arm, total, checkpointing=False):
    """Production update with per-arm LR schedule. Loss/cap/order unchanged."""
    if not rows:
        raise ValueError("Empty training batch")
    scale = lr_scale(step, arm, total)
    for group in g.param_groups:
        group["lr"] = float(arm["g_lr"]) * scale
    for group in d.param_groups:
        group["lr"] = float(arm["d_lr"]) * scale
    device = next(critic.parameters()).device
    regularizer = make_regularizer()
    critic.requires_grad_(True)
    d.zero_grad(set_to_none=True)
    real = []
    fake = []
    with torch.no_grad():
        for row in rows:
            neutral = row["neutral"].to(device)
            with network.scaled(1.0):
                pred = backend.hidden(row["ids"])[:, row["prefix_len"] - 1].float()
            real.append(row["targets"].to(device) - neutral)
            fake.append(pred - neutral)
    real = torch.cat(real)
    fake = torch.cat(fake)
    cap_scale = critic.input_scale.detach()
    cap, stats = regularizer.penalty(critic.net, real / cap_scale, fake / cap_scale, step=step)
    d_loss = rp_d_loss(critic(real), critic(fake)) + cap
    if not torch.isfinite(d_loss):
        raise FloatingPointError("Non-finite D loss")
    d_loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in critic.parameters()):
        raise FloatingPointError("Non-finite D gradient")
    d.step()
    d.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    with torch.no_grad():
        real_scores = critic(real)
    g.zero_grad(set_to_none=True)
    totals = dict(g_adv=0.0, cos_pos=0.0)
    for i, row in enumerate(rows):
        neutral = row["neutral"].to(device)
        with network.scaled(1.0):
            pred = game.forward(backend, row, checkpointing)
            target = row["targets"].to(device)
            adv = rp_g_loss(real_scores[i : i + 1], critic(pred - neutral))
            loss = adv / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite G loss")
            loss.backward()
        totals["g_adv"] += float(adv.detach()) / len(rows)
        cos = F.cosine_similarity(pred - neutral, target - neutral, dim=-1).mean()
        totals["cos_pos"] += float(cos.detach()) / len(rows)
    import math

    norm = param_grad_norm(network.parameters())
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite G gradient")
    torch.nn.utils.clip_grad_value_(network.parameters(), 1.0)
    g.step()
    return dict(
        totals,
        loss=totals["g_adv"],
        d_loss=float(d_loss.detach()),
        d_pen=float(cap.detach()),
        grad_norm=norm,
        penalty_center=stats["center"],
        lr_scale=float(scale),
    )


@isolated_seed("seed")
def run_cell(cell, arm_name, *, steps=BUDGETS, seed=0):
    if not steps or min(steps) < 1:
        raise ValueError("Positive step budgets required")
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm {arm_name!r}")
    arm = ARMS[arm_name]
    total = max(steps)
    field = PLUS_NEU_CELLS[cell](seed=seed)
    network = ToySlider(field.dim)
    backend = ToyBackend(field, network)
    rows = []
    for i in range(field.rows):
        positive, _, neutral = field.poles(i)
        rows.append(dict(ids=[i], prefix_len=1, neutral=neutral[None], targets=positive[None]))
    critic, g, d = build_game_variant(backend, network, rows, arm)
    ema = ResidualEMA(network, arm["ema"]) if float(arm["ema"]) > 0 else None
    checkpoints = []
    history = []
    for step in range(1, total + 1):
        order = torch.randperm(len(rows)).tolist()
        metrics = update_variant(
            backend, network, critic, g, d, [rows[i] for i in order],
            step=step, arm=arm, total=total, checkpointing=False,
        )
        if ema is not None:
            ema.update(network)
        history.append(dict(step=step, **metrics))
        if step in steps:
            snap = ema.snapshot() if ema is not None else network.snapshot()
            score = score_plus_neu_residual(
                arm_name, field, snap, teacher="faithful_plus_neu", plus_only=True,
            )
            checkpoints.append(dict(step=step, **score))
    control = score_plus_neu_exam(
        "main_supervised_control", field, teacher="faithful_plus_neu", plus_neu=True, steps=400, seed=seed,
    )
    return dict(
        cell=cell, seed=seed, arm=arm_name, batch=len(rows),
        knobs={k: arm[k] for k in ("g_lr", "d_lr", "beta2", "schedule", "ema", "critic")},
        recipe=dict(game.RECIPE),
        checkpoints=checkpoints, control=control, history=history,
    )


def summarize(results, *, steps, seeds=AUDIT_SEEDS):
    """Per-arm PASS/FAIL at each budget (both required cells, every seed)."""
    out = []
    for arm in ARMS:
        row = {"arm": arm, "identity": ARMS[arm]["identity"], "budgets": {}}
        for step in steps:
            ok = accepted(
                [r for r in results if r["arm"] == arm], steps=[step], seeds=seeds,
            )
            row["budgets"][str(step)] = "PASS" if ok else "FAIL"
        out.append(row)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=[a for a in ARMS if ARMS[a]["propose_only"]])
    parser.add_argument("--steps", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(AUDIT_SEEDS))
    parser.add_argument("--cells", nargs="+", choices=tuple(PLUS_NEU_CELLS), default=["divergent", "close"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for seed in args.seeds:
            for cell in args.cells:
                result = run_cell(cell, arm, steps=tuple(args.steps), seed=seed)
                results.append(result)
                args.out.write_text(json.dumps(dict(results=results), indent=2, allow_nan=False) + "\n")
                for row in result["checkpoints"]:
                    print(
                        json.dumps(
                            dict(
                                arm=arm, cell=cell, seed=seed,
                                **{k: row[k] for k in ["step", "cover", "off_caption", "neu_hold", "hit", "pole_cos"]},
                            )
                        ),
                        flush=True,
                    )
    ok = all(
        accepted([r for r in results if r["arm"] == arm], steps=args.steps, seeds=args.seeds)
        for arm in args.arms
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
