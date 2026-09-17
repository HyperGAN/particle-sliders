"""Propose-only +/0 conditional RpGAN, scored on the unipolar leaderboard.

Use the same free-origin student as the supervised uni board. Neutral hold
must be learned here; it is not made automatic by removing that parameter.
The shared game is also available to explicitly requested music experiments.
No locked default or Music argv is changed by collecting this arm.
"""
from contextlib import contextmanager

import torch
from torch import nn

from analysis.slider2d.plus_neu_exam import OriginResidual
from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders import unipolar_gan as game

NAME = 'rpgan_bcap_plus_neu'
PROPOSE_ONLY = True
MERGE_TO_TRAINER = False
TOY_LR = .005


class Student(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.origin = nn.Parameter(torch.zeros(dim))
        self.scale = 0.

    def delta(self, scale):
        return scale * self.odd + abs(scale) * self.even + self.origin

    @contextmanager
    def scaled(self, scale):
        if scale not in game.SCALES:
            raise ValueError('Only +1 and 0 can be trained')
        previous = self.scale
        self.scale = scale
        try:
            yield
        finally:
            self.scale = previous

    def snapshot(self):
        return OriginResidual(self.odd.detach().clone(), self.even.detach().clone(),
                              self.origin.detach().clone())


@isolated_seed('seed')
def fit(field, *, steps=400, seed=0):
    network = Student(field.dim)
    real = torch.stack([field.poles(i)[0] - field.poles(i)[2] for i in range(field.rows)])
    critic, g, d = game.build_game(network, real, lr=TOY_LR)
    def predict(i, scale, checkpointing):
        return network.delta(scale)[None]
    history = []
    for step in range(1, steps + 1):
        history.append(game.update(network, critic, g, d, real, predict,
            step=step, total_steps=steps, checkpointing=False))
    return network.snapshot(), history


def score_bipolar(field, residual):
    """Read the SAME unipolar fit on the existing bipolar continuation gates.

    No negative training and no antipodal gate. This diagnostic demonstrates
    why a unipolar HIT must not be promoted as a bipolar winner.
    """
    from analysis.slider2d.exam import (
        teacher_rollouts, teacher_self_match, rollout_report, exam_verdicts,
    )
    from analysis.slider2d.field import cosine
    head = field.readout()
    delta = getattr(residual, 'delta_for_row', lambda scale, row: residual.delta(scale))
    positive, negative, corpus = teacher_rollouts(field, head)
    def report(plus, minus):
        return rollout_report(field, plus, minus, readout=head,
            teacher_plus=positive, teacher_minus=negative, corpus=corpus)
    row = report([field.poles(i)[2] + delta(1., i) for i in range(field.rows)],
                 [field.poles(i)[2] + delta(-1., i) for i in range(field.rows)])
    ceiling = report([field.poles(i)[0] for i in range(field.rows)],
                     [field.poles(i)[1] for i in range(field.rows)])
    row.update(name=NAME, teacher='faithful_plus_neu; trained +/0 only',
        roll_swing_kept=row['roll_swing'] / (abs(ceiling['roll_swing']) + 1e-8),
        roll_match_kept=row['roll_match'] / (teacher_self_match(positive, negative) + 1e-8),
        collapse=cosine(delta(1., 0), delta(-1., 0)))
    row['axis'] = exam_verdicts(row)
    row['pass'] = all(value == 'right' for value in row['axis'].values())
    row['reason'] = 'Same unipolar weights; negative endpoint was not trained'
    return row


# ---------------------------------------------------------------------------
# UniPG-E: zero-at-zero RpGAN trainer + gate scoring (additive; #119 above kept)
# ---------------------------------------------------------------------------
"""Unipolar ParticleGAN toy trainer (GAN-only) + PairField gate scoring.

UniPG-E harness spine (appended additively onto the UniPG-C +/0 arm above;
both stay propose_only). Trains a zero-at-zero residual with paired Rp
logistic + GradRegularizer (default b_cap) against a RAW POSITIVE teacher
cloud only — no minus branch, no supervised MSE/FM/cover on G — then
scores the unipolar toy gates (cover@+1, leak@+1, neu_hold@0) on the
divergent + close PairField cells.

- Teacher: raw ``pos`` pole (the ``lm_faithful_plus_neu`` contract for
  what + is: ``pos`` itself, never leftover-gated).
- Scale 0 = exact base by construction (``delta(0) == 0``); neu_hold is
  therefore structural, reported honestly, never trained.
- Scale -1 is a logged canary only, never a gate. Antipodal cos never
  consulted.
- ``cover_weight`` / ``fm_weight`` must be 0 (fail-closed GAN-only); any
  nonzero raises instead of silently adding supervision.

CPU only. No Hub, no GPU, no Music weights. Does not touch the Music
bipolar trainer, ``locked_shared``, or the live ``--lm_target`` default.
"""
from dataclasses import dataclass, replace


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
from analysis.slider2d.exam import PairField
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


# Unipolar eval scales. Gates read 0 and 1; 0.5 is a diagnostic; -1 canary.
UNIPOLAR_EVAL_SCALES = (0.0, 0.5, 1.0)


@dataclass
class UniResidual:
    """Zero-at-zero residual: ``delta(s) = s*w_odd + |s|*w_even``."""

    w_odd: torch.Tensor
    w_even: torch.Tensor

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd + abs(float(scale)) * self.w_even

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_odd, self.w_even]

    def snapshot(self) -> "UniResidual":
        return UniResidual(self.w_odd.detach().clone(), self.w_even.detach().clone())


def uni_cfg(**overrides) -> AdvConfig:
    """GAN-only config: locked-shape defaults with supervision forced off.

    Starts from ``AdvConfig()`` (the locked #94 shape) and forces
    ``cover_weight=0`` + ``fm_weight=0`` unless the caller explicitly
    passes them (which the sweep harness refuses to do for MATCH arms).
    """
    base = AdvConfig()
    forced = {"cover_weight": 0.0, "fm_weight": 0.0}
    forced.update(overrides)
    return replace(base, **forced)


def _assert_gan_only(cfg: AdvConfig) -> None:
    bad = []
    if float(cfg.cover_weight) != 0.0:
        bad.append(f"cover_weight={cfg.cover_weight!r} (supervised MSE on G)")
    if float(cfg.fm_weight) != 0.0:
        bad.append(f"fm_weight={cfg.fm_weight!r} (feature-matching on G)")
    if bad:
        raise ValueError("GAN-only violation: " + "; ".join(bad))


@isolated_seed("cfg.seed", default=0)
def fit_uni_adv(
    field: PairField,
    *,
    cfg: AdvConfig | None = None,
    vicreg_fn=None,
) -> tuple[UniResidual, dict]:
    """Fit one zero-at-zero residual with unipolar RpGAN + GradRegularizer.

    ``vicreg_fn`` overrides the particle regularizer (default: locked
    ``vicreg_loss``); pass ``vicreg_faithful_loss`` for the upstream
    var+cov form. ``None`` keeps the locked path byte-identical.
    """
    cfg = cfg or uni_cfg()
    _assert_gan_only(cfg)
    vfn = vicreg_fn if vicreg_fn is not None else vicreg_loss
    dim = int(field.dim)
    residual = UniResidual(
        torch.zeros(dim, requires_grad=True),
        torch.zeros(dim, requires_grad=True),
    )
    prior = ParticlePrior(int(cfg.n_particles), dim)
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
    ema: EMA | None = None
    if float(cfg.ema) > 0.0:
        ema = EMA(residual.parameters(), decay=cfg.ema)

    poles_p, neus = [], []
    for row in range(int(field.rows)):
        pos, _neg, neu = field.poles(row)
        poles_p.append(pos.flatten())
        neus.append(neu.flatten())
    poles_p = torch.stack(poles_p)
    neus_t = torch.stack(neus)
    logs = {"d": [], "g": [], "cap": [], "grad_real": [], "grad_fake": []}
    reg = make_grad_regularizer(cfg)

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step, total=cfg.steps, delay=cfg.delay, min_ratio=cfg.min_lr_ratio
        )
        opt_g.param_groups[0]["lr"] = g_lr * scale
        opt_g.param_groups[1]["lr"] = prior_lr * scale
        for group in opt_d.param_groups:
            group["lr"] = d_lr * scale

    def fake_batch(n: int) -> torch.Tensor:
        idx = torch.randint(0, neus_t.shape[0], (n,))
        return (
            neus_t[idx] + residual.delta(1.0) + prior.sample(n, jitter=cfg.particle_jitter)
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
            fake = fake_batch(int(cfg.batch)).detach()
            real_g = real.detach()
            d_real = critic(real_g)
            d_fake = critic(fake)
            cap, _ = reg.penalty(critic, real_g, fake, step=step + 1)
            d_loss = rp_d_loss(d_real, d_fake) + cap
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        fake = fake_batch(int(cfg.batch))
        d_real = critic(real.detach())
        d_fake = critic(fake)
        g_adv = rp_g_loss(d_real, d_fake)
        g_extra = residual.w_odd.new_zeros(())
        parts = prior.particles
        if float(cfg.vicreg_weight) > 0.0:
            g_extra = g_extra + float(cfg.vicreg_weight) * vfn(parts)
        if float(cfg.particle_l2) > 0.0:
            g_extra = g_extra + float(cfg.particle_l2) * parts.pow(2).mean()
        g_loss = g_adv + g_extra
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        if ema is not None:
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

    if ema is not None:
        ema.copy_to(residual.parameters())
    snap = residual.snapshot()
    stats = {
        "d_loss": logs["d"][-1] if logs["d"] else None,
        "g_loss": logs["g"][-1] if logs["g"] else None,
        "cap": logs["cap"][-1] if logs["cap"] else None,
        "grad_real": logs["grad_real"][-1] if logs["grad_real"] else None,
        "grad_fake": logs["grad_fake"][-1] if logs["grad_fake"] else None,
        "teacher": "raw_positive",
        "b_cap": float(cfg.b_cap),
        "steps": int(cfg.steps),
        "log": logs,
    }
    return snap, stats


def score_uni_residual(name: str, field: PairField, residual: UniResidual) -> dict:
    """Score a fitted zero-at-zero residual on the unipolar toy gates."""
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    d_plus = residual.delta(1.0)
    d_half = residual.delta(0.5)
    d_minus = residual.delta(-1.0)
    d_zero = residual.delta(0.0)
    assert float(d_zero.norm()) == 0.0, "unipolar residual must hold 0 exactly"
    head = field.readout()
    overlap_rows, off_rows, blend_rows, cover_rows = [], [], [], []
    neu_overlap_rows, neu_drift_rows, neu_hold_rows = [], [], []
    half_rows = []
    canary_overlap, canary_off, canary_landed = [], [], []
    sings_plus, sings_zero, sings_minus = [], [], []
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
        cover = plus_cover(overlap, blend)
        neu_ov = _token_share(zero_seqs, neu_bag)
        drift = drift_from_neu(student_zero, neu, pos, mid)
        hold = neu_hold(neu_ov, drift)
        half_rows.append(plus_cover(_token_share(half_seqs, bags["pos"]),
                                    blend_toward_mid(student_half, pos, mid, neg)))
        overlap_rows.append(overlap)
        off_rows.append(off)
        blend_rows.append(blend)
        cover_rows.append(cover)
        neu_overlap_rows.append(neu_ov)
        neu_drift_rows.append(drift)
        neu_hold_rows.append(hold)
        canary_overlap.append(_token_share(minus_seqs, bags["neg"]))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(student_minus, pos, neu, neg))
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
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold,
        "overlap_pos": sum(overlap_rows) / len(overlap_rows),
        "blend_toward_mid": sum(blend_rows) / len(blend_rows),
        "overlap_neu": sum(neu_overlap_rows) / len(neu_overlap_rows),
        "drift_from_neu": sum(neu_drift_rows) / len(neu_drift_rows),
        "half_cover": sum(half_rows) / len(half_rows),
        "hit": hit,
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
        "canary": {
            "scored": False,
            "minus_overlap_neg": sum(canary_overlap) / len(canary_overlap),
            "minus_off_caption": canary_off_mean,
            "minus_landed": landed,
            "minus_sings": " | ".join(sings_minus),
            "dangerous": bool(landed == "pos" or canary_off_mean > PLUS_OFF_MAX),
        },
    }


def score_uni_adv(
    field: PairField,
    *,
    cfg: AdvConfig | None = None,
    vicreg_fn=None,
    name: str = "uni_rpgan_bcap",
) -> dict:
    """Fit unipolar GAN-only, then score the toy gates. Returns row + stats."""
    cfg = cfg or uni_cfg()
    residual, stats = fit_uni_adv(field, cfg=cfg, vicreg_fn=vicreg_fn)
    row = score_uni_residual(name, field, residual)
    row["teacher"] = "raw_positive"
    row["seed"] = int(cfg.seed)
    row["steps"] = int(cfg.steps)
    row["d_loss"] = stats.get("d_loss")
    row["g_loss"] = stats.get("g_loss")
    row["cap"] = stats.get("cap")
    return row
