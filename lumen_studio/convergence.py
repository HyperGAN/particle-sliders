"""Diagnostics on fixed inputs and disposable opponents; training is unchanged."""
import math

import numpy as np
import torch

from .game import grad_norm
from .metrics import residual_breakdown
from .vendor.reference import noise_std, particle_vic, rp_d_loss, rp_g_loss


def rms(x):
    return float(x.double().square().mean().sqrt())


def movement(before, after, teacher, scale, records):
    """Full-field change in a fixed function, not parameter distance or loss slope."""
    if before.shape != after.shape or before.shape != teacher.shape or before.shape != scale.shape:
        raise ValueError("Fixed predictions, targets and scales must have identical shapes")
    if len(records) != len(before) or not torch.all(scale > 0):
        raise ValueError("Invalid fixed records or normalization")
    if any(not torch.isfinite(x).all() for x in (before, after, teacher, scale)):
        raise ValueError("Nonfinite convergence fixture")
    a, b, t = before.float() / scale, after.float() / scale, teacher.float() / scale
    change = b - a
    amount, previous, target = rms(change), rms(a), rms(t)
    return dict(change_rms=amount, previous_edit_rms=previous, current_edit_rms=rms(b),
        teacher_edit_rms=target,
        relative_to_previous_edit=amount / previous if previous > 1e-12 else None,
        relative_to_teacher_edit=amount / target if target > 1e-12 else None,
        per_field_p95=float(change.square().mean(-1).sqrt().quantile(.95)),
        direction_cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()),
        breakdown=residual_breakdown(change, records))


def paired_improvement(before, after, groups, seed=4701):
    """Positive means lower loss. Bootstrap whole prompt rows, never pixels/times."""
    if len(before) != len(after) or len(before) != len(groups) or not before:
        raise ValueError("Paired losses and row identities must match")
    differences = np.asarray(before, dtype=np.float64) - np.asarray(after, dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError("Nonfinite paired objective")
    unique = sorted(set(groups))
    cluster = np.array([differences[np.array(groups) == group].mean() for group in unique])
    gen = np.random.default_rng(seed)
    estimates = gen.choice(cluster, (2000, len(cluster)), replace=True).mean(1)
    return dict(before=float(np.mean(before)), after=float(np.mean(after)),
        improvement=float(cluster.mean()), ci95=list(map(float, np.quantile(estimates, [.025, .975]))),
        prompt_rows=len(unique), fields=len(groups),
        interval_scope="Prompt-row cluster bootstrap on fixed development rows and evaluation noise; not population-wide certainty")


def noise_and_scale(game, positions, indices, draws, step):
    ts = [positions[i] for i in indices]
    sigma = torch.tensor([noise_std(step - 1, start=game.noise_starts[t], decay_steps=1600, hold=1.)
                          for t in ts], device=game.device)
    return draws.to(game.device) * sigma[:, None], game.scale[ts]


def response_step(game, predict_residual, positions, step, side):
    """One production-equation D-only or G-only step, with the other side fixed.

    Draw both row/noise streams to retain production sampler ordering. The lazy
    cap still runs every four D updates; G retains the exact particle VIC draw.
    Nothing here changes the production Trainer or calls its save/update methods.
    """
    if side not in ("d", "g"):
        raise ValueError("Response side must be d or g")
    dimension = game.scale.shape[-1]
    di, dn = game.sampler.draw("d", dimension)
    gi, gn = game.sampler.draw("g", dimension)
    game.d.zero_grad(set_to_none=True)
    game.g.zero_grad(set_to_none=True)
    game.critic.requires_grad_(side == "d")
    ids, draws = (di, dn) if side == "d" else (gi, gn)
    adv_total = penalty_total = 0.
    for start in range(0, len(ids), game.microbatch):
        batch = ids[start:start + game.microbatch]
        noise, scale = noise_and_scale(game, positions, batch, draws[start:start+len(batch)], step)
        if side == "d":
            with torch.no_grad():
                fake = noise + predict_residual(batch) / scale
            adv = rp_d_loss(game.critic(noise), game.critic(fake))
            penalty, _ = game.capper.penalty(game.critic, noise, fake, step=step, collect_stats=False)
        else:
            with torch.no_grad():
                real = game.critic(noise)
            fake = noise + predict_residual(batch) / scale
            adv = rp_g_loss(real, game.critic(fake))
            penalty = torch.zeros((), device=game.device)
        loss = (adv + penalty) * len(batch) / len(ids)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite frozen-opponent loss")
        loss.backward()
        adv_total += float(adv.detach()) * len(batch) / len(ids)
        penalty_total += float(penalty.detach()) * len(batch) / len(ids)
    vic = 0.
    if side == "g":
        selection = torch.randperm(128, generator=game.sampler.generators["vic"])[:64].to(game.device)
        value = particle_vic(game.adapter.particles[selection])
        value.backward()
        vic = float(value.detach())
    parameters = game.critic.parameters() if side == "d" else game.adapter.parameters()
    norm = grad_norm(parameters)
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite frozen-opponent gradient")
    (game.d if side == "d" else game.g).step()
    return dict(step=step, side=side, adversarial=adv_total, cap=penalty_total, vic=vic, gradient_norm=norm)


def objective_rows(game, residuals, positions, *, seed=3803, noise_repeats=2):
    """Evaluate both current objectives under fixed common noise on unseen rows.

    Cap is evaluated exactly then divided by lazy_k to express its expected
    per-update weight. All selected late checkpoints use sigma=1, as does this
    diagnostic. VIC uses four fixed 64-particle subsets. Critic evaluation is
    microbatch one, matching the accepted production execution.
    """
    if len(residuals) != len(positions):
        raise ValueError("Residual and timestep coverage differs")
    generator = torch.Generator().manual_seed(seed)
    d_adv, d_cap, g_adv = [], [], []
    game.critic.requires_grad_(False)
    for residual, position in zip(residuals, positions, strict=True):
        values = []
        for _ in range(noise_repeats):
            noise = torch.randn(1, residual.numel(), generator=generator).to(game.device)
            error = residual.reshape(1, -1).to(game.device) / game.scale[position]
            fake = noise + error
            with torch.no_grad():
                real_score, fake_score = game.critic(noise), game.critic(fake)
                da = float(rp_d_loss(real_score, fake_score))
                ga = float(rp_g_loss(real_score, fake_score))
            # Input derivatives are needed even though critic parameters are frozen.
            with torch.enable_grad():
                cap, _ = game.capper.penalty(game.critic, noise, fake, step=4, collect_stats=False)
            values.append((da, float(cap.detach()) / game.capper.lazy_k, ga))
        d_adv.append(sum(v[0] for v in values) / len(values))
        d_cap.append(sum(v[1] for v in values) / len(values))
        g_adv.append(sum(v[2] for v in values) / len(values))
    vic_generator = torch.Generator().manual_seed(seed+1)
    with torch.no_grad():
        vic = sum(float(particle_vic(game.adapter.particles[
            torch.randperm(128, generator=vic_generator)[:64].to(game.device)])) for _ in range(4)) / 4
    return dict(d_adversarial=d_adv, d_cap=d_cap,
        d_total=[a+b for a, b in zip(d_adv, d_cap, strict=True)],
        g_adversarial=g_adv, g_vic=vic, g_total=[x+vic for x in g_adv],
        evaluation_noise=1., noise_repeats=noise_repeats, evaluation_seed=seed)
