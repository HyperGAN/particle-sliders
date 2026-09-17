"""Shared +/0 RpGAN game for CPU fixtures and opt-in music training.

The scale label conditions D; its gradient cap differentiates hidden states
only. G has only the paired adversarial objective, averaged over +1 and 0.
No MSE, endpoint regularizer, particle prior, EMA, or negative-scale training.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from analysis.slider2d.adv import delayed_cosine, make_grad_regularizer, rp_d_loss, rp_g_loss
from conceptmod.textsliders.lm_adv import param_grad_norm

SCALES = (0., 1.)
BETAS = (0., .99)


class ScaleCritic(nn.Module):
    def __init__(self, dim, teacher, hidden=256):
        super().__init__()
        scale = teacher.detach().float().square().mean().sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError('Teacher must have finite, nonzero RMS')
        self.register_buffer('input_scale', scale)
        self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.LeakyReLU(.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, 1))

    def calibrated(self, z, scale):
        if scale not in SCALES:
            raise ValueError('Only scale 0 and +1 are trained')
        label = torch.full_like(z[:, :1], scale)
        return self.net(torch.cat([z, label], dim=-1)).squeeze(-1)

    def forward(self, delta, scale):
        return self.calibrated(delta.float() / self.input_scale, scale)


def build_game(network, real_plus, *, lr):
    critic = ScaleCritic(real_plus.shape[-1], real_plus).to(real_plus.device)
    g = torch.optim.Adam(network.parameters(), lr=lr, betas=BETAS)
    d = torch.optim.Adam(critic.parameters(), lr=lr, betas=BETAS)
    # Store base rates in optimizer state so checkpoint resume is exact.
    for optimizer in (g, d):
        optimizer.param_groups[0]['initial_lr'] = lr
    return critic, g, d


def update(network, critic, g, d, real_plus, predict, *, step, total_steps,
           checkpointing=True):
    """One D then one G update; predict(row, scale, checkpointing) returns delta.

    Keep the scale context through backward, including native checkpoint
    recomputation. Zero outputs with no trainable path (multiplier LoRA)
    contribute the exact constant GAN term without an unused model forward.
    """
    if not 1 <= step <= total_steps or len(real_plus) == 0:
        raise ValueError('Invalid update budget or empty batch')
    factor = delayed_cosine(step - 1, total=total_steps, delay=80, min_ratio=.05)
    for optimizer in (g, d):
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
    reg = make_grad_regularizer(arm='b_cap', coeff=1., kappa=1., norm='l2',
                               lazy_k=1, target_anneal='none')
    real = {0.: torch.zeros_like(real_plus), 1.: real_plus}
    fake = {}
    with torch.no_grad():
        for scale in SCALES:
            with network.scaled(scale):
                fake[scale] = torch.cat([predict(i, scale, False) for i in range(len(real_plus))])
    critic.requires_grad_(True)
    d.zero_grad(set_to_none=True)
    d_loss = 0.
    cap_total = 0.
    for scale in SCALES:
        cap, _ = reg.penalty(lambda z: critic.calibrated(z, scale),
            real[scale] / critic.input_scale, fake[scale] / critic.input_scale, step=step)
        d_loss = d_loss + .5 * (rp_d_loss(critic(real[scale], scale), critic(fake[scale], scale)) + cap)
        cap_total = cap_total + .5 * cap.detach()
    if not torch.isfinite(d_loss):
        raise FloatingPointError('Non-finite D loss')
    d_loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in critic.parameters()):
        raise FloatingPointError('Non-finite D gradient')
    d.step()
    d.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    with torch.no_grad():
        real_scores = {scale: critic(real[scale], scale) for scale in SCALES}
    g.zero_grad(set_to_none=True)
    losses = {0.: 0., 1.: 0.}
    cos = 0.
    zero_norm = 0.
    for scale in SCALES:
        for i in range(len(real_plus)):
            with network.scaled(scale):
                pred = predict(i, scale, checkpointing)
                adv = rp_g_loss(real_scores[scale][i:i+1], critic(pred, scale))
                loss = .5 * adv / len(real_plus)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite G loss')
                if loss.requires_grad:
                    loss.backward()
            losses[scale] += float(adv.detach()) / len(real_plus)
            if scale == 1.:
                cos += float(F.cosine_similarity(pred.detach(), real_plus[i:i+1], dim=-1).mean()) / len(real_plus)
            else:
                zero_norm += float(pred.detach().norm()) / len(real_plus)
    norm = param_grad_norm(network.parameters())
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite G gradient')
    g.step()
    adv = .5 * (losses[0.] + losses[1.])
    return dict(loss=adv, g_adv=adv, g_pos=losses[1.], g_zero=losses[0.],
        d_loss=float(d_loss.detach()), d_pen=float(cap_total), cos_pos=cos,
        zero_delta_norm=zero_norm, grad_norm=norm, penalty_center=1.,
        g_lr=g.param_groups[0]['lr'], d_lr=d.param_groups[0]['lr'])
