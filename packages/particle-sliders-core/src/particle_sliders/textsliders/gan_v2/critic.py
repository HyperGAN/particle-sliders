"""Conditional, ordered span critic plus full-width preservation constraints."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..lm_adv import SpanTransformerD, cap_penalty
from ..lm_gan import feature_mean_surrogate


class SpanCritic(SpanTransformerD):
    def __init__(self, in_dim, *, width=128, layers=2, heads=4,
                 conditioned=True, ordered=True, normalized_features=True):
        super().__init__(in_dim, width=width, n_layers=layers, n_heads=heads,
                         in_mode="scaled", readout="mean_last")
        self.conditioned, self.ordered = conditioned, ordered
        self.normalized_features = normalized_features
        if conditioned:
            self.condition_proj = nn.Linear(in_dim, width, bias=False)
            nn.init.xavier_uniform_(self.condition_proj.weight)

    def _context_pooled(self, seq, mask=None, condition=None):
        if not self.conditioned and not self.ordered:
            return super()._pooled(seq, mask)
        if mask is None:
            mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
        if mask.shape != seq.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("Every critic sequence needs valid positions")
        m = mask.unsqueeze(-1).to(seq.dtype)
        h = self.proj(self._embed(seq) * m)
        if self.conditioned:
            if condition is None or condition.shape != seq.shape or condition.requires_grad:
                raise ValueError("Critic needs a detached, aligned neutral hidden condition")
            # Fixed neutral context, shared by both poles, with no row-ID table.
            c = condition.float()
            c = c / c.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            h = h + self.condition_proj(c * m)
        if self.ordered:
            # Ordinals within the common lyric span. Padding and the absolute
            # caption offset have no effect. Audio-start has its own readout.
            ordinal = (mask.long().cumsum(1) - 1).clamp_min(0).float()
            frequency = torch.exp(torch.arange(0, self.width, 2, device=seq.device).float()
                                  * (-math.log(10000.) / self.width))
            phase = ordinal[..., None] * frequency
            position = torch.zeros_like(h)
            position[..., 0::2] = phase.sin()
            position[..., 1::2] = phase[..., :position[..., 1::2].shape[-1]].cos()
            h = h + .1 * position * m
        for block in self.blocks:
            h = block(h, ~mask.bool())
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        indices = torch.arange(h.shape[1], device=h.device).expand(mask.shape)
        last_index = indices.masked_fill(~mask, -1).max(1).values
        last = h[torch.arange(len(h), device=h.device), last_index]
        return torch.cat([mean, last], -1)

    def forward(self, seq, mask=None, condition=None):
        return self.head(self.out_norm(self._context_pooled(seq, mask, condition))).squeeze(-1)

    def features(self, seq, mask=None, condition=None):
        pooled = self._context_pooled(seq, mask, condition)
        # No learned norm gain: otherwise the gain/head could rescale inversely
        # and recreate an unbounded FM scale despite "normalized" features.
        return pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-3) if self.normalized_features else pooled


class _FixedCondition(nn.Module):
    in_mode = "scaled"

    def __init__(self, critic, condition):
        super().__init__()
        self.critic, self.condition = critic, condition

    @property
    def input_scale(self):
        return self.critic.input_scale

    def forward(self, seq, mask=None):
        return self.critic(seq, mask, self.condition)


def conditional_cap(critic, real, fake, mask, condition, *, coefficient=1., kappa=1.):
    # The legacy cap keeps every row in scaled coordinates, including zero.
    return cap_penalty(_FixedCondition(critic, condition), real, fake,
                       coeff=coefficient, kappa=kappa, mask_real=mask, mask_fake=mask)


def matching_loss(features, fake_mean, real_mean, paired_target, *, paired_fraction):
    if not math.isfinite(paired_fraction) or not 0 <= paired_fraction <= 1:
        raise ValueError("Paired FM fraction must be between zero and one")
    batch = feature_mean_surrogate(features, fake_mean, real_mean)
    paired = F.mse_loss(features, paired_target.detach())
    return (1 - paired_fraction) * batch + paired_fraction * paired


def full_width_guard(fake, real, scale, *, radius=1.):
    """Paired, ordered trust tube in all hidden coordinates, not projected ones.

    Within the tube it adds no teacher-imitation gradient. Outside it penalizes
    displacement regardless of the critic projection's nullspace. Radius is a
    declared research tolerance, not a perceptually calibrated quality limit.
    """
    if not math.isfinite(radius) or radius < 0 or float(scale) <= 0:
        raise ValueError("Invalid full-width guard")
    error = (fake.float() - real.detach().float()) / scale
    distance = error.square().mean(-1).add(1e-12).sqrt()
    return F.relu(distance - radius).square().mean()


def bounded_policy_kl(student_logits, teacher_logits, *, tolerance=.05):
    """Teacher-to-student KL hinge; protects more than the EOS margin.

    Shared histories and allowed semantic-plus-EOS vocabulary are required.
    It does not constrain downstream audio paths or certify lyric accuracy.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Invalid policy tolerance")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[-1] < 2:
        raise ValueError("Policy comparisons require identical distributions")
    logp = teacher_logits.detach().float().log_softmax(-1)
    logq = student_logits.float().log_softmax(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)
    return F.relu(kl - tolerance).mean(), kl.detach()


def pad_sequences(sequences):
    lengths = [x.shape[1] for x in sequences]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Empty span batch")
    width = sequences[0].shape[-1]
    batch = sequences[0].new_zeros(len(sequences), max(lengths), width)
    mask = torch.zeros(batch.shape[:2], device=batch.device, dtype=torch.bool)
    for i, (value, length) in enumerate(zip(sequences, lengths)):
        if value.shape != (1, length, width):
            raise ValueError("Inconsistent span geometry")
        batch[i, :length] = value[0]
        mask[i, :length] = True
    return batch, mask
