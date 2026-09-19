"""Readout-aware teacher matching on frozen continuation histories.

This is a smooth local policy surrogate. The native sampler also applies a
conditional top-50 mask and samples depth codes; neither operation is represented
by this loss. A soft target retains all semantic-code probabilities rather than
substituting a single teacher token. It does not establish trajectory diversity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from analysis.gan_bcap.objective_20260905.distribution import paired_mmd


def continuation_policy_log_probs(hidden, readout, *, span_length, stride=4, cfg_scale=1.5):
    """Return conditional and guided log probabilities over codes plus EOS.

    ``hidden`` contains aligned lyric states followed by audio-start and cached
    audio states, with conditional and unconditional branches in that order.
    ``readout`` contains only the fixed semantic-code and EOS output-head rows.
    The audio-start position is included exactly once.
    """
    if hidden.ndim != 3 or hidden.shape[0] != 2:
        raise ValueError('Expected aligned conditional and unconditional states')
    if not 1 <= span_length <= hidden.shape[1] or stride < 1:
        raise ValueError('Invalid continuation selection')
    if readout.ndim != 2 or readout.shape[1] != hidden.shape[2]:
        raise ValueError('Readout and hidden width differ')
    selected = hidden[:, span_length-1::stride]
    logits = F.linear(selected, readout)
    conditional, unconditional = logits[0], logits[1]
    guided = unconditional+(conditional-unconditional)*cfg_scale
    return F.log_softmax(torch.stack((conditional, guided)), dim=-1)


def paired_policy_kl(student_log_probs, teacher_log_probs):
    """Forward KL, equal weight per position and per conditional/guided policy."""
    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.ndim != 3:
        raise ValueError('Student and frozen teacher policies must align')
    teacher = teacher_log_probs.detach()
    return (teacher.exp()*(teacher-student_log_probs)).sum(-1).mean()


@torch.no_grad()
def policy_diagnostics(student_log_probs, teacher_log_probs):
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError('Student and teacher policies must align')
    teacher = teacher_log_probs.exp()
    student = student_log_probs.exp()
    return dict(
        kl_by_policy=(teacher*(teacher_log_probs-student_log_probs)).sum(-1).mean(-1).cpu().tolist(),
        teacher_entropy_by_policy=(-(teacher*teacher_log_probs).sum(-1).mean(-1)).cpu().tolist(),
        student_entropy_by_policy=(-(student*student_log_probs).sum(-1).mean(-1)).cpu().tolist(),
        top1_agreement_by_policy=(student.argmax(-1)==teacher.argmax(-1)).float().mean(-1).cpu().tolist(),
        total_variation_by_policy=(.5*(teacher-student).abs().sum(-1)).mean(-1).cpu().tolist())


class PolicyObjective:
    """Frozen semantic-policy KL with a small paired hidden-MMD regularizer.

    Both positive constants are measured once from the starting training batch.
    Thus the initial terms have weights 1 and ``hidden_weight``; the denominators
    never adapt during optimization. No held-out data sets these constants.
    """
    def __init__(self, readout, *, hidden_scale, bandwidths, stride=4, cfg_scale=1.5,
                 hidden_weight=.1, policy_calibration=1., hidden_calibration=1.):
        if min(hidden_scale, policy_calibration, hidden_calibration) <= 0 or hidden_weight < 0:
            raise ValueError('Invalid frozen objective calibration')
        self.readout = readout.detach()
        self.hidden_scale, self.bandwidths = hidden_scale, bandwidths
        self.stride, self.cfg_scale, self.hidden_weight = stride, cfg_scale, hidden_weight
        self.policy_calibration, self.hidden_calibration = policy_calibration, hidden_calibration

    def log_probs(self, hidden, row):
        return continuation_policy_log_probs(hidden, self.readout,
            span_length=row['real'].shape[1], stride=self.stride, cfg_scale=self.cfg_scale)

    @torch.no_grad()
    def teacher(self, row):
        device = self.readout.device
        positive = row['energy_neutral'].to(device)+row['energy_target'].to(device)
        return self.log_probs(positive, row).cpu()

    def terms(self, fake, row):
        hidden = fake+row['energy_neutral'].to(fake)
        student = self.log_probs(hidden, row)
        policy = paired_policy_kl(student, row['policy_teacher_log_probs'].to(fake))
        hidden_loss = paired_mmd(fake, row['energy_target'].to(fake), scale=self.hidden_scale,
                                 bandwidths=self.bandwidths)
        return policy, hidden_loss

    def __call__(self, fake, row):
        policy, hidden = self.terms(fake, row)
        return policy/self.policy_calibration+self.hidden_weight*hidden/self.hidden_calibration
