import pytest
import torch

from analysis.gan_bcap.mmd_game_20260905.policy_geometry import (
    continuation_policy_log_probs, paired_policy_kl, policy_diagnostics, PolicyObjective)


def test_cfg_readout_and_audio_start_alignment():
    hidden = torch.arange(2*7*3, dtype=torch.float64).reshape(2, 7, 3)/30
    weight = torch.tensor([[1., 0., 0.], [0., 1., -1.], [-.2, .3, .5]], dtype=torch.float64)
    got = continuation_policy_log_probs(hidden, weight, span_length=3, stride=2)
    logits = hidden[:, [2, 4, 6]]@weight.T
    expected = torch.stack((logits[0], 1.5*logits[0]-.5*logits[1])).log_softmax(-1)
    torch.testing.assert_close(got, expected)


def test_soft_teacher_kl_minimum_and_teacher_detached():
    teacher = torch.tensor([[[.2, .3, .5]], [[.1, .8, .1]]], dtype=torch.float64).log().requires_grad_()
    logits = torch.zeros_like(teacher, requires_grad=True)
    loss = paired_policy_kl(logits.log_softmax(-1), teacher)
    loss.backward()
    assert teacher.grad is None
    assert logits.grad is not None
    assert paired_policy_kl(teacher, teacher).item() == 0.
    assert loss.item() > 0
    updated = (logits-0.1*logits.grad).log_softmax(-1)
    assert paired_policy_kl(updated, teacher) < loss


def test_policy_is_invariant_to_common_logit_offset():
    hidden = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    readout = torch.eye(3, dtype=torch.float64)
    shifted = hidden+17
    a = continuation_policy_log_probs(hidden, readout, span_length=1, stride=1)
    b = continuation_policy_log_probs(shifted, readout, span_length=1, stride=1)
    torch.testing.assert_close(a, b)
    diagnostics = policy_diagnostics(a, b)
    assert max(diagnostics['total_variation_by_policy']) < 1e-14


def test_reject_missing_cfg_branch():
    with pytest.raises(ValueError, match='conditional'):
        continuation_policy_log_probs(torch.zeros(1, 4, 3), torch.eye(3), span_length=2)


def test_frozen_policy_objective_has_zero_at_teacher_and_attached_student_gradient():
    gen = torch.Generator().manual_seed(13)
    neutral = torch.randn(2, 5, 3, generator=gen)
    target = torch.randn(2, 5, 3, generator=gen)*.2
    row = dict(real=torch.zeros(1, 2, 3), energy_neutral=neutral, energy_target=target)
    objective = PolicyObjective(torch.randn(7, 3, generator=gen), hidden_scale=.3,
                                bandwidths=[.03, .1, .3, 1., 3.], stride=1)
    row['policy_teacher_log_probs'] = objective.teacher(row)
    assert objective(target, row).abs().item() < 1e-7
    fake = torch.zeros_like(target, requires_grad=True)
    policy, hidden = objective.terms(fake, row)
    objective.policy_calibration = policy.detach().item()
    objective.hidden_calibration = hidden.detach().item()
    initial = objective(fake, row)
    assert initial.item() == pytest.approx(1.1, abs=1e-6)
    initial.backward()
    assert fake.grad.norm() > 0
    assert torch.isfinite(fake.grad).all()
    assert not row['policy_teacher_log_probs'].requires_grad
