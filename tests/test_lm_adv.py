"""Gates for the training-only LM adversarial head (RpGAN + b_cap).

No model weights, no GPU. Covers ``conceptmod.textsliders.lm_adv`` only —
the live trainer imports it but these tests never construct the 8B LM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytest

from conceptmod.textsliders.lm_adv import (
    LMDiscriminator,
    cap_penalty,
    flat_param_grad,
    param_grad_norm,
    rp_d_loss,
    rp_g_loss,
)


class _LinearD(nn.Module):
    """D(x) = w.x with fixed ||w||: grad norm is known exactly."""

    def __init__(self, dim: int, norm: float) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(norm / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


def test_forward_shape():
    d = LMDiscriminator(32, hidden_dim=16)
    assert d(torch.randn(5, 32)).shape == (5,)


def test_forward_is_scale_invariant_and_finite():
    # Gain shortcut is gone by construction: 100x the delta, same score.
    d = LMDiscriminator(32, hidden_dim=16)
    x = torch.randn(4, 32)
    assert torch.allclose(d(x), d(100.0 * x), atol=1e-4)
    assert bool(torch.isfinite(d(torch.zeros(2, 32))).all())


def test_unit_lognorm_sees_magnitude():
    d = LMDiscriminator(32, hidden_dim=16, in_mode="unit_lognorm")
    x = torch.randn(4, 32)
    assert d(x).shape == (4,)
    # Same direction, different scale -> different score (magnitude visible).
    assert not torch.allclose(d(x), d(100.0 * x), atol=1e-4)
    assert bool(torch.isfinite(d(torch.zeros(2, 32))).all())
    # b_cap stays quiet on sane inputs (grads O(1), no phantom penalty).
    pen, stat = cap_penalty(d, torch.randn(6, 32) * 20.0, torch.randn(6, 32) * 20.0)
    assert stat["applied"] is True
    assert float(pen.detach()) < 5.0


def test_bad_in_mode_rejected():
    try:
        LMDiscriminator(8, in_mode="raw")
    except ValueError:
        return
    raise AssertionError("expected ValueError for in_mode='raw'")


def test_forward_rejects_bad_dim():
    try:
        LMDiscriminator(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for in_dim=0")


def test_cap_free_below_margin():
    d = _LinearD(8, norm=0.5)
    pen, stat = cap_penalty(d, torch.randn(6, 8), torch.randn(6, 8))
    assert stat["applied"] is True
    assert float(pen.detach()) == 0.0


def test_cap_exact_value_above_margin():
    # ||w|| = 2 -> relu(2-1)^2 = 1 per sample, both sides:
    # pen = (coeff/2) * (1 + 1) = coeff.
    d = _LinearD(8, norm=2.0)
    pen, _ = cap_penalty(d, torch.randn(6, 8), torch.randn(6, 8), coeff=2.0)
    assert abs(float(pen.detach()) - 2.0) < 1e-5


def test_cap_disabled():
    d = _LinearD(8, norm=2.0)
    pen, stat = cap_penalty(d, torch.randn(4, 8), torch.randn(4, 8), coeff=0.0)
    assert stat["applied"] is False
    assert float(pen) == 0.0


def test_cap_ignores_degenerate_rows():
    # Exact-zero rows (fresh-LoRA fakes) must not forge phantom gradients
    # through the unit-norm eps clamp, and must not silence the healthy side.
    d = _LinearD(8, norm=2.0)
    real = torch.randn(4, 8)
    fake = torch.zeros(3, 8)
    pen, stat = cap_penalty(d, real, fake, coeff=2.0)
    assert stat["applied"] is True
    # Lone healthy side keeps per-sample weight: coeff * mean(phi) = 2*1.
    assert abs(float(pen.detach()) - 2.0) < 1e-5
    pen.backward()
    assert bool(torch.isfinite(d.lin.weight.grad).all())

    both_dead, stat = cap_penalty(d, torch.zeros(2, 8), torch.zeros(2, 8))
    assert stat["applied"] is False
    assert float(both_dead) == 0.0


def test_cap_is_differentiable():
    d = _LinearD(8, norm=2.0)
    pen, _ = cap_penalty(d, torch.randn(4, 8), torch.randn(4, 8))
    pen.backward()
    assert d.lin.weight.grad is not None
    assert bool((d.lin.weight.grad != 0).any())


def test_rp_tie_is_ln2():
    tied = torch.zeros(7)
    assert abs(float(rp_d_loss(tied, tied)) - math.log(2)) < 1e-6
    assert abs(float(rp_g_loss(tied, tied)) - math.log(2)) < 1e-6


def test_rp_rewards_correct_ranking():
    real = torch.full((8,), 3.0)
    fake = torch.full((8,), -3.0)
    assert float(rp_d_loss(real, fake)) < 0.05
    assert float(rp_d_loss(fake, real)) > 2.0
    assert float(rp_g_loss(fake, real)) > 2.0
    assert float(rp_g_loss(real, fake)) < 0.05


def test_g_step_touches_student_not_d():
    torch.manual_seed(0)
    d = LMDiscriminator(16, hidden_dim=16)
    for p in d.parameters():
        p.requires_grad_(False)
    fake = torch.randn(3, 16, requires_grad=True)
    real = torch.randn(3, 16)
    loss = rp_g_loss(d(fake), d(real))
    loss.backward()
    assert fake.grad is not None
    assert bool((fake.grad != 0).any())
    for p in d.parameters():
        assert p.grad is None


def test_grad_accounting_helpers():
    lin = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    (lin(x).pow(2).mean()).backward()
    assert param_grad_norm(lin.parameters()) > 0.0
    flat = flat_param_grad(lin.parameters())
    assert flat.shape == (lin.weight.numel() + lin.bias.numel(),)
    assert bool(torch.isfinite(flat).all())
    lin.zero_grad(set_to_none=True)
    assert param_grad_norm(lin.parameters()) == 0.0
    assert bool((flat_param_grad(lin.parameters()) == 0).all())


def test_scaled_teacher_calibration_preserves_geometry_and_origin_gradient():
    torch.manual_seed(7)
    d = LMDiscriminator(32, hidden_dim=16, in_mode="scaled")
    real = torch.randn(4, 32) * 8
    scale = d.calibrate_input_scale(real)
    assert scale == pytest.approx(float(real.square().mean().sqrt()))
    assert d._embed(real).square().mean() == pytest.approx(1.0)
    assert torch.allclose(d._embed(2 * real), 2 * d._embed(real))
    zero = torch.zeros_like(real, requires_grad=True)
    gradient = torch.autograd.grad(d(zero).sum(), zero)[0]
    assert torch.isfinite(gradient).all()
    assert 0 < float(gradient.norm()) < 10


def test_scaled_cap_is_invariant_to_hidden_units_and_includes_zero_fakes():
    torch.manual_seed(8)
    d = LMDiscriminator(32, hidden_dim=16, in_mode="scaled")
    real = torch.randn(4, 32)
    fake = torch.zeros_like(real)
    d.calibrate_input_scale(real)
    pen, stat = cap_penalty(d, real, fake, kappa=0.01)
    assert stat["fake_kept"] == len(fake)
    assert stat["fake_grad_mean"] > 0
    d.calibrate_input_scale(real * 20)
    pen2, stat2 = cap_penalty(d, real * 20, fake, kappa=0.01)
    assert float(pen2.detach()) == pytest.approx(float(pen.detach()), rel=1e-5)
    assert stat2["fake_grad_mean"] == pytest.approx(stat["fake_grad_mean"], rel=1e-5)


def test_scaled_calibration_excludes_padding_and_rejects_bad_teachers():
    d = LMDiscriminator(4, hidden_dim=4, in_mode="scaled")
    real = torch.tensor([[[2., 2., 2., 2.], [999., 999., 999., 999.]]])
    assert d.calibrate_input_scale(real, torch.tensor([[True, False]])) == 2
    for bad in (torch.zeros(1, 4), torch.full((1, 4), float("nan")), torch.empty(0, 4)):
        with pytest.raises(ValueError):
            d.calibrate_input_scale(bad)
    for scale in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            LMDiscriminator(4, in_mode="scaled", input_scale=scale)
