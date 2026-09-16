"""Conditional row correspondence without changing b_cap input geometry."""

import math

import pytest
import torch

from conceptmod.textsliders.lm_adv import (
    LMDiscriminator, RowConditionalD, SpanTransformerD, cap_penalty, rp_d_loss,
)


def _critic(arch):
    torch.manual_seed(7)
    if arch == "mlp":
        base = LMDiscriminator(16, hidden_dim=12, in_mode="scaled")
        shape, feature_dim = (3, 16), 12
    else:
        base = SpanTransformerD(16, width=12, n_heads=3, n_layers=1,
                                in_mode="scaled", readout="mean_last")
        shape, feature_dim = (3, 7, 16), 24
    return RowConditionalD(base, 3, feature_dim), shape


@pytest.mark.parametrize("arch", ["mlp", "tx"])
def test_conditional_single_rows_match_batch_without_cross_row_gradient(arch):
    d, shape = _critic(arch)
    x = torch.randn(shape, requires_grad=True)
    ids = torch.tensor([2, 0, 1])
    scores = d(x, row_ids=ids)
    singles = torch.cat([d(x[i:i+1], row_ids=ids[i:i+1]) for i in range(3)])
    torch.testing.assert_close(scores, singles, atol=1e-5, rtol=1e-5)
    grad = torch.autograd.grad(scores[0], x)[0]
    assert torch.count_nonzero(grad[1:]) == 0
    torch.testing.assert_close(d.features(x), d.base.features(x))


@pytest.mark.parametrize("arch", ["mlp", "tx"])
def test_identical_real_fake_with_same_condition_has_no_condition_shortcut(arch):
    d, shape = _critic(arch)
    x = torch.randn(shape)
    ids = torch.tensor([0, 1, 2])
    loss = rp_d_loss(d(x, row_ids=ids), d(x.clone(), row_ids=ids))
    assert float(loss.detach()) == pytest.approx(math.log(2), abs=1e-6)
    grad = torch.autograd.grad(loss, d.embedding.weight)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("arch", ["mlp", "tx"])
def test_conditional_cap_reaches_embedding_at_zero_fakes_in_calibrated_units(arch):
    d, shape = _critic(arch)
    real, fake = torch.randn(shape), torch.zeros(shape)
    ids = torch.tensor([2, 0, 1])
    mask = None if arch == "mlp" else torch.ones(shape[:2], dtype=torch.bool)
    if mask is not None:
        mask[0, -2:] = False
    scale = d.calibrate_input_scale(real, mask)
    assert d.in_mode == "scaled" and scale > 0
    assert d.input_scale.data_ptr() == d.base.input_scale.data_ptr()
    pen, stats = cap_penalty(d, real, fake, kappa=0.01, mask_real=mask, mask_fake=mask,
                             condition_real=ids, condition_fake=ids)
    pen.backward()
    assert stats["fake_kept"] == len(fake)
    assert d.embedding.weight.grad is not None
    assert torch.isfinite(d.embedding.weight.grad).all()
    assert torch.count_nonzero(d.embedding.weight.grad) > 0
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    d.calibrate_input_scale(20 * real, mask)
    rescaled, _ = cap_penalty(d, 20 * real, fake, kappa=0.01, mask_real=mask, mask_fake=mask,
                             condition_real=ids, condition_fake=ids)
    assert float(rescaled.detach()) == pytest.approx(float(pen.detach()), rel=1e-5)


def test_conditional_projection_distinguishes_a_teacher_row_permutation():
    d, _ = _critic("mlp")
    real = torch.randn(2, 16)
    ids = torch.tensor([0, 1])
    with torch.no_grad():
        # Constant unconditional scores cannot tell this permutation apart.
        d.base.net[-1].weight.zero_()
        d.base.net[-1].bias.zero_()
        difference = d.features(real)[0] - d.features(real)[1]
        d.embedding.weight[0].copy_(difference)
        d.embedding.weight[1].copy_(-difference)
    matched = d(real, row_ids=ids)
    permuted = d(real.flip(0), row_ids=ids)
    assert bool((matched > permuted).all())
    assert float(rp_d_loss(matched, permuted).detach()) < math.log(2)


def test_conditional_head_requires_valid_ids():
    d, shape = _critic("mlp")
    x = torch.randn(shape)
    for ids in (None, torch.tensor([0]), torch.tensor([[0, 1, 2]]),
                torch.tensor([0., 1., 2.]), torch.tensor([-1, 0, 1]), torch.tensor([0, 1, 3])):
        with pytest.raises(ValueError):
            d(x, row_ids=ids)
    with pytest.raises(ValueError):
        RowConditionalD(d.base, 0, 12)
    with pytest.raises(ValueError):
        RowConditionalD(d.base, 3, 11)(x, row_ids=torch.arange(3))
