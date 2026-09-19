"""Regression checks for adaptive row mining and latent VICReg.

No model weights, no GPU. Covers ``conceptmod.textsliders.lm_particles``.
"""

from __future__ import annotations

import torch
import pytest

from conceptmod.textsliders.lm_particles import ParticleBatch, VICRegLikeLoss


def test_vicreg_collapsed_triggers_hinge():
    vic = VICRegLikeLoss(target_std=1.0)
    z = torch.full((8, 4), 0.5)  # zero variance in every dim
    # Hinge ceiling is 1 - sqrt(eps), not 1: the eps inside sqrt(var + eps)
    # keeps the std differentiable at collapse.
    assert abs(float(vic(z)) - (1.0 - vic.eps**0.5)) < 1e-5


def test_vicreg_spread_is_free():
    vic = VICRegLikeLoss(target_std=1.0)
    torch.manual_seed(0)
    # Std AT target, ~decorrelated: both terms ~free. (Covariance is not
    # scale-invariant, so std-2 data would pay covariance, not hinge.)
    z = torch.randn(32, 4)
    assert float(vic(z)) < 0.2


def test_vicreg_correlated_pays_covariance():
    vic = VICRegLikeLoss(target_std=1.0)
    torch.manual_seed(0)
    base = torch.randn(32, 1) * 2.0
    z = torch.cat([base, base], dim=1)  # perfectly correlated pair
    assert float(vic(z)) > 0.5


def test_vicreg_single_dim_has_no_covariance():
    vic = VICRegLikeLoss(target_std=1.0)
    z = torch.full((8, 1), 0.5)
    assert abs(float(vic(z)) - (1.0 - vic.eps**0.5)) < 1e-5


def test_vicreg_one_particle_is_finite_and_pays_missing_spread():
    vic = VICRegLikeLoss()
    z = torch.randn(1, 4, requires_grad=True)
    loss = vic(z)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(1.0 - vic.eps**0.5)
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())


def test_particles_reject_bad_shapes():
    for bad in (lambda: ParticleBatch(0, 4), lambda: ParticleBatch(8, 0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_particle_weights_uniform_at_init():
    torch.manual_seed(0)
    p = ParticleBatch(64, 4, init_std=1e-6)  # ~zero logits -> uniform
    idx = p.sample_indices(8)
    assert idx.shape == (8,)
    assert bool(((idx >= 0) & (idx < 64)).all())
    w = p.weights(idx)
    assert w.shape == (4,)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert bool(((w - 0.25).abs() < 1e-3).all())


def test_particle_weights_sharpen_with_temp():
    torch.manual_seed(0)
    p = ParticleBatch(64, 4)
    idx = p.sample_indices(8)
    w_hot = p.weights(idx, temp=0.1)
    w_cold = p.weights(idx, temp=10.0)
    assert float(w_hot.max()) > float(w_cold.max())
    try:
        p.weights(idx, temp=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for temp=0")


def test_game_grad_reaches_only_sampled_particles():
    # D-loss small = D separates easily = student WEAK there. Minimizing the
    # weighted gaps must shift mass toward the weak row (row 0 here).
    torch.manual_seed(0)
    p = ParticleBatch(8, 4)
    opt = torch.optim.SGD(p.parameters(), lr=1.0)
    idx = torch.tensor([0, 1, 2, 3])
    gaps = torch.tensor([0.0, 0.7, 0.7, 0.7])
    w_before = p.weights(idx).detach()
    w = p.weights(idx)
    (w * gaps).sum().backward()
    grad = p.logits.grad
    assert grad is not None
    # Sampled rows carry game grads; unsampled rows get exactly none here.
    assert bool((grad[:4].abs().sum() > 0).item())
    assert bool((grad[4:].abs().sum() == 0).item())
    opt.step()
    w_after = p.weights(idx).detach()
    assert float(w_after[0]) > float(w_before[0])


def test_uniform_floor_survives_shared_logit_shift():
    # The old VICReg(logits) did not notice this collapse: it centers each
    # column, which deletes the very bias that concentrates the row weights.
    torch.manual_seed(7)
    p = ParticleBatch(64, 4)
    idx = torch.arange(p.num_particles)
    vic = VICRegLikeLoss()
    before_vic = vic(p.logits).detach()
    before_balance = p.balance_loss(idx).detach()
    with torch.no_grad():
        p.logits[:, 0] += 1000.0
    assert torch.allclose(vic(p.logits).detach(), before_vic, atol=1e-5)
    raw = p.weights(idx, uniform_mix=0.0).detach()
    assert raw[0] == 1.0
    w = p.weights(idx).detach()
    assert torch.allclose(w, torch.tensor([0.925, 0.025, 0.025, 0.025]))
    assert p.balance_loss(idx).detach() > before_balance + 100.0


def test_balance_restores_rows_from_saturated_logits_and_only_updates_samples():
    p = ParticleBatch(8, 4, init_std=0.0)
    idx = torch.tensor([0, 2, 4, 6])
    with torch.no_grad():
        p.logits[:, 0] = 1000.0
    loss = p.balance_loss(idx)
    loss.backward()
    grad = p.logits.grad
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(grad).all())
    # Gradient descent lowers dominant logits and raises omitted rows even
    # though the unfloored softmax has rounded exactly to [1, 0, 0, 0].
    assert bool((grad[idx, 0] > 0).all())
    assert bool((grad[idx, 1:] < 0).all())
    assert torch.equal(grad[torch.tensor([1, 3, 5, 7])], torch.zeros(4, 4))


def test_balance_allows_specialized_particles_when_rows_are_covered():
    p = ParticleBatch(4, 4, init_std=0.0)
    with torch.no_grad():
        p.logits.copy_(torch.eye(4) * 40.0)
    idx = torch.arange(4)
    assert float(p.balance_loss(idx).detach()) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(p.weights(idx), torch.full((4,), 0.25))


def test_single_row_mining_remains_finite():
    p = ParticleBatch(1, 1)
    idx = torch.tensor([0])
    assert torch.equal(p.weights(idx), torch.ones(1))
    loss = p.balance_loss(idx)
    loss.backward()
    assert loss == 0 and bool(torch.isfinite(p.logits.grad).all())


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_uniform_mixture_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        ParticleBatch(4, 4, uniform_mix=bad)
    p = ParticleBatch(4, 4)
    with pytest.raises(ValueError):
        p.weights(torch.arange(4), uniform_mix=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_mining_rejects_invalid_temperatures(bad):
    p = ParticleBatch(4, 4)
    with pytest.raises(ValueError):
        p.weights(torch.arange(4), temp=bad)
    with pytest.raises(ValueError):
        p.balance_loss(torch.arange(4), temp=bad)
