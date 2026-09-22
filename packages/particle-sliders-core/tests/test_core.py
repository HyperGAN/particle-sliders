import pytest
import torch

from particle_sliders import (
    RoutedMLP, GlobalMixErrorCritic, particle_vic, rp_d_loss, rp_g_loss,
    fit_routed_down,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(71)
    torch.set_num_threads(1)


def test_particle_permutation_and_gradients():
    model = RoutedMLP(8, 8)
    x = torch.randn(2, 5, 8, requires_grad=True)
    particles = torch.randn(128, 4, requires_grad=True)
    expected = model(x, particles)
    actual = model(x, particles[torch.randperm(128)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    (expected.square().mean() + particle_vic(particles)).backward()
    for value in (x, particles, *model.parameters()):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_critic_supports_gradient_penalty_double_backward():
    critic = GlobalMixErrorCritic(torch.randn(16, 24), tokens=4, width=16, heads=4)
    x = torch.randn(8, 24, requires_grad=True)
    scores = critic(x)
    gradient, = torch.autograd.grad(scores.sum(), x, create_graph=True)
    gradient.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.parameters())
    assert scores.abs().max() <= 8


def test_losses_prefer_opposite_orderings():
    real, fake = torch.ones(8), -torch.ones(8)
    assert rp_d_loss(real, fake) < rp_d_loss(fake, real)
    assert rp_g_loss(real, fake) > rp_g_loss(fake, real)


def test_dual_fit_matches_independent_primal_solution_and_heldout():
    x = torch.randn(96, 12, dtype=torch.float64)
    known = torch.randn(8, 12, dtype=torch.float64)
    y = x @ known.T
    down, ridge = fit_routed_down(x, y)
    primal = torch.linalg.solve(x.T @ x + ridge * torch.eye(12, dtype=x.dtype), x.T @ y).T
    torch.testing.assert_close(down, primal, rtol=1e-10, atol=1e-10)
    heldout = torch.randn(32, 12, dtype=x.dtype)
    error = (heldout @ (down - known).T).square().sum()
    assert error / (heldout @ known.T).square().sum() < 1e-4


def test_zero_and_underdetermined_calibration():
    down, ridge = fit_routed_down(torch.zeros(4, 12), torch.randn(4, 8))
    assert torch.count_nonzero(down) == 0 and ridge > 0
    down, _ = fit_routed_down(torch.randn(4, 12), torch.randn(4, 8))
    assert down.shape == (8, 12) and torch.isfinite(down).all()


@pytest.mark.parametrize("case", ["rows", "nan", "ridge", "empty", "dtype"])
def test_rejects_invalid_calibration(case):
    x, y, ridge = torch.randn(8, 12), torch.randn(8, 4), 0.01
    if case == "rows": y = y[:3]
    if case == "nan": x[0, 0] = float("nan")
    if case == "ridge": ridge = -1
    if case == "empty": x, y = x[:0], y[:0]
    if case == "dtype": y = y.double()
    with pytest.raises(ValueError):
        fit_routed_down(x, y, ridge_fraction=ridge)
