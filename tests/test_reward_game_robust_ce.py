import torch
import pytest
from conceptmod.textsliders.reward_game.robust_ce import value_and_weight


def test_scalar_pullback_matches_balanced_nonlinear_objective():
    current = torch.tensor([7.1, 7.8, 8.0, 8.2, 7.7, 7.9, 6.8, 8.4], dtype=torch.float64, requires_grad=True)
    baseline = torch.tensor([7.5, 7.8, 7.9, 8.0, 7.4, 7.7, 7.0, 8.1], dtype=torch.float64)
    objective = (.1*torch.nn.functional.softplus((.1-(current-baseline))/.1)).mean()
    expected, = torch.autograd.grad(objective, current)
    values, weights = zip(*(value_and_weight(float(x), float(b)) for x, b in zip(current.detach(), baseline)))
    torch.testing.assert_close(torch.tensor([-w/8 for w in weights], dtype=torch.float64), expected)
    assert sum(values)/8 == pytest.approx(float(objective))
    assert weights[0] > weights[1] > weights[3]


def test_extreme_finite_comparisons_and_nonfinite_rejection():
    assert value_and_weight(0., 10.)[1] == 1.
    assert value_and_weight(10., 0.)[1] < 1e-40
    with pytest.raises(ValueError, match='Nonfinite'):
        value_and_weight(float('nan'), 7.)
