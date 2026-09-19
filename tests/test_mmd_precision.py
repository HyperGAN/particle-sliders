import pytest
import torch

from analysis.gan_bcap.mmd_game_20260905.precision import directional_probe
from analysis.gan_bcap.mmd_game_20260905.precision_memory import directional_probe as memory_probe


@pytest.mark.parametrize("probe", [directional_probe, memory_probe])
def test_directional_probe_matches_quadratic_derivative_and_restores_state(probe):
    p = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
    p.grad = torch.tensor([3., 4.], dtype=torch.float64)
    initial = p.detach().clone()
    rng = torch.get_rng_state().clone()
    def losses():
        yield p.square().sum()/2
    report = probe([p], losses)
    assert report['no_grad_repeats'] == [2.5]*3
    assert report['grad_forward_repeats'] == [2.5]*2
    for trial in report['trials']:
        assert trial['finite_difference'] == pytest.approx(-5**.5, abs=1e-10)
        assert trial['sides']['1']['change'] < 0
    assert torch.equal(p, initial)
    assert torch.equal(p.grad, torch.tensor([3., 4.], dtype=torch.float64))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("probe", [directional_probe, memory_probe])
def test_probe_restores_parameters_and_gradients_after_failed_evaluation(probe):
    p = torch.nn.Parameter(torch.tensor([2.]))
    initial = p.detach().clone()
    calls = 0
    def losses():
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RuntimeError('evaluation failed')
        yield p.square().sum()
    with pytest.raises(RuntimeError, match='evaluation failed'):
        probe([p], losses)
    assert torch.equal(p, initial)
    assert p.grad is None
