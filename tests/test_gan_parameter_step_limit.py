import math

import pytest
import torch

from analysis.gan_bcap.parameter_step_limit import bound_step


def test_global_parameter_step_bound_preserves_direction():
    before = [torch.tensor([1., -2.], dtype=torch.float64), torch.tensor([3.], dtype=torch.float64)]
    deltas = [torch.tensor([3., 4.], dtype=torch.float64), torch.tensor([12.], dtype=torch.float64)]
    params = [torch.nn.Parameter(a+b) for a,b in zip(before,deltas)]
    record = bound_step(params, before, 2.)
    assert record['proposed_update_norm'] == pytest.approx(13.)
    assert record['actual_update_norm'] == pytest.approx(2.)
    for p,a,d in zip(params,before,deltas):
        assert torch.allclose(p-a, d*(2/13))


def test_inactive_parameter_bound_is_bitwise_identity():
    before = [torch.tensor([1., -2.])]
    params = [torch.nn.Parameter(torch.tensor([1.1,-1.9]))]
    unchanged = params[0].detach().clone()
    record = bound_step(params,before,2.)
    assert record['factor'] == 1.
    assert torch.equal(unchanged,params[0])
    for invalid in [0., -1., math.inf, math.nan]:
        with pytest.raises(ValueError):
            bound_step(params,before,invalid)
