"""OFF bypasses generic adapter math even if its branch is nonfinite."""
import importlib

import pytest
import torch
from torch import nn


KINDS = ['native', 'legacy', 'dora']


def adapter(kind, shape):
    torch.manual_seed(12)
    host = nn.Linear(3, 4) if shape == 'linear' else nn.Conv2d(3, 4, 3, padding=1)
    original = host.forward
    module = importlib.import_module({
        'native': 'particle_sliders.native_lora',
        'legacy': 'particle_sliders.textsliders.lora',
        'dora': 'particle_sliders.textsliders.dora',
    }[kind])
    cls = module.DoRAModule if kind == 'dora' else module.LoRAModule
    branch = cls('test', host, multiplier=0., lora_dim=2, alpha=2.)
    branch.apply_to()
    return host, original, branch


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('shape', ['linear', 'conv2d'])
@pytest.mark.parametrize('value', [1., float('nan'), float('inf'), -float('inf')])
def test_zero_multiplier_returns_exact_host_and_skips_nonfinite_branch(kind, shape, value, monkeypatch):
    host, original, branch = adapter(kind, shape)
    with torch.no_grad():
        for parameter in branch.parameters():
            parameter.fill_(value)
    monkeypatch.setattr(branch.lora_down, 'forward', lambda *args: pytest.fail('OFF evaluated adapter'))
    if kind == 'dora':
        monkeypatch.setattr(branch, 'apply_weight_decompose', lambda *args: pytest.fail('OFF normalized adapter'))
    x = torch.randn(2, 3) if shape == 'linear' else torch.randn(2, 3, 5, 5)
    x.requires_grad_(True)
    expected = original(x)
    actual = host(x)
    assert torch.equal(actual, expected) and torch.isfinite(actual).all()
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert all(parameter.grad is None for parameter in branch.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('shape', ['linear', 'conv2d'])
def test_nonzero_multiplier_keeps_the_existing_linear_delta_strength(kind, shape):
    host, original, branch = adapter(kind, shape)
    with torch.no_grad():
        branch.lora_down.weight.fill_(.2)
        branch.lora_up.weight.fill_(.3)
    x = torch.randn(2, 3) if shape == 'linear' else torch.randn(2, 3, 5, 5)
    base = original(x)
    branch.multiplier = 1.
    full = host(x)
    assert not torch.equal(full, base)
    branch.multiplier = -.5
    torch.testing.assert_close(host(x), base - .5 * (full - base))
