"""Rejected envelope configuration leaves pending and effective state intact."""
import importlib
from importlib.metadata import distribution
from pathlib import Path

import pytest
import torch
from torch import nn


def adapter_module(name, kind):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        # 0.14 intentionally excludes embedded products. Never hide a missing
        # generic module, a missing lora/dora module, or an internal dependency.
        if kind != 'native' and error.name == 'particle_sliders.textsliders':
            import particle_sliders
            installed = distribution('particle-sliders-core')
            origin = Path(installed.locate_file('particle_sliders/__init__.py')).resolve()
            if installed.version == particle_sliders.__version__ and Path(particle_sliders.__file__).resolve() == origin:
                pytest.skip('installed shared core excludes legacy textsliders; retained source 0.13 runs this case')
        raise


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3)

    def forward(self, x):
        return self.proj(x)


def network(kind):
    module = adapter_module('particle_sliders.' + ('native_lora' if kind == 'native' else 'textsliders.lora'), kind)
    host = Attention()
    value = module.LoRANetwork(host, rank=2, alpha=2., target_replace=['Attention'])
    with torch.no_grad():
        value.unet_loras[0].lora_up.weight.fill_(.2)
    value.set_seq_gain([.5, 1.], mode='prefix')
    return host, value


BAD = [([], 'stretch'), (torch.empty(0), 'prefix'), ([float('nan')], 'stretch'),
       ([float('inf')], 'prefix'), ([-float('inf')], 'stretch'),
       (torch.tensor([1e100], dtype=torch.float64), 'prefix'),
       ([1.], 'unknown'), (None, 'unknown')]


@pytest.mark.parametrize('kind', ['native', 'legacy'])
@pytest.mark.parametrize('gain,mode', BAD)
@pytest.mark.parametrize('active', [False, True])
def test_invalid_envelope_preserves_all_configuration_and_effective_values(kind, gain, mode, active):
    host, value = network(kind)
    x = torch.randn(1, 4, 3)
    if active:
        value.__enter__()
        value.set_seq_gain([2., 3.], mode='stretch')  # Pending next entry only.
    try:
        before = host(x).detach().clone()
        configured = value.seq_gain, value.seq_gain_mode
        layers = [(layer, layer.seq_gain, layer.seq_gain_mode, layer.multiplier) for layer in value.unet_loras]
        stack = list(getattr(value, '_activation_stack', []))
        with pytest.raises(ValueError, match='Sequence gain'):
            value.set_seq_gain(gain, mode=mode)
        assert value.seq_gain is configured[0] and value.seq_gain_mode == configured[1]
        for layer, envelope, previous_mode, strength in layers:
            assert layer.seq_gain is envelope and layer.seq_gain_mode == previous_mode
            assert layer.multiplier == strength
        assert getattr(value, '_activation_stack', []) == stack
        assert torch.equal(host(x), before)
        if active:
            with value:
                assert all(layer.seq_gain is configured[0] for layer in value.unet_loras)
            assert torch.equal(host(x), before)
    finally:
        if active:
            value.__exit__(None, None, None)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind', ['native', 'legacy'])
def test_valid_prefix_stretch_scalar_and_uniform_inputs_keep_their_mapping(kind):
    _, value = network(kind)
    layer = value.unet_loras[0]
    value.set_seq_gain([1., 2.], mode='prefix')
    assert torch.equal(layer._seq_gain_broadcast(torch.ones(1, 4, 3)), torch.tensor([1., 2., 2., 2.]).reshape(1, 4, 1))
    value.set_seq_gain([1., 3.], mode='stretch')
    assert torch.equal(layer._seq_gain_broadcast(torch.ones(1, 3, 3)), torch.tensor([1., 2., 3.]).reshape(1, 3, 1))
    value.set_seq_gain(torch.tensor(.5))
    assert layer._seq_gain_broadcast(torch.ones(1, 3, 3)).item() == .5
    value.set_seq_gain(None)
    assert layer._seq_gain_broadcast(torch.ones(1, 3, 3)) == 1.
