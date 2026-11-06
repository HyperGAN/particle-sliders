"""CPU-only regression checks for nested effective adapter activation."""
import importlib

import pytest
import torch
from torch import nn

from particle_sliders.textsliders import ltx25_backend as ltx
from particle_sliders.textsliders import minimax_h3_backend as h3


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3, 3)
        self.to_k = nn.Linear(3, 3)

    def forward(self, x):
        return self.to_q(x) + self.to_k(x)


class MiniMaxH3Attention(Attention):
    pass


def custom_network(kind):
    if kind in ('native', 'legacy', 'dora'):
        path = {'native': 'particle_sliders.native_lora',
                'legacy': 'particle_sliders.textsliders.lora',
                'dora': 'particle_sliders.textsliders.dora'}[kind]
        module = importlib.import_module(path)
        host = Attention()
        cls = module.DoRANetwork if kind == 'dora' else module.LoRANetwork
        network = cls(host, rank=2, alpha=2., target_replace=['Attention'])
        return host, network, network.unet_loras
    host = nn.Module()
    host.attn = MiniMaxH3Attention()
    if kind == 'h3':
        network = h3.AttnLoRANetwork(host, rank=2, alpha=2.)
    elif kind == 'ltx-attn':
        # Match LTX's native video-attention path filter.
        host.transformer_blocks = nn.ModuleList([nn.Module()])
        host.transformer_blocks[0].attn1 = host.attn
        del host.attn
        network = ltx.AttnLoRANetwork(host, rank=2, alpha=2.)
    else:
        network = ltx.ModuleLoRANetwork(
            [('attn.to_q', host.attn.to_q), ('attn.to_k', host.attn.to_k)], rank=2, alpha=2.)
    leaf = host.transformer_blocks[0].attn1 if kind == 'ltx-attn' else host.attn
    return leaf, network, network.loras


@pytest.mark.parametrize('kind', ['native', 'legacy', 'dora', 'h3', 'ltx-attn', 'ltx-module'])
def test_nested_scope_restores_effective_outputs_and_heterogeneous_scales(kind):
    host, network, adapters = custom_network(kind)
    assert len(adapters) == 2
    with torch.no_grad():
        for index, adapter in enumerate(adapters):
            adapter.lora_up.weight.fill_(.2 * (index + 1))
    for adapter, value in zip(adapters, [.25, -.5]):
        adapter.multiplier = value
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    before = host(values).detach().clone()
    network.lora_scale = .75
    with network:
        outer = host(values).detach().clone()
        assert all(a.multiplier == (1. if kind == 'dora' else .75) for a in adapters)
        if kind != 'dora':
            network.set_lora_slider(-.5)
        with pytest.raises(ValueError, match='body failure'):
            with network:
                assert all(a.multiplier == (1. if kind == 'dora' else -.5) for a in adapters)
                raise ValueError('body failure')
        assert torch.equal(host(values), outer)
    assert [a.multiplier for a in adapters] == [.25, -.5]
    assert torch.equal(host(values), before)


@pytest.mark.parametrize('kind', ['native', 'legacy'])
def test_scope_restores_exact_per_layer_sequence_envelopes(kind):
    _, network, adapters = custom_network(kind)
    originals = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
    for adapter, envelope in zip(adapters, originals):
        adapter.seq_gain, adapter.seq_gain_mode = envelope, 'prefix'
    network.seq_gain = torch.tensor([.5])
    network.seq_gain_mode = 'stretch'
    with network:
        outer = network.seq_gain
        assert all(a.seq_gain is outer for a in adapters)
        network.set_seq_gain([.2, .4], mode='prefix')
        assert all(a.seq_gain is outer for a in adapters)
        with network:
            assert all(a.seq_gain is network.seq_gain for a in adapters)
        assert all(a.seq_gain is outer for a in adapters)
    for adapter, envelope in zip(adapters, originals):
        assert adapter.seq_gain is envelope
        assert adapter.seq_gain_mode == 'prefix'


class ScaleLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.lora_A = nn.Linear(3, 3, bias=False)
        self.scaling = {'slider': factor, 'other': factor / 2}

    def forward(self, value):
        return value * sum(self.scaling.values())


@pytest.mark.parametrize('embed', [False, True])
def test_peft_scope_restores_dictionary_objects_values_and_nested_output(embed):
    first, second = nn.Sequential(ScaleLayer(2.)), nn.Sequential(ScaleLayer(3.))
    network = (ltx._EmbedPeftLoRANetwork(first, second, 2, 2.) if embed
               else ltx.PeftLoRANetwork(nn.Sequential(first, second), 2, 2.))
    layers = [first[0], second[0]]
    dictionaries = [layer.scaling for layer in layers]
    for layer, value in zip(layers, [.25, -.5]):
        layer.scaling['slider'] = value
    original = [dict(layer.scaling) for layer in layers]
    network.lora_scale = .75
    with network:
        outer = [dict(layer.scaling) for layer in layers]
        network.set_lora_slider(-.5)
        assert [layer.scaling for layer in layers] == outer
        with network:
            assert [layer.scaling['slider'] for layer in layers] == [-1., -1.5]
        assert [layer.scaling for layer in layers] == outer
    for layer, dictionary, expected in zip(layers, dictionaries, original):
        assert layer.scaling is dictionary
        assert layer.scaling == expected


def test_peft_failed_entry_restores_all_touched_settings():
    host = nn.Sequential(ScaleLayer(2.), ScaleLayer(3.))
    network = ltx.PeftLoRANetwork(host, 2, 2.)
    original = [dict(layer.scaling) for layer in host]

    def fail(scale):
        host[0].scaling['slider'] = 999.
        raise ValueError('entry failure')

    network._apply_scale = fail
    with pytest.raises(ValueError, match='entry failure'):
        with network:
            pytest.fail('failed entry must not execute the body')
    assert [layer.scaling for layer in host] == original
    assert network._activation_stack == []
