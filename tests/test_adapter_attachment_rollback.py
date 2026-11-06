"""CPU-only failures after native host mutation must leave a reusable host."""
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.adapter_attachment import attachment_transaction
from particle_sliders.textsliders import ltx25_backend as ltx
from particle_sliders.textsliders import minimax_h3_backend as h3
from particle_sliders.textsliders.supra_model import attach_supra_lora


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3, 3)
        self.to_k = nn.Linear(3, 3)

    def forward(self, x):
        return self.to_q(x) + self.to_k(x)


class MiniMaxH3Attention(Attention):
    pass


def record_host(host):
    return ([(m, 'forward' in vars(m), vars(m).get('forward')) for m in host.modules()],
            [(p, p.requires_grad) for p in host.parameters()],
            [(name, module) for name, module in host.named_modules()],
            {key: value.clone() for key, value in host.state_dict().items()},
            torch.get_rng_state().clone())


def assert_unchanged(host, before):
    forwards, gradients, modules, state, rng = before
    for module, owned, forward in forwards:
        assert ('forward' in vars(module)) == owned
        assert vars(module).get('forward') is forward
    assert all(parameter.requires_grad == required for parameter, required in gradients)
    assert list(host.named_modules()) == modules
    assert host.state_dict().keys() == state.keys()
    assert all(torch.equal(host.state_dict()[key], value) for key, value in state.items())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('kind', ['native', 'legacy', 'dora', 'h3', 'ltx-attn', 'ltx-module'])
def test_partial_network_attachment_restores_host_forwards_and_rng(kind, monkeypatch):
    host = nn.Module()
    host.attn = MiniMaxH3Attention() if kind == 'h3' else Attention()
    # Exercise preservation of both class-defined and instance-owned forwards.
    host.attn.to_q.forward = host.attn.to_q.forward
    next(host.parameters()).requires_grad_(False)
    if kind in ('native', 'legacy', 'dora'):
        path = {'native': 'particle_sliders.native_lora',
                'legacy': 'particle_sliders.textsliders.lora',
                'dora': 'particle_sliders.textsliders.dora'}[kind]
        module = importlib.import_module(path)
        cls = module.DoRANetwork if kind == 'dora' else module.LoRANetwork
        construct = lambda: cls(host, rank=2, alpha=2., target_replace=['Attention'])
    elif kind == 'h3':
        cls = h3.AttnLoRANetwork
        construct = lambda: cls(host, rank=2, alpha=2.)
    elif kind == 'ltx-attn':
        cls = ltx.AttnLoRANetwork
        host.transformer_blocks = nn.ModuleList([nn.Module()])
        host.transformer_blocks[0].attn1 = host.attn
        del host.attn
        construct = lambda: cls(host, rank=2, alpha=2.)
    else:
        cls = ltx.ModuleLoRANetwork
        construct = lambda: cls([('q', host.attn.to_q), ('k', host.attn.to_k)], rank=2, alpha=2.)

    def fail_after_registration(self, name, module):
        nn.Module.add_module(self, name, module)
        raise RuntimeError('failure after first host patch')

    monkeypatch.setattr(cls, 'add_module', fail_after_registration)
    before = record_host(host)
    with pytest.raises(RuntimeError, match='after first host patch'):
        construct()
    assert_unchanged(host, before)


def test_supra_missing_later_target_undoes_earlier_module_replacement():
    host = nn.Module()
    host.attn = Attention()
    next(host.parameters()).requires_grad_(False)
    before = record_host(host)
    with pytest.raises(RuntimeError, match='missing'):
        attach_supra_lora(host, rank=2, alpha=2., targets=['attn.to_q', 'missing'])
    assert_unchanged(host, before)


def test_music3_routed_install_failure_undoes_nested_lora_attachment(monkeypatch):
    from particle_sliders import music3_native as music

    class Qwen3Attention(Attention):
        pass

    host = nn.Sequential(Qwen3Attention())
    before = record_host(host)
    original = music.ParticleNetwork._install

    def fail(self, lora, bridge):
        original(self, lora, bridge)
        raise RuntimeError('routed rebind failure')

    monkeypatch.setattr(music.ParticleNetwork, '_install', fail)
    with pytest.raises(RuntimeError, match='routed rebind failure'):
        music.ParticleNetwork(host)
    assert_unchanged(host, before)


def test_yue2_last_attachment_failure_restores_base_gradient_flags():
    from particle_sliders.yue2_native import ParticleSlider

    class FailingLinear(nn.Linear):
        def __setattr__(self, name, value):
            if name == 'forward':
                raise RuntimeError('projection patch failure')
            super().__setattr__(name, value)

    host = nn.Module()
    host.config = SimpleNamespace(model_type='yue2', num_hidden_layers=1)
    host.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        setattr(layer.self_attn, name, (FailingLinear if name == 'k_proj' else nn.Linear)(3, 3))
    host.model.layers = nn.ModuleList([layer])
    before = record_host(host)
    with pytest.raises(RuntimeError, match='projection patch failure'):
        ParticleSlider(host)
    assert_unchanged(host, before)


def test_transaction_restores_module_containers_and_new_metadata():
    host = nn.Sequential(nn.Linear(3, 3))
    original_modules = host._modules
    before = record_host(host)
    with pytest.raises(ValueError, match='mutation failure'):
        with attachment_transaction([host]):
            host[0] = nn.Sequential(host[0], nn.Linear(3, 3))
            host.peft_config = {'default': object()}
            host.requires_grad_(False)
            raise ValueError('mutation failure')
    assert host._modules is original_modules
    assert not hasattr(host, 'peft_config')
    assert_unchanged(host, before)


def test_successful_attachment_keeps_rng_progress_and_published_module():
    host = nn.Module()
    host.attn = Attention()
    rng = torch.get_rng_state().clone()
    adapters = attach_supra_lora(host, rank=2, alpha=2., targets=['attn.to_q'])
    assert host.attn.to_q is adapters[0]
    assert not torch.equal(torch.get_rng_state(), rng)
    assert adapters[0].down.weight.requires_grad
    assert not adapters[0].base.weight.requires_grad
