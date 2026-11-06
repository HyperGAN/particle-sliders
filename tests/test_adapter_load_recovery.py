"""CPU-only checkpoint and PEFT construction failure recovery."""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.adapter_loading import load_adapter_states, optional_peft
from particle_sliders.textsliders import ltx25_backend as ltx
from particle_sliders.textsliders import minimax_h3_backend as h3


def copied_state(module):
    return {key: value.clone() for key, value in module.state_dict().items()}


def assert_state(module, expected):
    assert module.state_dict().keys() == expected.keys()
    assert all(torch.equal(module.state_dict()[key], value) for key, value in expected.items())


def custom_network(kind):
    if kind == 'h3':
        return h3.AttnLoRANetwork(h3.DummyOmniTransformer(), rank=2, alpha=2.)
    if kind == 'ltx':
        return ltx.AttnLoRANetwork(ltx.DummyLTX2Transformer(), rank=2, alpha=2.)
    return ltx.ModuleLoRANetwork([('te.q', nn.Linear(3, 3)), ('conn.q', nn.Linear(3, 3))],
                                 rank=2, alpha=2.)


@pytest.mark.parametrize('kind', ['h3', 'ltx', 'ltx-embed'])
@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape', 'nonfinite'])
def test_custom_checkpoint_refusal_never_changes_existing_weights(kind, damage, monkeypatch):
    network = custom_network(kind)
    before = copied_state(network)
    payload = {key: torch.full_like(value, .75) for key, value in before.items()}
    key = next(iter(payload))
    if damage == 'missing':
        payload.pop(list(payload)[-1])
    elif damage == 'unexpected':
        payload['foreign.weight'] = torch.ones(1)
    elif damage == 'shape':
        payload[key] = torch.ones(99)
    else:
        payload[key].fill_(float('nan'))
    monkeypatch.setattr('safetensors.torch.load_file', lambda path: payload)
    with pytest.raises(ValueError):
        network.load_weights('bad.safetensors')
    assert_state(network, before)


class PeftHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.lora_A = nn.Linear(3, 2, bias=False)
        self.lora_B = nn.Linear(2, 3, bias=False)
        self.scaling = {'default': 1.}


def test_peft_checkpoint_requires_every_attached_adapter_key(monkeypatch):
    host = PeftHost()
    network = ltx.PeftLoRANetwork(host, 2, 2.)
    before = copied_state(host)
    monkeypatch.setattr('safetensors.torch.load_file',
                        lambda path: {'lora_A.weight': torch.full_like(host.lora_A.weight, .75)})
    with pytest.raises(ValueError, match='keys differ'):
        network.load_weights('missing.safetensors')
    assert_state(host, before)


def test_embed_checkpoint_validates_both_modules_before_copying(monkeypatch):
    encoder, connectors = PeftHost(), PeftHost()
    network = ltx._EmbedPeftLoRANetwork(encoder, connectors, 2, 2.)
    before = copied_state(encoder), copied_state(connectors)
    payload = {key: torch.full_like(value, .75) for key, value in network.state_dict().items()
               if not key.startswith('conn.')}
    monkeypatch.setattr('safetensors.torch.load_file', lambda path: payload)
    with pytest.raises(ValueError, match='keys differ'):
        network.load_weights('missing-connector.safetensors')
    assert_state(encoder, before[0])
    assert_state(connectors, before[1])


def test_failed_later_module_loader_rolls_back_both_adapter_subsets(monkeypatch):
    encoder, connectors = PeftHost(), PeftHost()
    network = ltx._EmbedPeftLoRANetwork(encoder, connectors, 2, 2.)
    before = copied_state(encoder), copied_state(connectors)
    payload = {key: torch.full_like(value, .75) for key, value in network.state_dict().items()}
    monkeypatch.setattr('safetensors.torch.load_file', lambda path: payload)

    def fail_after_copy(module, incompatible):
        raise RuntimeError('loader hook failure')

    connectors.register_load_state_dict_post_hook(fail_after_copy)
    with pytest.raises(RuntimeError, match='loader hook failure'):
        network.load_weights('complete.safetensors')
    assert_state(encoder, before[0])
    assert_state(connectors, before[1])


def test_finite_checkpoint_that_overflows_target_precision_is_refused():
    module = nn.Linear(2, 2, bias=False).half()
    before = copied_state(module)
    with pytest.raises(ValueError, match='invalid adapter'):
        load_adapter_states([(module, {'weight': torch.full((2, 2), 1e30)}, ['weight'])])
    assert_state(module, before)


def test_successful_peft_load_changes_adapters_and_leaves_base_untouched(monkeypatch):
    host = PeftHost()
    network = ltx.PeftLoRANetwork(host, 2, 2.)
    base = copied_state(host.base)
    payload = {key: torch.full_like(value, .75) for key, value in network.state_dict().items()}
    monkeypatch.setattr('safetensors.torch.load_file', lambda path: payload)
    network.load_weights('complete.safetensors')
    assert all(torch.equal(value, payload[key]) for key, value in network.state_dict().items())
    assert_state(host.base, base)


def bare_backend():
    backend = object.__new__(ltx.LTX25Backend)
    backend.lora_rank, backend.lora_alpha, backend.lora_up_init_std = 2, 2., 0.
    backend.device = backend.encoder_device = torch.device('cpu')
    backend.te_last_n, backend._peft = 1, False
    return backend


def test_peft_failure_after_host_mutation_is_rolled_back_and_not_fallback(monkeypatch):
    backend = bare_backend()
    host = ltx.DummyLTX2Transformer()
    backend.transformer = host
    backend.pipe = SimpleNamespace(transformer=host)
    before = copied_state(host)
    modules = list(host.named_modules())
    rng = torch.get_rng_state().clone()

    def fail(root, config):
        root.extra = nn.Linear(3, 3)
        root.requires_grad_(False)
        raise RuntimeError('peft mutated then failed')

    monkeypatch.setattr(ltx, 'optional_peft', lambda: (lambda **kwargs: kwargs, fail))
    monkeypatch.setattr(ltx, 'AttnLoRANetwork', lambda *a, **k: pytest.fail('unsafe fallback'))
    with pytest.raises(RuntimeError, match='mutated then failed'):
        backend._attach_lora(host)
    assert list(host.named_modules()) == modules
    assert_state(host, before)
    assert all(parameter.requires_grad for parameter in host.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    assert backend.transformer is backend.pipe.transformer is host
    assert not backend._peft


def test_failed_connector_wrap_does_not_publish_partial_text_encoder(monkeypatch):
    backend = bare_backend()
    backend.encoder, backend.connectors = nn.Sequential(nn.Linear(3, 3)), nn.Sequential(nn.Linear(3, 3))
    encoder, connectors = backend.encoder, backend.connectors
    backend.pipe = SimpleNamespace(text_encoder=encoder, connectors=connectors)
    before = copied_state(encoder), copied_state(connectors)
    monkeypatch.setattr(ltx, 'embed_lora_hosts', lambda *a, **k: [('te.q', encoder[0])])
    monkeypatch.setattr(ltx, 'te_last_n_attn_targets', lambda *a, **k: ['0'])
    monkeypatch.setattr(ltx, 'video_connector_lora_targets', lambda *a, **k: ['0'])

    def wrap(root, config):
        root.extra = nn.Linear(3, 3)
        if root is connectors:
            raise RuntimeError('connector wrap failure')
        return nn.Sequential(root)

    monkeypatch.setattr(ltx, 'optional_peft', lambda: (lambda **kwargs: kwargs, wrap))
    monkeypatch.setattr(ltx, 'ModuleLoRANetwork', lambda *a, **k: pytest.fail('unsafe fallback'))
    with pytest.raises(RuntimeError, match='connector wrap failure'):
        backend._attach_embed_lora()
    assert backend.encoder is backend.pipe.text_encoder is encoder
    assert backend.connectors is backend.pipe.connectors is connectors
    assert not hasattr(encoder, 'extra') and not hasattr(connectors, 'extra')
    assert_state(encoder, before[0])
    assert_state(connectors, before[1])
    assert not backend._peft


def test_missing_optional_peft_uses_the_custom_adapter_without_mixed_wrapping(monkeypatch):
    backend = bare_backend()
    host = ltx.DummyLTX2Transformer()
    backend.transformer = host
    backend.pipe = SimpleNamespace(transformer=host)
    monkeypatch.setattr(ltx, 'optional_peft', lambda: None)
    network = backend._attach_lora(host)
    assert isinstance(network, ltx.AttnLoRANetwork)
    assert network.loras
    assert backend.transformer is backend.pipe.transformer is host
    assert not backend._peft


def test_successful_peft_construction_publishes_only_the_completed_wrapper(monkeypatch):
    backend = bare_backend()
    host, wrapped = nn.Sequential(nn.Linear(3, 3)), PeftHost()
    backend.transformer = host
    backend.pipe = SimpleNamespace(transformer=host)
    monkeypatch.setattr(ltx, 'video_attn_lora_targets', lambda root: ['0'])
    monkeypatch.setattr(ltx, 'optional_peft',
                        lambda: (lambda **kwargs: kwargs, lambda root, config: wrapped))
    network = backend._attach_lora(host)
    assert isinstance(network, ltx.PeftLoRANetwork)
    assert network.transformer is backend.transformer is backend.pipe.transformer is wrapped
    assert backend._peft


@pytest.mark.parametrize('missing, fallback', [('peft', True), ('pkg_resources', False)])
def test_only_missing_peft_package_allows_fallback(missing, fallback, monkeypatch):
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == 'peft':
            raise ModuleNotFoundError('simulated missing module', name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_module)
    if fallback:
        assert optional_peft() is None
    else:
        with pytest.raises(ModuleNotFoundError) as failure:
            optional_peft()
        assert failure.value.name == missing


def test_embed_load_preserves_encoder_device_instead_of_moving_to_dit(monkeypatch):
    backend = bare_backend()
    backend.embed_recipe = True
    backend.device = torch.device('cuda:9')  # Sentinel only; no CUDA operation.
    placements = []
    backend.network = SimpleNamespace(load_weights=lambda path: None,
                                      to=lambda device: placements.append(device))
    monkeypatch.setattr(ltx, 'resolve_ltx_lora_path', lambda path: Path('complete.safetensors'))
    backend.load_trained('checkpoint')
    assert placements == [torch.device('cpu')]
