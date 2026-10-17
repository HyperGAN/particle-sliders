"""Public scale contexts, central image checkpoint binding and atomic head reload."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.pg09_backends import lora_scale_context, _lora_scale_context
from particle_sliders.pg09_images import IrisResponseAdapter, QwenImage21ResponseAdapter
from particle_sliders.bonsai_native import BonsaiParticleHead


def test_public_lora_scale_context_preserves_heterogeneous_nested_scales():
    assert lora_scale_context is _lora_scale_context
    local = nn.Module()
    local.down, local.up = nn.Linear(1, 1), nn.Linear(1, 1)
    local.multiplier = 2.5
    peft = nn.Module()
    peft.lora_A, peft.scaling = {}, {'first': 3., 'second': .25}
    context = lora_scale_context([('local', local), ('peft', peft)])
    with context(-2.):
        assert local.multiplier == -5. and peft.scaling == {'first': -6., 'second': -.5}
        with context(0.):
            assert local.multiplier == 0. and peft.scaling == {'first': 0., 'second': 0.}
        assert local.multiplier == -5. and peft.scaling == {'first': -6., 'second': -.5}
    assert local.multiplier == 2.5 and peft.scaling == {'first': 3., 'second': .25}
    with pytest.raises(RuntimeError):
        with context(-.5):
            raise RuntimeError('forward failed')
    assert local.multiplier == 2.5 and peft.scaling == {'first': 3., 'second': .25}


@pytest.mark.parametrize('adapter_type', [IrisResponseAdapter, QwenImage21ResponseAdapter])
def test_native_image_constructors_bind_signed_repeatable_checkpoint_once(adapter_type):
    scale = [0.]
    particles = nn.Parameter(torch.randn(128, 4))

    class Bank(nn.Module):
        particle_shape = (128, 4)

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.7))

        @contextmanager
        def active(self, actual_particles, strength):
            assert actual_particles is particles
            previous, scale[0] = scale[0], strength
            try:
                yield
            finally:
                scale[0] = previous

    bank = Bank()

    class Model(nn.Module):
        is_gradient_checkpointing = True

        def __init__(self):
            super().__init__()
            self.base = nn.Parameter(torch.tensor(.1), requires_grad=False)
            self.bind_calls = 0
            self.forward_calls = 0

        def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
            self.bind_calls += 1
            self.checkpoint = gradient_checkpointing_func

        def plain(self, value):
            self.forward_calls += 1
            return torch.sin(value * (bank.weight + particles.mean()) * scale[0])

        def forward(self, value):
            return self.checkpoint(self.plain, value)

    model = Model()
    adapter = adapter_type(SimpleNamespace(model=model), bank, particles)
    assert model.bind_calls == 1
    expected, actual = [], []
    for sign in (1., -1.):
        with adapter.network.scaled(sign):
            value = torch.tensor([[.3, -.8]])
            expected.append(model.plain(value))
            actual.append(model(value))
    loss = lambda values: values[0].sum() + .3 * values[1].square().sum()
    leaves = (bank.weight, particles)
    expected_grad = torch.autograd.grad(loss(expected), leaves)
    total = loss(actual)
    for _ in range(2):
        calls = model.forward_calls
        actual_grad = torch.autograd.grad(total, leaves, retain_graph=True)
        assert model.forward_calls > calls
        for result, wanted in zip(actual_grad, expected_grad):
            torch.testing.assert_close(result, wanted)
        assert scale[0] == 0. and adapter.network.scale == 1.


@pytest.mark.parametrize('corruption', [None, 'shape', 'extra', 'integer', 'bf16', 'nonfinite', 'metadata'])
def test_bonsai_load_validates_exact_fp32_schema_and_preserves_rng(tmp_path, corruption, monkeypatch):
    pytest.importorskip('safetensors')
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    head = BonsaiParticleHead()
    with torch.no_grad():
        head.up.weight.fill_(.0123)
    path = tmp_path / 'head.safetensors'
    head.save(path, {'test': True})
    state = load_file(str(path))
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
    if corruption == 'shape':
        state['particles'] = state['particles'][:-1]
    elif corruption == 'extra':
        state['unexpected'] = torch.zeros(1)
    elif corruption == 'integer':
        state['particles'] = state['particles'].long()
    elif corruption == 'bf16':
        state['particles'] = state['particles'].bfloat16()
    elif corruption == 'nonfinite':
        state['particles'][0, 0] = float('nan')
    elif corruption == 'metadata':
        record = json.loads(metadata['conceptmod'])
        record['rank'] = 4
        metadata['conceptmod'] = json.dumps(record)
    save_file(state, str(path), metadata=metadata)
    rng = torch.get_rng_state().clone()
    if corruption is not None:
        def forbidden(*args, **kwargs):
            raise AssertionError('invalid export constructed a native head')
        monkeypatch.setattr(BonsaiParticleHead, '__init__', forbidden)
        with pytest.raises(ValueError):
            BonsaiParticleHead.load(path)
    else:
        restored, record = BonsaiParticleHead.load(path)
        assert record['format'] == 'conceptmod-bonsai-gguf-particle-v1'
        assert restored.multiplier == 0. and restored.scale == head.scale
        for name, value in head.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name])
        values = torch.randn(1, 5120, generator=torch.Generator().manual_seed(7))
        for sign in (-1., 0., .5, 1.):
            with head.scaled(sign), restored.scaled(sign):
                assert torch.equal(head(values), restored(values))
    assert torch.equal(torch.get_rng_state(), rng)


def test_bonsai_fp32_load_ignores_process_default_dtype(tmp_path):
    pytest.importorskip('safetensors')
    head = BonsaiParticleHead().float()
    path = tmp_path / 'head.safetensors'
    head.save(path, {})
    previous = torch.get_default_dtype()
    rng = torch.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float64)
        restored, _ = BonsaiParticleHead.load(path)
    finally:
        torch.set_default_dtype(previous)
    assert all(value.dtype == torch.float32 and torch.equal(value, restored.state_dict()[key])
               for key, value in head.state_dict().items())
    assert all(value.dtype == torch.float32 for value in restored.state_dict().values())
    assert torch.equal(torch.get_rng_state(), rng)
