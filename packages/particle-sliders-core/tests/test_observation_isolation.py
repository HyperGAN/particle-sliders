"""Observation state/RNG regressions; all execution stays on CPU."""
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

import particle_sliders.lifecycle as lifecycle
from particle_sliders.inference import InferenceBinding, InferenceSession
from particle_sliders.lifecycle import Capabilities
from test_image_lifecycle import session as training_session


def inference_session(network, draw, *, rng_devices=()):
    return InferenceSession(InferenceBinding(network, {}, lambda: None, render=draw),
        Capabilities('adapter_gan', True, render_outputs=('latent',)),
        {'selection': 'raw'}, 'latent', lambda *args: None, rng_devices=rng_devices)


@pytest.mark.parametrize('kind', ['training', 'inference'])
@pytest.mark.parametrize('persistent', [True, False])
@pytest.mark.parametrize('fails', [True, False])
def test_all_adapter_buffers_restored(kind, persistent, fails):
    if kind == 'training':
        active = training_session()
        network = active.game.network
    else:
        network = nn.Linear(2, 2)
    network.register_buffer('observation_count', torch.zeros(()), persistent=persistent)
    network.child = nn.Module()
    network.child.register_buffer('native_count', torch.zeros(()), persistent=False)
    original = network.observation_count
    def draw(**kwargs):
        network.observation_count.add_(1)
        network.child.native_count.add_(2)
        network.train()
        torch.rand(3)
        if fails:
            raise RuntimeError('native failure after buffer mutation')
        return network.observation_count.clone()
    if kind == 'training':
        active.binding.render = draw
    else:
        active = inference_session(network, draw)
    modes = [(module, module.training) for module in network.modules()]
    rng = torch.get_rng_state().clone()
    with active:
        if fails:
            with pytest.raises(RuntimeError, match='native failure'):
                active.render()
        else:
            assert active.render().item() == 1
        assert original is network.observation_count
        assert network.observation_count.item() == network.child.native_count.item() == 0
        assert all(module.training == previous for module, previous in modes)
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('kind', ['training', 'inference'])
def test_buffer_rebinding_restores_original_registration(kind):
    active = training_session() if kind == 'training' else None
    network = active.game.network if active else nn.Linear(2, 2)
    network.register_buffer('transient', torch.ones(2), persistent=False)
    original = network.transient
    def draw(**kwargs):
        network.transient = torch.zeros(3)
        network.register_buffer('new_buffer', torch.zeros(2))
        raise RuntimeError('rebound')
    if active:
        active.binding.render = draw
    else:
        active = inference_session(network, draw)
    with active, pytest.raises(RuntimeError, match='rebound'):
        active.render()
    assert network.transient is original and torch.equal(original, torch.ones(2))
    assert 'transient' in network._non_persistent_buffers_set
    assert not hasattr(network, 'new_buffer')


@pytest.mark.parametrize('kind', ['training', 'inference'])
def test_buffer_only_cuda_device_is_forked_and_seeded_without_cuda_execution(kind, monkeypatch):
    active = training_session() if kind == 'training' else None
    network = active.game.network if active else nn.Linear(2, 2)
    with FakeTensorMode():
        simulated = torch.empty(2, device='cuda:3')
    network.register_buffer('native_state', simulated, persistent=False)
    if active is None:
        active = inference_session(network, lambda **kw: None, rng_devices=(2,))
    forked, seeded = [], []
    original_fork, original_seed = torch.random.fork_rng, lifecycle._seed_observation
    @contextmanager
    def cpu_fork(*, devices):
        forked.append(devices)
        with original_fork(devices=[]):
            yield
    def cpu_seed(seed, devices):
        seeded.append(devices)
        original_seed(seed, [])
    original_copy = torch.Tensor.copy_
    def cpu_copy(target, source, *args, **kwargs):
        if target is simulated:
            # Metadata-only device-selection proof: a CPU-only Torch build
            # cannot copy a simulated CUDA buffer. Real restoration is tested
            # above with actual CPU buffers for both persistence policies.
            assert source.shape == simulated.shape and source.device == simulated.device
            return target
        return original_copy(target, source, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', cpu_copy)
    monkeypatch.setattr(torch.random, 'fork_rng', cpu_fork)
    monkeypatch.setattr(lifecycle, '_seed_observation', cpu_seed)
    with active, active._observation(7, {'device': 'cuda:4'}):
        pass
    expected = [3, 4] if kind == 'training' else [2, 3, 4]
    assert forked == seeded == [expected]
    assert not torch.cuda.is_initialized()
