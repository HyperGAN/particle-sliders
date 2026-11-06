"""CPU-only fault injection for old full-state loaders; no averaged weights."""
from copy import deepcopy
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.textsliders import lm_gan_state
from particle_sliders.textsliders.gan_v2 import state as gan_state


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def engine(seed):
    torch.manual_seed(seed)
    network, critic = nn.Linear(2, 2), nn.Linear(2, 1)
    g, d = torch.optim.Adam(network.parameters()), torch.optim.Adam(critic.parameters())
    for module, opt in ((network, g), (critic, d)):
        module(torch.ones(1, 2)).sum().backward()
        opt.step()
    return SimpleNamespace(network=network, critic=critic, g_optimizer=g, d_optimizer=d, completed=1)


def snapshot(value):
    return deepcopy(dict(network=value.network.state_dict(), critic=value.critic.state_dict(),
        g=value.g_optimizer.state_dict(), d=value.d_optimizer.state_dict(), completed=value.completed))


def lm_args(value):
    return dict(run_signature={'test': 1}, modules={'g': value.network, 'd': value.critic},
                optimizers={'g': value.g_optimizer, 'd': value.d_optimizer})


def test_late_optimizer_failure_restores_modules_moments_flags_buffers_and_rng(tmp_path, monkeypatch):
    source, target = engine(1), engine(2)
    for value in (source, target):
        value.network.register_buffer('ephemeral', torch.tensor(3.), persistent=False)
    path = tmp_path / 'lm.pt'
    lm_gan_state.save(path, **lm_args(source), history=[{'step': 1}])
    before = snapshot(target)
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    flags = [(p, p.requires_grad, p.grad, p.grad.clone()) for p in target.network.parameters()]
    original, calls = target.d_optimizer.load_state_dict, []
    def fail_once(packet):
        original(packet)
        calls.append(True)
        if len(calls) == 1:
            target.network.eval().requires_grad_(False)
            target.network.gradient_checkpointing = True
            target.network.ephemeral.add_(5)
            torch.rand(3)
            random.random()
            raise RuntimeError('late live optimizer hook')
    monkeypatch.setattr(target.d_optimizer, 'load_state_dict', fail_once)
    with pytest.raises(RuntimeError, match='late live optimizer hook'):
        lm_gan_state.restore(path, **lm_args(target))
    assert len(calls) == 2
    equal(snapshot(target), before)
    assert target.network.training and target.network.ephemeral == 3
    assert not hasattr(target.network, 'gradient_checkpointing')
    for parameter, flag, gradient, value in flags:
        assert parameter.requires_grad == flag and parameter.grad is gradient
        assert torch.equal(parameter.grad, value)
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng


def test_cuda_topology_rejection_precedes_every_live_load(tmp_path, monkeypatch):
    source, target = engine(1), engine(2)
    path = tmp_path / 'lm.pt'
    lm_gan_state.save(path, **lm_args(source), history=[{'step': 1}])
    packet = torch.load(path, weights_only=True)
    packet['cuda_rng'] = [torch.zeros(8, dtype=torch.uint8)]
    torch.save(packet, path)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(target.network, 'load_state_dict', lambda *args, **kwargs: pytest.fail('live load before CUDA admission'))
    with pytest.raises(ValueError, match='same number of visible CUDA devices'):
        lm_gan_state.restore(path, **lm_args(target))


def test_failed_rollback_preserves_primary_attempts_other_restorers_and_quarantines(tmp_path, monkeypatch):
    source, target = engine(1), engine(2)
    path = tmp_path / 'lm.pt'
    lm_gan_state.save(path, **lm_args(source), history=[{'step': 1}])
    primary = RuntimeError('permanent live load failure')
    rng = torch.get_rng_state().clone()
    def fail(module, incompatible):
        torch.rand(4)
        raise primary
    handle = target.network.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            lm_gan_state.restore(path, **lm_args(target))
        assert caught.value is primary
        assert any('rollback failed' in note for note in primary.__notes__)
        assert torch.equal(torch.get_rng_state(), rng)
        with pytest.raises(RuntimeError, match='failed training restore rollback'):
            target.g_optimizer.step()
        with pytest.raises(RuntimeError, match='failed training restore rollback'):
            lm_gan_state.save(tmp_path / 'bad.pt', **lm_args(target), history=[{'step': 1}])
    finally:
        handle.remove()
    assert lm_gan_state.restore(path, **lm_args(target)) == [{'step': 1}]
    target.g_optimizer.step()  # A complete restore recovers the existing optimizer.


def test_gan_v2_late_sampler_failure_rolls_back_counter_and_all_components(tmp_path):
    source, target = engine(1), engine(2)
    sa, sb = gan_state.RowSampler(3, 2, 1), gan_state.RowSampler(3, 2, 2)
    sa.next()
    sb.next()
    path = tmp_path / 'gan.pt'
    signature = {'test': 1}
    gan_state.save(path, source, sa, None, signature, [{'step': 1}])
    packet = torch.load(path, weights_only=True)
    packet['sampler']['cursor'] = 99
    torch.save(packet, path)
    before, sampler_before = snapshot(target), deepcopy(sb.state_dict())
    with pytest.raises(ValueError, match='sampler cursor'):
        gan_state.restore(path, target, sb, None, signature)
    equal(snapshot(target), before)
    equal(sb.state_dict(), sampler_before)


def test_gan_v2_raw_restore_preserves_exact_optimizer_and_sampler_continuation(tmp_path):
    source, target = engine(1), engine(2)
    sa, sb = gan_state.RowSampler(3, 2, 1), gan_state.RowSampler(3, 2, 2)
    sa.next()
    path = tmp_path / 'gan.pt'
    signature, history = {'test': 1}, [{'step': 1}]
    gan_state.save(path, source, sa, None, signature, history)
    assert gan_state.restore(path, target, sb, None, signature) == history
    equal(snapshot(source), snapshot(target))
    assert sa.next() == sb.next()
    for value in (source, target):
        value.g_optimizer.zero_grad(set_to_none=True)
        value.network(torch.ones(1, 2)).square().sum().backward()
        value.g_optimizer.step()
    equal(snapshot(source), snapshot(target))


def test_legacy_initialization_rolls_back_after_partial_live_load(tmp_path, monkeypatch):
    source, target = engine(1), engine(2)
    path = tmp_path / 'legacy.pt'
    lm_gan_state.save(path, run_signature={'sources': {}},
        modules={'lora': source.network, 'critic': source.critic},
        optimizers={'lora': source.g_optimizer, 'critic': source.d_optimizer}, history=[{'step': 1}])
    before = snapshot(target)
    original, calls = target.d_optimizer.load_state_dict, []
    def fail_once(packet):
        original(packet)
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('legacy import late failure')
    monkeypatch.setattr(target.d_optimizer, 'load_state_dict', fail_once)
    with pytest.raises(RuntimeError, match='legacy import late failure'):
        gan_state.initialize_from_legacy(path, target, reuse_critic=True)
    equal(snapshot(target), before)


@pytest.mark.parametrize('loader', ['lm', 'gan_v2'])
def test_failed_restore_keeps_update_latch_until_successful_full_restore(tmp_path, monkeypatch, loader):
    from particle_sliders._training import guard_training_update, require_healthy_update
    source, target = engine(1), engine(2)
    path = tmp_path / 'state.pt'
    if loader == 'lm':
        lm_gan_state.save(path, **lm_args(source), history=[{'step': 1}])
        def restore():
            return lm_gan_state.restore(path, **lm_args(target))
        owners = target.g_optimizer, target.d_optimizer
    else:
        sa, sb = gan_state.RowSampler(3, 2, 1), gan_state.RowSampler(3, 2, 2)
        gan_state.save(path, source, sa, None, {'test': 1}, [{'step': 1}])
        def restore():
            return gan_state.restore(path, target, sb, None, {'test': 1})
        owners = target, target.g_optimizer, target.d_optimizer
    with pytest.raises(RuntimeError, match='interrupted update'):
        with guard_training_update(*owners):
            raise RuntimeError('interrupted update')
    original, calls = target.d_optimizer.load_state_dict, []
    def fail_once(packet):
        original(packet)
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('recovery attempt failed')
    with monkeypatch.context() as patch:
        patch.setattr(target.d_optimizer, 'load_state_dict', fail_once)
        with pytest.raises(RuntimeError, match='recovery attempt failed'):
            restore()
    with pytest.raises(RuntimeError, match='failed training update'):
        require_healthy_update(*owners)
    restore()
    require_healthy_update(*owners)
