"""Failure transactions use actual shared sessions; no native weights or GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particle_sliders.lifecycle import CheckpointError, native_cleanup, _restore_all
from test_image_lifecycle import cpu_only, equal, seal, session


def test_ema_transfer_failure_precedes_game_mutation_and_preserves_rng(monkeypatch):
    source, target = session(), session()
    source.train(stop_after=2)
    target.train(stop_after=1)
    packet = source.checkpoint()
    before, rng = target.checkpoint(), torch.get_rng_state().clone()
    bad_value = next(iter(packet['payload']['ema']['values'].values()))
    original_to = torch.Tensor.to
    calls = []
    original_load = target.game.load_state_dict
    def load(state):
        calls.append('game mutation')
        return original_load(state)
    def transfer(value, *args, **kwargs):
        if value is bad_value:
            torch.rand(3)  # A failing native transfer/hook must not leak RNG.
            raise RuntimeError('EMA native transfer failed')
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(target.game, 'load_state_dict', load)
    monkeypatch.setattr(torch.Tensor, 'to', transfer)
    with pytest.raises(CheckpointError, match='EMA'):
        target.restore(packet)
    assert calls == []
    equal(before, target.checkpoint())
    assert torch.equal(rng, torch.get_rng_state())


def test_one_shot_live_module_restore_failure_rolls_back_every_session_value():
    source, target = session(), session()
    source.train(stop_after=2)
    target.train(stop_after=1)
    before, rng = target.checkpoint(), torch.get_rng_state().clone()
    modes = [m.training for m in target.game.network.modules()]
    grads = [p.requires_grad for p in target.game.network.parameters()]
    marker = RuntimeError('late native load hook')
    calls = []
    def fail_after_copy(module, incompatible):
        calls.append('after-copy')
        if len(calls) == 1:
            module.eval().requires_grad_(False)
            module.gradient_checkpointing = True
            torch.rand(3)
            raise marker
    handle = target.game.network.register_load_state_dict_post_hook(fail_after_copy)
    try:
        with pytest.raises(RuntimeError) as caught:
            target.restore(source.checkpoint())
        assert caught.value is marker
    finally:
        handle.remove()
    assert len(calls) == 2  # Incoming live copy, then restoration of the old state.
    equal(before, target.checkpoint())
    assert torch.equal(rng, torch.get_rng_state())
    assert [m.training for m in target.game.network.modules()] == modes
    assert [p.requires_grad for p in target.game.network.parameters()] == grads
    assert not hasattr(target.game.network, 'gradient_checkpointing')
    assert not target.game._failed and not target.closed


def test_failed_rollback_preserves_primary_error_and_disables_game():
    source, target = session(), session()
    source.train(stop_after=2)
    marker = RuntimeError('persistent live restore failure')
    def fail(module, incompatible):
        raise marker
    handle = target.game.network.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            target.restore(source.checkpoint())
        assert caught.value is marker
        assert any('rollback' in note for note in getattr(marker, '__notes__', []))
        assert target.game._failed
        with pytest.raises(RuntimeError, match='failed update'):
            target.game.step()
    finally:
        handle.remove()
    target.restore(source.checkpoint())
    assert not target.game._failed


class RestoreFault(nn.Linear):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'gradient_checkpointing' and getattr(self, 'armed', False):
            raise self.failure


@pytest.mark.parametrize('active_primary', [False, True])
def test_native_cleanup_attempts_all_attributes_modes_and_gradient_flags(active_primary):
    fault = RestoreFault(2, 2)
    fault.gradient_checkpointing = False
    original_function = lambda *args: None
    fault._gradient_checkpointing_func = original_function
    root = nn.Sequential(fault, nn.Linear(2, 2))
    root[1].eval()
    root[1].weight.requires_grad_(False)
    modes = [m.training for m in root.modules()]
    grads = [p.requires_grad for p in root.parameters()]
    restore = native_cleanup(root)
    root.eval().requires_grad_(False)
    fault.gradient_checkpointing = True
    fault._gradient_checkpointing_func = lambda *args: 'changed'
    root[1].gradient_checkpointing = True
    root[1]._gradient_checkpointing_func = 'owned temporary'
    cleanup_error = RuntimeError('checkpoint attribute restoration failed')
    fault.failure, fault.armed = cleanup_error, True
    primary = RuntimeError('original native failure')
    try:
        if active_primary:
            with pytest.raises(RuntimeError) as caught:
                try:
                    raise primary
                except RuntimeError as operation:
                    _restore_all([restore], primary=operation)
                    raise
            assert caught.value is primary
            assert any('checkpoint attribute' in note for note in primary.__notes__)
        else:
            with pytest.raises(RuntimeError) as caught:
                restore()
            assert caught.value is cleanup_error
    finally:
        fault.armed = False
    assert fault.gradient_checkpointing is False
    assert fault._gradient_checkpointing_func is original_function
    assert not hasattr(root[1], 'gradient_checkpointing')
    assert not hasattr(root[1], '_gradient_checkpointing_func')
    assert [m.training for m in root.modules()] == modes
    assert [p.requires_grad for p in root.parameters()] == grads


def test_invalid_pg_packet_rejects_before_live_hooks_even_when_rollback_is_available():
    source, target = session(), session()
    source.train(stop_after=2)
    packet = source.checkpoint()
    packet['payload']['game']['row_cursors']['g'] = 999
    seal(packet)
    before, calls = target.checkpoint(), []
    handle = target.game.network.register_load_state_dict_post_hook(lambda *args: calls.append('live load'))
    try:
        with pytest.raises(CheckpointError):
            target.restore(packet)
    finally:
        handle.remove()
    assert calls == []
    equal(before, target.checkpoint())


def test_update_cleanup_error_does_not_mask_original_native_failure():
    target = session()
    primary = RuntimeError('native forward failure')
    cleanup = RuntimeError('native close failure')
    def fail(prepared):
        raise primary
    def close():
        raise cleanup
    target.game.backend.generated, target.binding.close = fail, close
    with pytest.raises(RuntimeError) as caught:
        target.step()
    assert caught.value is primary
    assert any('native close failure' in note for note in primary.__notes__)
    assert not target.closed  # Failed cleanup stays available for a later retry.
    target.binding.close = lambda: None
    target.close()
    assert target.closed


def test_prepared_ema_is_owned_and_detached_from_input_packet():
    source, target = session(), session()
    source.train(stop_after=2)
    packet = source.checkpoint()
    for value in packet['payload']['ema']['values'].values():
        if value.is_floating_point():
            value.requires_grad_(True)
    seal(packet)
    target.restore(packet)
    assert all(not value.requires_grad for value in target.ema.values.values())
    saved = deepcopy(target.ema.state_dict())
    with torch.no_grad():
        for value in packet['payload']['ema']['values'].values():
            value.zero_()
    equal(saved, target.ema.state_dict())


def test_observation_restore_error_keeps_primary_and_attempts_modes_and_buffers():
    target = session()
    target.game.network.register_buffer('temporary_counter', torch.tensor(3.), persistent=False)
    before = target.checkpoint()
    mode = target.game.network.training
    primary, cleanup = RuntimeError('render operation failed'), RuntimeError('observation load hook failed')
    def render(**kwargs):
        target.game.network.temporary_counter.add_(5)
        next(target.game.network.parameters()).add_(10)
        raise primary
    def after_load(module, incompatible):
        raise cleanup
    target.binding.render = render
    handle = target.game.network.register_load_state_dict_post_hook(after_load)
    try:
        with pytest.raises(RuntimeError) as caught:
            target.render()
        assert caught.value is primary
        assert any('observation load hook' in note for note in primary.__notes__)
    finally:
        handle.remove()
    equal(before, target.checkpoint())
    assert target.game.network.training == mode
    assert target.game.network.temporary_counter == 3


def inference_session():
    from particle_sliders.inference import InferenceBinding, InferenceSession
    from particle_sliders.lifecycle import Capabilities
    return InferenceSession(InferenceBinding(nn.Linear(2, 2), {}, lambda: None),
                            Capabilities('adapter_gan', False), {'selection': 'raw'}, None, lambda *args: None)


@pytest.mark.parametrize('factory', [session, inference_session])
def test_failed_close_retries_actual_native_restoration(factory):
    target = factory()
    network = target.game.network if hasattr(target, 'game') else target.binding.network
    original = native_cleanup(network)
    modes = [m.training for m in network.modules()]
    grads = [p.requires_grad for p in network.parameters()]
    network.eval().requires_grad_(False)
    calls = []
    marker = RuntimeError('native close fails once')
    def close():
        calls.append(True)
        if len(calls) == 1:
            raise marker
        original()
    target.binding.close = close
    with pytest.raises(RuntimeError) as caught:
        target.close()
    assert caught.value is marker and not target.closed
    target.close()
    target.close()
    assert target.closed and len(calls) == 2
    assert [m.training for m in network.modules()] == modes
    assert [p.requires_grad for p in network.parameters()] == grads


@pytest.mark.parametrize('factory', [session, inference_session])
@pytest.mark.parametrize('operation_fails', [False, True])
def test_context_cleanup_uses_only_its_own_operation_exception(factory, operation_fails):
    target = factory()
    unrelated, operation, cleanup = (RuntimeError(name) for name in ('unrelated caller', 'operation', 'cleanup'))
    def close():
        raise cleanup
    target.binding.close = close
    try:
        raise unrelated
    except RuntimeError:
        with pytest.raises(RuntimeError) as caught:
            with target:
                if operation_fails:
                    raise operation
        assert caught.value is (operation if operation_fails else cleanup)
        assert not getattr(unrelated, '__notes__', [])
        if operation_fails:
            assert any('cleanup' in note for note in operation.__notes__)
    assert not target.closed
    target.binding.close = lambda: None
    target.close()


def test_normal_observation_inside_unrelated_except_raises_cleanup_error():
    target = session()
    unrelated, cleanup = RuntimeError('caller'), RuntimeError('observation cleanup')
    def after_load(*args):
        raise cleanup
    hook = target.game.network.register_load_state_dict_post_hook(after_load)
    try:
        try:
            raise unrelated
        except RuntimeError:
            with pytest.raises(RuntimeError) as caught:
                with target._observation(0):
                    pass
            assert caught.value is cleanup
            assert not getattr(unrelated, '__notes__', [])
    finally:
        hook.remove()


def test_published_actual4abc_packet_refuses_before_hooks_without_state_or_rng_changes():
    root = Path(__file__).parent / 'fixtures/lifecycle'
    provenance = json.loads((root / 'provenance.json').read_text())
    data = root / 'core4abc-training.pt'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == provenance['fixture_sha256']
    packet = torch.load(data, map_location='cpu', weights_only=True)
    assert packet['manifest']['lifecycle_sha256'] == provenance['lifecycle_sha256']
    assert packet['payload']['game']['completed_steps'] == 2
    target = session()
    target.train(stop_after=1)
    target.game.network.eval()
    next(target.game.network.parameters()).requires_grad_(False)
    modules = [*target.game.network.modules(), *target.game.critic.modules()]
    parameters = [*target.game.network.parameters(), *target.game.critic.parameters()]
    modes, grads = [m.training for m in modules], [p.requires_grad for p in parameters]
    before, rng, calls = target.checkpoint(), torch.get_rng_state().clone(), []
    assert packet['manifest']['lifecycle_sha256'] != target.manifest['lifecycle_sha256']
    hooks = [module.register_load_state_dict_post_hook(lambda *args: calls.append(True))
             for module in (target.game.network, target.game.critic)]
    try:
        with pytest.raises(CheckpointError, match='source'):
            target.restore(packet)
    finally:
        for hook in hooks:
            hook.remove()
    equal(before, target.checkpoint())
    assert torch.equal(rng, torch.get_rng_state()) and calls == []
    assert [m.training for m in modules] == modes
    assert [p.requires_grad for p in parameters] == grads


def test_session_restore_validates_incoming_game_only_once(monkeypatch):
    source, target = session(), session()
    source.train(stop_after=2)
    calls = []
    original = target.game._validate_state_dict
    def validate(state):
        calls.append(True)
        return original(state)
    monkeypatch.setattr(target.game, '_validate_state_dict', validate)
    target.restore(source.checkpoint())
    assert calls == [True]


def test_shadow_optimizer_failure_cannot_publish_incoming_thermal_state(monkeypatch):
    from particlegan.bispf import _LAST_THERMAL_SCALE
    source, target = session(), session()
    source.train(stop_after=2)
    before, rng = target.checkpoint(), torch.get_rng_state().clone()
    _LAST_THERMAL_SCALE[target.game.device] = .713
    original = type(target.game.opt_d).load_state_dict
    marker = RuntimeError('shadow load fails after publishing')
    def fail_shadow(optimizer, state):
        original(optimizer, state)
        if optimizer is not target.game.opt_d:
            _LAST_THERMAL_SCALE[target.game.device] = .001
            raise marker
    monkeypatch.setattr(type(target.game.opt_d), 'load_state_dict', fail_shadow)
    with pytest.raises(RuntimeError) as caught:
        target.restore(source.checkpoint())
    assert caught.value is marker
    assert _LAST_THERMAL_SCALE[target.game.device] == .713
    equal(before, target.checkpoint())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('rollback_fails', [False, True])
def test_live_failure_restores_thermal_and_diagnostics_even_if_rollback_fails(rollback_fails):
    from particlegan.bispf import _LAST_THERMAL_SCALE
    source, target = session(), session()
    source.train(stop_after=2)
    target.train(stop_after=1)
    game, calls = target.game, []
    stats, cooler = deepcopy(game.penalty.last_stats), deepcopy(game.penalty.regularizer.cooler.state_dict())
    _LAST_THERMAL_SCALE[game.device] = .731
    marker = RuntimeError('live failure mutates runtime')
    def fail(module, incompatible):
        calls.append(True)
        if len(calls) == 1 or rollback_fails:
            _LAST_THERMAL_SCALE[game.device] = .002
            game.penalty.last_stats = {'damaged': True}
            game.penalty.collect_stats = False
            game.penalty.regularizer.cooler.ema_residual = 999.
            game.opt_d.record._in_penalty = True
            raise marker
    hook = game.network.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            target.restore(source.checkpoint())
        assert caught.value is marker and len(calls) == 2
    finally:
        hook.remove()
    assert _LAST_THERMAL_SCALE[game.device] == .731
    equal(stats, game.penalty.last_stats)
    equal(cooler, game.penalty.regularizer.cooler.state_dict())
    assert game.penalty.collect_stats and not game.opt_d.record._in_penalty
    assert game._failed is rollback_fails
