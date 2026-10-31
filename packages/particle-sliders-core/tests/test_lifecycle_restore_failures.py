"""Failure transactions use actual shared sessions; no native weights or GPU."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particle_sliders.lifecycle import CheckpointError, native_cleanup
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
                finally:
                    restore()
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
