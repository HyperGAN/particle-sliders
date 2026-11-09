"""Faults after live mutation must restore exact state or block further use."""
from copy import deepcopy
from contextlib import contextmanager

import pytest
import torch

from particle_sliders.lifecycle import LifecycleError, UnsupportedOperation
from test_image_lifecycle import cpu_only, equal, session


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt, SystemExit])
def test_direct_game_interruption_after_critic_commit_requires_full_restore(monkeypatch, interrupt):
    active = session()
    game = active.game
    previous = game.state_dict()
    original = game.opt_d.step
    marker = interrupt('interrupted after critic commit')

    def stop_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise marker

    monkeypatch.setattr(game.opt_d, 'step', stop_after_commit)
    with pytest.raises(interrupt) as caught:
        game.step()
    assert caught.value is marker and game.completed_steps == 0
    assert any(not torch.equal(value, previous['critic'][name])
               for name, value in game.critic.state_dict().items())
    for operation in (game.step, game.state_dict, game.export_state_dict):
        with pytest.raises(RuntimeError, match='failed update'):
            operation()
    game.load_state_dict(previous)
    equal(previous, game.state_dict())
    active.close()


@pytest.mark.parametrize('direct', [False, True])
def test_late_export_load_failure_restores_training_weights_flags_and_rng(direct):
    source, target = session(), session()
    source.train(stop_after=2)
    target.train(stop_after=1)
    previous, rng = target.checkpoint(), torch.get_rng_state().clone()
    modules = list(target.game.network.modules())
    parameters = list(target.game.network.parameters())
    modes, flags = [m.training for m in modules], [p.requires_grad for p in parameters]
    calls = []
    marker = RuntimeError('export hook failed after live copy')

    def fail_once(module, incompatible):
        calls.append(True)
        if len(calls) == 1:
            module.eval().requires_grad_(False)
            module.gradient_checkpointing = True
            torch.rand(4)
            raise marker

    handle = target.game.network.register_load_state_dict_post_hook(fail_once)
    try:
        with pytest.raises(RuntimeError) as caught:
            if direct:
                target.game.load_export_state_dict(source.game.export_state_dict())
            else:
                target.load_inference(source.export())
        assert caught.value is marker
    finally:
        handle.remove()
    assert len(calls) == 2
    equal(previous, target.checkpoint())
    assert torch.equal(rng, torch.get_rng_state())
    assert [m.training for m in modules] == modes
    assert [p.requires_grad for p in parameters] == flags
    assert not hasattr(target.game.network, 'gradient_checkpointing')
    assert not target.inference_only
    target.step()
    assert target.game.completed_steps == 2
    source.close()
    target.close()


def test_failed_inference_load_rollback_quarantines_session():
    source, target = session(), session()
    source.train(stop_after=1)
    marker = RuntimeError('persistent export load hook failure')

    def fail(module, incompatible):
        with torch.no_grad():
            next(module.parameters()).add_(10)
        raise marker

    handle = target.game.network.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            target.load_inference(source.export())
        assert caught.value is marker
        assert any('restoration failed' in note for note in marker.__notes__)
    finally:
        handle.remove()
    for operation in (target.step, target.checkpoint, target.export, target.render,
                      lambda: target.restore(source.checkpoint())):
        with pytest.raises(LifecycleError, match='requires cleanup'):
            operation()
    source.close()
    target.close()


def test_admission_rollback_failure_quarantines_even_after_primary_rejection():
    target = session()
    modes = [m.training for m in target.game.network.modules()]

    def refuse(request):
        target.game.network.eval()
        with torch.no_grad():
            next(target.game.network.parameters()).add_(2)
        return False

    def fail(module, incompatible):
        raise RuntimeError('admission rollback hook failure')

    target.binding.before_update = refuse
    handle = target.game.network.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(UnsupportedOperation) as caught:
            target.step()
        assert any('admission rollback hook failure' in note for note in caught.value.__notes__)
    finally:
        handle.remove()
    assert [m.training for m in target.game.network.modules()] == modes
    for operation in (target.step, target.checkpoint, target.export, target.render):
        with pytest.raises(LifecycleError, match='requires cleanup'):
            operation()
    target.close()


def test_direct_full_restore_rolls_back_late_failure_without_session_help():
    source, target = session(), session()
    source.train(stop_after=2)
    target.train(stop_after=1)
    before = target.game.state_dict()
    calls = []

    def fail_once(module, incompatible):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('full state hook failure')

    handle = target.game.network.register_load_state_dict_post_hook(fail_once)
    try:
        with pytest.raises(RuntimeError, match='full state hook failure'):
            target.game.load_state_dict(source.game.state_dict())
    finally:
        handle.remove()
    equal(before, target.game.state_dict())
    source.close()
    target.close()


def test_engine_signature_is_an_independent_snapshot():
    target = session()
    original = deepcopy(target.game.signature)
    snapshot = target.game.signature
    snapshot['runtime']['torch'] = 'cannot-rebind'
    assert target.game.signature == original
    assert target.manifest['game_signature'] == original
    target.close()


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('valid', [False, True])
def test_export_load_preflight_hooks_preserve_rng_on_success_and_rejection(direct, valid):
    from test_image_lifecycle import seal
    source, target = session(), session()
    packet = source.export()
    if not valid:
        packet['payload']['adapter']['network'] = {}
        seal(packet)
    def consume_rng(module, state, prefix, metadata):
        torch.rand(3)
    handle = target.game.network.register_state_dict_post_hook(consume_rng)
    before = torch.get_rng_state().clone()
    try:
        def load():
            if direct:
                target.game.load_export_state_dict(packet['payload']['adapter'])
            else:
                target.load_inference(packet)
        if valid:
            load()
        else:
            with pytest.raises(ValueError if direct else RuntimeError):
                load()
    finally:
        handle.remove()
    assert torch.equal(before, torch.get_rng_state())
    if not valid:
        target.checkpoint()
    source.close()
    target.close()


@pytest.mark.parametrize('operation', ['admission', 'full', 'export'])
def test_transactions_include_buffer_only_cuda_devices_without_cuda_execution(operation, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    source, target = session(), session()
    full, exported = source.game.state_dict(), source.game.export_state_dict()
    with FakeTensorMode():
        simulated = torch.empty(2, device='cuda:3')
    target.game.network.register_buffer('native_state', simulated, persistent=False)
    original_fork, forked = torch.random.fork_rng, []
    @contextmanager
    def cpu_fork(*, devices):
        forked.append(devices)
        with original_fork(devices=[]):
            yield
    monkeypatch.setattr(torch.random, 'fork_rng', cpu_fork)
    if operation == 'admission':
        target.binding.before_update = lambda request: False
        with pytest.raises(UnsupportedOperation):
            target.step()
    elif operation == 'full':
        target.game.load_state_dict(full)
    else:
        target.game.load_export_state_dict(exported)
    assert forked and all(devices == [3] for devices in forked)
    assert not torch.cuda.is_initialized()
    source.close()
    target.close()


@pytest.mark.parametrize('disabled', ['failed', 'inference'])
@pytest.mark.parametrize('loader', ['full', 'export'])
@pytest.mark.parametrize('late_failure', [False, True])
def test_disabled_games_restore_or_remain_poisoned_until_full_recovery(disabled, loader, late_failure, monkeypatch):
    source, target = session(), session()
    source.train(stop_after=1)
    expected = source.game.state_dict()
    if disabled == 'failed':
        original = target.game.opt_g.step
        def fail_generator(*args, **kwargs):
            raise KeyboardInterrupt('interrupted after critic commit')
        monkeypatch.setattr(target.game.opt_g, 'step', fail_generator)
        with pytest.raises(KeyboardInterrupt):
            target.game.step()
        monkeypatch.setattr(target.game.opt_g, 'step', original)
    else:
        target.game.load_export_state_dict(source.game.export_state_dict())
    modes = [m.training for m in target.game.network.modules()]
    flags = [p.requires_grad for p in target.game.network.parameters()]
    def late_hook(module, incompatible):
        torch.rand(3)
        module.eval().requires_grad_(False)
        raise KeyboardInterrupt('late load interruption')
    handle = target.game.network.register_load_state_dict_post_hook(late_hook) if late_failure else None
    before_rng = torch.get_rng_state().clone()
    try:
        def load():
            if loader == 'full':
                target.game.load_state_dict(expected)
            else:
                target.game.load_export_state_dict(source.game.export_state_dict())
        if late_failure:
            with pytest.raises(KeyboardInterrupt) as caught:
                load()
            assert getattr(caught.value, '_particle_sliders_restoration_failed', False)
        else:
            load()
    finally:
        if handle is not None:
            handle.remove()
    assert torch.equal(before_rng, torch.get_rng_state())
    assert [m.training for m in target.game.network.modules()] == modes
    assert [p.requires_grad for p in target.game.network.parameters()] == flags
    if late_failure:
        for operation in (target.game.step, target.game.state_dict, target.game.export_state_dict):
            with pytest.raises(RuntimeError):
                operation()
    elif loader == 'export':
        with pytest.raises(RuntimeError, match='inference export'):
            target.game.state_dict()
    target.game.load_state_dict(expected)
    equal(expected, target.game.state_dict())
    source.close()
    target.close()
