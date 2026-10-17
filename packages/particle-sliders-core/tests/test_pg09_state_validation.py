"""Checkpoint update laws, transactional preflight, and export provenance."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from test_conditional_pg09 import Backend, equal, fixture


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_caller_owned_training_precision_matches_actual_float32_execution():
    game = fixture(overrides={'training_precision': 'float32'})
    assert game.recipe.training_precision == 'float32'
    assert game.recipe.optimizer_svd_backend == 'batched'
    assert game._signature['recipe']['training_precision'] == 'float32'
    assert all(parameter.dtype == torch.float32 for parameter in game.network.parameters())
    with pytest.raises(ValueError, match='contradicts actual training_precision'):
        fixture(overrides={'training_precision': 'bf16_channels_last'})


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('optimizer', ['opt_g', 'opt_d'])
@pytest.mark.parametrize('setting,value', [
    ('betas', (1., .9)),
    ('betas', (.1, .9)),
    ('eps', -1.),
    ('eps', float('nan')),
    ('lr', float('nan')),
    ('lr', float('inf')),
    ('lr', -1.),
    ('lr', 0.),
    ('lr', True),
])
def test_restore_rejects_changed_update_law_before_mutation(preset, optimizer, setting, value):
    source = fixture(preset=preset)
    source.step()
    packet = source.state_dict()
    packet[optimizer]['param_groups'][0][setting] = value
    receiver = fixture(preset=preset)
    before = receiver.state_dict()
    with pytest.raises(ValueError):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_valid_optimizer_restore_preserves_snapshot_and_exact_next_update(preset):
    source = fixture(preset=preset)
    source.step()
    packet = source.state_dict()
    snapshot = deepcopy(packet)
    expected = source.step()
    expected_state = source.state_dict()
    equal(packet, snapshot)
    receiver = fixture(preset=preset)
    receiver.load_state_dict(packet)
    assert receiver.step() == expected
    equal(receiver.state_dict(), expected_state)
    equal(packet, snapshot)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('corruption', ['sampler', 'spread', 'generator_optimizer'])
def test_rejected_preflight_does_not_publish_candidate_critic_thermal_state(preset, corruption):
    # PG's public critic restore publishes this device-level state even when
    # performed on a deep-copied optimizer. Rejection must precede that restore.
    from particlegan.bispf import _LAST_THERMAL_SCALE

    game = fixture(preset=preset)
    game.step()
    packet = game.state_dict()
    record = packet['opt_d']['regularizer']['record']
    controller = record if preset == 'bispf' else record['bispf']
    controller['thermal_scale'] = .123
    if corruption == 'sampler':
        packet['streams']['g_rows'] = torch.zeros(2, dtype=torch.uint8)
    elif corruption == 'spread':
        packet['spread'] = {'invalid': 'state'}
    else:
        packet['opt_g']['regularizer'] = {'invalid': 'state'}
    before = game.state_dict()
    published_before = dict(_LAST_THERMAL_SCALE)
    try:
        _LAST_THERMAL_SCALE[game.device] = .789
        with pytest.raises((ValueError, RuntimeError)):
            game.load_state_dict(packet)
        assert _LAST_THERMAL_SCALE[game.device] == .789
        equal(before, game.state_dict())
    finally:
        _LAST_THERMAL_SCALE.clear()
        _LAST_THERMAL_SCALE.update(published_before)


def test_failed_partial_update_cannot_export_previous_step_provenance(monkeypatch):
    game = fixture()
    initial_critic = deepcopy(game.critic.state_dict())

    def fail_after_critic_commit(*args, **kwargs):
        raise RuntimeError('forced generator optimizer failure')

    monkeypatch.setattr(game.opt_g, 'step', fail_after_critic_commit)
    with pytest.raises(RuntimeError, match='forced generator'):
        game.step()
    assert game.completed_steps == 0
    assert any(not torch.equal(value, initial_critic[name])
               for name, value in game.critic.state_dict().items())
    with pytest.raises(RuntimeError, match='failed update'):
        game.export_state_dict()
    qualified = fixture().export_state_dict()
    game.load_export_state_dict(qualified)
    equal(qualified, game.export_state_dict())


def test_inference_export_reload_preserves_completed_step_provenance():
    source = fixture()
    source.step()
    source.step()
    packet = source.export_state_dict()
    receiver = fixture()
    receiver.load_export_state_dict(packet)
    assert receiver.completed_steps == source.completed_steps == 2
    equal(packet, receiver.export_state_dict())
    assert torch.equal(source.evaluate(0, seed=31), receiver.evaluate(0, seed=31))
    with pytest.raises(RuntimeError, match='inference export'):
        receiver.state_dict()


@pytest.mark.parametrize('steps', [-1, 6, True, 1.5])
def test_export_invalid_step_rejected_before_weights_are_loaded(steps):
    source = fixture()
    source.step()
    packet = source.export_state_dict()
    packet['completed_steps'] = steps
    receiver = fixture()
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='export step'):
        receiver.load_export_state_dict(packet)
    equal(before, receiver.state_dict())


class BatchNormBackend(Backend):
    def __init__(self):
        super().__init__()
        self.network.normalization = nn.BatchNorm2d(2, affine=False)

    def generated(self, prepared):
        assert all(not module.training for module in self.network.modules())
        return self.network.normalization(super().generated(prepared))


@pytest.mark.parametrize('fail', [False, True])
def test_evaluate_preserves_batchnorm_buffers_rng_and_mixed_training_modes(fail, monkeypatch):
    base = fixture()
    backend = BatchNormBackend()
    game = ConditionalPG09(backend, base.rows, base.critic, conditions=base.conditions,
                          particles=backend.network.particles, batch_size=4, total_steps=5)
    game.network.train()
    game.network.routed.eval()
    modes = [(module, module.training) for module in game.network.modules()]
    before = game.state_dict()
    ambient_rng = torch.get_rng_state().clone()
    if fail:
        original = game.backend.generated

        def fail_after_forward(prepared):
            original(prepared)
            torch.randn(7)
            raise RuntimeError('forced evaluation failure')

        monkeypatch.setattr(game.backend, 'generated', fail_after_forward)
        with pytest.raises(RuntimeError, match='forced evaluation'):
            game.evaluate(0, seed=31)
    else:
        first = game.evaluate(0, seed=31)
        assert torch.equal(first, game.evaluate(0, seed=31))
    assert all(module.training == training for module, training in modes)
    equal(before, game.state_dict())
    assert torch.equal(ambient_rng, torch.get_rng_state())
