"""Checkpoint update laws, transactional preflight, and export provenance."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from test_conditional_pg09 import Backend, Critic, equal, fixture


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
@pytest.mark.parametrize('optimizer', ['opt_g', 'opt_d'])
@pytest.mark.parametrize('moment', ['exp_avg_sq', 'max_exp_avg_sq'])
def test_negative_second_moments_rejected_before_mutation(preset, optimizer, moment):
    source = fixture(preset=preset)
    source.step()
    packet = source.state_dict()
    key = packet[optimizer]['param_groups'][0]['params'][0]
    parameter = getattr(source, optimizer).param_groups[0]['params'][0]
    packet[optimizer]['state'][key][moment] = torch.full_like(parameter, -1.)
    receiver = fixture(preset=preset)
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='second moment is negative'):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('optimizer', ['opt_g', 'opt_d'])
@pytest.mark.parametrize('step', [-1, torch.tensor(-1.), torch.tensor(.5),
                                  torch.tensor(float('nan')), torch.tensor(float('inf')),
                                  torch.ones(2), torch.tensor(True)])
def test_invalid_optimizer_clocks_rejected_before_mutation(preset, optimizer, step):
    source = fixture(preset=preset)
    source.step()
    packet = source.state_dict()
    key = packet[optimizer]['param_groups'][0]['params'][0]
    packet[optimizer]['state'][key]['step'] = step
    receiver = fixture(preset=preset)
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='optimizer|nonfinite'):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


@pytest.mark.parametrize('optimizer', ['opt_g', 'opt_d'])
@pytest.mark.parametrize('change', ['missing_exp_avg', 'missing_exp_avg_sq', 'missing_step',
                                   'extra_field', 'unexpected_amsgrad_history'])
def test_original_adam_history_requires_complete_exact_fields(optimizer, change):
    source = fixture(preset='bispf')
    source.step()
    packet = source.state_dict()
    key = packet[optimizer]['param_groups'][0]['params'][0]
    slots = packet[optimizer]['state'][key]
    if change.startswith('missing_'):
        del slots[change.removeprefix('missing_')]
    elif change == 'extra_field':
        slots['unexpected'] = 1
    else:
        slots['max_exp_avg_sq'] = torch.zeros_like(slots['exp_avg_sq'])
    receiver = fixture(preset='bispf')
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='Adam history fields differ'):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_initial_empty_optimizer_history_remains_resumable(preset):
    source = fixture(preset=preset)
    packet = source.state_dict()
    assert packet['opt_g']['state'] == packet['opt_d']['state'] == {}
    receiver = fixture(preset=preset)
    receiver.load_state_dict(packet)
    assert receiver.step() == source.step()
    equal(receiver.state_dict(), source.state_dict())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('optimizer', ['opt_g', 'opt_d'])
@pytest.mark.parametrize('change', ['absent', 'empty'])
def test_started_optimizer_history_cannot_be_dropped_or_emptied(preset, optimizer, change):
    source = fixture(preset=preset)
    source.step()
    packet = source.state_dict()
    key = packet[optimizer]['param_groups'][0]['params'][0]
    assert packet[optimizer]['state'][key]
    if change == 'absent':
        del packet[optimizer]['state'][key]
    else:
        packet[optimizer]['state'][key] = {}
    receiver = fixture(preset=preset)
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='history|histories'):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_started_generator_cannot_reset_all_history_even_with_changed_schema(preset):
    source = fixture(preset=preset)
    source.step()
    packet = source.state_dict()
    packet['opt_g']['state'] = {}
    packet['optimizer_history_schema']['g'] = {
        key: None for group in packet['opt_g']['param_groups'] for key in group['params']}
    receiver = fixture(preset=preset)
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='started generator'):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


def test_optimizer_history_presence_record_is_mandatory():
    source = fixture()
    packet = source.state_dict()
    del packet['optimizer_history_schema']
    receiver = fixture()
    before = receiver.state_dict()
    with pytest.raises(ValueError, match='history presence'):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())


class SparseCritic(Critic):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.ones(2))


def sparse_critic_fixture(preset):
    base = fixture(preset=preset)
    with torch.random.fork_rng():
        torch.manual_seed(123)
        critic = SparseCritic()
    return ConditionalPG09(base.backend, base.rows, critic, conditions=base.conditions,
                          particles=base.network.particles, preset=preset, batch_size=4, total_steps=5)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_never_used_critic_parameter_preserves_absent_history_and_exact_resume(preset):
    source = sparse_critic_fixture(preset)
    source.step()
    packet = source.state_dict()
    unused = next(key for saved, actual in zip(packet['opt_d']['param_groups'], source.opt_d.param_groups)
                  for key, parameter in zip(saved['params'], actual['params']) if parameter is source.critic.unused)
    assert unused not in packet['opt_d']['state']
    assert packet['optimizer_history_schema']['d'][unused] is None
    expected = source.step()
    receiver = sparse_critic_fixture(preset)
    receiver.load_state_dict(packet)
    assert receiver.step() == expected
    equal(receiver.state_dict(), source.state_dict())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('optimizer,module', [('opt_g', 'network'), ('opt_d', 'critic')])
def test_nonfinite_optimizer_result_marks_update_failed_before_completion(preset, optimizer, module, monkeypatch):
    game = fixture(preset=preset)
    update = getattr(game, optimizer)
    original = update.step

    def corrupt_after_update(*args, **kwargs):
        result = original(*args, **kwargs)
        with torch.no_grad():
            next(getattr(game, module).parameters()).fill_(float('nan'))
        return result

    monkeypatch.setattr(update, 'step', corrupt_after_update)
    with pytest.raises(FloatingPointError, match='nonfinite parameters after optimizer'):
        game.step()
    assert game._failed and game.completed_steps == 0
    assert not any(int(counts.sum()) for counts in game.exposures.values())
    assert all(parameter.requires_grad for parameter in game.critic.parameters())
    with pytest.raises(RuntimeError, match='failed update'):
        game.state_dict()
    with pytest.raises(RuntimeError, match='failed update'):
        game.export_state_dict()
    with pytest.raises(RuntimeError, match='failed update'):
        game.step()


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
