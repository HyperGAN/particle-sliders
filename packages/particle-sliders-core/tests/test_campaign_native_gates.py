"""Native geometry, teacher parity and rejected campaign construction gates."""
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_moss_campaign import CampaignFieldCritic, build_campaign_run
from particle_sliders.pg09_remaining_campaigns import teacher_parity
from test_conditional_pg09 import Backend, Prepared, fixture


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def parity_run(*, neutral=2., edit=0., target_error=0., base_error=0.):
    base = torch.full((1, 2, 2, 2), neutral)
    target = base + edit
    cached = Prepared({}, target, base)
    actual = Prepared({}, target + target_error, base + base_error)
    return SimpleNamespace(rows=[cached], backend=SimpleNamespace(prepare=lambda row: actual))


@pytest.mark.parametrize('neutral', [0., 2.])
def test_preservation_teacher_parity_allows_float_roundoff(neutral):
    run = parity_run(neutral=neutral, target_error=2.5e-7, base_error=2.5e-7)
    result, = teacher_parity(run, [0])
    assert result['normalization'] == 'frozen_neutral_velocity_rms'
    assert result['target_relative_rms'] < result['relative_rms_limit']
    assert result['target_rms_error'] <= result['rms_tolerance']


@pytest.mark.parametrize('field', ['target_error', 'base_error'])
@pytest.mark.parametrize('error', [.1, float('nan'), float('inf')])
def test_preservation_teacher_parity_rejects_corrupt_or_nonfinite_response(field, error):
    with pytest.raises(ValueError, match='teacher/cache parity'):
        teacher_parity(parity_run(**{field: error}), [0])


def test_zero_velocity_preservation_uses_bounded_absolute_tolerance():
    with pytest.raises(ValueError, match='teacher/cache parity'):
        teacher_parity(parity_run(neutral=0., target_error=2e-5), [0])


def test_edited_teacher_parity_retains_edit_scale_even_with_large_frozen_velocity():
    run = parity_run(neutral=1000., edit=.1, target_error=.01)
    with pytest.raises(ValueError, match='teacher/cache parity') as rejected:
        teacher_parity(run, [0], absolute_rms_limit=1.)
    result, = rejected.value.diagnostics
    assert result['normalization'] == 'frozen_edit_rms'
    assert result['absolute_rms_limit'] == 0.
    assert result['target_relative_rms'] > result['relative_rms_limit']


@pytest.mark.parametrize('options', [
    {'relative_rms_limit': 0.}, {'relative_rms_limit': float('nan')},
    {'relative_rms_limit': float('inf')}, {'relative_rms_limit': True},
    {'absolute_rms_limit': -1.}, {'absolute_rms_limit': float('nan')},
])
def test_teacher_parity_rejects_invalid_tolerances(options):
    with pytest.raises(ValueError, match='RMS limits'):
        teacher_parity(parity_run(), [0], **options)


@pytest.mark.parametrize('temporal', [False, True])
def test_field_critic_keeps_native_field_condition_gradients_and_raw_logits(temporal):
    torch.manual_seed(71)
    critic = CampaignFieldCritic(2, 3, width=8)
    response = torch.randn(1, 2, 3, 3).expand(2, -1, -1, -1).clone()
    if temporal:
        response = response.unsqueeze(2)
    response.requires_grad_()
    condition = torch.tensor([[0., 0., 0.], [2., 0., 0.]], requires_grad=True)
    with torch.no_grad():
        critic.score[-1].bias.fill_(100.)
    scores = critic(response, condition)
    assert scores.shape == (2,) and (scores > 90).all()
    assert scores[0] != scores[1]
    scores.sum().backward()
    assert response.grad.shape == response.shape
    assert not torch.equal(response.grad[0], response.grad[1])
    assert torch.count_nonzero(condition.grad)
    if temporal:
        torch.testing.assert_close(scores, critic(response.detach().squeeze(2), condition.detach()), rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(2, 2, 2, 3, 3), (2, 2, 3), (2, 2, 1, 1, 3, 3)])
def test_field_critic_rejects_non_singleton_temporal_or_wrong_rank(shape):
    with pytest.raises(ValueError, match='singleton temporal'):
        CampaignFieldCritic(2, 3, width=8)(torch.randn(shape), torch.zeros(2, 3))


class SingletonTemporalBackend(Backend):
    def generated(self, prepared):
        output = self.network(prepared.row['input']).reshape(1, 2, 1, 1, 1)
        return prepared.neutral + output.expand_as(prepared.target) * prepared.row['sign']


def test_native_singleton_temporal_field_executes_shared_pg09_update():
    original = fixture()
    backend = SingletonTemporalBackend()
    rows = [Prepared(row.row, row.target.unsqueeze(2), row.neutral.unsqueeze(2)) for row in original.rows]
    critic = CampaignFieldCritic(2, original.conditions.shape[1], width=8)
    seen = []
    critic.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
    game = ConditionalPG09(backend, rows, critic, conditions=original.conditions,
                          particles=backend.network.particles, batch_size=4, total_steps=2)
    metrics = game.step()
    assert metrics['d_pen'] > 0 and game.completed_steps == 1
    assert seen and all(shape == (4, 2, 1, 10, 10) for shape in seen)
    assert game.response_shape == (1, 2, 1, 10, 10)
    assert not torch.cuda.is_initialized()


class BuilderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])


@pytest.fixture
def builder_inputs(monkeypatch):
    class TrackingSlider(nn.Module):
        def __init__(self, transformer, **kwargs):
            super().__init__()
            transformer.requires_grad_(False)
            self.weight = nn.Parameter(torch.ones(1))
            self.handle = transformer.register_forward_hook(lambda module, args, output: output)

        @contextmanager
        def at(self, value):
            yield

        def detach(self):
            self.handle.remove()

    native = ModuleType('moss_sliders')
    native.__path__ = []
    lora = ModuleType('moss_sliders.lora')
    lora.Slider = TrackingSlider
    monkeypatch.setitem(sys.modules, 'moss_sliders', native)
    monkeypatch.setitem(sys.modules, 'moss_sliders.lora', lora)
    transformer = BuilderTransformer()
    provider = SimpleNamespace(split='train', response_shape=(1, 2, 2, 2), condition_width=3,
        device='unassigned', identity=lambda: {'fixture': 'bounded CPU cache'},
        recipe=dict(model_revision='fixture-v1', settings=dict(seed=7, rank=2, alpha=2)))
    identity = dict(model='bounded CPU MOSS construction fixture', revision='fixture-v1')
    return transformer, provider, identity


@pytest.mark.parametrize('invalid', ['version', 'domain'])
def test_campaign_runtime_domain_preflight_precedes_adapter_mutation(builder_inputs, monkeypatch, invalid):
    import particle_sliders.pg09 as core
    transformer, provider, identity = builder_inputs
    if invalid == 'version':
        monkeypatch.setattr(core, 'version', lambda name: 'unsupported')
        expected = RuntimeError
    else:
        provider.response_shape = (1, 200)
        expected = ValueError
    flags = [parameter.requires_grad for parameter in transformer.parameters()]
    with pytest.raises(expected):
        build_campaign_run(transformer, provider, model_identity=identity, response_scale=1.)
    assert not transformer._forward_hooks
    assert [parameter.requires_grad for parameter in transformer.parameters()] == flags
    assert provider.device == 'unassigned'
    assert all('forward' not in block.__dict__ for block in transformer.layers)


def test_rejected_core_build_detaches_slider_and_restores_grad_flags_and_provider(builder_inputs, monkeypatch):
    import particle_sliders.pg09 as core
    transformer, provider, identity = builder_inputs
    flags = [parameter.requires_grad for parameter in transformer.parameters()]

    def reject(*args, **kwargs):
        assert transformer._forward_hooks
        assert all(not parameter.requires_grad for parameter in transformer.parameters())
        raise ValueError('forced core configuration rejection')

    monkeypatch.setattr(core, 'ConditionalPG09', reject)
    with pytest.raises(ValueError, match='forced core'):
        build_campaign_run(transformer, provider, model_identity=identity, response_scale=1.)
    assert not transformer._forward_hooks
    assert [parameter.requires_grad for parameter in transformer.parameters()] == flags
    assert provider.device == 'unassigned'
    assert all('forward' not in block.__dict__ for block in transformer.layers)


def test_failed_checkpoint_install_restores_existing_forwards_and_markers(builder_inputs, monkeypatch):
    import particle_sliders.pg09 as core
    import particle_sliders.pg09_moss_campaign as campaign
    transformer, provider, identity = builder_inputs
    previous_forward = lambda value: value
    transformer.layers[0].forward = previous_forward
    transformer.layers[0]._pg09_scale_checkpoint = False
    flags = [parameter.requires_grad for parameter in transformer.parameters()]
    monkeypatch.setattr(core, 'ConditionalPG09', lambda *args, **kwargs: SimpleNamespace())
    original_checkpoint = campaign.checkpoint_with_adapter_scale

    def install_then_fail(block, network):
        original_checkpoint(block, network)
        if block is transformer.layers[1]:
            raise ValueError('forced checkpoint hook rejection')

    monkeypatch.setattr(campaign, 'checkpoint_with_adapter_scale', install_then_fail)
    with pytest.raises(ValueError, match='forced checkpoint'):
        build_campaign_run(transformer, provider, model_identity=identity, response_scale=1.)
    assert transformer.layers[0].forward is previous_forward
    assert transformer.layers[0]._pg09_scale_checkpoint is False
    assert 'forward' not in transformer.layers[1].__dict__
    assert '_pg09_scale_checkpoint' not in transformer.layers[1].__dict__
    assert not transformer._forward_hooks
    assert [parameter.requires_grad for parameter in transformer.parameters()] == flags
    assert provider.device == 'unassigned'
