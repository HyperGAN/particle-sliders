"""Real tiny PEFT mixed-adapter contracts; no pretrained model or downloads."""
from copy import deepcopy

import pytest
import torch
from torch import nn

peft = pytest.importorskip('peft')  # Optional native dependency, not core runtime.

from particle_sliders.peft_scaling import peft_adapter_scale_context
from particle_sliders.pg09_backends import (
    AdapterParameters, checkpoint_with_adapter_scale, lora_scale_context,
)

FIXED = 'slider_fixed'  # Deliberate prefix overlap: selection must use exact keys.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Already found a.*peft_config.*:UserWarning:peft.tuners.tuners_utils')


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(1)
            torch.manual_seed(20261101)
            yield
    finally:
        torch.set_num_threads(previous)


def model_fixture(*, rslora=False):
    host = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 3))
    for name, alpha in [(FIXED, 2), ('slider', 6)]:
        peft.inject_adapter_in_model(peft.LoraConfig(r=2, lora_alpha=alpha,
            target_modules=['0', '2'], lora_dropout=0., use_rslora=rslora and name == 'slider'),
            host, adapter_name=name)
    for layer in layers(host):
        layer.set_adapter([FIXED, 'slider'])
        with torch.no_grad():
            for name in (FIXED, 'slider'):
                layer.lora_B[name].weight.normal_(std=.15)
    host.eval().requires_grad_(False)
    for layer in layers(host):
        layer.lora_A['slider'].requires_grad_(True)
        layer.lora_B['slider'].requires_grad_(True)
    return host


def layers(host):
    return host[0], host[2]


def scope_fixture(host):
    scope = peft_adapter_scale_context([('native', host)], adapter_names=('slider',))
    network = AdapterParameters([(name, p) for name, p in host.named_parameters() if p.requires_grad], scope)
    return scope, network


def scale_state(host):
    return [(layer.scaling, dict(layer.scaling)) for layer in layers(host)]


def assert_scales_restored(host, before):
    for layer, (original, values) in zip(layers(host), before):
        assert layer.scaling is original
        assert layer.scaling == values


def fixed_only(host, value):
    for layer in host:
        if isinstance(layer, nn.Tanh):
            value = layer(value)
        else:
            value = layer.base_layer(value) + layer.lora_B[FIXED](
                layer.lora_A[FIXED](value)) * layer.scaling[FIXED]
    return value


@pytest.mark.parametrize('strength', [0., .5, -.75, 1.])
def test_explicit_selection_preserves_fixed_companion_and_original_dicts(strength):
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before, weights = scale_state(host), deepcopy(host.state_dict())
    x = torch.randn(2, 4)
    expected_zero = fixed_only(host, x)
    with scope(strength):
        for layer, (_, values) in zip(layers(host), before):
            assert layer.scaling['slider'] == values['slider'] * strength
            assert layer.scaling[FIXED] == values[FIXED]
        if strength == 0:
            torch.testing.assert_close(host(x), expected_zero, rtol=0, atol=0)
    assert_scales_restored(host, before)
    assert all(torch.equal(weights[key], value) for key, value in host.state_dict().items())


def test_overlapping_roots_deduplicate_and_metadata_binds_exact_scope():
    host = model_fixture()
    scope = peft_adapter_scale_context([('native', host), ('same_leaf', host[0])],
                                       adapter_names=('slider',))
    receipt = scope.metadata()
    assert receipt['adapter_names'] == ['slider']
    assert [target['path'] for target in receipt['targets']] == ['native.0', 'native.2']
    assert receipt['targets'][0]['selected_baseline_scaling'] == {'slider': 3.}
    assert receipt['targets'][0]['fixed_scaling'] == {FIXED: 1.}
    assert len(receipt['source_sha256']) == 64
    receipt['adapter_names'].append(FIXED)
    assert scope.metadata()['adapter_names'] == ['slider']
    with scope(.5):
        assert host[0].scaling['slider'] == 1.5  # alpha/r = 6/2, applied once
        assert host[0].scaling[FIXED] == 1.


def test_native_rs_lora_factors_are_captured_without_assuming_alpha_over_rank():
    host = model_fixture(rslora=True)
    before = scale_state(host)
    scope, _ = scope_fixture(host)
    with scope(.5):
        assert host[0].scaling['slider'] == (6 / (2 ** .5)) * .5
        assert host[0].scaling[FIXED] == 1.
    assert_scales_restored(host, before)


def test_nested_exception_restores_entry_scales_without_division():
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before = scale_state(host)
    with pytest.raises(KeyboardInterrupt, match='fixture'):
        with scope(.5):
            outer = scale_state(host)
            with scope(0.):
                assert all(layer.scaling['slider'] == 0 for layer in layers(host))
            assert_scales_restored(host, outer)
            with scope(-.75):
                assert host[0].scaling['slider'] == -2.25
            assert_scales_restored(host, outer)
            raise KeyboardInterrupt('fixture')
    assert_scales_restored(host, before)


def test_cleanup_preflight_interruption_still_restores_every_owned_dictionary(monkeypatch):
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before, original = scale_state(host), scope._preflight
    calls = []

    def interrupt_cleanup():
        calls.append(True)
        if len(calls) == 2:
            raise KeyboardInterrupt('cleanup preflight')
        original()

    monkeypatch.setattr(scope, '_preflight', interrupt_cleanup)
    with pytest.raises(KeyboardInterrupt, match='cleanup preflight'):
        with scope(.5):
            pass
    assert_scales_restored(host, before)


@pytest.mark.parametrize('strength', [0., .5, -.75, 1.])
def test_captured_checkpoint_gradients_match_after_scope_and_handle_exit(strength):
    plain = model_fixture()
    wrapped = deepcopy(plain)
    _, expected_owner = scope_fixture(plain)
    _, actual_owner = scope_fixture(wrapped)
    x = torch.randn(2, 4, requires_grad=True)
    replay_x = x.detach().clone().requires_grad_()
    before = scale_state(wrapped)
    with expected_owner.scaled(strength):
        expected = plain(x)
    with checkpoint_with_adapter_scale(wrapped, actual_owner):
        with actual_owner.scaled(strength):
            actual = wrapped(replay_x)
    # Both the forward scale scope and checkpoint installation have closed.
    # Existing capture machinery must replay the original strength, then restore
    # this unrelated outer strength rather than recompute at restored scale 1.
    expected.square().sum().backward()
    with actual_owner.scaled(.125):
        actual.square().sum().backward()
        assert actual_owner.scale == .125
        assert wrapped[0].scaling['slider'] == .375
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(replay_x.grad, x.grad, rtol=0, atol=0)
    for expected_p, actual_p in zip(expected_owner.parameters(), actual_owner.parameters()):
        assert actual_p.grad is not None and torch.isfinite(actual_p.grad).all()
        torch.testing.assert_close(actual_p.grad, expected_p.grad, rtol=0, atol=0)
    assert_scales_restored(wrapped, before)


def test_selected_optimizer_step_leaves_fixed_and_base_weights_unchanged():
    host = model_fixture()
    _, owner = scope_fixture(host)
    before, scaling = deepcopy(host.state_dict()), scale_state(host)
    optimizer = torch.optim.SGD(owner.parameters(), lr=.1)
    with owner.scaled(.5):
        host(torch.randn(3, 4)).square().sum().backward()
    selected = {name for name, p in host.named_parameters() if p.requires_grad}
    for name, parameter in host.named_parameters():
        if name in selected:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
    optimizer.step()
    assert any(not torch.equal(before[name], parameter) for name, parameter in host.named_parameters() if name in selected)
    assert all(torch.equal(before[name], parameter) for name, parameter in host.named_parameters() if name not in selected)
    assert_scales_restored(host, scaling)


def test_distinct_forward_strengths_and_repeated_autograd_grad_replay():
    plain = model_fixture()
    wrapped = deepcopy(plain)
    _, expected_owner = scope_fixture(plain)
    _, actual_owner = scope_fixture(wrapped)
    x = torch.randn(2, 4)
    expected, actual = [], []
    with checkpoint_with_adapter_scale(wrapped, actual_owner):
        for strength in (.5, -.75):
            with expected_owner.scaled(strength):
                expected.append(plain(x).square().sum())
            with actual_owner.scaled(strength):
                actual.append(wrapped(x).square().sum())
    for retain in (True, False):
        reference = torch.autograd.grad(sum(expected), tuple(expected_owner.parameters()), retain_graph=retain)
        replayed = torch.autograd.grad(sum(actual), tuple(actual_owner.parameters()), retain_graph=retain)
        for left, right in zip(reference, replayed):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('names', [(), None, 'slider', ('slider', 'slider'), ('',), ('slide',)])
def test_invalid_or_inexact_names_refuse_without_scale_mutation(names):
    host = model_fixture()
    before = scale_state(host)
    with pytest.raises(ValueError, match='adapter'):
        peft_adapter_scale_context([('native', host)], adapter_names=names)
    assert_scales_restored(host, before)


def test_missing_name_on_one_leaf_refuses_entire_scope_before_mutation():
    host = model_fixture()
    del host[2].lora_A['slider']
    before = scale_state(host)
    with pytest.raises(ValueError, match='incomplete selected'):
        peft_adapter_scale_context([('native', host)], adapter_names=('slider',))
    assert_scales_restored(host, before)
    # A present-but-not-placed selection must also fail; no partial coverage.
    with pytest.raises(ValueError, match='missing from layer'):
        peft_adapter_scale_context([('native', host)], adapter_names=('slider', 'not_here'))
    assert_scales_restored(host, before)


def test_partial_placement_requires_explicit_narrower_roots():
    host = model_fixture()
    host[2].delete_adapter('slider')
    before = scale_state(host)
    with pytest.raises(ValueError, match='missing from layer'):
        peft_adapter_scale_context([('native', host)], adapter_names=('slider',))
    assert_scales_restored(host, before)
    scope = peft_adapter_scale_context([('native.0', host[0])], adapter_names=('slider',))
    with scope(.5):
        assert host[0].scaling['slider'] == 1.5
        assert host[2].scaling is before[1][0]
        assert host[2].scaling == before[1][1]


def test_selected_embedding_leaf_is_refused_without_mutation():
    host = nn.Sequential(nn.Embedding(16, 4))
    peft.inject_adapter_in_model(peft.LoraConfig(r=2, target_modules=['0']), host, adapter_name='slider')
    before, original = dict(host[0].scaling), host[0].scaling
    with pytest.raises(ValueError, match='stock PEFT Linear'):
        peft_adapter_scale_context([('native', host)], adapter_names=('slider',))
    assert host[0].scaling is original and host[0].scaling == before


def test_distinct_layers_cannot_share_a_provenance_path():
    host = model_fixture()
    before = scale_state(host)
    with pytest.raises(ValueError, match='uniquely identify'):
        peft_adapter_scale_context([('same_path', host[0]), ('same_path', host[2])],
                                   adapter_names=('slider',))
    assert_scales_restored(host, before)


def test_later_layer_scaled_value_overflow_refuses_before_first_write():
    host = model_fixture()
    host[2].scaling['slider'] = 1e308
    scope, _ = scope_fixture(host)
    before = scale_state(host)
    with pytest.raises(ValueError, match='scaled value must be a finite'):
        with scope(2.):
            pass
    assert_scales_restored(host, before)


@pytest.mark.parametrize('case', ['disabled', 'merged', 'inactive', 'variant', 'replacement', 'rank', 'new_layer'])
def test_entry_preflight_checks_all_layers_before_any_mutation(case):
    host = model_fixture()
    scope, _ = scope_fixture(host)
    if case == 'disabled':
        host[2].enable_adapters(False)
    elif case == 'merged':
        host[2].merge(adapter_names=[FIXED])
    elif case == 'inactive':
        host[2].set_adapter([FIXED])
    elif case == 'variant':
        host[2].use_dora[FIXED] = True
    elif case == 'replacement':
        host[2].lora_B['slider'] = nn.Linear(2, 3, bias=False)
    elif case == 'rank':
        host[2].r['slider'] = 3
    else:
        host.add_module('another', deepcopy(host[2]))
    before = scale_state(host)
    with pytest.raises(ValueError, match='PEFT'):
        with scope(.5):
            raise AssertionError('must refuse before body')
    assert_scales_restored(host, before)


@pytest.mark.parametrize('strength', [float('nan'), float('inf'), -float('inf'), True])
def test_invalid_strength_refuses_without_scale_mutation(strength):
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before = scale_state(host)
    with pytest.raises(ValueError, match='finite real scalar'):
        with scope(strength):
            pass
    assert_scales_restored(host, before)


@pytest.mark.parametrize('case', ['fixed_scale', 'topology'])
def test_nested_external_mutation_raises_and_restores_owned_scale_dicts(case):
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before = scale_state(host)
    with pytest.raises(ValueError, match='changed'):
        with scope(.5):
            if case == 'fixed_scale':
                host[2].scaling[FIXED] *= 2
            else:
                host[2].lora_B['slider'] = nn.Linear(2, 3, bias=False)
            with scope(.25):
                raise AssertionError('mutation must be refused')
    assert_scales_restored(host, before)


def test_foreign_dictionary_replacement_is_reported_and_not_overwritten():
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before = scale_state(host)
    with pytest.raises(RuntimeError, match='ownership changed'):
        with scope(.5):
            foreign = dict(host[0].scaling)
            host[0].scaling = foreign
    assert host[0].scaling is foreign
    assert host[2].scaling is before[1][0]


@pytest.mark.parametrize('name', ['slider', FIXED])
def test_body_scaling_mutation_is_reported_without_needing_nested_entry(name):
    host = model_fixture()
    scope, _ = scope_fixture(host)
    before = scale_state(host)
    with pytest.raises((ValueError, RuntimeError), match='changed'):
        with scope(.5):
            host[0].scaling[name] *= 2
    assert_scales_restored(host, before)


def test_numeric_zero_peft_scaling_is_not_a_true_branch_bypass():
    host = model_fixture()
    scope, _ = scope_fixture(host)
    with torch.no_grad():
        host[0].lora_B['slider'].weight.fill_(float('inf'))
    with scope(0.):
        assert not torch.isfinite(host(torch.ones(1, 4))).all()


def test_historical_full_host_scaler_retains_its_all_names_behavior():
    host = model_fixture()
    before = scale_state(host)
    legacy = lora_scale_context([('native', host)])
    with legacy(.5):
        assert host[0].scaling['slider'] == 1.5
        assert host[0].scaling[FIXED] == .5  # Historical hazard, not new-helper policy.
    assert all(layer.scaling == values for layer, (_, values) in zip(layers(host), before))
