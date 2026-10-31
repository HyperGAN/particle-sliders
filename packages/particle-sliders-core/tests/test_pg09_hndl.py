"""CPU qualification of the optional released HNDL conditional critic."""
from copy import deepcopy
from dataclasses import dataclass
import io
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic


@pytest.fixture(autouse=True)
def optional_hndl():
    pytest.importorskip('hndl', minversion='0.8.2')
    from importlib.metadata import version
    if version('hndl') != '0.8.2':
        pytest.skip('legacy readout requires the separate hndl==0.8.2 extra; modern hypergan-hndl uses0.10.0')
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@dataclass(frozen=True)
class Prepared:
    target: torch.Tensor
    neutral: torch.Tensor
    inputs: torch.Tensor
    sign: int


class Backend:
    def __init__(self):
        self.network = nn.Linear(2, 8, bias=False)

    def metadata(self):
        return {'model': 'hndl-cpu-fixture', 'protocol': 'native-signed-response-v1'}

    def generated(self, row):
        return row.neutral + row.sign * self.network(row.inputs).reshape(row.neutral.shape)


def game(*, source=None, preset='bispf_optimized', legacy=False, domain='continuous_field'):
    from particle_sliders.pg09 import ConditionalPG09
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        backend = Backend()
        critic = (LegacyCritic() if legacy else
                  HNDLConditionalReadoutCritic(8, 2, hidden=6, source=source))
        shape = (1, 2, 2, 2) if domain == 'continuous_field' else (1, 8)
        rows = [Prepared(torch.full(shape, sign * .2), torch.zeros(shape),
                         torch.tensor([[i / 4, 1.]]), sign)
                for i, sign in enumerate((1, -1, 1, -1))]
        conditions = torch.tensor([[row.sign, i / 4] for i, row in enumerate(rows)])
        return ConditionalPG09(backend, rows, critic, conditions=conditions,
            response_domain=domain, batch_size=2, total_steps=3, preset=preset)


class LegacyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10, 6), nn.SiLU(), nn.Linear(6, 1))

    def forward(self, response, condition):
        return self.layers(torch.cat((response.flatten(1), condition), 1)).squeeze(-1)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) == type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('shape', [(3, 8), (3, 2, 2, 2), (1, 8)])
def test_named_inputs_keep_batch_and_return_raw_scalar(shape):
    critic = HNDLConditionalReadoutCritic(8, 2)
    assert critic(torch.randn(shape), torch.randn(shape[0], 2)).shape == (shape[0],)
    critic.network['logits'].weight.data.zero_()
    critic.network['logits'].bias.data.fill_(12.)
    assert torch.equal(critic(torch.zeros(shape), torch.zeros(shape[0], 2)), torch.full((shape[0],), 12.))


def test_matched_weights_scores_first_and_second_derivatives_equal_legacy():
    torch.manual_seed(53)
    critic = HNDLConditionalReadoutCritic(8, 2, hidden=6).double()
    legacy = LegacyCritic().double()
    critic.network['hidden'].load_state_dict(legacy.layers[0].state_dict())
    critic.network['logits'].load_state_dict(legacy.layers[2].state_dict())
    response = torch.randn(3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    condition = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    actual, expected = critic(response, condition), legacy(response, condition)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.sum(), (response, condition), create_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (response, condition), create_graph=True)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual_second = torch.autograd.grad(actual_grads[0].square().sum(), response)[0]
    expected_second = torch.autograd.grad(expected_grads[0].square().sum(), response)[0]
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
    other = torch.autograd.grad(critic(response, -condition).sum(), response)[0]
    assert not torch.equal(actual_grads[0], other)


@pytest.mark.parametrize('preset', ['bispf_optimized', 'bispf'])
@pytest.mark.parametrize('domain', ['continuous_field', 'continuous_vector'])
def test_public_gan_penalty_updates_and_exact_resume(preset, domain):
    original = game(preset=preset, domain=domain)
    before_g = deepcopy(original.network.state_dict())
    before_d = deepcopy(original.critic.state_dict())
    native_shapes = []
    original.critic.register_forward_pre_hook(lambda module, args: native_shapes.append(tuple(args[0].shape)))
    metrics = original.step()
    assert metrics['d_pen'] > 0
    assert metrics['g_loss'] == pytest.approx(metrics['g_adv'])
    assert any(not torch.equal(before_g[k], v) for k, v in original.network.state_dict().items())
    assert any(not torch.equal(before_d[k], v) for k, v in original.critic.state_dict().items())
    assert set(native_shapes) == {(2, 2, 2, 2) if domain == 'continuous_field' else (2, 8)}
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    resumed = game(preset=preset, domain=domain)
    resumed.load_state_dict(checkpoint)
    assert original.step() == resumed.step()
    equal(original.state_dict(), resumed.state_dict())
    assert not torch.cuda.is_initialized()


def test_architecture_record_and_same_shape_source_change_reject_resume_transactionally():
    original = game()
    original.step()
    source = original.critic.architecture()['source']
    record = original.critic.architecture()
    assert record['hndl_version'] == '0.8.2' and record['hndl_sources']
    assert record['plan']['input_shape'] == ['B', 8]
    json.dumps(record, allow_nan=False)
    record['source'] = 'mutated external record'
    assert original.critic.architecture()['source'] == source
    changed = game(source=source.replace('silu(', 'tanh('))
    assert original.critic.extra_repr() != changed.critic.extra_repr()
    before = changed.state_dict()
    with pytest.raises(ValueError, match='mismatch'):
        changed.load_state_dict(original.state_dict())
    equal(before, changed.state_dict())
    legacy = game(legacy=True)
    with pytest.raises(ValueError, match='mismatch'):
        legacy.load_state_dict(original.state_dict())


def test_optional_import_is_lazy_and_missing_extra_is_actionable():
    code = '''
import sys
import particle_sliders.pg09_hndl as adapter
assert 'hndl' not in sys.modules
sys.modules['hndl'] = None
try:
    adapter.HNDLConditionalReadoutCritic(8, 2)
except ImportError as exc:
    assert 'particle-sliders-core[hndl]' in str(exc)
else:
    raise AssertionError('missing optional dependency accepted')
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('sizes', [(0, 2, 16), (8, 0, 16), (8, 2, False), (8.5, 2, 16)])
def test_invalid_widths_fail_before_model_allocation(sizes):
    with pytest.raises(ValueError, match='positive integers'):
        HNDLConditionalReadoutCritic(*sizes)


def test_output_bounds_and_pretrained_sources_are_rejected_before_build():
    with pytest.raises(ValueError, match='raw-logit'):
        HNDLConditionalReadoutCritic(8, 2, source='concat(response, condition)\nlinear(1)\ntanh()')
    with pytest.raises(ValueError, match='readout operations'):
        HNDLConditionalReadoutCritic(8, 2, source='pretrained("hf://unused")')
    with pytest.raises(ValueError, match='shape contract'):
        HNDLConditionalReadoutCritic(8, 2)(torch.zeros(2, 8), torch.zeros(1, 2))


def test_indented_triple_quoted_source_matches_public_resolver_and_keeps_original_identity():
    from hndl import resolve

    source = '''
        joined = concat(response, condition)
        features = linear(joined, 4)
        features = silu(features)
        linear(features, 1)
    '''
    plan = resolve(source,
        input_shape={'response': ('B', 8), 'condition': ('B', 2)}, output_shape=('B', 1))
    critic = HNDLConditionalReadoutCritic(8, 2, source=source)
    assert critic.architecture()['source'] == source
    assert critic.architecture()['plan'] == json.loads(plan.to_json())
    response, condition = torch.randn(3, 8), torch.randn(3, 2)
    assert critic(response, condition).shape == (3,)
    with pytest.raises(ValueError, match='readout operations'):
        HNDLConditionalReadoutCritic(8, 2, source='''
            pretrained("hf://unused")
        ''')
