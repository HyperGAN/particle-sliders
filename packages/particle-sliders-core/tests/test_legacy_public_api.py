"""Public 0.3 constructor/stamp contract on the official PG0.9.1 lane."""
from dataclasses import FrozenInstanceError, asdict
import json
import pickle
from pathlib import Path

import pytest
import torch
from particlegan import grad_regularizers as upstream
from particle_sliders import GradRegularizer, GradientPenalty, particlegan_locked_shared
from particle_sliders.grad_regularizers import finite_difference_norm, UnsupportedLegacyRegularizer


def test_stamp_restores_frozen_historical_fields_and_dictionary_copy():
    stamp = particlegan_locked_shared()['stamp']
    source = Path(__import__('particle_sliders').__file__).with_name('legacy_stamp.json')
    document = json.loads(source.read_text())
    assert document['commit'] == '82b72661c03377f348ccb76de37cd4ca02d45a8d'
    assert document['sha256'] == 'd6e0b484691de311b31866b3fa52d6f388809648117926d2d47d25f302316610'
    assert stamp.cover_weight == 1.5 and stamp.n_particles == 12
    assert stamp.critic == 'host' and stamp.reg_norm == 'l2'
    assert asdict(stamp) == stamp.to_dict() == document['fields']
    assert len(asdict(stamp)) == 18
    copy = stamp.to_dict()
    copy['cover_weight'] = 0
    assert stamp.cover_weight == 1.5
    with pytest.raises(FrozenInstanceError):
        stamp.cover_weight = 0
    assert pickle.loads(pickle.dumps(stamp)) == stamp


@pytest.mark.parametrize('arm', ['b_cap', 'a_r1r2'])
@pytest.mark.parametrize('step', [1, 4])
def test_legacy_positional_constructor_and_call_delegate_exactly(arm, step):
    # Original positional order: arm, coeff, kappa, lazy_k, norm,
    # target_anneal, total_steps, method, fd_eps.
    regularizer = GradRegularizer(arm, 1.3, .2, 4, 'l2', 'none', 99, 'autograd', .07)
    assert GradRegularizer is GradientPenalty
    assert regularizer.norm == 'l2' and regularizer.method == 'autograd'
    assert regularizer.target_anneal == 'none' and regularizer.total_steps == 99
    assert regularizer.fd_eps == .07
    generator = torch.Generator().manual_seed(42)
    critic = torch.nn.Linear(3, 1).double()
    real = torch.arange(12, dtype=torch.double).reshape(4, 3) / 10
    fake = -real
    direct = upstream.GradientPenalty(arm=arm, coeff=1.3, kappa=.2, lazy_k=4)
    actual, stats = regularizer.penalty(critic, real, fake, step, generator, False)
    expected, expected_stats = direct.penalty(critic, real, fake, step, collect_stats=False)
    assert torch.equal(actual, expected) and stats == expected_stats == {}
    called = regularizer(critic, real, fake, step, generator)
    assert torch.equal(actual, called)
    if actual.requires_grad:
        args = (critic.weight,)
        ga = torch.autograd.grad(actual, args)
        gb = torch.autograd.grad(expected, args)
        assert all(torch.equal(a, b) for a, b in zip(ga, gb))


@pytest.mark.parametrize('options', [
    {'norm': 'l1'}, {'norm': 'linf'}, {'method': 'finite_difference'},
    {'target_anneal': 'linear', 'total_steps': 10},
    {'target_anneal': 'delayed', 'total_steps': 10},
    {'arm': 'f_none'}, {'arm': 'c_wgan_gp'},
])
def test_retired_modes_fail_before_upstream_state_or_critic_mutation(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before upstream construction')
    monkeypatch.setattr(upstream.GradientPenalty, '__init__', forbidden)
    rng = torch.get_rng_state().clone()
    with pytest.raises(UnsupportedLegacyRegularizer, match='17cd484'):
        GradRegularizer(**options)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('eps', [0, -1, float('nan'), float('inf')])
def test_inactive_fd_argument_keeps_historical_validation(eps):
    with pytest.raises(ValueError, match='fd_eps'):
        GradRegularizer(fd_eps=eps)


def test_finite_difference_import_is_actionable_and_cannot_touch_critic():
    def forbidden(value):
        raise AssertionError('retired kernel must not run a forward')
    with pytest.raises(UnsupportedLegacyRegularizer, match='finite_difference_norm'):
        finite_difference_norm(forbidden, torch.ones(1, 3), .05)


def test_finite_difference_wrapper_only_forwards_an_available_upstream_kernel(monkeypatch):
    sentinel = object()
    args = (object(), object(), .05)
    def supplied(*values):
        assert values == args
        return sentinel
    monkeypatch.setattr(upstream, 'finite_difference_norm', supplied, raising=False)
    assert finite_difference_norm(*args) is sentinel


def test_documented_legacy_example_is_complete_and_cpu_only():
    # The guide ships in the sdist; run its entire Python block unchanged.
    guide = Path(__file__).parents[1] / 'LEGACY-MIGRATION.md'
    text = guide.read_text()
    snippet = text.split('```python\n', 1)[1].split('```', 1)[0]
    namespace = {}
    exec(compile(snippet, 'LEGACY-MIGRATION.md', 'exec'), namespace)
    assert namespace['stats']['step'] == 1
    assert torch.isfinite(torch.tensor(namespace['stats']['g_loss']))
    assert not torch.cuda.is_initialized()
