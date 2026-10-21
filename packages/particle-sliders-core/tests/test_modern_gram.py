"""Modern source/profile defaults, upstream solver semantics and full game state."""
from copy import deepcopy
from dataclasses import asdict, replace
import inspect

import pytest
import torch
from particlegan import get_recipe
from particlegan.optim.gram_polar import GramGuardError, guarded_gram_polar
from particlegan.optim import gram_svd_fallback as upstream
from particle_sliders.optimizer_policy import configure_dense_profile, GRAM_PROFILE, POLICY
from particle_sliders.pg09 import ConditionalPG09, PG_VERSION, PG_SOURCE_PROFILE
from test_conditional_pg09 import fixture, equal


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def game(profile='auto', *, particles=True, preset='bispf_optimized'):
    original = fixture(particles=particles, preset=preset)
    return ConditionalPG09(original.backend, original.rows, original.critic,
        conditions=original.conditions,
        particles=original.network.particles if particles else None,
        preset=preset, batch_size=4, total_steps=5, optimizer_dense_profile=profile)


def reference(matrix):
    # Independent FP64 SVD value oracle; strict ORIGINAL FP32 cutoff and damping.
    value = matrix.double()
    u, singular, vh = torch.linalg.svd(value, full_matrices=False)
    cutoff = max(matrix.shape) * torch.finfo(torch.float32).eps * singular[0]
    weights = torch.where(singular > cutoff,
        singular / torch.hypot(singular, singular.new_tensor(.001)), 0.)
    return ((u * weights) @ vh).float()


def test_default_is_supported_upstream_profile_and_changes_only_one_recipe_field():
    recipe = get_recipe('bispf_optimized', total_steps=5)
    resolved, declaration = configure_dense_profile(recipe)
    assert PG_VERSION == '0.9.3' and PG_SOURCE_PROFILE == 'release-0.9.3'
    assert resolved.optimizer_dense_profile == GRAM_PROFILE
    before, after = asdict(recipe), asdict(resolved)
    assert {k for k in before if before[k] != after[k]} == {'optimizer_dense_profile'}
    assert declaration['schema'] == POLICY and declaration['eligible']
    assert not declaration['differentiated_solver']


@pytest.mark.parametrize('requested', ['default', GRAM_PROFILE, 'gram_fp64_guarded_v1'])
def test_explicit_supported_profiles_are_preserved(requested):
    resolved, _ = configure_dense_profile(get_recipe('bispf_optimized'), requested)
    assert resolved.optimizer_dense_profile == requested


def test_adam_arm_keeps_original_law_and_reports_why_gram_is_unavailable():
    original = get_recipe('bispf')
    resolved, declaration = configure_dense_profile(original)
    assert asdict(original) == asdict(resolved)
    assert declaration['resolved'] == 'default' and not declaration['eligible']
    with pytest.raises(ValueError, match='full DualNorm'):
        configure_dense_profile(original, GRAM_PROFILE)


@pytest.mark.parametrize('change', [dict(optimizer_momentum=.5, constraint_geometry_mode='none'), dict(optimizer_smoothing=.02),
                                    dict(optimizer_svd_backend='cpu', optimizer_convolution_rule='polar')])
def test_unsupported_optimizer_config_is_not_rewritten(change):
    recipe = replace(get_recipe('bispf_optimized'), **change)
    result, declaration = configure_dense_profile(recipe)
    assert asdict(result) == asdict(recipe) and declaration['resolved'] == 'default'
    with pytest.raises(ValueError):
        configure_dense_profile(recipe, GRAM_PROFILE)


@pytest.mark.parametrize('value', ['gram', 'native', '', None, [], {}])
def test_unknown_profile_refused(value):
    with pytest.raises(ValueError):
        configure_dense_profile(get_recipe('bispf_optimized'), value)


@pytest.mark.parametrize('shape', [(9, 3), (3, 9), (5, 5), (128, 4), (1, 8)])
@pytest.mark.parametrize('kind', ['random', 'zero', 'rank_deficient', 'small', 'large'])
def test_values_against_full_fp64_svd_and_input_rng_preservation(shape, kind):
    generator = torch.Generator().manual_seed(18)
    matrix = torch.randn(shape, generator=generator)
    if kind == 'zero':matrix.zero_()
    elif kind == 'rank_deficient':matrix[:, -1] = matrix[:, 0]
    elif kind == 'small':matrix *= 1e-15
    elif kind == 'large':matrix *= 1e15
    before, rng = matrix.clone(), torch.get_rng_state().clone()
    actual = upstream.solve(matrix).direction
    torch.testing.assert_close(actual, reference(matrix), rtol=4e-5, atol=3e-6)
    assert torch.isfinite(actual).all() and actual.dtype == torch.float32
    assert torch.equal(before, matrix) and torch.equal(rng, torch.get_rng_state())


def test_rank_ambiguity_uses_only_upstream_strict_svd_fallback(monkeypatch):
    def ambiguous(*args, **kwargs):
        raise GramGuardError(upstream.AMBIGUOUS_MESSAGE)
    monkeypatch.setattr(upstream, '_original_guard', ambiguous)
    matrix = torch.diag(torch.tensor([1., 2**-22]))
    counters = upstream.Counters()
    decision = upstream.solve(matrix, counters=counters)
    assert decision.path == 'cpu_fp64_svd_fallback'
    assert decision.diagnostics['rank_decision_policy'] == 'public_strict_computed_svd'
    assert decision.diagnostics['uncertified_svd_rank_decision']
    assert counters.rank_ambiguities == counters.svd_attempts == counters.fallback_passes == 1
    assert counters.gram_passes == 0
    assert torch.equal(decision.direction, reference(matrix))


@pytest.mark.parametrize('error', [GramGuardError('other numerical guard'), RuntimeError('technical solver failure')])
def test_other_failures_propagate_without_solver_substitution(monkeypatch, error):
    def fail(*args, **kwargs):raise error
    monkeypatch.setattr(upstream, '_original_guard', fail)
    counters = upstream.Counters()
    with pytest.raises(type(error), match=str(error)):
        upstream.solve(torch.ones(5, 3), counters=counters)
    assert counters.svd_attempts == 0 and counters.propagated_other_errors == 1


@pytest.mark.parametrize('kind', ['nan', 'inf', 'double', 'requires_grad'])
def test_invalid_solver_inputs_refused_without_graph_or_input_changes(kind):
    matrix = torch.ones(5, 3)
    if kind == 'nan':matrix[0, 0] = float('nan')
    elif kind == 'inf':matrix[0, 0] = float('inf')
    elif kind == 'double':matrix = matrix.double()
    else:matrix.requires_grad_()
    version = matrix._version
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):upstream.solve(matrix)
    assert matrix._version == version and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('particles', [True, False])
def test_actual_generator_critic_dispatch_and_exact_split_resume(particles):
    full, split = game(particles=particles), game(particles=particles)
    for _ in range(4):full.step()
    for _ in range(2):split.step()
    packet = deepcopy(split.state_dict())
    resumed = game(particles=particles)
    resumed.load_state_dict(packet)
    for _ in range(2):resumed.step()
    equal(full.state_dict(), resumed.state_dict())
    for name in ('generator', 'critic'):
        diagnostics = full.optimizer_diagnostics[name]
        assert diagnostics['profile'] == GRAM_PROFILE
        calls = diagnostics['aggregate']['calls']
        assert calls > 0
        assert calls == diagnostics['aggregate']['gram_passes'] + diagnostics['aggregate']['fallback_passes']
    assert 'optimizer_policy' in packet['config']
    # Upstream diagnostics deliberately reset on successful load; training state does not.
    resumed.load_state_dict(packet)
    assert resumed.optimizer_diagnostics['generator']['aggregate']['calls'] == 0


def test_profile_checkpoint_mismatch_is_rejected_before_mutation():
    source, target = game(), game('default')
    source.step();packet = source.state_dict();before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='mismatch'):target.load_state_dict(packet)
    equal(before, target.state_dict())
    # Same resolved solver but a different declared policy remains explicit identity.
    explicit = game(GRAM_PROFILE)
    with pytest.raises(ValueError, match='mismatch'):explicit.load_state_dict(packet)


def test_failure_closes_training_state_until_complete_matching_restore(monkeypatch):
    from particlegan.optim import dualnorm
    active = game();before = deepcopy(active.state_dict())
    def fail(*args, **kwargs):raise RuntimeError('injected solver HOLD')
    with monkeypatch.context() as context:
        context.setattr(dualnorm, 'svd_fallback_polar', fail)
        with pytest.raises(RuntimeError, match='HOLD'):active.step()
    with pytest.raises(RuntimeError, match='failed'):active.state_dict()
    with pytest.raises(RuntimeError, match='failed'):active.export_state_dict()
    active.load_state_dict(before)
    equal(before, active.state_dict())
    active.step()


def test_training_gradients_are_preserved_for_same_weights_before_update():
    native, gram = game('default'), game()
    inputs = torch.tensor([[.3, .4]])
    for candidate in (native, gram):
        candidate.network.zero_grad(set_to_none=True)
        with torch.random.fork_rng():
            torch.manual_seed(19)
            candidate.network(inputs).square().sum().backward()
    for a, b in zip(native.network.parameters(), gram.network.parameters()):
        if a.grad is None:assert b.grad is None
        else:assert torch.equal(a.grad, b.grad)


def test_core_policy_contains_no_matrix_factorization_kernel():
    import particle_sliders.optimizer_policy as module
    source = inspect.getsource(module)
    assert 'torch.linalg' not in source and 'eigh(' not in source and 'svd(' not in source
    assert not torch.cuda.is_initialized()


def test_authoritative_moss112_field_recipe_preserved_except_selected_dense_profile():
    import json
    from pathlib import Path
    from particlegan.recipes import Recipe
    document = json.loads((Path(__file__).parent/'fixtures/pg093_v92_recipe.json').read_text())
    values = document['recipe']
    assert len(values) == 112 and values['optimizer_dense_profile'] == 'default'
    recipe = Recipe(**values)
    resolved, _ = configure_dense_profile(recipe)
    before, after = asdict(recipe), asdict(resolved)
    assert len(after) == 112
    assert {key for key in before if before[key] != after[key]} == {'optimizer_dense_profile'}
    assert after['optimizer_dense_profile'] == GRAM_PROFILE


def test_modern_fibo_hndl_factory_uses_gram_and_preserves_exact_resume(tmp_path):
    import hashlib
    from particle_sliders.fibo_native import (
        FiboParticleSlider, FiboEditResponseAdapter, FiboCachedRows, write_cache, build_fibo_game,
    )
    from particle_sliders.fibo_hndl import PROFILE, HNDLFieldCritic
    from test_fibo_native import Pipeline, IDENTITY, references
    def construct(folder):
        pipeline = Pipeline()
        slider = FiboParticleSlider(pipeline.transformer, network_profile=PROFILE)
        adapter = FiboEditResponseAdapter(pipeline, slider, model_identity=IDENTITY)
        prepared = adapter.capture_pair('keep jacket', 'make jacket red', references())
        write_cache(folder, prepared, model_identity=IDENTITY)
        digest = hashlib.sha256((folder/'index.json').read_bytes()).hexdigest()
        provider = FiboCachedRows(folder, index_sha256=digest, model_identity=IDENTITY)
        return build_fibo_game(adapter, provider, response_scale=.1, total_steps=3)
    full, split = construct(tmp_path/'full'), construct(tmp_path/'split')
    assert isinstance(full.critic, HNDLFieldCritic)
    assert full.recipe.optimizer_dense_profile == GRAM_PROFILE
    for _ in range(3):full.step()
    split.step();packet = deepcopy(split.state_dict())
    resumed = construct(tmp_path/'resumed');resumed.load_state_dict(packet)
    resumed.step();resumed.step()
    equal(full.state_dict(), resumed.state_dict())
    assert full.optimizer_diagnostics['generator']['aggregate']['calls'] > 0
    assert full.optimizer_diagnostics['critic']['aggregate']['calls'] > 0
    assert not torch.cuda.is_initialized()
