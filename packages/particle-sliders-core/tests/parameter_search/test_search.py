from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import pytest

from particle_sliders.parameter_search.fixture import FixtureBackend
from particle_sliders.parameter_search.metric import digest, file_hash
from particle_sliders.parameter_search.runner import Run, ResourceLimit, atomic_json, audit, plan, read_json, validated
from particle_sliders.parameter_search.space import Space


REPO = Path(__file__).resolve().parents[4]


def config(name='vector'):
    value = read_json(REPO / 'examples/parameter_search' / (name + '.json'))
    # Short real training tests, retaining two independently varied rate axes.
    for name in ('lr.particle', 'regularizer.l2_alpha'):
        value['fixed'][name] = value['space'].pop(name)['initial']
    value['search'].update(initial_candidates=6, refinement_candidates=1, budgets=[2, 4],
                           training_seeds=[7], confirmation_seeds=[17])
    return value


def execute(tmp_path, value=None):
    value = config() if value is None else value
    run = Run(value, tmp_path)
    with run.lock():
        report = run.execute()
    return run, report


@pytest.mark.parametrize('name', ['vector', 'coupled'])
def test_real_training_replay_resume_full_audit_and_distinct_geometries(tmp_path, name):
    value = config(name)
    run, report = execute(tmp_path, value)
    assert report['nominations']
    assert all(row['status'] == 'confirmed_raw_metric' and not row['diagnostics']['native_evidence'] for row in report['nominations'])
    assert audit(tmp_path)['verified_evaluations'] > 10
    updates = read_json(tmp_path / 'progress.json')['attempted_updates']
    # Identical command is idempotent: all completed budgets/evidence are reused.
    run2, again = execute(tmp_path, value)
    assert read_json(tmp_path / 'progress.json')['attempted_updates'] == updates
    assert again['nominations'] == report['nominations']
    cap = run.cap
    surfaces = {p['position']['surface'] for e in cap.grids['dev'] for p in e['positions']}
    assert surfaces == ({'hidden'} if name == 'vector' else {'image', 'audio'})
    assert not list(tmp_path.glob('qualification/*'))  # optional controls never gate nomination


def test_rejects_unsupported_noop_parameters_ema_splits_and_native_allocation():
    value = config()
    value['fixed']['regularizer.invented_alpha'] = .1
    with pytest.raises(ValueError, match='no-op'):
        plan(value)
    value = config(); value['backend_options']['ema_decay'] = .8
    with pytest.raises(ValueError, match='EMA'):
        plan(value)
    backend = FixtureBackend('fixture-vector')
    cap = backend.describe(config()['backend_options'])
    with pytest.raises(ValueError, match='overlap'):
        replace(cap, grids=dict(cap.grids, heldout=cap.grids['dev']))
    with pytest.raises(ValueError, match='objective'):
        replace(cap.parameters[3], cohort='shared')


def test_deterministic_joint_search_refinement_zero_and_structural_cohorts():
    value = config()
    _, cap, space, values = validated(value)
    assert values == space.explore(6, 41)
    assert any(sum(p[n] != space.initial[n] for n in space.search) >= 2 for p in values)
    assert space.refine([values[0]], values, 0) == []
    refined = space.refine([values[0]], values, 2)
    assert refined and not {digest(v) for v in refined} & {digest(v) for v in values}
    baseline = space.initial
    assert space.cohort(cap.identity, baseline) != space.cohort(cap.identity, dict(baseline, **{'particles.population': 4}))
    assert space.cohort(cap.identity, baseline) != space.cohort(cap.identity, dict(baseline, **{'regularizer.l2_alpha': 0.}))
    fixed = dict(value['fixed']); fixed.pop('particles.population')
    other = Space(cap, fixed, dict(value['space'], **{'particles.population': {'values': [2, 4], 'initial': 2}}))
    assert 'particles.population' in other.boundaries(dict(other.initial, **{'particles.population': 4}))
    with pytest.raises(ValueError, match='joint'):
        space.explore(5, 41)


@pytest.mark.parametrize('name', ['lr.generator', 'lr.critic', 'lr.particle', 'regularizer.l2_alpha', 'particles.population', 'particles.temperature'])
def test_every_advertised_fixture_knob_changes_actual_state_or_response(name):
    backend = FixtureBackend('fixture-coupled')
    value = config('coupled')
    _, cap, space, _ = validated(value)
    initial = space.initial
    altered = dict(initial)
    altered[name] = {'lr.generator': .2, 'lr.critic': .2, 'lr.particle': .2,
        'regularizer.l2_alpha': .2, 'particles.population': 4, 'particles.temperature': 2.}[name]
    a = backend.open(value['backend_options'], initial, 7, 'test')
    b = backend.open(value['backend_options'], altered, 7, 'test')
    a.step(); b.step()
    assert a.state_digest() != b.state_digest()
    a.close(); b.close()


def test_interrupted_unsaved_updates_resume_from_actual_checkpoint(tmp_path, monkeypatch):
    value = config()
    run = Run(value, tmp_path)
    parameters = run.initial[0]
    with run.lock():
        original = run.update
        calls = 0
        def interrupt(active, params):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise KeyboardInterrupt()
            original(active, params)
        monkeypatch.setattr(run, 'update', interrupt)
        with pytest.raises(KeyboardInterrupt):
            run.trial(parameters, 7, 4)
    progress = read_json(tmp_path / 'progress.json')
    row = next(iter(progress['trials'].values()))
    assert row['latest'] == '1'  # unsaved step2 must never be counted as saved
    attempted = progress['attempted_updates']
    resumed = Run(value, tmp_path)
    with resumed.lock():
        result = resumed.trial(parameters, 7, 4)
    assert result['binding']['completed_steps'] == 4
    assert read_json(tmp_path / 'progress.json')['attempted_updates'] == attempted + 3
    assert audit(tmp_path)['verified_evaluations'] == 1


@pytest.mark.parametrize('artifact', ['responses.bin', 'responses.jsonl', 'score.json', 'checkpoint', 'export', 'results'])
def test_tampering_never_reblessed_on_audit_or_resume(tmp_path, artifact):
    value = config()
    _, report = execute(tmp_path, value)
    progress = read_json(tmp_path / 'progress.json')
    row = next(iter(progress['trials'].values()))
    if artifact == 'checkpoint':
        path = tmp_path / row['checkpoints']['1']['path']
    elif artifact == 'export':
        path = tmp_path / next(iter(row['exports'].values()))['path']
    elif artifact == 'results':
        result = read_json(tmp_path / 'results.json')
        next(iter(result['rungs'][0]['groups'].values()))[0]['error'] += 1
        atomic_json(tmp_path / 'results.json', result)
        path = None
    else:
        directory = tmp_path / next(iter(row['evaluations'].values()))
        path = directory / artifact
    if path is not None:
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1; path.write_bytes(data)
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        audit(tmp_path)
    if artifact == 'export':
        run = Run(value, tmp_path)
        with run.lock(), pytest.raises(ValueError, match='export checksum'):
            run.execute()


def test_state_mutating_evaluation_and_export_and_effective_config_fail_closed(tmp_path, monkeypatch):
    import particle_sliders.parameter_search.runner as module
    backend = FixtureBackend('fixture-vector')
    original = backend.open
    fault = ['evaluate']
    def opening(*args, **kwargs):
        trial = original(*args, **kwargs)
        if fault[0] == 'evaluate':
            def responses(*args, **kwargs):
                trial.generator[0] += .1
                return original_responses(*args, **kwargs)
            original_responses = trial.responses
            trial.responses = responses
        elif fault[0] == 'export':
            original_export = trial.export
            def export(*args, **kwargs):
                original_export(*args, **kwargs)
                trial.generator[0] += .1
            trial.export = export
        else:
            trial.effective_parameters = lambda: {}
        return trial
    backend.open = opening
    monkeypatch.setattr(module, 'resolve', lambda name: backend)
    for mode in ('evaluate', 'export', 'config'):
        fault[0] = mode
        directory = tmp_path / mode
        run = Run(config(), directory)
        with run.lock(), pytest.raises(ValueError, match='advanced|effective'):
            run.trial(run.initial[0], 7, 2)
        if mode == 'evaluate':
            assert list(directory.rglob('*.failed/failure.json'))
            assert not list(directory.rglob('receipt.json'))


def test_resource_and_manifest_guards_and_crash_wall_accounting(tmp_path):
    value = config(); value['resources']['max_updates'] = 1
    run = Run(value, tmp_path)
    with run.lock(), pytest.raises(ResourceLimit):
        run.trial(run.initial[0], 7, 2)
    assert read_json(tmp_path / 'progress.json')['attempted_updates'] == 1
    altered = deepcopy(value); altered['resources']['max_trials'] += 1
    with pytest.raises(ValueError, match='source/config'):
        with Run(altered, tmp_path).lock():
            pass
    progress = read_json(tmp_path / 'progress.json')
    progress['phase_started_utc'] = time.time() - 1000
    atomic_json(tmp_path / 'progress.json', progress)
    resumed = Run(value, tmp_path)
    with resumed.lock():
        resumed.resources()  # elapsed time is diagnostic, never a deadline
    assert read_json(tmp_path / 'progress.json')['elapsed_seconds'] > 999


def test_strict_metric_order_preserves_every_lower_error_without_tolerance(tmp_path):
    run = Run(config(), tmp_path)
    candidates = run.initial[:3]
    # All distinct FP64 scores rank strictly, even closer than the old tolerance.
    errors = {digest(p): .5 + i * .75e-6 for i, p in enumerate(reversed(candidates))}
    run.trial = lambda p, seed, budget: dict(error=errors[digest(p)], diagnostics={'aligned_better_than_off': True})
    expected = run.rank(candidates, 2)
    assert expected == run.rank(list(reversed(candidates)), 2)
    rows = next(iter(expected.values()))
    assert [row['error'] for row in rows] == sorted(errors.values())


@pytest.mark.parametrize('field', ['confirmation_errors', 'heldout_errors', 'diagnostics', 'boundary_parameters', 'status', 'overall', 'ranking_order', 'missing_candidate', 'replay', 'missing_dev'])
def test_full_nomination_and_qualification_audit_recomputed(tmp_path, field):
    execute(tmp_path)
    path = tmp_path / 'results.json'
    result = read_json(path)
    nominee = result['nominations'][0]
    if field in ('confirmation_errors', 'heldout_errors'):
        nominee[field][0] += .1
    elif field == 'diagnostics':
        nominee['diagnostics']['native_evidence'] = True
    elif field == 'boundary_parameters':
        nominee['boundary_parameters'] = ['invented']
    elif field == 'status':
        nominee['status'] = 'fixture_only'
    elif field == 'overall':
        result['overall_nomination']['candidate'] = 'other'
    elif field in ('ranking_order', 'missing_candidate'):
        rows = next(iter(result['rungs'][0]['groups'].values()))
        if field == 'ranking_order': rows.reverse()
        else: rows.pop()
    else:
        progress = read_json(tmp_path / 'progress.json')
        row = progress['trials'][next(iter(nominee['evidence']))]
        if field == 'replay':
            row['replay_proof']['observed_state_sha256'] = 'incorrect'
        else:
            key = next(k for k in row['evaluations'] if k == '00000004-dev-raw')
            del row['evaluations'][key]
        atomic_json(tmp_path / 'progress.json', progress)
    atomic_json(path, result)
    with pytest.raises(ValueError):
        audit(tmp_path)


def test_native_plan_does_not_allocate_but_execution_requires_explicit_flag(tmp_path, monkeypatch):
    import particle_sliders.parameter_search.runner as module
    backend = FixtureBackend('fixture-vector')
    cap = backend.describe(config()['backend_options'])
    cap = replace(cap, identity=dict(cap.identity, device='cuda:7'), evidence='native')
    class NativeDescription:
        def describe(self, options): return cap
        def open(self, *args, **kwargs): pytest.fail('allocation while inspecting plan')
    monkeypatch.setattr(module, 'resolve', lambda name: NativeDescription())
    assert plan(config())['capabilities']['identity']['device'] == 'cuda:7'
    with pytest.raises(ValueError, match='allow-native'):
        Run(config(), tmp_path)


@pytest.mark.parametrize('mutation', ['selection', 'decay', 'missing_decay', 'old_schema', 'wall_limit'])
def test_raw_admission_happens_before_product_backend_resolution(mutation, monkeypatch):
    import particle_sliders.parameter_search.runner as module
    value = config()
    if mutation == 'selection': value['selection'] = 'ema'
    elif mutation == 'decay': value['backend_options']['ema_decay'] = 0.
    elif mutation == 'missing_decay': del value['backend_options']['ema_decay']
    elif mutation == 'old_schema': value['schema'] = 'particle-sliders-parameter-search-v1'
    else: value['resources']['max_wall_seconds'] = 1
    if mutation == 'wall_limit':
        with pytest.raises(ValueError, match='resource'):
            plan(value)
        return
    monkeypatch.setattr(module, 'resolve', lambda name: pytest.fail('resolved a product before raw admission'))
    with pytest.raises(ValueError):
        plan(value)


def test_raw_fixture_has_no_shadows_and_rejects_historical_packets(tmp_path):
    value = config()
    _, cap, space, _ = validated(value)
    backend = FixtureBackend('fixture-vector')
    trial = backend.open(value['backend_options'], space.initial, 7, 'test')
    assert not hasattr(trial, 'ema')
    trial.step()
    path = tmp_path / 'full.json'
    trial.checkpoint(path)
    packet = read_json(path)
    assert 'ema' not in packet['state']
    trial.export(tmp_path / 'raw.json', 'raw')
    assert read_json(tmp_path / 'raw.json')['weights'] == trial._weights()
    for selection in ('ema', 'averaged'):
        with pytest.raises(ValueError, match='raw'):
            trial.responses('dev', selection)
        with pytest.raises(ValueError, match='raw'):
            trial.export(tmp_path / 'forbidden.json', selection)
    assert not (tmp_path / 'forbidden.json').exists()
    packet.pop('schema')
    atomic_json(path, packet)
    with pytest.raises(ValueError, match='checkpoint'):
        backend.open(value['backend_options'], space.initial, 7, 'test', checkpoint=path)
    with pytest.raises(ValueError):
        replace(cap, selections=('raw', 'ema'))


def test_actual_lowest_raw_sse_wins_with_failed_alignment_boundary_media_and_native_diagnostics(tmp_path, monkeypatch):
    import particle_sliders.parameter_search.runner as module
    backend = FixtureBackend('fixture-vector')
    original_describe, original_open = backend.describe, backend.open
    backend.describe = lambda options: replace(original_describe(options), trained_off=False)
    def opening(options, parameters, *args, **kwargs):
        active = original_open(options, parameters, *args, **kwargs)
        responses = active.responses
        def observe(split, selection, *, off=False):
            for chunk in responses(split, selection, off=off):
                # The boundary candidate has the smallest mean SSE (.605),
                # despite reversing one scene. Others have .64 everywhere.
                gain = (-.1 if chunk['example']['scene'] == 'room' else 1.) if parameters['lr.generator'] == .3 else .2
                chunk['student'] = [n + gain * (t - n) for t, n in zip(chunk['target'], chunk['neutral'])]
                yield chunk
        active.responses = observe
        return active
    backend.open = opening
    monkeypatch.setattr(module, 'resolve', lambda name: backend)
    run = Run(config(), tmp_path)
    with run.lock():
        report = run.execute({'invalid_optional_review': True})
    winner = next(row for row in report['nominations'] if row['candidate'] == report['overall_nomination']['candidate'])
    assert winner['parameters']['lr.generator'] == .3
    assert winner['error'] == pytest.approx(.605)
    assert winner['error'] < .64
    assert not winner['diagnostics']['training_aligned_better_than_off']
    assert not winner['diagnostics']['confirmation_aligned_better_than_off']
    assert not winner['diagnostics']['heldout_aligned_better_than_off']
    assert winner['boundary_parameters'] and not winner['diagnostics']['native_evidence']
    assert not winner['diagnostics']['trained_off_available']
    assert winner['diagnostics']['matched_media_review']['error']
    assert winner['status'] == 'confirmed_raw_metric'
    assert audit(tmp_path)['verified_evaluations'] > 0
