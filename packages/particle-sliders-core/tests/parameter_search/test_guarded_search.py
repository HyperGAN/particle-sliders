"""Real fixture Run/execute/audit with an explicit catastrophic-output guard."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from particle_sliders.candidate_failure import (
    CandidateFailure, CandidateGuard, FileRecord, FinishedOutput, GuardUnmeasured, ObservationCache, QualifiedDetector,
)
from particle_sliders.parameter_search import __main__ as cli
from particle_sliders.parameter_search.fixture import FixtureTrial
from particle_sliders.parameter_search.metric import digest
from particle_sliders.parameter_search.runner import Run, atomic_json, audit, read_json


REPO = Path(__file__).resolve().parents[4]


def config(cohorts=False, training_seeds=(7,)):
    value = read_json(REPO / 'examples/parameter_search/vector.json')
    for name in ('lr.particle', 'regularizer.l2_alpha'):
        value['fixed'][name] = value['space'].pop(name)['initial']
    count = 6
    if cohorts:
        value['space']['particles.population'] = dict(values=[2, 4], initial=value['fixed'].pop('particles.population'))
        count = 8
    value['search'].update(initial_candidates=count, refinement_candidates=1, budgets=[2, 4], keep_per_cohort=2,
                           training_seeds=list(training_seeds), confirmation_seeds=[17])
    return value


class Harness:
    """Synthetic qualified observer; judge(info) returns a failure probability."""
    def __init__(self, root):
        self.root = Path(root)
        self.contract = self.file('contract.json', json.dumps(dict(schema='SYNTHETIC_TEST_CONTRACT',
            thresholds=dict(fail_greater_than_or_equal=.95, pass_less_than_or_equal=.05))))
        self.instrument = dict(name='SYNTHETIC_TEST_OBSERVER', source='test_guarded_search.py')
        self.qualification = self.file('qualification.json', json.dumps(dict(
            schema='Universal-catastrophic-detector-actual-qualification-v1', accepted=True, qualified=True,
            detector_contract=asdict(self.contract), actual_instrument_source=self.instrument,
            judgment_kind='probability', covered_media=['text'])))
        self.detector = QualifiedDetector(self.contract, self.instrument, self.qualification, 'probability')
        self.cache = ObservationCache(self.root / 'firsts', self.detector, self.observe)
        self.judge = lambda info: .01
        self.observed = []

    def file(self, name, body):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        return FileRecord.from_path(path)

    def observe(self, source, media, native):
        info = json.loads(Path(source.path).read_text())
        self.observed.append(info)
        p = self.judge(info)
        return dict(status='fail' if p >= .95 else 'pass' if p <= .05 else 'unmeasured', failure_probability=p,
                    reason='synthetic fixture judgment', output_source=asdict(source),
                    detector_contract=asdict(self.contract), actual_instrument_source=self.instrument)

    def guard(self, row):
        return CandidateGuard(self.root / 'guards' / (row['candidate'] + '.json'), candidate_id=row['candidate'],
            scientific_protocol=dict(scope='SYNTHETIC_SEARCH_TEST', raw=True), media='text', cache=self.cache)

    def state(self, candidate):
        return json.loads((self.root / 'guards' / (candidate + '.json')).read_text())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'guard')
    def finished_outputs(self, split, selection):
        info = dict(candidate=digest(self.parameters)[:16], seed=self.seed, steps=self.completed_steps,
                    state=self.state_digest())
        yield FinishedOutput(h.file('outputs/' + digest(info) + '.json', json.dumps(info)), 'text', dict(allowed_empty=False))
    monkeypatch.setattr(FixtureTrial, 'finished_outputs', finished_outputs, raising=False)
    return h


def execute(directory, value, guard=None):
    run = Run(value, directory, failure_guard=guard)
    with run.lock():
        return run, run.execute()


def ids(report, rung=0):
    return [r['candidate'] for rows in report['rungs'][rung]['groups'].values() for r in rows]


@pytest.mark.parametrize('kind', ['catastrophic_output', 'unmeasured_output_guard'])
def test_guarded_initial_exclusion_executes_and_audits(tmp_path, harness, kind):
    value = config(training_seeds=(7, 8))
    _, reference = execute(tmp_path / 'reference', value)
    leaders = {rows[0]['candidate'] for rows in reference['rungs'][0]['groups'].values()}
    run = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    victim = next(digest(p)[:16] for p in run.initial if digest(p)[:16] not in leaders)
    # Seed 7 passes; the candidate-wide guard trips on the second training seed.
    harness.judge = lambda info: (.99 if kind == 'catastrophic_output' else .5) if (info['candidate'], info['seed']) == (victim, 8) else .01
    with run.lock():
        report = run.execute()
    assert victim not in ids(report)
    assert ids(report) == [c for c in ids(reference) if c != victim]
    assert [r for rows in report['rungs'][0]['groups'].values() for r in rows] == [
        r for rows in reference['rungs'][0]['groups'].values() for r in rows if r['candidate'] != victim]
    progress = read_json(tmp_path / 'run' / 'progress.json')
    assert progress['trials'][f'{victim}-seed-8']['failure']['kind'] == kind
    # The refinement pass re-ranks initial candidates; seed 7 is then refused
    # before reopening and records the same retained guard outcome.
    assert progress['trials'][f'{victim}-seed-7']['failure']['kind'] == kind
    assert progress['trials'][f'{victim}-seed-7']['evaluations']
    result = audit(tmp_path / 'run')
    assert result['diagnostics']['guard_exclusions'] == [dict(candidate=victim, rung=2, kind=kind, trial=f'{victim}-seed-7')]
    assert audit(tmp_path / 'reference')['diagnostics']['guard_exclusions'] == []


def test_all_pass_guard_keeps_unguarded_ranking_and_nominations(tmp_path, harness):
    value = config(cohorts=True)
    _, reference = execute(tmp_path / 'reference', value)
    _, guarded = execute(tmp_path / 'run', value, harness.guard)
    assert harness.observed
    assert guarded['rungs'] == reference['rungs']
    assert guarded['nominations'] == reference['nominations']
    assert guarded['overall_nomination'] == reference['overall_nomination']
    assert audit(tmp_path / 'run')['diagnostics']['guard_exclusions'] == []


def test_explicit_resolution_readmits_a_transiently_unmeasured_candidate(tmp_path, harness, monkeypatch):
    value = config(training_seeds=(7, 8))
    _, reference = execute(tmp_path / 'reference', value)
    leaders = {rows[0]['candidate'] for rows in reference['rungs'][0]['groups'].values()}
    run = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    victim = next(digest(p)[:16] for p in run.initial if digest(p)[:16] not in leaders)
    render, failures = FixtureTrial.finished_outputs, []
    def flaky(self, split, selection):
        # One render error on the second seed, before any observer call.
        if (digest(self.parameters)[:16], self.seed) == (victim, 8) and not failures:
            failures.append(self.completed_steps)
            raise RuntimeError('transient render error')
        return render(self, split, selection)
    monkeypatch.setattr(FixtureTrial, 'finished_outputs', flaky)
    with run.lock():
        report = run.execute()
    assert len(failures) == 1 and victim not in ids(report)
    row = read_json(tmp_path / 'run' / 'progress.json')['trials'][f'{victim}-seed-8']
    assert (row['outcome'], row['failure']['kind']) == ('unmeasured', 'unmeasured_output_guard')
    assert harness.state(victim)['status'] == 'unmeasured'
    assert audit(tmp_path / 'run')['diagnostics']['guard_exclusions']
    # Nothing re-admits it automatically: the identical command excludes it again.
    _, again = execute(tmp_path / 'run', value, harness.guard)
    assert victim not in ids(again) and len(failures) == 1
    harness.guard(row).resolve_unmeasured(reason='render error before any observer call', resolved_by='test operator')
    # Seed 7's earlier passing checkpoints stay selectable; seed 8's superseded
    # checkpoint is probed again before any further update.
    _, resolved = execute(tmp_path / 'run', value, harness.guard)
    assert resolved['rungs'] == reference['rungs']
    progress = read_json(tmp_path / 'run' / 'progress.json')
    assert progress['trials'][f'{victim}-seed-8']['outcome'] == 'completed'
    assert 'failure' not in progress['trials'][f'{victim}-seed-8']
    assert harness.state(victim)['status'] == 'pass' and len(harness.state(victim)['resolutions']) == 1
    assert audit(tmp_path / 'run')['diagnostics']['guard_exclusions'] == []


def test_excluded_finalist_has_no_nomination_and_reexecution_audits(tmp_path, harness):
    value = config(cohorts=True)
    _, reference = execute(tmp_path / 'reference', value)
    assert len(reference['nominations']) == 2
    finalist, other = reference['nominations']
    harness.judge = lambda info: .99 if (info['candidate'], info['seed']) == (finalist['candidate'], 17) else .01
    run, report = execute(tmp_path / 'run', value, harness.guard)
    assert report['rungs'] == reference['rungs']
    assert [(n['candidate'], n['cohort']) for n in report['nominations']] == [(other['candidate'], other['cohort'])]
    assert report['overall_nomination']['candidate'] == other['candidate']
    result = audit(tmp_path / 'run')
    assert result['diagnostics']['guard_exclusions'] == [dict(candidate=finalist['candidate'], rung='nomination',
        kind='catastrophic_output', trial=f"{finalist['candidate']}-seed-17")]
    # Idempotent re-execution: the failed guard now excludes it from every rung.
    _, again = execute(tmp_path / 'run', value, harness.guard)
    assert finalist['candidate'] not in ids(again) + ids(again, 1)
    assert finalist['candidate'] not in {n['candidate'] for n in again['nominations']}
    exclusions = audit(tmp_path / 'run')['diagnostics']['guard_exclusions']
    assert dict(candidate=finalist['candidate'], rung=2, kind='catastrophic_output',
                trial=f"{finalist['candidate']}-seed-7") in exclusions


def test_all_candidates_excluded_audits_without_nomination(tmp_path, harness):
    harness.judge = lambda info: .99
    _, report = execute(tmp_path / 'run', config(), harness.guard)
    assert report['nominations'] == [] and report['overall_nomination'] is None
    assert all(rung['groups'] == {} for rung in report['rungs'])
    result = audit(tmp_path / 'run')
    assert len(result['diagnostics']['guard_exclusions']) == 6
    assert {e['rung'] for e in result['diagnostics']['guard_exclusions']} == {2}
    results = read_json(tmp_path / 'run' / 'results.json')
    results['overall_nomination'] = dict(candidate='invented')
    atomic_json(tmp_path / 'run' / 'results.json', results)
    with pytest.raises(ValueError, match='overall'):
        audit(tmp_path / 'run')


@pytest.mark.parametrize('tamper', ['drop_healthy_row', 'drop_failure', 'technical_failure', 'drop_nomination', 'foreign_nomination'])
def test_guarded_audit_stays_strict(tmp_path, harness, tamper):
    value = config(cohorts=True)
    run = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    victim = digest(run.initial[-1])[:16]
    harness.judge = lambda info: .99 if info['candidate'] == victim else .01
    with run.lock():
        run.execute()
    assert audit(tmp_path / 'run')['diagnostics']['guard_exclusions']
    results, progress = read_json(tmp_path / 'run' / 'results.json'), read_json(tmp_path / 'run' / 'progress.json')
    row = progress['trials'][f'{victim}-seed-7']
    if tamper == 'drop_healthy_row':
        next(iter(results['rungs'][0]['groups'].values())).pop()
    elif tamper == 'drop_failure':
        del row['failure']
    elif tamper == 'technical_failure':
        row['failure']['kind'] = 'technical'
    elif tamper == 'drop_nomination':
        results['nominations'].pop()
    else:
        results['nominations'].append(dict(results['nominations'][0], cohort='invented'))
    atomic_json(tmp_path / 'run' / 'results.json', results)
    atomic_json(tmp_path / 'run' / 'progress.json', progress)
    with pytest.raises(ValueError):
        audit(tmp_path / 'run')


def test_unguarded_missing_evidence_is_a_value_error(tmp_path):
    execute(tmp_path, config())
    progress = read_json(tmp_path / 'progress.json')
    results = read_json(tmp_path / 'results.json')
    trial = f"{ids(results)[0]}-seed-7"
    del progress['trials'][trial]
    atomic_json(tmp_path / 'progress.json', progress)
    with pytest.raises(ValueError, match='missing retained trial evidence'):
        audit(tmp_path)


def test_resume_probes_checkpoint_persisted_before_its_probe(tmp_path, harness, monkeypatch):
    value = config()
    run = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    parameters = run.initial[0]
    candidate = digest(parameters)[:16]
    original = Run.check_finished_outputs
    def killed(self, active, row, receipt):
        if active.completed_steps == 1:
            raise KeyboardInterrupt('killed after receipt persist, before probe commit')
        return original(self, active, row, receipt)
    monkeypatch.setattr(Run, 'check_finished_outputs', killed)
    with run.lock(), pytest.raises(KeyboardInterrupt):
        run.trial(parameters, 7, 2)
    monkeypatch.setattr(Run, 'check_finished_outputs', original)
    assert read_json(tmp_path / 'run' / 'progress.json')['trials'][f'{candidate}-seed-7']['latest'] == '1'
    assert harness.state(candidate)['status'] == 'pending' and harness.observed == []
    events = []
    judge = harness.judge
    harness.judge = lambda info: events.append(('probe', info['steps'])) or judge(info)
    resumed = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    update = resumed.update
    resumed.update = lambda active, p: (events.append(('update', active.completed_steps)), update(active, p))
    with resumed.lock():
        result = resumed.trial(parameters, 7, 2)
    assert result['binding']['completed_steps'] == 2
    assert events[0] == ('probe', 1)  # assessed before any further update
    assert harness.state(candidate)['status'] == 'pass'
    row = read_json(tmp_path / 'run' / 'progress.json')['trials'][f'{candidate}-seed-7']
    assert row['outcome'] == 'completed' and 'failure' not in row


@pytest.mark.parametrize('verdict', ['failed', 'unmeasured'])
def test_failed_or_unmeasured_guard_still_refuses_before_checksum_or_open(tmp_path, harness, monkeypatch, verdict):
    value = config()
    run = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    parameters = run.initial[0]
    if verdict == 'failed':
        harness.judge = lambda info: .99
    else:
        def interrupted(info):
            raise KeyboardInterrupt('killed while the committed probe was running')
        harness.judge = interrupted
    with run.lock(), pytest.raises(CandidateFailure if verdict == 'failed' else KeyboardInterrupt):
        run.trial(parameters, 7, 2)
    assert harness.state(digest(parameters)[:16])['status'] == verdict
    harness.judge = lambda info: .01
    import particle_sliders.parameter_search.runner as module
    resumed = Run(value, tmp_path / 'run', failure_guard=harness.guard)
    opened, hashed = [], []
    resumed.open_trial = lambda *a, **kw: opened.append(a)
    file_hash = module.file_hash
    monkeypatch.setattr(module, 'file_hash', lambda p: hashed.append(p) or file_hash(p))
    with resumed.lock(), pytest.raises(CandidateFailure if verdict == 'failed' else GuardUnmeasured):
        resumed.trial(parameters, 7, 2)
    assert opened == [] and hashed == []


def test_cli_retains_resolved_interruption_receipt(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config()))
    out = tmp_path / 'run'
    original = Run.execute
    def interrupted(self, review=None):
        raise RuntimeError('simulated lease expiry')
    monkeypatch.setattr(Run, 'execute', interrupted)
    assert cli.main(['run', '--config', str(path), '--out', str(out)]) == 2
    receipt = (out / 'interruption.json').read_bytes()
    assert json.loads(receipt)['no_confirmed_result'] is True
    monkeypatch.setattr(Run, 'execute', original)
    assert cli.main(['run', '--config', str(path), '--out', str(out)]) == 0
    assert not (out / 'interruption.json').exists()
    resolved = list(out.glob('interruption-resolved-*.json'))
    assert len(resolved) == 1 and resolved[0].read_bytes() == receipt
    assert cli.main(['run', '--config', str(path), '--out', str(out)]) == 0
    assert list(out.glob('interruption*.json')) == resolved
    assert cli.main(['audit', '--out', str(out)]) == 0
    capsys.readouterr()
