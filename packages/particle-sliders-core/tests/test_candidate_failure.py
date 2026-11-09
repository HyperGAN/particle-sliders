"""No models/numerical frameworks: genuine controller paths over fake sessions."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from particle_sliders.candidate_failure import (
    CandidateFailure, CandidateGuard, FileRecord, FinishedOutput, GuardedSession,
    GuardUnmeasured, ObservationCache, QualifiedDetector, open_guarded_session,
)
from particle_sliders.parameter_search.runner import Run


class Harness:
    def __init__(self, root, *, kind='probability', media=('image', 'text', 'discrete')):
        self.root = Path(root)
        self.contract = self.file('contract.json', json.dumps(dict(
            schema='SYNTHETIC_TEST_CONTRACT_NOT_ACTUAL_QUALIFICATION',
            thresholds=dict(fail_greater_than_or_equal=.95, pass_less_than_or_equal=.05))))
        self.instrument = dict(name='SYNTHETIC_TEST_OBSERVER', source='test_candidate_failure.py')
        qualification = dict(schema='Universal-catastrophic-detector-actual-qualification-v1',
            accepted=True, qualified=True, detector_contract=asdict(self.contract),
            actual_instrument_source=self.instrument, judgment_kind=kind, covered_media=list(media))
        self.qualification = self.file('qualification.json', json.dumps(qualification))
        self.detector = QualifiedDetector(self.contract, self.instrument, self.qualification, kind)
        self.calls = 0
        self.probability = .01
        self.category = False
        self.error = None
        self.cache = ObservationCache(self.root / 'firsts', self.detector, self.observe)

    def file(self, name, body):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return FileRecord.from_path(p)

    def output(self, name='output.txt', body='usable depiction', media='image', native=None):
        return FinishedOutput(self.file(name, body), media, native or dict(allowed_empty=False))

    def observe(self, source, media, native):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.detector.judgment_kind == 'probability':
            status = 'fail' if self.probability >= .95 else 'pass' if self.probability <= .05 else 'unmeasured'
            fields = dict(failure_probability=self.probability)
        else:
            status = 'unmeasured' if self.category is None else 'fail' if self.category else 'pass'
            fields = dict(failure_probability=None, categorical_failure=self.category)
        return dict(status=status, reason='synthetic fixture judgment', output_source=asdict(source),
                    detector_contract=asdict(self.contract), actual_instrument_source=self.instrument, **fields)

    def guard(self, candidate='candidate', media='image', cache=None):
        return CandidateGuard(self.root / (candidate + '.state.json'), candidate_id=candidate,
            scientific_protocol=dict(scope='SYNTHETIC_CONTROLLER_TEST', raw=True), media=media,
            cache=cache or self.cache)


class FakeSession:
    def __init__(self, total=5):
        self.game = SimpleNamespace(completed_steps=0, recipe=SimpleNamespace(total_steps=total))
        self.closed = False
        self.restores = 0

    def step(self):
        self.game.completed_steps += 1
        return dict(step=self.game.completed_steps)

    def train(self, *, stop_after=None, on_step=None):
        try:
            while self.game.completed_steps < (stop_after or self.game.recipe.total_steps):
                metrics = self.step()
                if on_step:
                    on_step(metrics)
            return metrics
        except BaseException:
            self.close()
            raise

    def restore(self, packet):
        self.restores += 1

    def close(self):
        self.closed = True


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.h = Harness(self.temp.name)

    def test_failure_stops_next_update_persists_and_blocks_open_resume_selection(self):
        h = self.h
        h.probability = .99
        guard = h.guard()
        session = FakeSession()
        checkpoints = []
        def checkpoint(s):
            cp = h.file('full-%s.json' % s.game.completed_steps, json.dumps(dict(step=s.game.completed_steps)))
            checkpoints.append(cp)
            return cp
        def probe(s, cp):
            self.assertEqual(s.game.completed_steps, 2)
            self.assertTrue(Path(cp.path).exists())  # Full state committed BEFORE any probe.
            yield h.output()
            self.fail('requested another finished output after detected catastrophe')
        wrapped = GuardedSession(session, guard, checkpoint=checkpoint, probe=probe, every=2)
        with self.assertRaises(CandidateFailure):
            wrapped.train(stop_after=5)
        self.assertEqual(session.game.completed_steps, 2)
        self.assertTrue(session.closed)
        self.assertEqual(json.loads(Path(checkpoints[0].path).read_text())['step'], 2)
        restarted = h.guard()
        opened = []
        with self.assertRaises(CandidateFailure):
            open_guarded_session(lambda: opened.append(True), restarted,
                                 checkpoint=checkpoint, probe=probe, every=2)
        self.assertEqual(opened, [])
        with self.assertRaises(CandidateFailure):
            wrapped.restore({})
        self.assertEqual(session.restores, 0)
        with self.assertRaises(CandidateFailure):
            restarted.require_selection(checkpoints[0])
        self.assertEqual(h.calls, 1)

    def test_unknown_is_unmeasured_not_model_failed_and_blocks_extra_update(self):
        h = self.h
        h.probability = .5
        session = FakeSession()
        wrapped = GuardedSession(session, h.guard(), every=1,
            checkpoint=lambda s: h.file('full.json', str(s.game.completed_steps)),
            probe=lambda s, cp: [h.output()])
        with self.assertRaises(GuardUnmeasured):
            wrapped.train(stop_after=5)
        self.assertEqual(session.game.completed_steps, 1)
        self.assertEqual(json.loads(h.guard().path.read_text())['status'], 'unmeasured')
        with self.assertRaises(GuardUnmeasured):
            wrapped.step()
        self.assertEqual(session.game.completed_steps, 1)

    def test_allowed_empty_is_not_automatically_failed(self):
        h = self.h
        output = h.output(body='', media='text', native=dict(allowed_empty=True))
        guard = h.guard(media='text')
        cp = h.file('full.json', 'committed full fake state')
        guard.check_checkpoint(cp, lambda: [output])
        guard.require_selection(cp)
        self.assertEqual(h.calls, 1)
        guard.check_checkpoint(cp, lambda: self.fail('repeated finished-output probe'))
        self.assertEqual(h.calls, 1)

    def test_no_qualification_then_first_actual_call_and_retained_no_call_diagnostic(self):
        h = self.h
        absent = QualifiedDetector(h.contract, h.instrument)
        first = ObservationCache(h.root / 'firsts', absent, h.observe)
        output = h.output()
        result = first.assess(output)
        self.assertEqual(result['status'], 'unmeasured')
        self.assertFalse(result['call_performed'])
        self.assertEqual(h.calls, 0)
        self.assertEqual(len(list((h.root / 'firsts').rglob('observation.json'))), 0)
        self.assertEqual(len(list((h.root / 'firsts').rglob('no-call-*.json'))), 1)
        self.assertEqual(h.cache.assess(output)['status'], 'pass')
        self.assertEqual(h.calls, 1)
        self.assertEqual(first.assess(output)['status'], 'unmeasured')
        self.assertEqual(h.calls, 1)

    def test_missing_qualification_or_uncovered_media_stops_before_product_open(self):
        h = self.h
        for media, cache in [('audio', h.cache), ('image', ObservationCache(h.root / 'none',
                QualifiedDetector(h.contract, h.instrument), h.observe))]:
            guard = h.guard('no-' + media, media=media, cache=cache)
            opened = []
            with self.assertRaises(GuardUnmeasured):
                open_guarded_session(lambda: opened.append(True), guard, every=1,
                                     checkpoint=lambda s: None, probe=lambda s, cp: [])
            self.assertEqual(opened, [])
        self.assertEqual(h.calls, 0)

    def test_actual_output_bytes_verified_before_observer_and_can_recover_before_first_call(self):
        h = self.h
        output = h.output()
        Path(output.output_source.path).write_text('wrong bytes')
        self.assertEqual(h.cache.assess(output)['status'], 'unmeasured')
        self.assertEqual(h.calls, 0)
        Path(output.output_source.path).write_text('usable depiction')
        self.assertEqual(h.cache.assess(output)['status'], 'pass')
        self.assertEqual(h.calls, 1)

    def test_actual_observer_error_is_durable_and_never_rerated(self):
        h = self.h
        h.error = RuntimeError('fixture instrument unavailable')
        output = h.output()
        first = h.cache.assess(output)
        h.error = None
        self.assertEqual(first['status'], 'unmeasured')
        self.assertEqual(h.cache.assess(output), first)
        self.assertEqual(h.calls, 1)

    def test_committed_ambiguous_call_is_not_repeated(self):
        h = self.h
        h.error = KeyboardInterrupt()
        output = h.output()
        with self.assertRaises(KeyboardInterrupt):
            h.cache.assess(output)
        h.error = None
        result = h.cache.assess(output)
        self.assertEqual(result['status'], 'unmeasured')
        self.assertIsNone(result['failure_probability'])
        self.assertIn('no automatic replay', result['reason'])
        self.assertEqual(h.calls, 1)

    def test_categorical_bool_never_masquerades_as_probability_or_integer_confidence(self):
        h = Harness(self.h.root / 'categorical', kind='categorical')
        for n, category, status in [(0, False, 'pass'), (1, True, 'fail'), (2, None, 'unmeasured'),
                                    (3, 0, 'unmeasured'), (4, 1, 'unmeasured')]:
            h.category = category
            result = h.cache.assess(h.output('out-%s' % n, body='distinct fixture %s' % n))
            self.assertEqual(result['status'], status)
            self.assertIsNone(result['failure_probability'])
        self.assertEqual(h.calls, 5)

    def test_identical_payload_at_another_path_reuses_first_observation(self):
        h = self.h
        first = h.cache.assess(h.output('first.txt'))
        h.probability = .99
        second = h.cache.assess(h.output('second.txt'))
        self.assertEqual(second, first)
        self.assertEqual(h.calls, 1)

    def test_missing_assessment_cannot_resume_a_saved_checkpoint(self):
        h = self.h
        cp = h.file('unassessed-full.json', 'already completed state')
        opened = []
        with self.assertRaises(GuardUnmeasured):
            open_guarded_session(lambda: opened.append(True), h.guard(), checkpoint=lambda s: cp,
                                 probe=lambda s, c: [], every=1, resume_checkpoint=cp)
        self.assertEqual(opened, [])

    def test_import_surface_does_not_load_numerical_frameworks(self):
        # Other tests may legitimately load numerical libraries in the test
        # runner; prove this module's import boundary in its own process.
        source = str(Path(__file__).parents[1] / 'src')
        code = ("import sys; sys.path.insert(0, " + repr(source) + "); "
                "import particle_sliders.candidate_failure; "
                "import particle_sliders.parameter_search.runner; "
                "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules")
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_restore_requires_the_exact_assessed_checkpoint_not_another_packet(self):
        h = self.h
        guard = h.guard()
        assessed = h.file('assessed-full.json', 'assessed full state')
        unassessed = h.file('other-full.json', 'different unassessed state')
        guard.check_checkpoint(assessed, lambda: [h.output()])
        session = FakeSession()
        wrapped = GuardedSession(session, guard, checkpoint=lambda s: assessed,
                                 probe=lambda s, c: [h.output()], every=1)
        loaded = []
        with self.assertRaises(GuardUnmeasured):
            wrapped.restore(unassessed, load_packet=lambda p: loaded.append(p))
        self.assertEqual(loaded, [])
        self.assertEqual(session.restores, 0)
        with self.assertRaises(GuardUnmeasured):
            wrapped.restore({'unbound': 'packet'})
        self.assertEqual(session.restores, 0)
        wrapped.restore(assessed, load_packet=lambda p: dict(exact_source=p))
        self.assertEqual(session.restores, 1)

    def test_reject_candidate_or_expected_label_leakage_to_observer(self):
        for native in [dict(lr=.036), dict(nested=dict(expected_label='fail'))]:
            with self.assertRaises(ValueError):
                self.h.output(native=native)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.h = Harness(self.temp.name)

    def test_rank_excludes_failed_unknown_keeps_other_candidate_and_metric(self):
        run = Run.__new__(Run)
        run.config = dict(search=dict(training_seeds=[1]))
        run.cap = SimpleNamespace(identity={})
        run.space = SimpleNamespace(cohort=lambda identity, p: 'same', simplicity=lambda p: ())
        def trial(parameters, seed, budget):
            if parameters['id'] == 'failed':
                raise CandidateFailure('retained catastrophic failure')
            if parameters['id'] == 'unknown':
                raise GuardUnmeasured('unknown observer')
            return dict(error=.123, diagnostics=dict(aligned_better_than_off=True))
        run.trial = trial
        ranked = run.rank([dict(id='failed'), dict(id='unknown'), dict(id='healthy')], 32)
        self.assertEqual([r['parameters']['id'] for r in ranked['same']], ['healthy'])
        self.assertEqual(ranked['same'][0]['error'], .123)  # No combined/gated score.

    def test_native_vectors_are_not_used_as_finished_media(self):
        h = self.h
        run = Run.__new__(Run)
        run.root = h.root
        cp = h.file('committed.full', 'committed full fake state')
        run.config = dict(selection='raw')
        run.failure_guard = lambda row: h.guard('no-media')
        active = SimpleNamespace(completed_steps=1,
            responses=lambda *a, **kw: self.fail('native vectors are not finished outputs'))
        with self.assertRaises(GuardUnmeasured):
            run.check_finished_outputs(active, {}, dict(path='committed.full', sha256=cp.sha256))
        self.assertEqual(h.calls, 0)
        self.assertEqual(json.loads(h.guard('no-media').path.read_text())['status'], 'unmeasured')

    def test_no_eligible_candidates_produces_no_nomination(self):
        run = Run.__new__(Run)
        run.root = self.h.root / 'empty-search'
        run.root.mkdir()
        run.config = dict(search=dict(budgets=[32], refinement_candidates=0, confirmation_seeds=[]))
        run.initial = [dict(rate=.01)]
        run.rank = lambda candidates, budget: {}
        run.space = SimpleNamespace(simplicity=lambda p: ())
        run.identity, run.progress = 'fixture', {}
        run.cap = SimpleNamespace(evidence='fixture')
        run.config['selection'] = 'raw'
        result = run.execute()
        self.assertIsNone(result['overall_nomination'])
        self.assertEqual(result['nominations'], [])

    def test_real_trial_path_stops_after_committed_first_update_and_never_reopens(self):
        h = self.h
        h.probability = .99
        run = Run.__new__(Run)
        run.root = h.root / 'search'
        run.root.mkdir()
        run.config = dict(resources=dict(max_trials=10, checkpoint_every=1), selection='raw')
        run.progress = dict(trials={}, initials={}, attempted_updates=0, completed_update_calls=0)
        run.cap = SimpleNamespace(identity={}, selections=('raw',))
        run.space = SimpleNamespace(cohort=lambda identity, p: 'same')
        run.failure_guard = lambda row: h.guard(row['candidate'])
        run.persist = lambda: None
        run.resources = lambda **kwargs: None
        run.verify = lambda active, parameters: None
        opened = []
        updates = []
        class Trial:
            initial_state_sha256 = 'same-initial'
            def __init__(self, checkpoint=None):
                self.completed_steps = json.loads(Path(checkpoint).read_text())['step'] if checkpoint else 0
            def step(self):
                self.completed_steps += 1
                updates.append(self.completed_steps)
            def state_digest(self):
                return hashlib.sha256(str(self.completed_steps).encode()).hexdigest()
            def checkpoint(self, path):
                Path(path).write_text(json.dumps(dict(step=self.completed_steps)))
            def effective_parameters(self):
                return dict(rate=.01)
            def optimizer_groups(self):
                return []
            def finished_outputs(self, **kwargs):
                yield h.output()
            def close(self):
                pass
        def open_trial(parameters, seed, cohort, checkpoint=None):
            opened.append(checkpoint)
            return Trial(checkpoint)
        run.open_trial = open_trial
        with self.assertRaises(CandidateFailure):
            run.trial(dict(rate=.01), 1, 32)
        row = next(iter(run.progress['trials'].values()))
        row['evaluations']['old_scientific_metric'] = dict(error=.00001)
        self.assertEqual(updates, [1])
        self.assertEqual(row['outcome'], 'failed')
        self.assertTrue((run.root / row['checkpoints']['1']['path']).exists())
        with self.assertRaises(CandidateFailure):
            run.trial(dict(rate=.01), 1, 64)
        self.assertEqual(len(opened), 1)
        self.assertEqual(updates, [1])
        self.assertEqual(row['evaluations']['old_scientific_metric'], dict(error=.00001))


if __name__ == '__main__':
    unittest.main()
