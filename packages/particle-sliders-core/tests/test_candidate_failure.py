"""No models/numerical frameworks: genuine controller paths over fake sessions."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
from tempfile import TemporaryDirectory
import threading
import time
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


def run_in_thread(fn, timeout=None):
    """Run fn in a thread; returns (thread, result dict). Join with a timeout."""
    result = {}
    def target():
        try:
            result['value'] = fn()
        except BaseException as exc:
            result['error'] = exc
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    if timeout is not None:
        thread.join(timeout)
    return thread, result


PROBER = r"""
import json, sys, time
from dataclasses import asdict
from pathlib import Path
from particle_sliders.candidate_failure import *
a = json.loads(sys.argv[1])
detector = QualifiedDetector(FileRecord(**a['contract']), a['instrument'], FileRecord(**a['qualification']))
def observe(source, media, native):
    p = a['probability']
    return dict(status='fail' if p >= .95 else 'pass', reason='synthetic child judgment',
                output_source=asdict(source), detector_contract=a['contract'],
                actual_instrument_source=a['instrument'], failure_probability=p)
cache = ObservationCache(a['firsts'], detector, observe)
guard = CandidateGuard(a['state'], candidate_id=a['candidate'], scientific_protocol=a['protocol'],
                       media='image', cache=cache)
def probe():
    Path(a['ready']).write_text('probing')
    while not Path(a['go']).exists():
        time.sleep(.02)
    yield FinishedOutput(FileRecord(**a['output']), 'image', dict(allowed_empty=False))
guard.check_checkpoint(FileRecord(**a['checkpoint']), probe)
"""


class InFlightAndResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.h = Harness(self.temp.name)

    def wait_for(self, condition, timeout=20):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, 'timed out')
            time.sleep(.02)

    def thread_prober(self, probability):
        h = self.h
        h.probability = probability
        prober, sibling = h.guard('cand'), h.guard('cand')  # Two instances, one candidate-wide state.
        started, release = threading.Event(), threading.Event()
        def probe():
            started.set()
            self.assertTrue(release.wait(20))
            yield h.output('a-out.txt', 'a output')
        cp = h.file('a-full.json', 'a')
        worker, outcome = run_in_thread(lambda: prober.check_checkpoint(cp, probe))
        self.assertTrue(started.wait(20))
        self.assertEqual(json.loads(sibling.path.read_text())['status'], 'unmeasured')
        waiting, seen = run_in_thread(sibling.require_update, timeout=.3)
        self.assertTrue(waiting.is_alive())  # Waits for the live prober instead of aborting.
        release.set()
        waiting.join(20)
        worker.join(20)
        self.assertFalse(waiting.is_alive() or worker.is_alive())
        return seen, outcome, sibling, cp

    def test_sibling_thread_waits_for_live_probe_then_proceeds(self):
        seen, outcome, sibling, cp = self.thread_prober(.01)
        self.assertEqual(seen, dict(value=None))
        self.assertEqual(outcome['value']['status'], 'pass')
        sibling.require_selection(cp)

    def test_sibling_thread_sees_failure_after_failing_probe(self):
        seen, outcome, sibling, cp = self.thread_prober(.99)
        self.assertIsInstance(seen['error'], CandidateFailure)
        self.assertIsInstance(outcome['error'], CandidateFailure)

    def start_child_prober(self, probability):
        h = self.h
        guard = h.guard('cand')
        args = dict(contract=asdict(h.contract), instrument=h.instrument,
                    qualification=asdict(h.qualification), firsts=str(h.root / 'firsts'),
                    state=str(guard.path), candidate='cand', protocol=guard.binding['scientific_protocol'],
                    ready=str(h.root / 'ready'), go=str(h.root / 'go'), probability=probability,
                    output=asdict(h.output('child-out.txt', 'child output').output_source),
                    checkpoint=asdict(h.file('child-full.json', 'child state')))
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))
        child = subprocess.Popen([sys.executable, '-c', PROBER, json.dumps(args)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.wait_for(lambda: (h.root / 'ready').exists() or child.poll() is not None)
        self.assertIsNone(child.poll(), child.stderr.read() if child.poll() is not None else '')
        return guard, child

    def test_sibling_waits_for_probe_in_another_real_process(self):
        guard, child = self.start_child_prober(.01)
        waiting, seen = run_in_thread(guard.require_update, timeout=.3)
        self.assertTrue(waiting.is_alive())
        (self.h.root / 'go').write_text('go')
        waiting.join(20)
        self.assertFalse(waiting.is_alive())
        self.assertEqual(seen, dict(value=None))
        self.assertEqual(child.wait(20), 0)
        self.assertEqual(json.loads(guard.path.read_text())['status'], 'pass')

    def test_killed_prober_leaves_terminal_unmeasured(self):
        guard, child = self.start_child_prober(.01)
        child.send_signal(signal.SIGKILL)
        child.wait(20)
        value = json.loads(guard.path.read_text())
        self.assertEqual([r['status'] for r in value['checkpoints'].values()], ['probe_committed'])
        reader, seen = run_in_thread(guard.require_update, timeout=20)
        self.assertFalse(reader.is_alive())
        self.assertIsInstance(seen['error'], GuardUnmeasured)
        with self.assertRaises(GuardUnmeasured):
            self.h.guard('cand').require_selection()
        self.assertEqual(self.h.calls, 0)

    def test_nested_call_from_probing_thread_raises_without_hanging(self):
        h = self.h
        guard, other = h.guard('cand'), h.guard('cand')
        nested = []
        def probe():
            for call in (guard.require_update, other.require_resume, guard.require_selection,
                         lambda: other.check_checkpoint(h.file('x.json', 'x'), lambda: [])):
                try:
                    call()
                    nested.append('allowed')
                except GuardUnmeasured:
                    nested.append('unmeasured')
            yield h.output()
        worker, outcome = run_in_thread(lambda: guard.check_checkpoint(h.file('full.json', 'f'), probe),
                                        timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(nested, ['unmeasured'] * 4)
        self.assertEqual(outcome['value']['status'], 'pass')
        other.require_update()

    def test_transient_probe_error_resolved_explicitly_then_same_checkpoint_passes(self):
        h = self.h
        guard = h.guard()
        cp = h.file('full.json', 'committed state')
        attempts = []
        def probe():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('transient render error')
            yield h.output()
        with self.assertRaises(GuardUnmeasured):
            guard.check_checkpoint(cp, probe)
        superseded = json.loads(guard.path.read_text())['checkpoints']
        with self.assertRaises(GuardUnmeasured):
            guard.check_checkpoint(cp, probe)
        self.assertEqual(attempts, [1])
        for bad in [dict(reason='', resolved_by='op'), dict(reason='r', resolved_by=' ')]:
            with self.assertRaises(ValueError):
                guard.resolve_unmeasured(**bad)
        resolution = guard.resolve_unmeasured(reason='render OOM, no observer call', resolved_by='operator')
        self.assertEqual(resolution['superseded_checkpoints'], superseded)
        self.assertEqual(resolution['previous_status'], 'unmeasured')
        reopened = h.guard()
        reopened.require_update()
        with self.assertRaises(GuardUnmeasured):
            reopened.require_selection(cp)
        self.assertEqual(reopened.check_checkpoint(cp, probe)['status'], 'pass')
        reopened.require_selection(cp)
        value = json.loads(h.guard().path.read_text())
        self.assertEqual(value['resolutions'], [resolution])
        self.assertEqual(value['resolutions'][0]['superseded_checkpoints'], superseded)
        self.assertEqual(value['schema'], 'particle-sliders-candidate-failure-state-v1')
        self.assertEqual(h.calls, 1)
        with self.assertRaises(ValueError):
            reopened.resolve_unmeasured(reason='r', resolved_by='op')  # pass is not resolvable

    def test_resolution_keeps_earlier_pass_selectable_and_never_passes_the_superseded_checkpoint(self):
        h = self.h
        guard = h.guard()
        first, second = h.file('full-1.json', 'state 1'), h.file('full-2.json', 'state 2')
        self.assertEqual(guard.check_checkpoint(first, lambda: [h.output()])['status'], 'pass')
        def broken():
            raise RuntimeError('transient render error')
        with self.assertRaises(GuardUnmeasured):
            guard.check_checkpoint(second, broken)
        with self.assertRaises(GuardUnmeasured):
            guard.require_selection(first)  # Blocked while the later probe is unknown.
        guard.resolve_unmeasured(reason='render OOM, no observer call', resolved_by='operator')
        reopened = h.guard()
        self.assertEqual(json.loads(reopened.path.read_text())['status'], 'pass')
        reopened.require_update()
        reopened.require_selection(first)
        with self.assertRaises(GuardUnmeasured):
            reopened.require_selection(second)  # No row: needs its own assessment.
        self.assertEqual(reopened.check_checkpoint(second, lambda: [h.output('second.txt', 'second output')])['status'], 'pass')
        reopened.require_selection(second)

    def test_interrupted_probe_committed_row_is_resolvable(self):
        h = self.h
        guard = h.guard()
        cp = h.file('full.json', 'committed state')
        def probe():
            raise KeyboardInterrupt
            yield
        with self.assertRaises(KeyboardInterrupt):
            guard.check_checkpoint(cp, probe)
        with self.assertRaises(GuardUnmeasured):
            h.guard().require_resume()
        resolution = h.guard().resolve_unmeasured(reason='Ctrl-C before render', resolved_by='operator')
        self.assertEqual([r['status'] for r in resolution['superseded_checkpoints'].values()],
                         ['probe_committed'])
        self.assertEqual(guard.check_checkpoint(cp, lambda: [h.output()])['status'], 'pass')

    def test_failed_and_pending_cannot_be_resolved(self):
        h = self.h
        guard = h.guard()
        with self.assertRaises(ValueError):
            guard.resolve_unmeasured(reason='r', resolved_by='op')
        h.probability = .99
        with self.assertRaises(CandidateFailure):
            guard.check_checkpoint(h.file('full.json', 's'), lambda: [h.output()])
        before = guard.path.read_text()
        with self.assertRaises(CandidateFailure):
            guard.resolve_unmeasured(reason='r', resolved_by='op')
        self.assertEqual(guard.path.read_text(), before)

    def test_uncertain_observer_verdict_is_not_rerated_by_resolution(self):
        h = self.h
        h.probability = .5
        guard = h.guard()
        cp = h.file('full.json', 's')
        with self.assertRaises(GuardUnmeasured):
            guard.check_checkpoint(cp, lambda: [h.output()])
        guard.resolve_unmeasured(reason='retry', resolved_by='op')
        h.probability = .01
        with self.assertRaises(GuardUnmeasured):
            guard.check_checkpoint(cp, lambda: [h.output()])
        self.assertEqual(h.calls, 1)
        value = json.loads(guard.path.read_text())
        self.assertEqual(value['status'], 'unmeasured')
        self.assertEqual(len(value['resolutions']), 1)

    def test_journal_without_resolutions_stays_readable(self):
        h = self.h
        guard = h.guard()
        value = json.loads(guard.path.read_text())
        self.assertNotIn('resolutions', value)
        guard.require_update()


class SessionCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.h = Harness(self.temp.name)

    def test_close_failure_does_not_replace_guard_exception(self):
        h = self.h
        h.probability = .99
        session = FakeSession()
        def close():
            raise OSError('close failed')
        session.close = close
        wrapped = GuardedSession(session, h.guard(), every=1,
            checkpoint=lambda s: h.file('full.json', str(s.game.completed_steps)),
            probe=lambda s, cp: [h.output()])
        with self.assertRaises(CandidateFailure) as caught:
            wrapped.step()
        self.assertIn('close failed', ' '.join(caught.exception.__notes__))

    def test_enter_closes_open_session_when_resume_is_refused(self):
        h = self.h
        h.probability = .99
        guard = h.guard()
        with self.assertRaises(CandidateFailure):
            guard.check_checkpoint(h.file('full.json', 's'), lambda: [h.output()])
        for failing_close in (False, True):
            session = FakeSession()
            if failing_close:
                session.close = lambda: (_ for _ in ()).throw(OSError('close failed'))
            with self.assertRaises(CandidateFailure) as caught:
                with GuardedSession(session, guard, every=1, checkpoint=lambda s: None,
                                    probe=lambda s, cp: []):
                    self.fail('entered a failed candidate')
            self.assertEqual(session.closed, not failing_close)
            self.assertEqual(bool(getattr(caught.exception, '__notes__', None)), failing_close)

    def test_file_record_streams_with_same_identity(self):
        body = os.urandom(3 * 1024 * 1024 + 17)
        path = self.h.root / 'large.bin'
        path.write_bytes(body)
        record = FileRecord.from_path(path)
        self.assertEqual((record.bytes, record.sha256), (len(body), hashlib.sha256(body).hexdigest()))
        original = Path.read_bytes
        Path.read_bytes = lambda self: (_ for _ in ()).throw(AssertionError('whole-file read'))
        try:
            self.assertEqual(FileRecord.from_path(path), record)
        finally:
            Path.read_bytes = original


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
