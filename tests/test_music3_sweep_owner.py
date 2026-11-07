import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from particle_sliders.execution.process_identity import capture_process_identity, process_liveness
from scripts import sweep_music3_critic as sweep

CFG = {'name': 'fixture', 'critic': 'mlp'}


def status(run, name, completed=0):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'status.json').write_text(json.dumps({'status': name, 'completed': completed}))


def owner_record(process, code=None):
    return dict(schema=sweep.OWNER_SCHEMA, process=process,
                controller=capture_process_identity(), exit_code=code)


def dead_owner():
    owner = capture_process_identity()
    return dict(owner, start=str(int(owner['start']) + 1))


@pytest.mark.parametrize('training_marker', [True, False])
def test_incomplete_attempt_is_retained_and_not_relaunched(tmp_path, monkeypatch, training_marker):
    run = sweep.run_dir(tmp_path, CFG)
    status(run, 'training' if training_marker else 'starting')
    capture = run / 'captured-source.py'
    capture.write_bytes(b'original captured source')
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(sweep.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not allocate a trainer'))
    with pytest.raises(RuntimeError, match='retain.*immutable'):
        sweep.run_one(CFG, steps=10, root=tmp_path, gpu=0)
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}


@pytest.mark.parametrize('kind', ['dead', 'foreign', 'corrupt', 'missing'])
def test_stale_or_unverified_training_never_waits_or_deletes(tmp_path, monkeypatch, kind):
    run = sweep.run_dir(tmp_path, CFG)
    status(run, 'training')
    owner = dead_owner() if kind == 'dead' else capture_process_identity()
    if kind == 'foreign':
        owner['host'] += '-elsewhere'
    if kind == 'corrupt':
        owner['boot'] = 'bad'
    if kind != 'missing':
        sweep.write_owner(run, owner_record(owner))
    snapshot = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(sweep.time, 'sleep', lambda seconds: pytest.fail('must not wait for unproven work'))
    with pytest.raises(RuntimeError):
        sweep.prepare_attempt(run, 10)
    assert snapshot == {p.name: p.read_bytes() for p in run.iterdir()}


def test_wait_does_not_adopt_terminal_marker_before_child_exit_and_reap(tmp_path, monkeypatch):
    original_wait = sweep.wait_for_owner
    monkeypatch.setattr(sweep, "wait_for_owner", lambda run: original_wait(run, poll_seconds=.01))
    run = tmp_path / 'run'
    status(run, 'complete', 10)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.4)'])
    receipt = owner_record(capture_process_identity(proc.pid))
    sweep.write_owner(run, receipt)
    done = threading.Event()
    errors = []
    def wait():
        try:
            assert sweep.prepare_attempt(run, 10) == (10, True)
        except BaseException as error:
            errors.append(error)
        done.set()
    thread = threading.Thread(target=wait)
    thread.start()
    try:
        time.sleep(.1)
        assert proc.poll() is None and not done.is_set()
        proc.wait()
        # The child is gone, but a live controller has not certified its exit.
        time.sleep(.1)
        assert not done.is_set()
        sweep.write_owner(run, dict(receipt, exit_code=0))
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        sweep.write_owner(run, dict(receipt, exit_code=0))
        thread.join(timeout=6)
    assert done.is_set() and not errors


def test_dead_controller_cannot_certify_terminal_exit(tmp_path):
    run = tmp_path / 'run'
    status(run, 'complete', 10)
    record = owner_record(dead_owner())
    record['controller'] = dead_owner()
    sweep.write_owner(run, record)
    with pytest.raises(RuntimeError, match='certify'):
        sweep.prepare_attempt(run, 10)


@pytest.mark.parametrize('code', [1, -9])
def test_nonzero_reaped_exit_is_not_reused_or_rerolled(tmp_path, code):
    run = tmp_path / 'run'
    status(run, 'complete', 10)
    sweep.write_owner(run, owner_record(dead_owner(), code))
    with pytest.raises(RuntimeError, match='exited'):
        sweep.prepare_attempt(run, 10)


def test_legacy_terminal_reuse_and_explicit_checkpoint_resume_are_retained(tmp_path):
    run = tmp_path / 'run'
    status(run, 'complete', 10)
    assert sweep.prepare_attempt(run, 10) == (10, True)
    with pytest.raises(RuntimeError, match='immutable'):
        sweep.prepare_attempt(run, 20)
    (run / 'state.pt').write_bytes(b'full checkpoint loader admits actual schema')
    assert sweep.prepare_attempt(run, 20) == (10, False)
    status(run, "paused", 10)
    assert sweep.prepare_attempt(run, 20) == (10, False)


def test_live_legacy_directory_is_not_renamed(tmp_path):
    old = tmp_path / 'runs' / 'fixture-s10'
    status(old, 'training')
    with pytest.raises(RuntimeError):
        sweep.migrate_legacy_dirs(tmp_path, CFG)
    assert old.is_dir() and not sweep.run_dir(tmp_path, CFG).exists()


def test_extension_and_screen_share_owner_launch_and_exit_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, 'argv_for', lambda *a, **k: (
        [sys.executable, '-c', 'import time; time.sleep(.15)'], dict(os.environ)))
    attempt = sweep.start_attempt(CFG, steps=10, root=tmp_path, gpu=0)
    run = attempt['save_dir']
    receipt = sweep.read_owner(run)
    assert process_liveness(receipt['process']) == 'live'
    assert receipt['exit_code'] is None
    result = sweep.finish_attempt(attempt)
    assert result['returncode'] == 0
    assert sweep.read_owner(run)['exit_code'] == 0
    assert process_liveness(receipt['process']) == 'dead'
    # The exact extension path uses the same launch/reap machinery.
    other = dict(CFG, name='extension')
    assert sweep.run_one(other, steps=20, root=tmp_path, gpu=0)['returncode'] == 0
    assert sweep.read_owner(sweep.run_dir(tmp_path, other))['exit_code'] == 0


def test_parallel_preparation_cannot_overwrite_an_owner(tmp_path):
    with sweep.launch_lock(tmp_path, CFG):
        with pytest.raises(RuntimeError, match='Another controller'):
            with sweep.launch_lock(tmp_path, CFG):
                pytest.fail('must not obtain a second lock')


def test_immediate_owned_exit_can_be_certified_without_live_only_capture(tmp_path, monkeypatch):
    actual_capture = sweep.capture_process_identity
    def capture_after_exit(pid=None):
        if pid is not None:
            # Do not reap: readable zombie identity still proves which child ran.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                proof = actual_capture(pid)
                if proof is not None and process_liveness(proof) == 'dead':
                    return proof
                time.sleep(.005)
            pytest.fail('owned immediate child did not exit')
        return actual_capture()
    monkeypatch.setattr(sweep, 'capture_process_identity', capture_after_exit)
    monkeypatch.setattr(sweep, 'argv_for', lambda *a, **k: ([sys.executable, '-c', 'pass'], dict(os.environ)))
    assert sweep.run_one(CFG, steps=10, root=tmp_path, gpu=0)['returncode'] == 0
    assert sweep.read_owner(sweep.run_dir(tmp_path, CFG))['exit_code'] == 0


def test_first_board_failure_still_reaps_every_owned_screen_child(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, 'SCREEN', [CFG, dict(CFG, name='second')])
    monkeypatch.setattr(sweep, 'argv_for', lambda *a, **k: (
        [sys.executable, '-c', 'import time; time.sleep(.15)'], dict(os.environ)))
    monkeypatch.setattr(sys, 'argv', ['sweep', '--root', str(tmp_path), '--gpus', '0', '1'])
    actual_start = sweep.start_attempt
    children = []
    def start(*args, **kwargs):
        attempt = actual_start(*args, **kwargs)
        children.append(attempt)
        return attempt
    monkeypatch.setattr(sweep, 'start_attempt', start)
    writes = []
    def board(*args):
        writes.append(True)
        if len(writes) == 1:
            raise OSError('board storage failed')
    monkeypatch.setattr(sweep, 'write_board', board)
    with pytest.raises(OSError, match='board storage failed'):
        sweep.main()
    assert len(children) == 2 and len(writes) == 2
    assert all(child['proc'].returncode == 0 and child['handle'].closed for child in children)
    assert all(sweep.read_owner(child['save_dir'])['exit_code'] == 0 for child in children)
