"""Reporter data boundaries plus real pinned-HyperGAN subprocess integration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'src/particle_sliders/execution/reporting.py'
SPEC = importlib.util.spec_from_file_location('standalone_reporting', SCRIPT)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def test_measured_native_scalars_only():
    step, scalars, duration = reporting.metric_record(dict(completed_steps=7, seconds=.5,
        metrics=dict(step=7, g_loss=2., d_loss=1., grad_norm_generator=.4, enabled=True, text='ignored')))
    assert step == 7 and duration == .5
    assert scalars == {'loss/g_total': 2., 'loss/d_total': 1., 'grad_norm/generator': .4}
    assert 'progress/samples_seen' not in scalars
    with pytest.raises(ValueError, match='nonfinite'):
        reporting.metric_record({'step': 1, 'g_loss': float('nan')})
    with pytest.raises(ValueError, match='completed step'):
        reporting.metric_record({'step': True, 'g_loss': 1.})


def test_complete_lines_keep_partial_tail_and_detect_changed_prefix(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(b'{"step":1}\n{"step":')
    rows = list(reporting.complete_lines(path, 0, None))
    assert len(rows) == 1
    state = rows[0][1]
    with path.open('ab') as stream:
        stream.write(b'2}\n')
    assert list(reporting.complete_lines(path, state['offset'], state['anchor']))[0][0]['step'] == 2
    path.write_bytes(b'{"step":9}\n{"step":2}\n')
    with pytest.raises(ValueError, match='previously observed'):
        list(reporting.complete_lines(path, state['offset'], state['anchor']))


def checkpoint(root, step):
    path = root / 'artifacts/training/checkpoints' / f'step-{step:06d}'
    path.mkdir(parents=True)
    (path / 'state.pt').write_bytes(b'opaque native full-state bytes')
    dump(path / 'manifest.json', dict(completed_steps=step,
         weights_sha256=hashlib.sha256((path / 'state.pt').read_bytes()).hexdigest()))
    return path


def test_checkpoint_frontier_requires_matching_full_state_bytes(tmp_path):
    path = checkpoint(tmp_path, 3)
    assert reporting.checkpoint_record(path)['step'] == 3
    (path / 'state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint step or bytes'):
        reporting.checkpoint_record(path)


def test_runtime_pin_failure_precedes_import(tmp_path):
    with pytest.raises(ValueError, match='unsupported HyperGAN'):
        reporting.load_runtime(tmp_path, '0' * 40, reporting.HYPERGAN_SOURCE_SHA256)


@pytest.fixture
def native_runtime():
    runtime = os.environ.get('HYPERGAN_TEST_RUNTIME_PATH')
    python = os.environ.get('HYPERGAN_TEST_PYTHON', sys.executable)
    if not runtime:
        pytest.skip('set HYPERGAN_TEST_RUNTIME_PATH to run actual pinned reporting integration')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=runtime)
    command = [python, str(SCRIPT), '--runtime-path', runtime,
        '--runtime-commit', reporting.HYPERGAN_COMMIT,
        '--runtime-sha256', reporting.HYPERGAN_SOURCE_SHA256]
    return command, env, python


def test_real_hypergan_check_without_native_run(native_runtime, tmp_path):
    command, env, _ = native_runtime
    result = subprocess.run([*command, '--run', str(tmp_path / 'absent'), '--check'],
        env=env, capture_output=True, text=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'checked' and receipt['torch_imported'] is False
    assert not (tmp_path / 'absent').exists()


def test_real_hypergan_backfill_reattach_and_verified_frontier(native_runtime, tmp_path):
    command, env, python = native_runtime
    run = tmp_path / 'run'
    dump(run / 'request.json', {'resolved': {'target_steps': 2, 'semantic_sha256': 'test'},
        'training': {'native_recipe': 'kept verbatim'}})
    dump(run / 'progress.json', {'status': 'training'})
    metrics = run / 'artifacts/training/training.jsonl'
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"completed_steps":1,"seconds":0.5,"metrics":{"step":1,"g_loss":2.0}}\n{"completed_steps":')
    checkpoint(run, 1)
    before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    subprocess.run([*command, '--run', str(run)], env=env, capture_output=True, text=True, check=True)
    assert all(p.read_bytes() == content for p, content in before.items())
    state = json.loads((run / 'reporting/hypergan/reporter-state.json').read_text())
    assert state['observed_step'] == state['last_durable_step'] == 1
    telemetry = json.loads((run / 'reporting/hypergan/summary.json').read_text())['telemetry_trends']
    subprocess.run([*command, '--run', str(run)], env=env, capture_output=True, text=True, check=True)
    assert json.loads((run / 'reporting/hypergan/summary.json').read_text())['telemetry_trends'] == telemetry
    with metrics.open('a') as stream:
        stream.write('2,"seconds":0.25,"metrics":{"step":2,"g_loss":1.0}}\n')
    final = checkpoint(run, 2)
    dump(run / 'result.json', {'status': 'completed', 'completed_steps': 2, 'checkpoint': str(final)})
    subprocess.run([*command, '--run', str(run)], env=env, capture_output=True, text=True, check=True)
    dump(run / 'result.json', {'status': 'completed', 'completed_steps': 2, 'checkpoint': str(final),
                              'reporting': {'status': 'complete'}})
    subprocess.run([*command, '--run', str(run)], env=env, capture_output=True, text=True, check=True)
    code = '''import json,sys
from hypergan.run_events import read_event_page
from hypergan.metrics import read_catalog
from hypergan.run_info import read_run_info
from pathlib import Path
root=Path(sys.argv[1]);rows=read_event_page(root)['events']
assert [row['step'] for row in rows if row['event']=='train']==[1,2]
assert len([row for row in rows if row['event']=='complete'])==1
assert read_catalog(root)['metrics']['loss/g_total']['source']=='loss/g_total'
info=read_run_info(root)
assert info.status=='completed' and info.step==2 and info.latest_metrics['loss/g_total']==1.
assert len(info.checkpoints)==2 and all(row['external'] for row in info.checkpoints)
assert 'torch' not in sys.modules
print('actual HyperGAN reader/catalog/summary verified')
'''
    checked = subprocess.run([python, '-c', code, str(run / 'reporting/hypergan')],
        env=env, capture_output=True, text=True, check=True)
    assert 'verified' in checked.stdout


def test_real_hypergan_terminal_without_metric_rows_uses_verified_frontier(native_runtime, tmp_path):
    command, env, _ = native_runtime
    dump(tmp_path / 'request.json', {'resolved': {'target_steps': 2}, 'training': {}})
    final = checkpoint(tmp_path, 2)
    dump(tmp_path / 'result.json', {'status': 'completed', 'completed_steps': 2, 'checkpoint': str(final)})
    subprocess.run([*command, '--run', str(tmp_path)], env=env, capture_output=True, text=True, check=True)
    root = tmp_path / 'reporting/hypergan'
    state = json.loads((root / 'reporter-state.json').read_text())
    assert state['observed_step'] == 0 and state['last_durable_step'] == 2
    assert json.loads((root / 'summary.json').read_text())['step'] == 2
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    assert not any(row['event'] == 'train' for row in events)


def test_real_hypergan_ready_and_clean_sidecar_stop(native_runtime, tmp_path):
    command, env, _ = native_runtime
    dump(tmp_path / 'request.json', {'resolved': {'target_steps': 2}, 'training': {}})
    child = subprocess.Popen([*command, '--run', str(tmp_path), '--follow'], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        state = tmp_path / 'reporting/hypergan/reporter-state.json'
        deadline = time.monotonic() + 10
        while not state.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert state.exists() and json.loads(state.read_text())['status'] == 'ready'
        child.terminate()
        child.communicate(timeout=5)
        assert child.returncode == 0
        final = json.loads(state.read_text())
        assert final['status'] == 'complete' and final['terminal_observed'] is False
        assert final['last_durable_step'] is None
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
