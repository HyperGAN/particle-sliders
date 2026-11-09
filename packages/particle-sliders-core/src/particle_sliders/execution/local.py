"""Isolated local execution; no product or Torch imports in the controller."""
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .process_identity import capture_process_identity
from .request import materialize_training, validate_resolved, _target


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)



class ReportingError(RuntimeError):
    """The selected observer failed; retain native training state separately."""


class ControllerTerminated(KeyboardInterrupt):
    """SIGTERM delivered to the controller, unwound as a recorded cancellation."""


def _terminated(signum, frame):
    raise ControllerTerminated('controller received SIGTERM')


def _deferred(signum, frame):
    pass  # A repeated SIGTERM must not abort the terminal receipt in progress.


@contextmanager
def _sigterm_cancels():
    # Only replace the default disposition, and only where signal.signal works.
    if threading.current_thread() is not threading.main_thread() or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        yield
        return
    signal.signal(signal.SIGTERM, _terminated)
    try:
        yield
    finally:
        # Leave a handler that a callback installed during the run in place.
        if signal.getsignal(signal.SIGTERM) in (_terminated, _deferred):
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _defer_sigterm():
    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) is _terminated:
        signal.signal(signal.SIGTERM, _deferred)


def _bind_worker_lifetime(parent_pid):
    def bind():
        if sys.platform.startswith('linux'):
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent_pid:
                os._exit(126)
    return bind


def _reporter_command(request, output, *, check=False):
    reporting = request['reporting']
    command = [reporting['runtime_python'], str(Path(__file__).with_name('reporting.py')),
        '--run', str(output), '--runtime-path', reporting['runtime_path'],
        '--runtime-commit', reporting['runtime_commit'], '--runtime-sha256', reporting['runtime_sha256'],
        '--check' if check else '--follow']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    return command, env


def _check_reporting(request):
    if request.get('reporting') is None:
        return
    command, env = _reporter_command(request, Path(request['output']['root']), check=True)
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise ReportingError('reporting runtime check could not complete: ' + str(error)) from error
    if result.returncode:
        raise ReportingError('reporting runtime check failed: ' + (result.stderr or result.stdout).strip()[-2000:])


def _stop_reporter(child, log, *, graceful=False):
    if child is None:
        return
    try:
        if graceful:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                raise ReportingError('reporter did not finish after terminal training result') from None
            if child.returncode:
                raise ReportingError(f'reporter exited with code {child.returncode}; see reporter.log')
        elif child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        if not graceful or child.poll() is not None:
            log.close()


def _start_reporter(request, output):
    if request.get('reporting') is None:
        return None, None
    command, env = _reporter_command(request, output)
    log = (output / 'reporter.log').open('a')
    child = None
    try:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
            preexec_fn=_bind_worker_lifetime(os.getpid()))
        marker = output / 'reporting/hypergan/reporter-state.json'
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise ReportingError(f'reporter exited before ready with code {child.returncode}; see reporter.log')
            if marker.is_file():
                try:
                    state = json.loads(marker.read_text())
                except (ValueError, OSError):
                    state = {}
                if state.get('status') == 'ready':
                    return child, log
                if state.get('status') == 'failed':
                    raise ReportingError('reporter initialization failed; see reporter.log')
            time.sleep(.05)
        raise ReportingError('reporter did not become ready before training; see reporter.log')
    except BaseException:
        if child is not None:
            _stop_reporter(child, log)
        else:
            log.close()
        raise

def command_for(request, config, artifacts, *, check=False):
    spec = validate_resolved(request)
    runtime = spec.get('runtime', {})
    directory = Path(runtime.get('cwd', request['resolved']['profile_dir'])).resolve()
    python = runtime.get('python', sys.executable)
    if spec['config_kind'] == 'native':
        command = [python if value == '{python}' else value for value in spec['command']]
        command += ['--config', str(config), '--output', str(artifacts),
                    '--checkpoint-every', str(request['output']['checkpoint_every'])]
        if request['execution'].get('resume'):
            command += ['--resume', request['execution']['resume']]
        for key, flag in (('selection', '--selection'), ('export_format', '--export-format')):
            if key in request['output']:
                command += [flag, request['output'][key]]
        if spec['capabilities'].get('device', False) and request['execution'].get('device'):
            command += ['--device', request['execution']['device']]
        if request['execution'].get('stop_after'):
            raise ValueError('stop-after is only supported by session profiles; native recipes own their target')
    else:
        # Execute this standalone file, without adding the new core to PYTHONPATH.
        # It imports the selected runtime's installed lifecycle/PG unchanged.
        command = [python, str(Path(__file__).with_name('session_worker.py')), '--config', str(config),
                   '--output', str(artifacts), '--interface', spec.get('interface', 'image'),
                   '--checkpoint-every', str(request['output']['checkpoint_every']),
                   '--selection', request['output']['selection'], '--export-format', request['output']['export_format']]
        if request['execution'].get('resume'):
            command += ['--resume', request['execution']['resume']]
        if request['execution'].get('stop_after'):
            command += ['--stop-after', str(request['execution']['stop_after'])]
    if check:
        command.append('--check')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(runtime.get('env', {}), PYTHONUNBUFFERED='1')
    return command, directory, env


def _admit(request):
    spec = validate_resolved(request)
    if spec['status'] == 'blocked':
        raise ValueError(spec['name'] + ': ' + spec['reason'])
    training = request['training']
    if spec['config_kind'] == 'session':
        if set(training) != {'backend', 'cohort', 'options'} or training['backend'] != spec['backend']:
            raise ValueError('session config differs from selected backend')
        if spec['status'] == 'fixture' and not training['options'].get('dummy', False):
            raise ValueError('fixture profile requires explicit dummy=true')
    materialize_training(request)  # Check every input before creating output or starting a worker.
    return spec


def check_request(request):
    spec = _admit(request)
    _check_reporting(request)
    if not spec['capabilities'].get('check', spec['config_kind'] == 'session'):
        raise ValueError('profile has no allocation-free check operation; use plan')
    with tempfile.TemporaryDirectory(prefix='particle-slider-check-') as root:
        config = Path(root) / 'training.json'
        atomic_json(config, materialize_training(request))
        command, cwd, env = command_for(request, config, Path(root) / 'artifacts', check=True)
        result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError('profile check failed: ' + result.stderr.strip())
        return dict(status='checked', profile=spec['name'], qualification=spec['status'],
                    target_steps=_target(request['training'], spec), worker_output=result.stdout.strip())


def _known_checkpoints(root):
    # Native workers declare complete checkpoint markers; no partial tensor files.
    result = set()
    for pattern in ('**/checkpoints/step-*/checkpoint.json', '**/checkpoints/step-*/manifest.json'):
        result.update(p.parent for p in root.glob(pattern))
    return result


def _read_worker_output(stream, events):
    """Publish one terminal event even when reading or decoding output fails."""
    error = None
    try:
        for line in stream:
            events.put((line, None))
    except BaseException as failure:
        error = failure
    finally:
        try:
            stream.close()
        except BaseException as failure:
            if error is None:
                error = failure
        finally:
            events.put((None, error))


def _reap_worker(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass  # The owned worker may exit between poll and signal.
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    else:
        child.wait()


def execute_resolved_request(request, *, on_checkpoint=None, on_progress=None):
    with _sigterm_cancels():
        return _execute(request, on_checkpoint=on_checkpoint, on_progress=on_progress)


def _execute(request, *, on_checkpoint=None, on_progress=None):
    request = deepcopy(request)
    spec = _admit(request)
    _check_reporting(request)  # Explicit reporting failures occur before GPU/output allocation.
    output = Path(request['output']['root']).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('training output must be a new empty directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / 'artifacts'
    config = output / 'training.json'
    atomic_json(output / 'request.json', request)
    atomic_json(config, materialize_training(request))
    command, cwd, env = command_for(request, config, artifacts)
    progress = dict(status='starting', profile=spec['name'], qualification=spec['status'],
                    target_steps=_target(request['training'], spec), completed_steps=0,
                    controller_identity=capture_process_identity())
    child = None
    reporter, reporter_log = None, None
    training_result = None
    seen = set()
    atomic_json(output / 'progress.json', progress)
    try:
        reporter, reporter_log = _start_reporter(request, output)
        parent_pid = os.getpid()
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True, preexec_fn=_bind_worker_lifetime(parent_pid))
        progress['pid'] = child.pid
        progress['worker_identity'] = capture_process_identity(child.pid)
        atomic_json(output / 'progress.json', progress)
        lines = queue.Queue()
        thread = threading.Thread(target=_read_worker_output, args=(child.stdout, lines), daemon=True)
        thread.start()
        stream_closed = False
        with (output / 'train.log').open('w') as stream:
            while not stream_closed or child.poll() is None:
                if reporter is not None and reporter.poll() is not None:
                    raise ReportingError(f'reporter exited during training with code {reporter.returncode}; see reporter.log')
                try:
                    line, read_error = lines.get(timeout=.2)
                except queue.Empty:
                    line, read_error = '', None
                if read_error is not None:
                    raise RuntimeError('training worker output reader failed: ' + str(read_error)) from read_error
                if line is None:
                    stream_closed = True
                elif line:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                native_progress = artifacts / 'progress.json'
                if native_progress.is_file():
                    try:
                        current = json.loads(native_progress.read_text())
                    except (ValueError, OSError):
                        current = {}
                    if current:
                        progress['worker'] = current
                        progress['completed_steps'] = current.get('completed_steps', progress['completed_steps'])
                        # A running setup/collection phase is not claimed as training.
                        if current.get('phase') == 'train' or current.get('status') == 'training':
                            progress['status'] = 'training'
                        atomic_json(output / 'progress.json', progress)
                        if on_progress:
                            on_progress(deepcopy(progress))
                for checkpoint in sorted(_known_checkpoints(artifacts) - seen):
                    seen.add(checkpoint)
                    progress['last_checkpoint'] = str(checkpoint)
                    atomic_json(output / 'progress.json', progress)
                    if on_checkpoint:
                        on_checkpoint(str(checkpoint))
            child.wait()
        if child.returncode:
            raise RuntimeError(f'training worker exited with code {child.returncode}; see {output / "train.log"}')
        native = json.loads((artifacts / 'result.json').read_text())
        target = request['execution'].get('stop_after', progress['target_steps'])
        if native.get('status') not in ('completed', 'checkpointed') or native.get('completed_steps') != target:
            raise RuntimeError('worker did not complete the requested update budget')
        for key in ('selection', 'export_format'):
            if key in request['output'] and spec['config_kind'] == 'native' and native.get(key) != request['output'][key]:
                raise RuntimeError('native worker did not confirm requested ' + key)
        paths = {}
        for key in ('checkpoint', 'adapter'):
            if not isinstance(native.get(key), str):
                raise RuntimeError('worker completion requires ' + key + ' path')
            path = Path(native[key])
            if not path.is_absolute():
                path = artifacts / path
            if not path.exists() or path.is_dir() and not any(path.iterdir()):
                raise RuntimeError('missing completed ' + key + ': ' + str(path))
            paths[key] = str(path.resolve())
        for checkpoint in sorted(_known_checkpoints(artifacts) - seen):
            seen.add(checkpoint)
            if on_checkpoint:
                on_checkpoint(str(checkpoint))
        if on_checkpoint and Path(paths['checkpoint']) not in seen:
            on_checkpoint(paths['checkpoint'])
        status = 'completed' if target == progress['target_steps'] else 'checkpointed'
        result = dict(status=status, completed_steps=target, target_steps=progress['target_steps'],
                      profile=spec['name'], qualification=spec['status'], **paths,
                      semantic_sha256=request['resolved']['semantic_sha256'],
                      runtime_sha256=request['resolved']['runtime_sha256'], native_result=native,
                      controller_identity=progress['controller_identity'], worker_identity=progress['worker_identity'])
        training_result = result
        atomic_json(output / 'result.json', result)
        atomic_json(output / 'progress.json', result)
        if reporter is not None:
            _stop_reporter(reporter, reporter_log, graceful=True)
            result['reporting'] = dict(kind=request['reporting']['kind'], status='complete',
                directory=str(output / 'reporting/hypergan'), config_sha256=request['resolved']['reporting_sha256'])
            atomic_json(output / 'result.json', result)
            atomic_json(output / 'progress.json', result)
        if on_progress:
            on_progress(deepcopy(result))
        return result
    except BaseException as error:
        _defer_sigterm()
        status = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        if child is not None:
            _reap_worker(child)
        terminal = dict(status=status, error=str(error), target_steps=progress['target_steps'],
                        last_checkpoint=progress.get('last_checkpoint'),
                        worker_reaped=child is None or child.poll() is not None,
                        controller_identity=progress['controller_identity'],
                        worker_identity=progress.get('worker_identity'))
        if training_result is not None:
            terminal.update(training_status=training_result['status'], completed_steps=training_result['completed_steps'],
                            checkpoint=training_result['checkpoint'], adapter=training_result['adapter'])
        if isinstance(error, ReportingError):
            terminal['reporting'] = dict(status='failed', kind=request['reporting']['kind'],
                error=str(error), directory=str(output / 'reporting/hypergan'))
        atomic_json(output / 'result.json', terminal)
        atomic_json(output / 'progress.json', terminal)
        if reporter is not None:
            # Terminal result lets the observer flush the final journal boundary.
            try:
                _stop_reporter(reporter, reporter_log, graceful=True)
            except BaseException:
                _stop_reporter(reporter, reporter_log)
        if isinstance(error, ReportingError):
            # This is controller coordination status, never fabricated HG metrics.
            marker = output / 'reporting/hypergan/reporter-state.json'
            try:
                state = json.loads(marker.read_text()) if marker.is_file() else {}
            except (OSError, ValueError):
                state = {}
            state.update(status='failed', controller_error=str(error))
            atomic_json(marker, state)
        raise
