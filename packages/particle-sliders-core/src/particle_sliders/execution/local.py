"""Isolated local execution; no product or Torch imports in the controller."""
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

from .request import materialize_training, validate_resolved, _target


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


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


def execute_resolved_request(request, *, on_checkpoint=None, on_progress=None):
    request = deepcopy(request)
    spec = _admit(request)
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
                    target_steps=_target(request['training'], spec), completed_steps=0)
    atomic_json(output / 'progress.json', progress)
    child = None
    seen = set()
    try:
        parent_pid = os.getpid()
        def bind_lifetime():
            if sys.platform.startswith('linux'):
                # Kill only this worker when the controller unexpectedly dies.
                # Native wrappers remain responsible for their own child groups.
                libc = ctypes.CDLL(None, use_errno=True)
                if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
                    os._exit(126)
                if os.getppid() != parent_pid:
                    os._exit(126)
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True, preexec_fn=bind_lifetime)
        progress['pid'] = child.pid
        atomic_json(output / 'progress.json', progress)
        lines = queue.Queue()
        def reader():
            for line in child.stdout:
                lines.put(line)
            lines.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        stream_closed = False
        with (output / 'train.log').open('w') as stream:
            while not stream_closed or child.poll() is None:
                try:
                    line = lines.get(timeout=.2)
                except queue.Empty:
                    line = ''
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
            if on_checkpoint:
                on_checkpoint(str(checkpoint))
        if on_checkpoint and Path(paths['checkpoint']) not in seen:
            on_checkpoint(paths['checkpoint'])
        status = 'completed' if target == progress['target_steps'] else 'checkpointed'
        result = dict(status=status, completed_steps=target, target_steps=progress['target_steps'],
                      profile=spec['name'], qualification=spec['status'], **paths,
                      semantic_sha256=request['resolved']['semantic_sha256'],
                      runtime_sha256=request['resolved']['runtime_sha256'], native_result=native)
        atomic_json(output / 'result.json', result)
        atomic_json(output / 'progress.json', result)
        if on_progress:
            on_progress(deepcopy(result))
        return result
    except BaseException as error:
        status = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        terminal = dict(status=status, error=str(error), target_steps=progress['target_steps'],
                        last_checkpoint=progress.get('last_checkpoint'),
                        worker_reaped=child is None or child.poll() is not None)
        atomic_json(output / 'result.json', terminal)
        atomic_json(output / 'progress.json', terminal)
        raise
