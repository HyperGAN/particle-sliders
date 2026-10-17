"""Root-only future audio phase guardian; importing this module launches nothing.

Y20 continuation and M3S software smoke only. A root-reviewed card, verified
proposal, two free leases and fresh resources are mandatory. Rendering and
automatic phase chaining are deliberately unsupported.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import guarded_pg09_canary as resources

ROOT = Path('/mnt/nvme/sliders/qualification')
AUDIO = Path('/mnt/nvme/sliders/audio-adapters-pg09-20261017')
CORE = '3e94b13eb5d6cd1ceee3f018f019a68603f35696'
PATCH = 'c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571'
MOSS_RECIPES = {
    'bad-intent': 'f4ca175543f01b02801d459c626a7b0081b9ad34add1808ba81a489163e2bb67',
    'afterimage': '7bb1cff4d82337b4674ffe770edb46965498a42d9f42e137aa8b7bddf0eecfa3',
}
CARDS = {
    'Y20': dict(gpu=1, uuid='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69', residents={3196},
                executable='/ml2/sliders/.venv-yue2-retrain/bin/python',
                script=str(AUDIO/'scripts/qualify_audio_pg09_live.py'), family='yue2',
                output='yue2-continue20', startup_gpu=16, startup_host=12),
    'M3S': dict(gpu=0, uuid='GPU-72c1b506-891d-b8bc-b353-e020585e1c47', residents={5556, 9450},
                executable='/mnt/nvme/sliders/qualification/envs/pg090/bin/python',
                script=str(AUDIO/'scripts/run_audio_pg09_smoke.py'), family='music3',
                output='music3-smoke', startup_gpu=30, startup_host=24),
}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path, deadline=None):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('phase deadline reached during input verification')
            value.update(block)
    return value.hexdigest()


def argument(command, flag):
    if command.count(flag) != 1:
        raise ValueError('exactly one argument required: ' + flag)
    return command[command.index(flag)+1]


def validate_scope(card, proposal):
    """Pure scope validation, also exercised without GPU or native imports."""
    if (card.get('schema') != 'root-audio-next-admission-v1' or
            card.get('status') != 'REVIEWED_PENDING_FRESH_RESOURCES' or
            card.get('phase') not in CARDS):
        raise ValueError('only separately reviewed Y20 or M3S cards are accepted')
    phase = card['phase']
    fixed = CARDS[phase]
    admission = card.get('admission_id', '')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{5,95}', admission):
        raise ValueError('explicit new admission id required')
    leases = [str(ROOT/f"admission-gpu-{fixed['gpu']}.lock"),
              f"/mnt/nvme/sliders/locks/gpu-{fixed['gpu']}.lock"]
    if (card.get('physical_gpu') != fixed['gpu'] or card.get('uuid') != fixed['uuid'] or
            card.get('lease_paths') != leases):
        raise ValueError('physical card/lease scope differs')
    if (proposal.get('schema') != 'audio-pg09-next-stage-proposal-v1' or
            proposal.get('phase') != phase or proposal.get('family') != fixed['family'] or
            proposal.get('status') != 'PROPOSED_NOT_ADMITTED' or
            proposal.get('dispatch_allowed') is not False or proposal.get('admission_id') is not None or
            proposal.get('core_commit') != CORE or proposal.get('source_profile') != 'alz-continuous-v1' or
            proposal.get('penalty_input_domain') != 'continuous' or proposal.get('patch_sha256') != PATCH or
            len(proposal.get('pg_sources', {})) != 49):
        raise ValueError('native proposal/core/domain scope differs')
    limits = proposal['limits']
    expected = dict(compute_seconds=720, shutdown_grace_seconds=180, maximum_wall_seconds=900,
                    maximum_concurrent_cuda_children=1, host_running_available_gib_minimum=8,
                    gpu_running_free_mib_minimum=6144, nvme_free_gib_minimum=100,
                    gpu_abort_temperature_c=82, gpu_cooling_start_c=76,
                    gpu_cooling_target_c=70, gpu_cooling_maximum_seconds=180,
                    power_abort_fraction_of_limit=1.15, retry=False, automatic_next_phase=False,
                    startup_gpu_free_gib_minimum=fixed['startup_gpu'],
                    startup_host_available_gib_minimum=fixed['startup_host'],
                    sequential_cuda_children=1 if phase == 'Y20' else 3)
    if any(type(limits.get(k)) is not type(v) or limits.get(k) != v for k, v in expected.items()):
        raise ValueError('reviewed phase resource/serial limits differ')
    output = ROOT/'runs/audio-pg09-reviewed-20261017'/fixed['output']
    if Path(proposal['output_directory']) != output or output.resolve() != output:
        raise ValueError('fixed new phase output differs')
    command = list(proposal.get('command') or [])
    if command[:2] != [fixed['executable'], fixed['script']] or '--fixture' in command:
        raise ValueError('native phase executable differs; no fixture or render fallback')
    flags = command[2::2]
    allowed = {'--out', '--family', '--minimum-free-gib', '--model-dir', '--manifest',
               '--pg-source-profile', '--penalty-input-domain', '--preset', '--seed',
               '--total-steps', '--max-seq-len', '--device', '--admission-id',
               '--train-prompts', '--heldout-prompts'}
    if phase == 'Y20':
        allowed |= {'--product', '--resume', '--until'}
    if len(command) % 2 or len(flags) != len(set(flags)) or set(flags) != allowed:
        raise ValueError('exact phase argument set required')
    for flag in ('--manifest', '--train-prompts', '--heldout-prompts'):
        if argument(command, flag) not in proposal.get('input_pins', {}):
            raise ValueError('manifest/prompt input checksum required')
    expected_args = {'--out': str(output), '--family': fixed['family'], '--total-steps': '20',
                     '--device': 'cuda:0', '--seed': '7', '--preset': 'bispf_optimized',
                     '--max-seq-len': '512', '--pg-source-profile': 'alz-continuous-v1',
                     '--penalty-input-domain': 'continuous', '--minimum-free-gib': str(fixed['startup_gpu'])}
    if any(argument(command, key) != value for key, value in expected_args.items()):
        raise ValueError('fixed native arguments differ')
    required_sources = {str(AUDIO/'scripts/qualify_audio_pg09_live.py'),
                        str(AUDIO/'conceptmod/textsliders/pg09_audio.py'),
                        str(ROOT/'ops-music3/music3_resources.py')}
    if phase == 'M3S':
        required_sources.add(str(AUDIO/'scripts/run_audio_pg09_smoke.py'))
    if not required_sources.issubset(proposal.get('source_pins', {})):
        raise ValueError('executed native worker/adapter/resource sources missing')
    if phase == 'Y20':
        if (argument(command, '--until') != '20' or
                argument(command, '--resume') != str(ROOT/'runs/audio-pg09-reviewed-20261017/yue2-smoke/full3/checkpoint.pt') or
                proposal.get('initial_completed_steps') != 3 or proposal.get('additional_updates') != 17):
            raise ValueError('continuation must use the qualified step3 checkpoint')
        if argument(command, '--resume') not in proposal.get('input_pins', {}):
            raise ValueError('qualified resume packet checksum required')
    elif '--until' in command or '--resume' in command:
        raise ValueError('M3S requires the serial full3 versus 1+2 supervisor')
    placeholder = argument(command, '--admission-id')
    if not placeholder.startswith('<ROOT_ASSIGN_NEW_') or not placeholder.endswith('>'):
        raise ValueError('proposal must remain unadmitted')
    command[command.index('--admission-id')+1] = admission
    environment = dict(proposal['environment'])
    if (environment.get('CUDA_VISIBLE_DEVICES') != '<ROOT_ASSIGNED_SINGLE_PHYSICAL_GPU_AFTER_REVIEW>' or
            any(environment.get(k) != '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')) or
            environment.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or
            any('site-packages' in p for p in environment.get('PYTHONPATH', '').split(os.pathsep))):
        raise ValueError('native isolated environment differs')
    environment['CUDA_VISIBLE_DEVICES'] = str(fixed['gpu'])
    return fixed, output, leases, command, environment


def validate(card, deadline):
    proposal_path = Path(card['proposal_path'])
    if sha(proposal_path, deadline) != card['proposal_sha256']:
        raise ValueError('frozen proposal changed')
    proposal = load(proposal_path)
    result = validate_scope(card, proposal)
    priority_proofs(card)
    required = {str(Path(__file__).resolve()), str(Path(resources.__file__).resolve()),
                str(ROOT/'ops-sana-kernel-fixed/kernel_gpu_guard.py')}
    if not required.issubset(card.get('guardian_source_pins', {})):
        raise ValueError('all actual guardian/resource sources must be bound')
    for bindings in (card['guardian_source_pins'], proposal['source_pins'], proposal['input_pins']):
        for path, expected in bindings.items():
            if sha(path, deadline) != expected:
                raise ValueError('phase source/input changed: ' + path)
    for name, expected in proposal['pg_sources'].items():
        if sha(Path('/tmp/alz-pg09-domain-proposal/particlegan')/name, deadline) != expected:
            raise ValueError('reviewed PG profile changed')
    return result


def priority_proofs(card):
    proofs = card.get('completed_moss_evaluations', {})
    if set(proofs) != set(MOSS_RECIPES):
        raise ValueError('both prioritized MOSS completion/evaluation proofs required')
    for case, recipe in MOSS_RECIPES.items():
        path = ROOT/'runs/moss-pg09-original-runtime-v2-20261017'/case/'progress.json'
        item = proofs[case]
        if item.get('path') != str(path) or sha(path) != item.get('sha256'):
            raise ValueError('prioritized MOSS proof identity changed')
        proof = load(path)
        if (proof.get('recipe_sha256') != recipe or proof.get('mode') != 'evaluate' or
                proof.get('status') != 'evaluated' or proof.get('latest_step') != 3840 or
                not proof.get('frozen_base_unchanged')):
            raise ValueError('prioritized MOSS matched evaluation is incomplete')


def host_floor(snapshot):
    value = snapshot.get('host_free_mib')
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 8192:
        raise RuntimeError('new phase requires at least8GiB available host memory')


def owned_children(pids, residents, group):
    result = []
    for pid in pids-residents:
        try:
            actual = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if actual != group:
            raise RuntimeError('unowned CUDA process observed')
        result.append(dict(pid=pid, pgid=actual))
    if len(result) > 1:
        raise RuntimeError('more than one concurrent CUDA child')
    return result


def terminal_proof(phase, output, admission):
    stages = [output] if phase == 'Y20' else [output/name for name in ('full3', 'first1', 'resumed3')]
    for folder in stages:
        proof = load(folder/'progress.json')
        if (proof.get('status') != 'bounded_training_software_checks_passed' or
                proof.get('admission_id') != admission or
                not all(proof.get(k) for k in ('frozen_base_unchanged', 'zero_replay_exact',
                                               'export_replay_exact', 'actual_bank_weights_changed')) or
                any(type(proof.get(k)) is not int or proof[k] <= 0
                    for k in ('changed_adapter_tensors', 'changed_critic_tensors')) or
                any(not isinstance(proof.get(k), (int, float)) or not math.isfinite(proof[k]) or proof[k] <= 0
                    for k in ('actual_bank_delta_rms', 'bank_response_gradient_l1'))):
            raise RuntimeError('terminal learning/frozenbase/reload proof missing')
        expected = 20 if phase == 'Y20' else 1 if folder.name == 'first1' else 3
        if proof.get('completed_steps') != expected:
            raise RuntimeError('terminal update count differs')
    if phase == 'M3S':
        comparison = load(output/'comparison.json')
        if comparison.get('status') != 'exact_smoke_passed' or not comparison.get('exact_full_state_metrics_probes_export'):
            raise RuntimeError('full3 versus 1+2 exact proof missing')


def stop_owned(process, deadline):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(.01, deadline-time.monotonic()))
    except subprocess.TimeoutExpired:
        pass


def guardian(card_path):
    start = time.monotonic()
    epoch = time.time()
    card = load(card_path)
    fixed, output, names, command, environment = validate(card, start+720)
    admission = output.parent/(output.name+'-root-admission')
    if output.exists() or output.is_symlink() or admission.exists() or admission.is_symlink():
        raise FileExistsError('phase outcome exists; no automatic retry')
    if (not os.path.ismount('/mnt/nvme') or subprocess.check_output(
            ['findmnt', '-no', 'SOURCE', '--target', '/mnt/nvme'], text=True).strip() != '/dev/nvme0n1p2'):
        raise RuntimeError('expected NVMe output filesystem missing')
    hold = load(ROOT/'MSE-DISPATCH-HOLD.json')
    if hold.get('mse_only_training_allowed') is not False or hold.get('automatic_mse_restart_or_dispatch_allowed') is not False:
        raise RuntimeError('historical MSE dispatch hold changed')
    leases = []
    for name in names:
        handle = Path(name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(handle)
    resources.GPU, resources.UUID, resources.RESIDENTS = str(fixed['gpu']), fixed['uuid'], fixed['residents']
    initial = resources.status()
    host_floor(initial)
    if (initial['temperature_c'] >= 76 or initial['gpu_free_mib'] < fixed['startup_gpu']*1024 or
            initial['host_free_mib'] < fixed['startup_host']*1024 or resources.processes()-fixed['residents']):
        raise RuntimeError('fresh phase capacity/resident admission failed')
    resources.kernel_check(epoch)
    admission.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='root-audio-next-result-v1', phase=card['phase'], admission_id=card['admission_id'],
        card_sha256=sha(card_path), proposal_sha256=card['proposal_sha256'], command=command,
        uuid=fixed['uuid'], physical_gpu=fixed['gpu'], lease_paths=names, initial=initial,
        status='admitted', seen_cuda_children=[], quality_qualified=False, maximum_wall_seconds=900,
        host_running_available_gib_minimum=8)
    process = None
    def interrupted(signum, _frame):
        raise InterruptedError('guardian signal ' + str(signum))
    previous_signals = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (admission/'worker.log').open('x') as log:
            if time.monotonic()-start >= 720:
                raise TimeoutError('preflight exhausted compute budget before launch')
            process = subprocess.Popen(command, env=dict(os.environ, **environment), cwd=AUDIO,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(status='running', owned_pgid=process.pid)
            resources.write(admission/'receipt.json', receipt)
            while process.poll() is None:
                if time.monotonic()-start >= 720:
                    raise TimeoutError('compute deadline reserves180s for owned cleanup')
                snapshot = resources.status()
                host_floor(snapshot)
                children = owned_children(resources.processes(), fixed['residents'], process.pid)
                for child in children:
                    if child not in receipt['seen_cuda_children']:
                        receipt['seen_cuda_children'].append(child)
                resources.kernel_check(epoch)
                with (admission/'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(elapsed_seconds=time.monotonic()-start, children=children, **snapshot))+'\n')
                time.sleep(2)
            if process.returncode:
                raise RuntimeError('native phase failed; no retry')
            resources.kernel_check(epoch)
            host_floor(resources.status())
            if owned_children(resources.processes(), fixed['residents'], process.pid):
                raise RuntimeError('CUDA child survived supervisor completion')
            terminal_proof(card['phase'], output, card['admission_id'])
            receipt.update(status='completed', independent_artifact_review_pending=True)
    except BaseException as error:
        for signum in previous_signals:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None:
            stop_owned(process, start+900)
        receipt.update(status='failed_no_retry', error=repr(error))
        raise
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        receipt.update(elapsed_seconds=time.monotonic()-start,
                       exit_code=process.returncode if process is not None else None)
        resources.write(admission/'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, required=True)
    guardian(parser.parse_args().card)
