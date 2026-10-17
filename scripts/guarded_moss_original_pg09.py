"""Original-runtime phase admission for the reviewed pretrained MOSS GAN worker.

Owns only its worker process group and two per-card leases. There is no queue,
automatic retry, model loader, training objective, or MSE fallback here.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import guarded_pg09_canary as resources

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/nvme/sliders/qualification')
CARDS = {
    0: ('GPU-72c1b506-891d-b8bc-b353-e020585e1c47', {5556, 9450}),
    1: ('GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69', {3196}),
}


def phase_reason(recipe, mode, progress):
    if recipe.get('admission') != 'REVIEWED_GAN_RECIPE':
        return 'recipe has not reached reviewed admission status'
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if mode in ('resume-check', 'train', 'evaluate') and progress.get('recipe_sha256') != identity:
        return 'prior phase belongs to a different recipe/source/output'
    if mode == 'probe':
        if progress:
            return 'prior probe outcome must be inspected; no automatic retry'
    elif mode in ('train','resume-check'):
        if mode=='resume-check' and progress.get('fresh_process_resume_exact'):
            return 'existing fresh-process resume outcome must be inspected; no retry'
        if mode=='train' and not progress.get('fresh_process_resume_exact'):
            return 'separate-process GPU file resume has not passed'
        if (progress.get('status') != 'probe_passed' or progress.get('latest_step') != 3
                or not progress.get('frozen_base_unchanged') or not progress.get('generator_changed')
                or not progress.get('critic_changed') or not progress.get('exact_adapter_reload')
                or progress.get('exact_resume') != {'exact_loss': True, 'exact_full_state': True, 'exact_sample': True}):
            return 'pretrained three-step probe has not passed all release gates'
    elif mode == 'evaluate':
        if (progress.get('status') != 'trained' or progress.get('latest_step') != 3840
                or not progress.get('frozen_base_unchanged')):
            return 'full fixed GAN training budget and frozen base gate are incomplete'
    else:
        return 'unknown worker phase'
    return None


def verify_prior_checkpoint(contract, recipe, mode, progress, output):
    """Verify immutable checkpoint proof without Torch/model imports."""
    pointer = output / 'LATEST.json'
    if mode == 'probe':
        if pointer.exists() or (output / 'checkpoints').exists():
            raise ValueError('fresh probe must not inherit an unreviewed checkpoint')
        return
    latest = json.loads(pointer.read_text())
    step = 3840 if mode == 'evaluate' else 3
    if (latest.get('recipe_sha256') != recipe.identity or latest.get('step') != step
            or latest.get('directory') != f'step-{step:06d}'):
        raise ValueError('latest checkpoint differs from qualified phase/recipe')
    metadata = contract.verify_checkpoint(output / 'checkpoints' / latest['directory'], recipe)
    base = progress.get('frozen_base_sha256')
    if (not isinstance(base, str) or len(base) != 64 or metadata.get('step') != step
            or metadata.get('frozen_base_sha256') != base
            or progress.get('frozen_base_after_sha256') != base):
        raise ValueError('checkpoint frozen base differs from qualified phase proof')
    if mode=='train':
        proof_path=output/'fresh-process-resume.json'
        if contract.sha_file(proof_path)!=progress.get('fresh_process_resume_sha256'):
            raise ValueError('fresh-process resume proof checksum differs')
        proof=json.loads(proof_path.read_text())
        reference_sha=contract.sha_file(output/'probe-reference.pt')
        if (proof.get('schema')!='moss-fresh-process-gpu-resume-v1' or
                proof.get('recipe_sha256')!=recipe.identity or proof.get('verified_step')!=3 or
                proof.get('replayed_from_step')!=2 or proof.get('checkpoint_advanced') is not False or
                proof.get('reference_sha256')!=reference_sha or
                progress.get('probe_reference_sha256')!=reference_sha or
                proof.get('frozen_base_sha256')!=base or
                not all(proof.get(key) is True for key in
                        ('exact_loss','exact_full_state','exact_sample','frozen_base_unchanged'))):
            raise ValueError('fresh-process resume proof differs from qualified phase')


def worker_module():
    path = REPOSITORY / 'conceptmod/textsliders/pg09_moss_worker.py'
    spec = importlib.util.spec_from_file_location('reviewed_worker_contract', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('probe', 'resume-check', 'train', 'evaluate'))
    args = parser.parse_args(argv)
    contract = worker_module()
    recipe = contract.BoundRecipe.read(args.recipe)
    recipe.verify_sources()
    values = recipe.data
    from moss_original_runtime import PROFILE
    if values.get('runtime_profile') != PROFILE or sys.executable != PROFILE['executable']:
        raise ValueError('guardian must use exact original cache-producer runtime')
    gpu = values.get('physical_gpu')
    if type(gpu) is not int or gpu not in CARDS:
        raise ValueError('recipe must bind one known physical card')
    expected_leases = [f'/mnt/nvme/sliders/qualification/admission-gpu-{gpu}.lock',
                       f'/mnt/nvme/sliders/locks/gpu-{gpu}.lock']
    if values['lease_paths'] != expected_leases:
        raise ValueError('recipe lease paths differ from actual shared admission locks')
    if (not os.path.ismount('/mnt/nvme') or subprocess.check_output(
            ['findmnt', '-no', 'SOURCE', '--target', '/mnt/nvme'], text=True).strip() != '/dev/nvme0n1p2'):
        raise RuntimeError('expected NVMe is not mounted')
    hold = json.loads((ROOT / 'MSE-DISPATCH-HOLD.json').read_text())
    if hold.get('mse_only_training_allowed') is not False or hold.get('automatic_mse_restart_or_dispatch_allowed') is not False:
        raise RuntimeError('historical MSE dispatch hold must remain enforced')
    output = Path(values['output_directory'])
    prior = json.loads((output / 'progress.json').read_text()) if (output / 'progress.json').exists() else {}
    reason = phase_reason(values, args.mode, prior)
    if reason:
        raise RuntimeError(reason)
    verify_prior_checkpoint(contract, recipe, args.mode, prior, output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    admission = output / 'admission' / (args.mode + '-' + stamp)
    admission.mkdir(parents=True, exist_ok=False)
    leases = []
    for path in expected_leases:
        lease = Path(path).open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(lease)
    # Reuse exactly the canary's reviewed stdlib telemetry and kernel guards.
    resources.GPU = str(gpu)
    resources.UUID, resources.RESIDENTS = CARDS[gpu]
    start = time.time()
    initial = resources.status()
    if initial['temperature_c'] >= 76 or resources.processes() - resources.RESIDENTS:
        raise RuntimeError('cooling/unowned GPU process blocks admission')
    resources.kernel_check(start)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HOME='/mnt/nvme/sliders/cache/huggingface', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
        TMPDIR='/mnt/nvme/sliders/tmp',
        PYTHONPATH=os.pathsep.join((str(REPOSITORY), str(REPOSITORY / 'packages/particle-sliders-core/src'),
            '/ml2/codex-storage/future-workspaces/2026-10-16/task-6/moss-debug-dev/src',
            '/mnt/nvme/sliders/qualification/pg09-audit/wheel')))
    command = [sys.executable, '-u', str(REPOSITORY / 'scripts/moss_original_runtime.py'),
               '--recipe', str(args.recipe.resolve()), '--output', str(output), '--mode', args.mode]
    for lease in leases:
        command += ['--lease-fd', str(lease.fileno())]
    receipt = dict(schema='moss-pg09-admission-v1', phase=args.mode, concept=values['concept'],
        recipe_sha256=recipe.identity, recipe_path=str(args.recipe.resolve()), objective='pg09_gan_only',
        command=command, physical_gpu=gpu, uuid=resources.UUID, initial=initial,
        lease_paths=expected_leases, status='admitted', telemetry=[], mse_hold_preserved=True,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    process = None
    deadline = 1800 if args.mode in ('probe','resume-check') else values['wall_seconds']
    try:
        with (admission / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, env=environment, cwd=REPOSITORY,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(lease.fileno() for lease in leases))
            receipt.update(status='running', pid=process.pid)
            contract.atomic_json(admission / 'receipt.json', receipt)
            last_kernel = 0
            while process.poll() is None:
                if time.time() - start >= deadline:
                    raise RuntimeError('reviewed phase wall deadline reached')
                current = resources.status()
                if resources.processes() - resources.RESIDENTS - {process.pid}:
                    raise RuntimeError('new unowned GPU process observed')
                if time.time() - last_kernel >= 5:
                    resources.kernel_check(start)
                    last_kernel = time.time()
                current['elapsed_seconds'] = time.time() - start
                receipt['telemetry'].append(current)
                # Keep monitoring memory bounded while retaining periodic raw data.
                with (admission / 'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(current) + '\n')
                receipt['telemetry'] = receipt['telemetry'][-5:]
                time.sleep(2)
            if process.returncode:
                raise RuntimeError('worker exit ' + str(process.returncode))
            resources.kernel_check(start)
            completed = json.loads((output / 'progress.json').read_text())
            expected = {'probe': 'probe_passed', 'resume-check':'probe_passed', 'train': 'trained', 'evaluate': 'evaluated'}[args.mode]
            if completed.get('status') != expected or not completed.get('frozen_base_unchanged'):
                raise RuntimeError('worker did not pass declared phase outcome: ' + str(completed.get('status')))
            if args.mode=='resume-check' and not completed.get('fresh_process_resume_exact'):
                raise RuntimeError('fresh-process resume proof missing')
            receipt.update(status='completed', outcome=completed)
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        receipt.update(status='failed_no_retry', error=repr(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.time() - start, exit_code=process.returncode if process is not None else None)
        contract.atomic_json(admission / 'receipt.json', receipt)
    print(json.dumps(dict(status=receipt['status'], receipt=str(admission / 'receipt.json'))))


if __name__ == '__main__':
    main()
