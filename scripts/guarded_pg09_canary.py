"""Owned, bounded CUDA/BF16 GAN canary. No pretrained weights or MSE trainer.

The parent is stdlib-only, holds both existing admission locks and monitors
only its child. The worker uses tiny random official MOSS architecture.
"""
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/nvme/sliders/qualification')
REPOSITORY = Path(__file__).resolve().parents[1]
GPU = '0'
UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
RESIDENTS = {5556, 9450}


def write(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def resource_reason(temperature, free_mib, power, limit, host_mib, disk_gib):
    if any(not math.isfinite(x) or x < 0 for x in (temperature, free_mib, power, limit, host_mib, disk_gib)) or limit <= 0:
        return 'invalid resource telemetry'
    if temperature >= 82 or free_mib < 6144 or power > 1.15 * limit:
        return 'GPU temperature/memory/power guard'
    if host_mib < 4096 or disk_gib < 100:
        return 'host/NVMe resource guard'
    return None


def status():
    raw = subprocess.check_output(['nvidia-smi', '--id=' + GPU,
        '--query-gpu=uuid,temperature.gpu,memory.free,power.draw,power.limit',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    if raw[0].strip() != UUID:
        raise RuntimeError('physical GPU identity changed')
    temperature, free, power, limit = map(float, raw[1:])
    host = next(int(line.split()[1]) / 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
    disk = shutil.disk_usage('/mnt/nvme').free / 1024**3
    reason = resource_reason(temperature, free, power, limit, host, disk)
    if reason:
        raise RuntimeError(reason)
    return dict(temperature_c=temperature, gpu_free_mib=free, power_w=power,
                power_limit_w=limit, host_free_mib=host, disk_free_gib=disk)


def processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(line.split(',')[1]) for line in raw.splitlines() if line.split(',')[0].strip() == UUID}


def kernel_check(since):
    sys.path.insert(0, str(ROOT / 'ops-sana-kernel-fixed'))
    from kernel_gpu_guard import classify_kernel_gpu_query
    result = subprocess.run(['journalctl', '-k', '--since', '@' + str(int(since)), '--no-pager',
                             '-g', 'NVRM: Xid|GPU has fallen|nvidia.*[Ee]rror'],
                            capture_output=True, text=True, timeout=10)
    checked = classify_kernel_gpu_query(result.returncode, result.stdout, result.stderr)
    if checked.reason:
        raise RuntimeError(checked.reason + ': ' + '\n'.join(checked.error_lines))


def guardian(directory):
    if directory.exists():
        raise RuntimeError('existing canary outcome must be inspected; no automatic retry')
    directory.mkdir(parents=True)
    leases = []
    for path in (ROOT / 'admission-gpu-0.lock', Path('/mnt/nvme/sliders/locks/gpu-0.lock')):
        lease = path.open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(lease)
    initial = status()
    if initial['temperature_c'] >= 76 or processes() - RESIDENTS:
        raise RuntimeError('GPU cooling or unowned process blocks admission')
    start = time.time()
    kernel_check(start)
    # The worker command is fixed to this reviewed GAN-only script.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join((str(REPOSITORY / 'packages/particle-sliders-core/src'),
             str(REPOSITORY), '/ml2/codex-storage/future-workspaces/2026-10-16/task-6/moss-debug-dev/src',
             str(ROOT / 'ops-sana-kernel-fixed'))))
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(directory),
               *(str(lease.fileno()) for lease in leases)]
    with (directory / 'worker.log').open('x') as log:
        process = subprocess.Popen(command, env=env, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=tuple(lease.fileno() for lease in leases))
        receipt = dict(status='running', pid=process.pid, command=command,
            objective='public PG09 g_adv+required_prior; d_adv+public_penalty',
            pretrained=False, gpu=GPU, uuid=UUID, admission=initial,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            core_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip())
        write(directory / 'launch.json', receipt)
        checks = []
        try:
            while process.poll() is None:
                if time.time() - start > 120:
                    raise RuntimeError('120 second bounded canary deadline')
                current = status()
                if processes() - RESIDENTS - {process.pid}:
                    raise RuntimeError('new unowned GPU process observed')
                kernel_check(start)
                checks.append(current)
                time.sleep(1)
            if process.returncode:
                raise RuntimeError('worker failed with exit ' + str(process.returncode))
            receipt.update(status='completed', worker=json.loads((directory / 'result.json').read_text()))
        except BaseException as exc:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            receipt.update(status='failed_no_retry', error=str(exc))
            raise
        finally:
            receipt.update(elapsed_seconds=time.time() - start, telemetry=checks, exit_code=process.returncode)
            write(directory / 'launch.json', receipt)
    print(json.dumps(dict(status=receipt['status'], receipt=str(directory / 'launch.json'))))


def worker(directory, descriptors):
    expected = (ROOT / 'admission-gpu-0.lock', Path('/mnt/nvme/sliders/locks/gpu-0.lock'))
    for fd, path in zip(descriptors, expected):
        if Path('/proc/self/fd/' + str(fd)).resolve() != path:
            raise RuntimeError('worker requires inherited owned admission leases')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from torch import nn
    from moss_sliders.smoke import tiny_model
    from moss_sliders.lora import Slider
    from conceptmod.textsliders.pg09_backends import MossResponseAdapter, ResponseRow, checkpoint_with_adapter_scale
    from particle_sliders.pg09 import ConditionalPG09
    from music3_resources import ResourceGuard
    from dataclasses import replace
    from copy import deepcopy
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    guard = ResourceGuard()
    guard()
    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(256 + 19, 16), nn.SiLU(), nn.Linear(16, 1))
        def forward(self, field, condition):
            return self.net(torch.cat((field.flatten(1), condition), 1)).squeeze(-1)
    def make():
        torch.manual_seed(20261017)
        model = tiny_model().to(device='cuda', dtype=torch.bfloat16).requires_grad_(False)
        slider = Slider(model, rank=2)
        backend = MossResponseAdapter(model, slider, model_identity={'model': 'official tiny random MOSS', 'seed': 20261017, 'pretrained': False})
        for block in model.layers:
            checkpoint_with_adapter_scale(block, backend.network)
        guard.attach(model)
        contexts = [(torch.randn(1, 3, 16, device='cuda', dtype=torch.bfloat16),
                     torch.ones(1, 3, device='cuda', dtype=torch.bool)) for _ in range(2)]
        latent = torch.randn(1, 4, 8, 8, device='cuda', dtype=torch.bfloat16)
        base = ResponseRow(contexts[0], contexts[1], contexts[0], latent, timestep=torch.tensor(500., device='cuda'))
        rows = [backend.prepare(replace(base, scale=sign)) for sign in (1., -1.)]
        pooled = contexts[0][0].float().mean(1)
        conditions = torch.cat([torch.cat((pooled, torch.tensor([[.5, sign, 1.]], device='cuda')), 1) for sign in (1., -1.)])
        critic = Critic().cuda()
        guard.attach(critic)
        game = ConditionalPG09(backend, rows, critic, conditions=conditions, total_steps=4,
                               batch_size=2, response_domain='continuous_field')
        return model, backend, rows, game
    def equal(a, b):
        if torch.is_tensor(a):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    def forbidden(*args, **kwargs):
        raise AssertionError('MSE optimization is forbidden')
    torch.nn.functional.mse_loss = forbidden
    started = time.monotonic()
    model, backend, rows, game = make()
    frozen = deepcopy(model.state_dict())
    before = deepcopy(backend.network.state_dict())
    metrics = [game.step(), game.step()]
    packet = game.state_dict()
    torch.save(packet, directory / 'checkpoint.pt')
    expected = game.step()
    expected_state = game.state_dict()
    other_model, other, other_rows, restored = make()
    restored.load_state_dict(torch.load(directory / 'checkpoint.pt', weights_only=True))
    actual = restored.step()
    assert equal(expected, actual) and equal(expected_state, restored.state_dict())
    assert all(torch.equal(frozen[name], value) for name, value in model.state_dict().items())
    assert any(not torch.equal(before[name], value) for name, value in backend.network.state_dict().items())
    exported = game.export_state_dict()
    restored.load_export_state_dict(exported)
    assert all(torch.equal(backend.generated(row), other.generated(other_row)) for row, other_row in zip(rows, other_rows))
    torch.cuda.synchronize()
    guard()
    result = dict(backend='official MOSS architecture at tiny random dimensions', pretrained=False,
        objective='g_adv+required_prior; d_adv+public_penalty', generator_arm='adapter_gan',
        response_shape=list(rows[0].target.shape), signed_checkpoint_replay=True,
        metrics=metrics + [expected], exact_resume=True, exact_export_reload=True,
        frozen_base_unchanged=True, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic() - started, dtype='bfloat16 frozen base / float32 adapters+critic',
        core_signature=game._signature, quality_qualified=False)
    write(directory / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('exact_resume', 'exact_export_reload', 'peak_cuda_bytes', 'elapsed_seconds')}))


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), [int(sys.argv[3]), int(sys.argv[4])])
    elif len(sys.argv) == 2:
        guardian(Path(sys.argv[1]))
    else:
        raise SystemExit('usage: guarded_pg09_canary.py NEW_OUTPUT_DIRECTORY')
