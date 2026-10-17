"""Bounded read-only two-stack MOSS cache-parity diagnostic; no optimizer or adapters."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/nvme/sliders/qualification')
RUNTIMES = {
    'original': '/mnt/nvme/sliders/moss-image-sliders/.venv/bin/python',
    'pg090': '/mnt/nvme/sliders/qualification/envs/pg090/bin/python',
}
WHEEL = ROOT / 'pg09-audit/wheel'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_card(path):
    card = json.loads(Path(path).read_text())
    if (card.get('schema') != 'moss-readonly-runtime-diagnostic-v1' or
            card.get('runtimes') != RUNTIMES or card.get('indices') != [0, 39] or
            card.get('optimizer_updates') != 0 or card.get('wall_seconds') != 900):
        raise ValueError('diagnostic scope differs')
    if card.get('admission') != 'REVIEWED_READONLY_DIAGNOSTIC':
        raise ValueError('diagnostic review incomplete')
    pins = card['source_pins']
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('diagnostic source changed: ' + path)
    if str(Path(__file__).resolve()) not in pins:
        raise ValueError('executing diagnostic is not bound')
    recipe = REPO / 'qualification/moss-reruns/bad-intent.json'
    if sha(recipe) != card['training_recipe_file_sha256']:
        raise ValueError('reviewed source/cache recipe changed')
    from conceptmod.textsliders.pg09_moss_worker import BoundRecipe
    bound = BoundRecipe.read(recipe)
    bound.verify_sources()
    output = Path(card['output_directory'])
    if output.parent != ROOT / 'runs' or not output.name.startswith('moss-runtime-diagnostic-'):
        raise ValueError('diagnostic must use a separate new output')
    return card, bound


def tensor_sha(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((list(tensor.shape), str(tensor.dtype))).encode())
    h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def error_metrics(actual, expected, edit):
    import torch
    if not all(torch.isfinite(t).all() for t in (actual, expected, edit)):
        raise FloatingPointError('nonfinite raw diagnostic tensor')
    delta = actual.double() - expected.double()
    rms = float(delta.square().mean().sqrt())
    return dict(exact=torch.equal(actual, expected), error_rms=rms,
                base_relative_rms=rms / max(float(expected.double().square().mean().sqrt()), 1e-20),
                edit_relative_rms=rms / max(float(edit.double().square().mean().sqrt()), 1e-20),
                max_abs=float(delta.abs().max()))


def worker(card_path, name, fds):
    card, recipe = validate_card(card_path)
    from conceptmod.textsliders.pg09_moss_worker import inherited_leases, load_guard, parameter_sha
    inherited_leases(fds, recipe.data['lease_paths'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('diagnostic physical GPU must be zero')
    origins = recipe.verify_loaded_sources()
    import torch
    import diffusers
    import transformers
    import numpy
    from particle_sliders.pg09 import validate_pg09_domain
    from conceptmod.textsliders.pg09_moss_campaign import MossCampaignRows
    from moss_sliders.vendor.pipeline_moss_image import MOSSImageTransformer
    from safetensors.torch import save_file
    profile = validate_pg09_domain((1,128,32,32), response_domain='continuous_field')
    expected = {'original': ('2.9.1+cu128', '0.38.0', '5.3.0'), 'pg090': ('2.11.0+cu128', '0.40.0.dev0', '5.15.0')}[name]
    if tuple(str(m.__version__) for m in (torch, diffusers, transformers)) != expected:
        raise RuntimeError('actual runtime package versions differ from reviewed card')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    output = Path(card['output_directory']) / name
    output.mkdir(exist_ok=False)
    report = dict(schema=card['schema'], runtime=name, executable=sys.executable,
        modules={m.__name__: dict(version=str(m.__version__), path=m.__file__)
                 for m in (torch, diffusers, transformers, numpy)},
        cuda=str(torch.version.cuda), profile=profile, imported_sources=origins,
        optimizer_updates=0, cache_mutated=False, states=[])
    write(output / 'report.json', report)
    guard = load_guard(recipe)
    rows = MossCampaignRows(recipe.data['cache_directory'], split='train', device='cpu')
    guard()
    model = MOSSImageTransformer.from_pretrained(recipe.data['model_snapshot'], subfolder='transformer',
        local_files_only=True, torch_dtype=torch.bfloat16).requires_grad_(False).eval().to('cuda:0')
    guard.attach(model)
    report['frozen_parameter_sha256'] = parameter_sha(model, frozen_only=True, guard=guard)
    report['buffers'] = {key: dict(sha256=tensor_sha(value), dtype=str(value.dtype), shape=list(value.shape),
        rms=float(value.float().square().mean().sqrt())) for key, value in model.named_buffers()}
    report['config'] = dict(model.config)
    write(output / 'report.json', report)
    buffers = {key: value.detach().cpu().contiguous() for key, value in model.named_buffers()}
    if buffers:
        save_file(buffers, str(output / 'buffers.safetensors'))
    outputs = {}
    def flags():
        return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp=torch.backends.cuda.math_sdp_enabled())
    for policy in ('native-default', 'deterministic', 'math-sdpa'):
        if policy != 'native-default':
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
        if policy == 'math-sdpa':
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
        for index in card['indices']:
            cached = rows[index]
            row = cached.row
            data = rows._bundle(rows.address(index).bundle)
            address = rows.address(index)
            edit = data['delta'][address.path,address.position]
            record = dict(policy=policy, index=index, flags=flags(), native_time=float(row.timestep),
                input_sha256=dict(latent=tensor_sha(row.latent), timestep=tensor_sha(row.timestep),
                    neutral=tensor_sha(row.neutral[0]), positive=tensor_sha(row.positive[0]),
                    mask=tensor_sha(row.neutral[1])), outputs={})
            for role, context, expected in (('neutral', row.neutral, cached.neutral),
                    ('positive', row.positive, cached.neutral + edit)):
                raw = []
                for repeat in range(2):
                    guard()
                    with torch.inference_mode():
                        result = model(row.latent.to('cuda'), row.timestep.to('cuda'),
                            context[0].to('cuda'), context[1].to('cuda')).sample.float().cpu()
                    if not result.isfinite().all():
                        raise FloatingPointError('raw model output nonfinite')
                    raw.append(result)
                prefix = f'{policy}-{index}-{role}'
                outputs[prefix] = raw[0].contiguous()
                record['outputs'][role] = dict(sha256=tensor_sha(raw[0]), repeat_exact=torch.equal(*raw),
                    **error_metrics(raw[0], expected, edit))
            report['states'].append(record)
            write(output / 'report.json', report)
    report['frozen_parameter_after_sha256'] = parameter_sha(model, frozen_only=True, guard=guard)
    report['frozen_base_unchanged'] = report['frozen_parameter_sha256'] == report['frozen_parameter_after_sha256']
    if not report['frozen_base_unchanged']:
        raise RuntimeError('read-only diagnostic changed frozen weights')
    report['buffer_after_sha256'] = {key: tensor_sha(value) for key, value in model.named_buffers()}
    report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save_file(outputs, str(output / 'raw-outputs.safetensors'))
    report['raw_outputs_sha256'] = sha(output / 'raw-outputs.safetensors')
    report['status'] = 'diagnostic_complete'
    write(output / 'report.json', report)


def guardian(card_path):
    card, recipe = validate_card(card_path)
    import guarded_pg09_canary as resources
    output = Path(card['output_directory'])
    if output.exists():
        raise FileExistsError('diagnostic output exists; no retry')
    leases = []
    for path in recipe.data['lease_paths']:
        lease = Path(path).open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(lease)
    if not os.path.ismount('/mnt/nvme') or subprocess.check_output(
            ['findmnt','-no','SOURCE','--target','/mnt/nvme'],text=True).strip() != '/dev/nvme0n1p2':
        raise RuntimeError('expected NVMe missing')
    hold = json.loads((ROOT / 'MSE-DISPATCH-HOLD.json').read_text())
    if hold.get('mse_only_training_allowed') is not False or hold.get('automatic_mse_restart_or_dispatch_allowed') is not False:
        raise RuntimeError('MSE hold changed')
    initial = resources.status()
    if initial['temperature_c'] >= 76 or resources.processes() - resources.RESIDENTS:
        raise RuntimeError('cooling or unowned process blocks diagnostic')
    start = time.time()
    resources.kernel_check(start)
    output.mkdir()
    receipt = dict(schema=card['schema'], card_sha256=sha(card_path), optimizer_updates=0,
        initial=initial, children=[], status='running', lease_paths=recipe.data['lease_paths'])
    process = None
    try:
        for name, executable in RUNTIMES.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', HF_HOME='/mnt/nvme/sliders/cache/huggingface',
                TMPDIR='/mnt/nvme/sliders/tmp', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONPATH=os.pathsep.join((str(REPO),str(REPO/'packages/particle-sliders-core/src'),
                    '/ml2/codex-storage/future-workspaces/2026-10-16/task-6/moss-debug-dev/src',str(WHEEL))))
            command = [executable, '-u', str(Path(__file__).resolve()), '--card',str(card_path.resolve()),
                       '--worker',name]
            for lease in leases:
                command += ['--lease-fd', str(lease.fileno())]
            with (output / (name + '.log')).open('x') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    cwd=REPO, start_new_session=True, pass_fds=tuple(x.fileno() for x in leases))
                child = dict(runtime=name, pid=process.pid, command=command)
                receipt['children'].append(child)
                write(output/'admission.json',receipt)
                while process.poll() is None:
                    if time.time()-start >= 900:
                        raise RuntimeError('diagnostic total 900-second deadline')
                    current = resources.status()
                    if resources.processes() - resources.RESIDENTS - {process.pid}:
                        raise RuntimeError('unowned GPU process appeared')
                    resources.kernel_check(start)
                    with (output/'telemetry.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(runtime=name, elapsed_seconds=time.time()-start,**current))+'\n')
                    time.sleep(2)
                child['exit_code'] = process.returncode
                if process.returncode:
                    raise RuntimeError('diagnostic child failed; no retry')
                result = json.loads((output/name/'report.json').read_text())
                if result.get('status') != 'diagnostic_complete' or not result.get('frozen_base_unchanged'):
                    raise RuntimeError('diagnostic proof missing')
        resources.kernel_check(start)
        receipt['status'] = 'completed'
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
        receipt['elapsed_seconds'] = time.time()-start
        write(output/'admission.json',receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--worker', choices=RUNTIMES)
    parser.add_argument('--lease-fd', action='append', type=int)
    args = parser.parse_args()
    worker(args.card,args.worker,args.lease_fd) if args.worker else guardian(args.card)


if __name__ == '__main__':
    main()
