"""Guardian-invoked MOSS GAN worker. Import/config validation are standard-library only.

Execution loads only the admitted local transformer (and VAE for evaluation).
There is no admission logic, downloader, custom loss or optimizer in this file.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import time
import uuid

REVISION = 'fc5de0d56fc910cbca6f6625a70f5f582b26deb1'
SNAPSHOT = Path('/mnt/nvme/sliders/cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-SFT/snapshots') / REVISION
OLD_ROOT = Path('/mnt/nvme/sliders/runs/moss-nonlighting-campaign-v1')
SCALES = {'bad-intent': 0.18700603425178916, 'afterimage': 0.19194795196490175}
OBJECTIVE = 'pg09_gan_only'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.pending-' + uuid.uuid4().hex)
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class BoundRecipe:
    data: dict

    @classmethod
    def read(cls, path):
        return cls.validate(json.loads(Path(path).read_text()))

    @classmethod
    def validate(cls, value):
        if value.get('schema') != 'moss-pg09-worker-v1' or value.get('objective') != OBJECTIVE:
            raise ValueError('worker requires the reviewed GAN-only recipe schema')
        concept = value.get('concept')
        if concept not in SCALES:
            raise ValueError('single-case Bad Intent or Afterimage only')
        fixed = dict(model_revision=REVISION, pg_version='0.9.0', preset='bispf_optimized',
                     total_steps=3840, batch_size=8, training_seed=1000,
                     checkpoint_every=80, wall_seconds=7 * 24 * 3600,
                     signed_rows=15360, response_domain='continuous_field', sampling='balanced')
        for key, expected in fixed.items():
            if value.get(key) != expected or type(value.get(key)) is not type(expected):
                raise ValueError('recipe fixed field differs: ' + key)
        if not math.isclose(value.get('response_scale', -1), SCALES[concept], rel_tol=0, abs_tol=1e-14):
            raise ValueError('frozen training response scale differs')
        if Path(value.get('model_snapshot', '')).resolve() != SNAPSHOT.resolve():
            raise ValueError('only the official pinned local snapshot is allowed')
        cache = OLD_ROOT / concept
        if Path(value.get('cache_directory', '')).resolve() != cache.resolve():
            raise ValueError('original reviewed cache location differs')
        old = cache / 'checkpoints/step-001600/adapter'
        if Path(value.get('old_mse_adapter', '')).resolve() != old.resolve():
            raise ValueError('old comparison must use the preserved 1600-step adapter')
        output = Path(value.get('output_directory', '')).resolve()
        allowed_output = any(output.is_relative_to(root.resolve()) for root in
                             (Path('/mnt/nvme/sliders/runs'), Path('/mnt/nvme/sliders/qualification/runs')))
        if (not allowed_output or
                output.is_relative_to(OLD_ROOT.resolve()) or output.is_relative_to(SNAPSHOT.parent.resolve())):
            raise ValueError('output must be a separate new slider run, preserving original caches and model')
        for key in ('cache_identity_sha256', 'resolved_recipe_sha256', 'old_adapter_sha256', 'old_manifest_sha256'):
            if not isinstance(value.get(key), str) or len(value[key]) != 64:
                raise ValueError('missing reviewed hash: ' + key)
        if value.get('critic_width', 32) != 32:
            raise ValueError('reviewed live critic width must remain 32')
        guard = value.get('resource_guard', {})
        if not isinstance(guard.get('path'), str) or len(guard.get('sha256', '')) != 64:
            raise ValueError('source-pinned original ResourceGuard is required')
        pins = value.get('source_pins')
        if (not isinstance(pins, dict) or not pins or
                any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 for k, v in pins.items())):
            raise ValueError('worker/core/backend/guardian source pins required')
        required = {'pg09_moss_worker.py', 'pg09_moss_campaign.py', 'pg09_backends.py', 'pg09.py',
                    'pipeline_moss_image.py', 'lora.py'}
        if not required.issubset({Path(p).name for p in pins}):
            raise ValueError('source pins must cover worker, core, helper and native MOSS implementation')
        paths = value.get('lease_paths')
        if not isinstance(paths, list) or len(paths) != 2 or len(set(paths)) != 2:
            raise ValueError('both inherited lease paths must be bound')
        return cls(json.loads(json.dumps(value)))

    @property
    def identity(self):
        return digest(self.data)

    def verify_sources(self):
        if str(Path(__file__).resolve()) not in {str(Path(p).resolve()) for p in self.data['source_pins']}:
            raise ValueError('executing worker itself is not source-pinned')
        for path, expected in self.data['source_pins'].items():
            if sha_file(path) != expected:
                raise ValueError('reviewed source changed: ' + path)
        guard = self.data['resource_guard']
        if sha_file(guard['path']) != guard['sha256']:
            raise ValueError('ResourceGuard source changed')

    def verify_module_origin(self, module):
        origin = getattr(module, '__file__', None)
        if origin is None:
            raise ValueError('loaded module has no verifiable source origin')
        origin = str(Path(origin).resolve())
        pins = {str(Path(p).resolve()): h for p, h in self.data['source_pins'].items()}
        if origin not in pins or sha_file(origin) != pins[origin]:
            raise ValueError('actual loaded module source is not pinned: ' + origin)
        return origin

    def verify_loaded_sources(self):
        # Import definitions only, before any model loading or CUDA allocation.
        # Verify actual Python resolution; arbitrary listed files are insufficient.
        modules = ('conceptmod.textsliders.pg09_moss_campaign',
                   'conceptmod.textsliders.pg09_backends', 'particle_sliders.pg09',
                   'moss_sliders.vendor.pipeline_moss_image', 'moss_sliders.lora')
        return {name: self.verify_module_origin(importlib.import_module(name)) for name in modules}

    def verify_output(self, output):
        if Path(output).resolve() != Path(self.data['output_directory']).resolve():
            raise ValueError('output differs from reviewed recipe')


def inherited_leases(fds, expected_paths):
    """Validate inherited handles only; never acquire, release or replace leases."""
    if len(fds) != 2 or len(set(fds)) != 2:
        raise ValueError('guardian must pass two distinct inherited lease FDs')
    actual = []
    for fd in fds:
        fcntl.fcntl(fd, fcntl.F_GETFD)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('lease FD must refer to the original regular lock file')
        actual.append(str(Path(os.readlink(f'/proc/self/fd/{fd}')).resolve()))
    if set(actual) != {str(Path(p).resolve()) for p in expected_paths}:
        raise ValueError('inherited lease targets differ from guardian recipe')
    return actual


def tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.layout == b.layout and
                (torch.equal(a, b) if a.device == b.device else torch.equal(a.cpu(), b.cpu())))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def parameter_sha(module, *, frozen_only=False, guard=None):
    """Stream one parameter to CPU at a time; never clone the full base state."""
    import torch
    h = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if frozen_only and parameter.requires_grad:
                raise ValueError('pretrained base unexpectedly contains trainable parameters')
            if guard is not None:
                guard()
            tensor = parameter.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str((list(tensor.shape), str(tensor.dtype))).encode())
            raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
            for offset in range(0, len(raw), 1024 * 1024):
                h.update(raw[offset:offset + 1024 * 1024])
            del raw, tensor
    return h.hexdigest()


class GradientWitness:
    """Last observed backward norm per parameter; never contributes to a loss."""
    def __init__(self, module):
        self.last = {}
        self.handles = [p.register_hook(self._hook(i)) for i, p in enumerate(module.parameters()) if p.requires_grad]

    def _hook(self, index):
        def record(gradient):
            self.last[index] = gradient.detach().float().square().sum()
        return record

    def reset(self):
        self.last.clear()

    def norm(self):
        import torch
        return float(torch.stack(list(self.last.values())).sum().sqrt()) if self.last else 0.

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def no_optimized_mse():
    import torch.nn.functional as functional
    original = functional.mse_loss
    def reject(*args, **kwargs):
        raise RuntimeError('MSE is forbidden in the corrected GAN worker')
    functional.mse_loss = reject
    try:
        yield
    finally:
        functional.mse_loss = original


def save_checkpoint(output, recipe, run, base_sha):
    import torch
    step = run.game.completed_steps
    destination = Path(output) / 'checkpoints' / f'step-{step:06d}'
    packet = dict(schema='moss-pg09-worker-checkpoint-v1', recipe_sha256=recipe.identity,
                  frozen_base_sha256=base_sha, game=run.game.state_dict())
    pointer = Path(output) / 'LATEST.json'
    if pointer.exists() and json.loads(pointer.read_text())['step'] > step:
        raise ValueError('cannot move latest checkpoint backwards')
    if destination.exists():
        meta = verify_checkpoint(destination, recipe)
        existing = torch.load(destination / 'state.pt', map_location='cpu', weights_only=True)
        if meta['frozen_base_sha256'] != base_sha or not tree_equal(existing, packet):
            raise ValueError('divergent same-step checkpoint already exists')
        atomic_json(pointer, dict(directory=destination.name, step=step, recipe_sha256=recipe.identity))
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / ('.pending-' + uuid.uuid4().hex)
    temporary.mkdir()
    with (temporary / 'state.pt').open('wb') as stream:
        torch.save(packet, stream)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(temporary / 'checkpoint.json', dict(schema=packet['schema'], step=step,
                recipe_sha256=recipe.identity, frozen_base_sha256=base_sha,
                state_sha256=sha_file(temporary / 'state.pt')))
    temporary.rename(destination)
    atomic_json(Path(output) / 'LATEST.json', dict(directory=destination.name, step=step,
                                                recipe_sha256=recipe.identity))
    return destination


def verify_checkpoint(directory, recipe):
    directory = Path(directory)
    meta = json.loads((directory / 'checkpoint.json').read_text())
    if (meta.get('schema') != 'moss-pg09-worker-checkpoint-v1' or meta.get('recipe_sha256') != recipe.identity or
            directory.name != f"step-{meta['step']:06d}" or
            sha_file(directory / 'state.pt') != meta.get('state_sha256')):
        raise ValueError('GAN checkpoint checksum/schema/recipe differs; MSE resume forbidden')
    return meta


def restore_latest(output, recipe, run, base_sha):
    import torch
    output = Path(output)
    pointer = output / 'LATEST.json'
    published = json.loads(pointer.read_text()) if pointer.exists() else None
    if published is not None and (published.get('recipe_sha256') != recipe.identity or
                                 Path(published['directory']).name != published['directory']):
        raise ValueError('latest checkpoint identity/path differs')
    candidates = sorted((output / 'checkpoints').glob('step-*'), reverse=True)
    if not candidates:
        if published is not None:
            raise ValueError('published latest checkpoint is missing')
        return None
    directory = candidates[0]
    if directory.resolve().parent != (output / 'checkpoints').resolve():
        raise ValueError('checkpoint path escapes the new run')
    meta = verify_checkpoint(directory, recipe)
    if meta['frozen_base_sha256'] != base_sha:
        raise ValueError('frozen pretrained base differs')
    if published is not None and (published['step'] > meta['step'] or
                                 (published['step'] == meta['step'] and published['directory'] != directory.name)):
        raise ValueError('latest checkpoint pointer disagrees with immutable checkpoint')
    packet = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
    if (packet.get('schema') != meta['schema'] or packet.get('recipe_sha256') != recipe.identity or
            packet.get('frozen_base_sha256') != base_sha or
            packet['game']['completed_steps'] != meta['step']):
        raise ValueError('GAN checkpoint packet differs')
    run.game.load_state_dict(packet['game'])
    # Recover an interruption after complete directory publication but before LATEST.
    atomic_json(pointer, dict(directory=directory.name, step=meta['step'], recipe_sha256=recipe.identity))
    return directory


class TeacherParityError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('frozen teacher/cache parity failed before update: ' + json.dumps(diagnostics))


def parity_diagnostics(run, indices=(0, 39), *, relative_limit=1e-6, emit=None):
    import torch
    from dataclasses import replace
    results = []
    for index in indices:
        cached = run.rows[index]
        with torch.no_grad():
            recomputed = run.backend.prepare(cached.row)
        denominator = (cached.target - cached.neutral).float().square().mean().clamp_min(1e-20)
        relative = float((recomputed.target - cached.target.to(recomputed.target)).square().mean() / denominator)
        base_relative = float((recomputed.neutral - cached.neutral.to(recomputed.neutral)).square().mean() / denominator)
        rendering = replace(cached.row, negative=tuple(t.to(run.rows.device) for t in run.rows.context('')), guidance=3.)
        results.append(dict(index=index, sign=cached.row.scale, native_time=float(cached.row.timestep),
                            relative_target_mse=relative, relative_base_mse=base_relative,
                            relative_target_rms=math.sqrt(relative), relative_base_rms=math.sqrt(base_relative),
                            threshold_relative_mse=relative_limit, threshold_relative_rms=math.sqrt(relative_limit),
                            teacher_negative='neutral', cfg_negative='empty',
                            cfg_negative_is_explicit=not torch.equal(rendering.negative[0], rendering.neutral[0])))
        if emit is not None:
            emit('teacher_parity_state', diagnostic=results[-1])
    if any(not math.isfinite(r['relative_target_mse'] + r['relative_base_mse']) or
           max(r['relative_target_mse'], r['relative_base_mse']) > relative_limit for r in results):
        raise TeacherParityError(results)
    return results


def probe_replay(run, before_last, expected_metrics, expected_state):
    """Replay the final probe update on the same fixed full-budget recipe."""
    expected_sample = run.game.evaluate(0, seed=29001)
    run.game.load_state_dict(before_last)
    actual = run.step()
    if not tree_equal(actual, expected_metrics) or not tree_equal(run.game.state_dict(), expected_state):
        raise RuntimeError('probe exact loss/full-state replay failed')
    if not tree_equal(run.game.evaluate(0, seed=29001), expected_sample):
        raise RuntimeError('probe exact sample replay failed')
    return dict(exact_loss=True, exact_full_state=True, exact_sample=True)


def coverage(game):
    return {phase: dict(total_exposures=int(counts.sum()), unique_rows=int((counts > 0).sum()),
                        min_exposures=int(counts.min()), max_exposures=int(counts.max()),
                        cycle_cursor=game.row_cursors[phase], rows=len(counts))
            for phase, counts in game.exposures.items()}


def telemetry():
    import torch
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    current_rss = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmRSS:'))
    return dict(process_rss_kib=current_rss, rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                host_available_kib=available, cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def load_guard(recipe):
    source = recipe.data['resource_guard']
    spec = importlib.util.spec_from_file_location('reviewed_moss_resource_guard', source['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (Path(module.__file__).resolve() != Path(source['path']).resolve() or
            sha_file(module.__file__) != source['sha256']):
        raise ValueError('actual loaded ResourceGuard source differs')
    return module.ResourceGuard()


def load_admitted_run(recipe, guard):
    import torch
    from dataclasses import asdict
    from moss_sliders.vendor.pipeline_moss_image import MOSSImageTransformer
    from .pg09_moss_campaign import MossCampaignRows, build_campaign_run
    values = recipe.data
    rows = MossCampaignRows(values['cache_directory'], split='train', device='cpu')
    if (digest(rows.identity()) != values['cache_identity_sha256'] or len(rows) != 15360 or
            rows.response_shape != (1, 128, 32, 32) or rows.condition_width != 4101):
        raise ValueError('reviewed lazy cache/condition identity differs')
    guard()
    model = MOSSImageTransformer.from_pretrained(values['model_snapshot'], subfolder='transformer',
                                                local_files_only=True, torch_dtype=torch.bfloat16)
    model.requires_grad_(False).eval().to('cuda:0')
    guard.attach(model)
    base_sha = parameter_sha(model, frozen_only=True, guard=guard)
    identity = dict(model='OpenMOSS-Team/MOSS-Image-SFT', revision=REVISION, frozen_parameter_sha256=base_sha)
    run = build_campaign_run(model, rows, model_identity=identity, response_scale=values['response_scale'],
                             batch_size=8, total_steps=3840, seed=1000, checkpointing=True,
                             critic_width=values.get('critic_width', 32), recipe_overrides=values.get('recipe_overrides'))
    if digest(asdict(run.game.recipe)) != values['resolved_recipe_sha256']:
        raise ValueError('actual public PG recipe differs from reviewed resolved fields')
    guard.attach(run.game.critic)
    return run, base_sha


def evaluate_common_protocol(run, recipe, output, guard, emit):
    import torch
    import numpy as np
    from safetensors.torch import load_file
    from diffusers import AutoencoderKLFlux2
    from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler
    values = recipe.data
    if run.game.completed_steps != 3840:
        raise ValueError('sustained evaluation requires the completed fixed training budget')
    old = Path(values['old_mse_adapter'])
    if sha_file(old / 'adapter.safetensors') != values['old_adapter_sha256'] or sha_file(old / 'slider.json') != values['old_manifest_sha256']:
        raise ValueError('preserved old MSE comparison export changed')
    manifest = json.loads((old / 'slider.json').read_text())
    slider = run.backend.slider
    if (manifest['revision'] != REVISION or manifest['base_model'] != 'OpenMOSS-Team/MOSS-Image-SFT' or
            manifest['rank'] != slider.rank or manifest['alpha'] != slider.alpha or manifest['targets'] != slider.targets):
        raise ValueError('old MSE adapter architecture/model differs')
    old_state = load_file(str(old / 'adapter.safetensors'), device='cpu')
    current = slider.state_dict()
    if old_state.keys() != current.keys() or any(old_state[k].shape != current[k].shape or not old_state[k].isfinite().all() for k in current):
        raise ValueError('old MSE adapter schema/values differ')
    export = Path(output) / 'exports/step-003840.safetensors'
    if not export.exists():
        run.export_adapter(export)
    expected = run.game.evaluate(0, seed=29001)
    state = run.game.state_dict()
    run.load_adapter(export)
    if not tree_equal(expected, run.game.evaluate(0, seed=29001)):
        raise RuntimeError('GAN exported/reloaded response differs before rendering')
    run.game.load_state_dict(state)
    guard()
    vae = AutoencoderKLFlux2.from_pretrained(values['model_snapshot'], subfolder='vae',
                                            local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    guard.attach(vae)
    pipe = MOSSImagePipeline(run.backend.transformer, MOSSImageScheduler(), vae=vae)
    pipe.set_progress_bar_config(disable=True)
    gan_weights = {k: v.detach().clone() for k, v in slider.state_dict().items()}
    results = []
    try:
        for arm in ('gan', 'old-mse-common-protocol'):
            slider.load_state_dict(gan_weights if arm == 'gan' else old_state, strict=True)
            for case in run.render_cases():
                guard()
                target = Path(output) / 'evaluation' / arm / f"{case['row_id']}-{case['label']}.png"
                if target.exists() or target.with_suffix('.json').exists():
                    raise FileExistsError('evaluation outputs are immutable: ' + str(target))
                started = time.monotonic()
                image = run.render(pipe, case)
                if isinstance(image, list):
                    image = image[0]
                if not np.isfinite(np.asarray(image)).all():
                    raise FloatingPointError('nonfinite decoded image')
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name('.pending-' + uuid.uuid4().hex + '.png')
                image.save(pending, format='PNG')
                pending.replace(target)
                receipt = dict(case, arm=arm, recipe_sha256=recipe.identity, sha256=sha_file(target),
                               adapter_sha256=sha_file(export) if arm == 'gan' else values['old_adapter_sha256'],
                               seconds=time.monotonic() - started, model_revision=REVISION,
                               old_images_preserved=True, quality_qualified=False, **telemetry())
                atomic_json(target.with_suffix('.json'), receipt)
                results.append(receipt)
                emit('image_saved', file=str(target), arm=arm, completed_step=run.game.completed_steps)
    finally:
        slider.load_state_dict(gan_weights, strict=True)
    return results


def execute(recipe, output, mode, lease_fds, *, initial_qualifier=None):
    """Called only by the guardian; importing this module never reaches this path."""
    recipe.verify_output(output)
    recipe.verify_sources()
    leases = inherited_leases(lease_fds, recipe.data['lease_paths'])
    loaded_sources = recipe.verify_loaded_sources()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit():
        raise ValueError('guardian must expose exactly one physical CUDA card')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('admitted CUDA device unavailable; no CPU/MSE fallback')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / 'worker-recipe.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != recipe.data:
        raise ValueError('run output already belongs to a different reviewed recipe')
    if not identity_file.exists():
        atomic_json(identity_file, recipe.data)
    started = time.monotonic()
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    previous_signals = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    progress = dict(schema='moss-pg09-worker-progress-v1', mode=mode, recipe_sha256=recipe.identity,
                    pid=os.getpid(), physical_gpu=visible, lease_paths=leases,
                    loaded_sources=loaded_sources, status='loading', latest_step=0)
    def emit(event, **fields):
        progress.update(fields, event=event, elapsed_seconds=time.monotonic() - started)
        atomic_json(output / 'progress.json', progress)
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(progress, allow_nan=False) + '\n')
        print(json.dumps(progress, allow_nan=False), flush=True)
    run = None
    guard = load_guard(recipe)
    try:
        emit('loading', status='loading')
        run, base_sha = load_admitted_run(recipe, guard)
        restore_latest(output, recipe, run, base_sha)
        parity = parity_diagnostics(run, emit=emit)
        if initial_qualifier is not None and run.game.completed_steps == 0:
            initial_qualifier(run, emit=emit)
        if not (output / 'LATEST.json').exists():
            save_checkpoint(output, recipe, run, base_sha)
        emit('teacher_parity', status='ready', latest_step=run.game.completed_steps, parity=parity,
             frozen_base_sha256=base_sha, condition_provenance=run.rows.identity())
        g_before, d_before = parameter_sha(run.backend.network), parameter_sha(run.game.critic)
        witness_g, witness_d = GradientWitness(run.backend.network), GradientWitness(run.game.critic)
        before_last = None
        metrics = []
        with no_optimized_mse():
            if mode == 'evaluate':
                images = evaluate_common_protocol(run, recipe, output, guard, emit)
                emit('evaluation_complete', status='evaluated', images=len(images), quality_qualified=False)
            else:
                target_step = 3 if mode == 'probe' else 3840
                if run.game.completed_steps >= target_step:
                    raise ValueError('requested phase already reached its fixed stopping boundary')
                while run.game.completed_steps < target_step and not stop:
                    if time.monotonic() - started >= recipe.data['wall_seconds']:
                        stop = True
                        break
                    guard()
                    witness_g.reset()
                    witness_d.reset()
                    torch.cuda.reset_peak_memory_stats()
                    if mode == 'probe' and run.game.completed_steps == 2:
                        before_last = run.game.state_dict()
                    before = time.monotonic()
                    value = run.step()
                    torch.cuda.synchronize()
                    value.update(step_seconds=time.monotonic() - before,
                                 g_observed_backward_norm=witness_g.norm(), d_observed_backward_norm=witness_d.norm(),
                                 cooling_seconds=guard.cooling_seconds, **telemetry())
                    if not all(math.isfinite(value[k]) and value[k] > 0 for k in ('g_observed_backward_norm', 'd_observed_backward_norm')):
                        raise RuntimeError('actual generator/critic backward witness is zero or nonfinite')
                    metrics.append(value)
                    emit('optimizer_step', status='training', latest_step=run.game.completed_steps, metrics=value)
                    if run.game.completed_steps % 80 == 0 or stop:
                        checkpoint = save_checkpoint(output, recipe, run, base_sha)
                        emit('checkpoint_saved', checkpoint=str(checkpoint),
                             coverage=coverage(run.game))
                if mode == 'probe' and run.game.completed_steps == 3 and not stop:
                    expected = run.game.state_dict()
                    metric = {k: v for k, v in metrics[-1].items() if k not in ('step_seconds', 'g_observed_backward_norm', 'd_observed_backward_norm', 'cooling_seconds') and not k.startswith(('process_', 'rss_', 'host_', 'cuda_'))}
                    reference = dict(schema='moss-file-resume-reference-v1', recipe_sha256=recipe.identity,
                        frozen_base_sha256=base_sha, before=before_last, expected=expected,
                        metrics=metric, sample=run.game.evaluate(0, seed=29001))
                    reference_path = output / 'probe-reference.pt'
                    with reference_path.open('xb') as stream:
                        torch.save(reference, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    file_reference = torch.load(reference_path, map_location='cpu', weights_only=True)
                    replay = probe_replay(run, file_reference['before'], metric, expected)
                    export = output / 'exports/step-000003.safetensors'
                    run.export_adapter(export)
                    sample = run.game.evaluate(0, seed=29001)
                    run.load_adapter(export)
                    if not tree_equal(sample, run.game.evaluate(0, seed=29001)):
                        raise RuntimeError('probe adapter export/reload differs')
                    run.game.load_state_dict(expected)
                    emit('probe_replay', exact_resume=replay, exact_adapter_reload=True,
                         file_backed_resume_exact=True, probe_reference_sha256=sha_file(reference_path))
                checkpoint = save_checkpoint(output, recipe, run, base_sha)
                emit('checkpoint_saved', checkpoint=str(checkpoint), coverage=coverage(run.game))
                g_after, d_after = parameter_sha(run.backend.network), parameter_sha(run.game.critic)
                if not stop and (g_before == g_after or d_before == d_after):
                    raise RuntimeError('actual generator or critic parameters did not change')
                emit('phase_complete', status='stopped' if stop else ('probe_passed' if mode == 'probe' else 'trained'),
                     latest_step=run.game.completed_steps, generator_changed=g_before != g_after,
                     critic_changed=d_before != d_after, g_before_sha256=g_before, g_after_sha256=g_after,
                     d_before_sha256=d_before, d_after_sha256=d_after, **telemetry())
        final_base = parameter_sha(run.backend.transformer, frozen_only=True, guard=guard)
        if final_base != base_sha:
            raise RuntimeError('frozen pretrained parameter SHA changed')
        emit('frozen_base_verified', frozen_base_after_sha256=final_base, frozen_base_unchanged=True,
             guard_cooling_seconds=guard.cooling_seconds)
        witness_g.close()
        witness_d.close()
        return progress
    except BaseException as error:
        emit('failed', status='failed', error=repr(error),
             latest_step=run.game.completed_steps if run is not None else 0,
             diagnostics=getattr(error, 'diagnostics', None))
        raise
    finally:
        for s, handler in previous_signals.items():
            signal.signal(s, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['probe', 'train', 'evaluate'])
    parser.add_argument('--lease-fd', type=int, action='append', required=True)
    args = parser.parse_args(argv)
    execute(BoundRecipe.read(args.recipe), args.output, args.mode, args.lease_fd)


if __name__ == '__main__':
    main()
