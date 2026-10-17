"""Bounded native YuE2 rendering of a public ConditionalPG09 inference export.

Only a separately reviewed root card may run main(). Importing this module
loads no model. CPU tests use random AR weights and synthetic acoustic output.
No training update, optimizer, lease, retry or process dispatcher lives here.
"""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from conceptmod.textsliders.pg09_audio import (
    AudioResponseCritic, AudioRow, YuE2ARResponseAdapter, audio_conditions,
    require_audio_width,
)
from conceptmod.textsliders.pg09_audio_checks import matched_waveform_checks
from inspect_audio_pg09_assets import bootstrap_yue2_product, runtime_identity
import qualify_audio_pg09_live as producer

RUNS = Path('/mnt/nvme/sliders/qualification/runs/audio-pg09-reviewed-20261017')
PRODUCER_SHA = '019a56ef3475f2d876e6a3c0affee2c4029095d6e5e2bc92ada9ca2c8a4fa3ec'
OPTIONS = dict(pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
PROTOCOL = dict(cot='off', cfg_scale=1.0, backend='torch-eager', quantization='none',
    offload_ar=False, temperature=1.0, top_p=0.95, top_k=100,
    repetition_penalty=1.2, penalty_window=50, min_tokens=32, max_tokens=200,
    ode_steps=32, ode_method='midpoint', sample_rate=48000,
    maximum_audio_seconds=8.1, vae_core_frames=1024, memory_budget_gib=16)


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


@contextmanager
def isolated_rng(model, device=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = {p.device.index for p in model.parameters() if p.is_cuda}
    if device is not None and torch.device(device).type == 'cuda':
        devices.add(torch.device(device).index or 0)
    try:
        with torch.random.fork_rng(devices=sorted(devices)):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass
class Clip:
    tokens: list
    latents: np.ndarray
    audio: np.ndarray
    config: dict
    truncated: dict
    prefix: list


def render_one(pipe, network, request, check=lambda: None, *, scale=0.):
    from yue2.protocol import Sampling, token_prefixes
    if (scale not in (0., .5, 1.) or request.cot != 'off' or request.cfg_scale != 1.0 or request.abc is not None
            or pipe.backend != 'torch-eager' or pipe.quantization != 'none' or pipe.offload_ar):
        raise ValueError('renderer requires explicit cot off, CFG1, eager unquantized AR')
    sampling = Sampling(**{k:PROTOCOL[k] for k in (
        'temperature', 'top_p', 'top_k', 'repetition_penalty', 'penalty_window', 'min_tokens', 'max_tokens')})
    expected = token_prefixes(request, pipe.tokenizer)
    if len(expected) + sampling.max_tokens > 24576:
        raise ValueError('complete prefix plus semantic budget exceeds native context')
    with isolated_rng(pipe._model, getattr(pipe, 'device', None)), torch.inference_mode():
        check()
        with network.scaled(scale) if network else nullcontext():
            plan = pipe.plan(request=request)
            if list(plan.prefix) != expected or plan.request != request or plan.truncated:
                raise ValueError('native plan changed full request/prefix or truncated cot-off input')
            semantic = pipe.generate_semantic(plan, sampling=sampling, cancelled=lambda: (check() or False))
        tokens = list(semantic.tokens)
        if (semantic.plan is not plan or not 32 <= len(tokens) <= 200
                or any(type(v) is not int or not 0 <= v < 32768 for v in tokens)
                or type(semantic.truncated) is not bool):
            raise ValueError('native semantic token count/domain/provenance differs')
        check()
        with network.scaled(0) if network else nullcontext():
            latents = np.asarray(pipe.synthesize(semantic, cancelled=lambda: (check() or False)))
            if (latents.shape not in ((len(tokens), 64), (1, 64, len(tokens)))
                    or latents.dtype != np.float32 or not np.isfinite(latents).all()):
                raise ValueError('native NAR returned invalid latent coordinates')
            check()
            audio = np.asarray(pipe.decode(latents))
        if (audio.ndim != 2 or audio.shape[1] != 2 or not 0 < len(audio) <= int(8.1 * 48000)
                or abs(len(audio)-len(tokens)*1920) > 4800
                or not np.isfinite(audio).all() or audio.dtype != np.float32):
            raise ValueError('native VAE returned invalid stereo48kHz duration/dtype/values')
        config = pipe.effective_config(request, semantic_sampling=sampling)
        if config['cot'] != 'off' or config['cfg_scale'] != 1.0 or config['backend'] != 'torch-eager':
            raise ValueError('effective native protocol differs')
        if (config['generation']['ode_steps'] != 32 or config['generation']['ode_method'] != 'midpoint'
                or config['generation']['semantic'] != sampling.__dict__):
            raise ValueError('native sampling or NAR solver differs')
        check()
        return Clip(tokens, latents.copy(), audio.copy(), config,
                    dict(abc=plan.truncated, semantic=semantic.truncated), expected)


def load_public_export(game, adapter, heldout, exported, qualification, baseline):
    # Producer hashes bind these artifacts externally; this checks their relation.
    if (exported.get('completed_steps') != 20 or qualification['game']['completed_steps'] != 20
            or not exact(exported, qualification['export'])
            or not exact(exported['network'], qualification['game']['network'])
            or exported['config'] != qualification['game']['config']
            or qualification['frozen_base_sha256'] != baseline
            or len(qualification['probes']) != len(heldout)):
        raise ValueError('accepted20-step export/qualification/base provenance differs')
    game.load_export_state_dict(exported)  # Public transactional inference API only.
    if not all(torch.equal(adapter.sample(row), expected.to(adapter.particles.device))
               for row, expected in zip(heldout, qualification['probes'])):
        raise ValueError('public export failed exact producer heldout hidden probes')


def render_case(pipe, bind, exported, qualification, row, index, check=lambda: None, save=None):
    from yue2.protocol import SongRequest
    if index not in (0, 1):
        raise ValueError('exactly the two predeclared heldout cases are supported')
    if getattr(pipe, '_pg09_adapter_attached', False):
        raise ValueError('clean native baseline must precede adapter attachment')
    model = pipe._model
    start = time.monotonic()
    with isolated_rng(model, getattr(pipe, 'device', None)):
        request = SongRequest(style=row['neutral'], lyrics=row['lyrics'], cot='off',
                              cfg_scale=1.0, seed=1709 + index, id=f'heldout-{index}')
        baseline = producer.frozen_digest(model, torch.nn.Module(), check)
        clips = dict(base=render_one(pipe, None, request, check))
        if save:
            save('base', clips['base'], request)
        # decode() offloads the host. Restore that same object before attachment.
        if pipe._load_model() is not model:
            raise ValueError('native pipeline replaced the single reused AR host')
        adapter, game, heldout = bind()
        if adapter.model is not model or pipe._model is not model:
            raise ValueError('adapter/export bound to another native AR host')
        pipe._pg09_adapter_attached = True
        load_public_export(game, adapter, heldout, exported, qualification, baseline)
        before = tensor_digest(adapter.network.state_dict())
        bank_before = tensor_digest(dict(bank=adapter.particles))
        for name, scale, caption in (('off', 0., 'neutral'), ('half', .5, 'neutral'),
                                     ('on', 1., 'neutral'), ('positive_caption', 0., 'positive')):
            request = SongRequest(style=row[caption], lyrics=row['lyrics'], cot='off',
                                  cfg_scale=1.0, seed=1709 + index, id=f'heldout-{index}')
            clips[name] = render_one(pipe, adapter.network, request, check, scale=scale)
            if name == 'off' and (clips['base'].tokens != clips['off'].tokens
                    or not np.array_equal(clips['base'].latents, clips['off'].latents)
                    or not np.array_equal(clips['base'].audio, clips['off'].audio)):
                raise ValueError('native adapter-off token/latent/waveform replay differs from clean base')
            if before != tensor_digest(adapter.network.state_dict()):
                raise ValueError('adapter or actual bank changed during inference')
            if save:
                save(name, clips[name], request)
        if baseline != producer.frozen_digest(model, adapter.network, check):
            raise ValueError('frozen native host changed during rendering')
        checks = matched_waveform_checks({k:v.audio for k,v in clips.items()}, sample_rate=48000,
            maximum_seconds=8.1, truncated={k:v.truncated for k,v in clips.items()})
        checks.update(public_export_hidden_replay_exact=True, frozen_base_unchanged=True,
            frozen_base_sha256=baseline, adapter_unchanged=True, actual_bank_sha256=bank_before,
            native_zero_tokens_latents_waveform_exact=True, seed=1709+index,
            case=index, elapsed_seconds=time.monotonic()-start, protocol=PROTOCOL)
        checks['clips'] = {name:dict(semantic_tokens=len(clip.tokens), latent_shape=list(clip.latents.shape),
            audio_shape=list(clip.audio.shape), truncated=clip.truncated,
            prefix_sha256=hashlib.sha256(json.dumps(clip.prefix).encode()).hexdigest())
            for name,clip in clips.items()}
        return clips, checks


def bind_native(backend, manifest, train_path, heldout_path, check):
    """Rebuild the producer's public binding on the already loaded host; no trainer."""
    import yaml
    from particle_sliders import ConditionalPG09, winning_formulation
    from yue2.slider import ParticleSlider
    if producer.sha_file(producer.__file__) != PRODUCER_SHA:
        raise ValueError('admitted producer source changed')
    torch.manual_seed(7)
    slider = ParticleSlider(backend.model, winning_formulation())
    adapter = YuE2ARResponseAdapter(backend, slider,
        model_identity=dict(manifest=manifest['model_identity'], worker_sha256=PRODUCER_SHA),
        checkpoint_modules=list(backend.model.model.layers), **OPTIONS)
    fixed, heldout = [], []
    for path, required, offset, destination in ((train_path, 4, 0, fixed), (heldout_path, 2, 100, heldout)):
        rows = yaml.safe_load(Path(path).read_text())['rows']
        if len(rows) != required:
            raise ValueError('producer fixed row count differs')
        for index, row in enumerate(rows):
            check()
            neutral = backend.prefix(row['neutral'], row['lyrics'])
            positive = backend.prefix(row['positive'], row['lyrics'])
            with adapter.network.scaled(0):
                history = backend.continuation(neutral, 16, 1709+offset+index)
            packed = AudioRow(tuple(neutral+history), tuple(positive+history), history_tokens=16)
            if max(len(packed.neutral_ids), len(packed.positive_ids)) > 512:
                raise ValueError('complete producer row exceeds512 tokens')
            destination.append(adapter.prepare(packed))
    conditions = audio_conditions(fixed)
    critic = AudioResponseCritic(fixed[0].target.shape[-1], conditions.shape[1]).to(adapter.particles.device)
    game = ConditionalPG09(adapter, fixed, critic, conditions=conditions, particles=adapter.particles,
        response_domain='continuous_vector', preset='bispf_optimized', batch_size=2,
        seed=7, total_steps=20, **OPTIONS)
    return adapter, game, heldout


def validate_card(card):
    if (card.get('schema') != 'root-yue2-pg09-render-admission-v1'
            or card.get('status') != 'REVIEWED_BOUNDED_AUDIO_RENDER' or not card.get('admission_id')
            or card.get('protocol') != PROTOCOL or card.get('case') not in (0, 1)):
        raise ValueError('separate reviewed root renderer admission card required')
    index = card['case']
    if Path(card['output_directory']) != RUNS / f'yue2-matched-case-{index}':
        raise ValueError('bound fresh per-case output directory required')
    limits = card['limits']
    if (limits['host_running_available_gib_minimum'] < 8
            or limits['startup_host_available_gib_minimum'] < 16
            or limits['startup_gpu_free_gib_minimum'] < 20
            or limits['compute_seconds'] != 720 or limits['maximum_wall_seconds'] != 900
            or limits['shutdown_grace_seconds'] != 180):
        raise ValueError('root renderer resource/wall bounds differ')
    if any('site-packages' in p for p in os.environ.get('PYTHONPATH', '').split(':')):
        raise ValueError('whole site-packages injection prohibited')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('reviewed deterministic CUBLAS workspace required')
    if card.get('product') != '/mnt/nvme/sliders/audio-yue2-source-20261017':
        raise ValueError('exact frozen YuE product root required')
    identity = runtime_identity()
    keys = ('sys_executable', 'torch_version', 'torch_file', 'transformers_version',
            'transformers_file', 'particlegan_version', 'particlegan_file')
    expected = card.get('runtime_expected', {})
    if any(key not in expected or identity[key] != expected[key] for key in keys):
        raise ValueError('reviewed native renderer runtime origins differ')
    import particle_sliders.pg09 as core
    repository = Path(producer.__file__).resolve().parents[1]
    native = Path('/ml2/sliders/.venv-yue2-retrain/lib/python3.11/site-packages/yue2')
    required = [str(Path(__file__).resolve()), str(Path(producer.__file__).resolve()),
        str(repository / 'scripts/inspect_audio_pg09_assets.py'), str(Path(core.__file__).resolve()),
        str(Path(core.__file__).with_name('pg09_continuous_profile.json').resolve()),
        str(repository / 'conceptmod/textsliders/pg09_audio.py'),
        str(repository / 'conceptmod/textsliders/pg09_audio_checks.py'),
        str(repository / 'conceptmod/textsliders/pg09_backends.py'),
        '/mnt/nvme/sliders/qualification/ops-music3/music3_resources.py']
    required += [str(path) for path in native.glob('*.py')]
    product_sources = sorted((Path(card['product']) / 'yue2').glob('*.py'))
    if len(product_sources) != 9:
        raise ValueError('exact nine frozen YuE product sources required')
    required += [str(path) for path in product_sources]
    if any(path not in card['source_pins'] for path in required):
        raise ValueError('renderer and admitted producer sources must be bound')
    for field in ('manifest', 'model_dir', 'vae_dir', 'product', 'train_prompts', 'heldout_prompts',
                  'adapter_export', 'qualification'):
        if field not in card:
            raise ValueError('missing root renderer input: '+field)
    for field in ('manifest', 'train_prompts', 'heldout_prompts', 'adapter_export', 'qualification'):
        if card[field] not in card['input_pins']:
            raise ValueError('unbound renderer input: '+field)
    required_assets = [str(Path(card['model_dir']) / name) for name in
                       ('model.safetensors', 'config.json', 'qwen.tiktoken')]
    required_assets += [str(Path(card['vae_dir']) / name) for name in ('model.safetensors', 'config.json')]
    if any(path not in card['input_pins'] for path in required_assets):
        raise ValueError('all renderer model/tokenizer/VAE files must be bound')
    for group in ('source_pins', 'input_pins'):
        for path, expected in card[group].items():
            if producer.sha_file(path) != expected:
                raise ValueError('root renderer '+group+' identity differs: '+path)
    require_audio_width(2048, **OPTIONS)  # Public source gate before CUDA/model loading.
    return card


def host_floor(gib):
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < gib * 1024**2:
        raise RuntimeError(f'host available memory below{gib}GiB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, required=True)
    args = parser.parse_args()
    card = validate_card(json.loads(args.card.read_text()))
    if sys.executable != '/ml2/sliders/.venv-yue2-retrain/bin/python':
        raise ValueError('native YuE interpreter required')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('one root-admitted physical GPU must be visible')
    output = Path(card['output_directory'])
    if output.exists():
        raise FileExistsError('no overwrite or retry of renderer output')
    host_floor(card['limits']['startup_host_available_gib_minimum'])
    if (torch.cuda.device_count() != 1 or torch.cuda.mem_get_info(0)[0]
            < card['limits']['startup_gpu_free_gib_minimum'] * 1024**3):
        raise RuntimeError('fresh admitted renderer memory bound not met')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, '/mnt/nvme/sliders/qualification/ops-music3')
    from music3_resources import ResourceGuard
    resource = ResourceGuard()
    started = time.monotonic()
    def check(*unused):
        if time.monotonic()-started > 720:
            raise RuntimeError('renderer720-second compute budget exhausted')
        host_floor(card['limits']['host_running_available_gib_minimum'])
        resource()
    check()
    bootstrap_yue2_product(card['product'])
    from yue2.backend import YuE2Backend
    from yue2.pipeline import YuE2Pipeline
    from yue2.protocol import GenerationConfig, Sampling
    from yue2.storage import model_identity, resolve_model
    import yaml
    manifest = json.loads(Path(card['manifest']).read_text())
    for kind, field, identity in (('model', 'model_dir', 'model_identity'), ('vae', 'vae_dir', 'vae_identity')):
        record = manifest['records'][kind]
        resolved = Path(resolve_model(record['model'], revision=record['revision'],
            cache_dir=manifest['cache_dir'], local_files_only=True)).resolve()
        if resolved != Path(card[field]).resolve() or resolved != Path(record['path']).resolve():
            raise ValueError('native renderer staged directory differs')
        if model_identity(resolved) != manifest[identity]:
            raise ValueError('native renderer actual model files differ')
    exported = torch.load(card['adapter_export'], map_location='cpu', weights_only=True)
    qualification = torch.load(card['qualification'], map_location='cpu', weights_only=True)
    if exported.get('completed_steps') != 20:
        raise ValueError('accepted20-step public export required before native loading')
    pipe = YuE2Pipeline(card['model_dir'], card['vae_dir'], device='cuda:0', backend='torch-eager',
        quantization='none', offload_ar=False, memory_budget_gib=16, vae_core_frames=1024,
        generation_config=GenerationConfig(semantic=Sampling(min_tokens=32, max_tokens=200)), progress=False)
    record = manifest['records']['model']
    backend = YuE2Backend(record['model'], device='cuda:0', allow_hub=False,
        revision=record['revision'], cache_dir=manifest['cache_dir'])
    if backend.identity != manifest['model_identity']:
        raise ValueError('loaded backend identity differs')
    # Pinned native cache bridge: one host for clean baseline, adapter, NAR and decode.
    pipe._model = backend.model
    for layer in backend.model.model.layers:
        layer.register_forward_pre_hook(check)
    rows = yaml.safe_load(Path(card['heldout_prompts']).read_text())['rows']
    if len(rows) != 2:
        raise ValueError('exact predeclared heldout rows required')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, clip, request):
        import soundfile
        destination = output / name
        destination.mkdir()
        np.save(destination / 'semantic.npy', np.asarray(clip.tokens, dtype=np.int32))
        np.save(destination / 'latent.npy', clip.latents)
        soundfile.write(destination / 'audio.wav', clip.audio, 48000, subtype='FLOAT')
        producer.atomic_json(destination / 'request.json', request.to_dict())
        producer.atomic_json(destination / 'config.json', clip.config)
        producer.atomic_json(destination / 'truncation.json', clip.truncated)
    try:
        _, receipt = render_case(pipe, lambda: bind_native(backend, manifest,
            card['train_prompts'], card['heldout_prompts'], check), exported, qualification,
            rows[card['case']], card['case'], check, save)
        validate_card(card)  # Source/input drift check before acceptance.
        receipt.update(status='native_matched_render_software_checks_passed',
            admission_id=card['admission_id'], card_sha256=producer.sha_file(args.card),
            renderer_sha256=producer.sha_file(__file__), runtime=runtime_identity(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            artifacts={str(p.relative_to(output)):producer.sha_file(p) for p in output.rglob('*') if p.is_file()})
        producer.atomic_json(output / 'receipt.json', receipt)
    except Exception as error:
        producer.atomic_json(output / 'failure.json', dict(status='failed_no_retry', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
