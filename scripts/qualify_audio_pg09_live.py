"""Bounded shared-game worker. CUDA execution requires a core-owner lease.

The worker owns loading, fixed rows and receipts; ConditionalPG09 owns every
training update. It never invokes a legacy trainer or generates listening audio.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from inspect_audio_pg09_assets import bootstrap_yue2_product, runtime_identity
from conceptmod.textsliders.pg09_audio import (
    AudioRow, Music3ARResponseAdapter, YuE2ARResponseAdapter, audio_conditions,
    require_audio_width,
    AudioResponseCritic,
)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_digest(model, network, check=None):
    owned = {id(p) for p in network.parameters()}
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if check:
            check()
        if id(parameter) in owned:
            continue
        if parameter.requires_grad:
            raise ValueError('frozen host acquired an unowned trainable parameter')
        digest.update(name.encode())
        # One host parameter at a time; no second full base-model copy.
        array = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(array))
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_native(args, options, check):
    import yaml
    from particle_sliders import ConditionalPG09, winning_formulation
    manifest = json.loads(args.manifest.read_text())
    if args.family == 'yue2':
        bootstrap_yue2_product(args.product)
        from yue2.backend import YuE2Backend
        from yue2.slider import ParticleSlider
        from yue2.storage import model_identity, resolve_model
        record = manifest['records']['model']
        resolved = Path(resolve_model(record['model'], revision=record['revision'],
            cache_dir=manifest['cache_dir'], local_files_only=True)).resolve()
        if args.model_dir.resolve() != Path(record['path']).resolve() or resolved != args.model_dir.resolve():
            raise ValueError('YuE2 model directory differs from supplied staged manifest')
        if model_identity(resolved) != manifest['model_identity']:
            raise ValueError('actual YuE2 model files/config differ from supplied manifest')
        backend = YuE2Backend(record['model'], device=args.device, allow_hub=False,
            revision=record['revision'], cache_dir=manifest['cache_dir'])
        if backend.identity != manifest['model_identity']:
            raise ValueError('loaded YuE2 backend identity differs from supplied manifest')
        model = backend.model
        check()
        torch.manual_seed(args.seed)
        slider = ParticleSlider(model, winning_formulation())
        adapter = YuE2ARResponseAdapter(backend, slider,
            model_identity=dict(manifest=manifest['model_identity'],
                                worker_sha256=sha_file(__file__)),
            checkpoint_modules=list(model.model.layers), **options)
        def pack(row, index):
            neutral = backend.prefix(row['neutral'], row['lyrics'])
            positive = backend.prefix(row['positive'], row['lyrics'])
            with adapter.network.scaled(0):
                history = backend.continuation(neutral, 16, 1709 + index)
            return AudioRow(tuple(neutral + history), tuple(positive + history), history_tokens=16)
    else:
        from conceptmod.textsliders.music3_particle_bridge import Music3Backend, ParticleNetwork
        # Verify the staged native LM files against the already authorized manifest.
        for entry in manifest['files']:
            if entry['filename'].startswith(('language_model/', 'tokenizer/')):
                path = args.model_dir / entry['filename']
                if sha_file(path) != entry['sha256']:
                    raise ValueError('staged model/tokenizer digest differs: ' + str(path))
        backend = Music3Backend(str(args.model_dir), device=args.device, dummy=False)
        model = backend.model
        check()
        torch.manual_seed(args.seed)
        slider = ParticleNetwork(model)
        adapter = Music3ARResponseAdapter(model, slider,
            model_identity=dict(manifest_sha256=sha_file(args.manifest),
                                revision=manifest['revision'], worker_sha256=sha_file(__file__)),
            checkpoint_modules=list(model.model.layers), **options)
        def pack(row, index):
            del index
            ids, mask = backend.encode_text(row['neutral'], row['lyrics'])
            positive, positive_mask = backend.encode_text(row['positive'], row['lyrics'])
            return AudioRow(tuple(ids[0][mask[0].bool()].tolist()),
                            tuple(positive[0][positive_mask[0].bool()].tolist()))
    for layer in model.model.layers:
        layer.register_forward_pre_hook(check)
    fixed, heldout = [], []
    for split, path, destination in (('train', args.train_prompts, fixed),
                                      ('heldout', args.heldout_prompts, heldout)):
        rows = yaml.safe_load(path.read_text())['rows']
        required = 4 if split == 'train' else 2
        if len(rows) != required:
            raise ValueError(f'expected exactly {required} predeclared {split} rows')
        for index, row in enumerate(rows):
            packed = pack(row, index + (100 if split == 'heldout' else 0))
            if max(len(packed.neutral_ids), len(packed.positive_ids)) > args.max_seq_len:
                raise ValueError('complete native prompt exceeds token budget; no truncation')
            destination.append(adapter.prepare(packed))
    conditions = audio_conditions(fixed)
    contexts = {}
    for prepared, condition in zip(fixed, conditions):
        key = condition.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        previous = contexts.setdefault(key, prepared.row.neutral_ids)
        if previous != prepared.row.neutral_ids:
            raise ValueError('distinct native contexts collide in the complete fixed condition dataset')
    critic = AudioResponseCritic(fixed[0].target.shape[-1], conditions.shape[1]).to(args.device)
    game = ConditionalPG09(adapter, fixed, critic, conditions=conditions,
        particles=adapter.particles, response_domain='continuous_vector',
        preset=args.preset, batch_size=2, seed=args.seed, total_steps=args.total_steps, **options)
    return adapter, model, fixed, heldout, game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('yue2', 'music3'), required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--admission-id')
    parser.add_argument('--minimum-free-gib', type=float)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--product', type=Path)
    parser.add_argument('--train-prompts', type=Path)
    parser.add_argument('--heldout-prompts', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--until', type=int, required=True)
    parser.add_argument('--total-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--preset', choices=('bispf', 'bispf_optimized'), default='bispf_optimized')
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--pg-source-profile', default='release')
    parser.add_argument('--penalty-input-domain', default='auto')
    args = parser.parse_args()
    if not 1 <= args.until <= args.total_steps <= 20 or args.max_seq_len != 512:
        parser.error('bounded qualification requires at most20updates and max_seq_len512')
    if args.fixture:
        if args.device != 'cpu' or args.total_steps != 4:
            parser.error('CPU fixture requires --device cpu --total-steps4')
    else:
        if (args.device != 'cuda:0' or not args.admission_id or args.minimum_free_gib is None
                or any(p is None for p in (args.model_dir, args.manifest, args.train_prompts, args.heldout_prompts))):
            parser.error('native run requires explicit owner admission, CUDA/memory bound and all local input paths')
        if not 1 <= args.minimum_free_gib <= 64:
            parser.error('positive owner-specified free-memory bound required')
        if args.family == 'yue2' and args.product is None:
            parser.error('native YuE2 requires --product')
        if not args.out.resolve().is_relative_to('/mnt/nvme/sliders'):
            parser.error('native outputs must be isolated under /mnt/nvme/sliders')
    options = dict(pg_source_profile=args.pg_source_profile, penalty_input_domain=args.penalty_input_domain)
    config = (args.model_dir / ('config.json' if args.family == 'yue2' else 'language_model/config.json')) if not args.fixture else None
    width = json.loads(config.read_text())['hidden_size'] if config else 32
    require_audio_width(width, **options)  # Before CUDA initialization or loading weights.
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    guard = None
    if not args.fixture:
        if ',' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('one explicitly admitted physical GPU must be visible')
        if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info(0)[0] < args.minimum_free_gib * 1024**3:
            raise RuntimeError('admitted GPU does not meet the fresh free-memory bound')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, '/mnt/nvme/sliders/qualification/ops-music3')
        from music3_resources import ResourceGuard
        guard = ResourceGuard()
        guard()
    status = dict(status='initializing', family=args.family, pid=os.getpid(),
        admission_id=args.admission_id, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        runtime=runtime_identity(), worker_sha256=sha_file(__file__), completed_steps=0)
    atomic_json(args.out / 'progress.json', status)
    def check(*unused):
        if time.monotonic() - started > 900:
            raise RuntimeError('900-second owned worker budget exhausted')
        if guard:
            guard()
    try:
        if args.fixture:
            from pg09_audio_cpu_fixtures import make_audio
            adapter, model, fixed, game = make_audio(args.family, preset=args.preset,
                checkpointing=True, critic_class=AudioResponseCritic, **options)
            heldout = [adapter.prepare(replace(p.row, neutral_ids=(61,) + p.row.neutral_ids[1:])) for p in fixed]
        else:
            status['stage'] = 'loading_and_preparing_native_model'
            atomic_json(args.out / 'progress.json', status)
            adapter, model, fixed, heldout, game = load_native(args, options, check)
        if args.fixture:
            for layer in model.model.layers:
                layer.register_forward_pre_hook(check)
        baseline = frozen_digest(model, adapter.network, check)
        logs = []
        if args.resume:
            packet = torch.load(args.resume, map_location='cpu', weights_only=True)
            if packet['frozen_base_sha256'] != baseline:
                raise ValueError('frozen base digest differs from resume packet')
            game.load_state_dict(packet['game'])
            logs = packet['metrics']
        before = {n: p.detach().clone() for n, p in adapter.network.named_parameters()}
        bank_before = adapter.particles.detach().clone()
        critic_before = {n: p.detach().clone() for n, p in game.critic.named_parameters()}
        while game.completed_steps < args.until:
            check()
            logs.append(game.step())
            status.update(status='training', completed_steps=game.completed_steps,
                          elapsed_seconds=time.monotonic() - started, latest=logs[-1])
            atomic_json(args.out / 'progress.json', status)
            if game.completed_steps % 2 == 0 or game.completed_steps == args.until:
                torch.save(dict(game=game.state_dict(), metrics=logs, frozen_base_sha256=baseline),
                           args.out / 'checkpoint.pt')
        if frozen_digest(model, adapter.network, check) != baseline:
            raise ValueError('frozen base weights changed')
        if not all(torch.equal(adapter.sample(p, scale=0), p.neutral) for p in heldout):
            raise ValueError('heldout native scale0 replay failed')
        probes = [adapter.sample(p) for p in heldout]
        changed = [n for n, p in adapter.network.named_parameters() if not torch.equal(before[n], p)]
        critic_changes = [n for n, p in game.critic.named_parameters() if not torch.equal(critic_before[n], p)]
        response = adapter.generated(heldout[0])
        direction = torch.linspace(-1, 1, response.shape[-1], device=response.device)
        bank_gradient = torch.autograd.grad((response * direction).sum(), adapter.particles)[0]
        bank_gradient_l1 = float(bank_gradient.abs().sum())
        bank_changed = not torch.equal(bank_before, adapter.particles)
        bank_delta_rms = float((bank_before - adapter.particles.detach()).square().mean().sqrt())
        if (not changed or not critic_changes or not bank_changed or bank_gradient_l1 <= 0
                or not all(m['d_pen'] > 0 for m in logs)):
            raise ValueError('adapter/critic/bank-gradient/continuous-penalty learning gate failed')
        state = game.state_dict()
        exported = game.export_state_dict()
        torch.save(exported, args.out / 'adapter-export.pt')
        with torch.no_grad():
            next(adapter.network.parameters()).add_(.125)
        game.load_export_state_dict(exported)
        if not all(torch.equal(value, adapter.sample(p)) for value, p in zip(probes, heldout)):
            raise ValueError('adapter export reload changed heldout response')
        game.load_state_dict(state)
        # Retain exact probes/rows for external fresh-process comparison.
        torch.save(dict(game=state, metrics=logs, probes=probes, export=exported,
                        frozen_base_sha256=baseline), args.out / 'qualification.pt')
        atomic_json(args.out / 'rows.json', dict(train=[asdict(p.row) for p in fixed],
                                                heldout=[asdict(p.row) for p in heldout]))
        status.update(status='bounded_training_software_checks_passed',
            elapsed_seconds=time.monotonic() - started, frozen_base_unchanged=True,
            zero_replay_exact=True, export_replay_exact=True, changed_adapter_tensors=len(changed),
            changed_critic_tensors=len(critic_changes), bank_response_gradient_l1=bank_gradient_l1,
            actual_bank_weights_changed=bank_changed, actual_bank_delta_rms=bank_delta_rms,
            heldout_response_effect=[float((v - p.neutral).square().mean().sqrt()) for v, p in zip(probes, heldout)],
            quality_qualified=False, audio_rendered=False,
            peak_allocated_bytes=(torch.cuda.max_memory_allocated() if not args.fixture else 0),
            peak_reserved_bytes=(torch.cuda.max_memory_reserved() if not args.fixture else 0))
    except Exception as error:
        status.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        atomic_json(args.out / 'progress.json', status)
        raise
    atomic_json(args.out / 'progress.json', status)
    print(json.dumps({k: v for k, v in status.items() if k != 'runtime'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
