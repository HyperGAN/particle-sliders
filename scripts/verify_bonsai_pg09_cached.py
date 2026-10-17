"""Owner-admitted CPU verification using existing official cached readouts only.

Preflight is metadata-only. Historical produce/restore modes are retired in this
package cohort. The archival implementation performed four shared updates, checkpointing
after two; restore replays the final two in a separate process. An explicit
source-pinned --plan declared1200 updates and final20-update replay. It remains
available only for archival metadata preflight here. No server,
GGUF/model loader, network, CUDA, legacy trainer or installed-package mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import resource
import shutil
import time

CORE_SHA256 = '426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256'
ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = Path('/mnt/nvme/sliders/qualification')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(core):
    from prepare_text_video_pg09_verification import audit_manifest, ASSETS
    module = core / 'packages/particle-sliders-core/src/particle_sliders/pg09.py'
    if digest(module) != CORE_SHA256:
        raise ValueError('core source differs from the frozen owner/QA model-admission pin')
    assets = audit_manifest(QUALIFICATION / 'models/bonsai-b072e1d', ASSETS['bonsai'])
    preparation = ROOT / 'evidence/pg09-text-video/assets-prepared.json'
    selected = json.loads(preparation.read_text())['bonsai_cached_readout_metadata']
    recipe = QUALIFICATION / 'runs/bonsai-caption-native-1200-20261017/recipe.json'
    previous = json.loads(recipe.read_text())
    identity = previous['frozen_readout_identity']
    identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if identity_digest != previous['base_identity']:
        raise ValueError('archived official readout identity differs')
    cache = QUALIFICATION / 'models/bonsai-caption-readouts-20261017'
    if not cache.is_dir():
        raise FileNotFoundError('existing immutable readout cache required; never populate it here')
    for item in selected:
        path = Path(item['metadata_path'])
        if path.parent != cache or digest(path) != item['metadata_sha256']:
            raise ValueError('readout metadata differs from frozen preparation')
        metadata = json.loads(path.read_text())
        if (metadata['identity'] != identity_digest or not metadata.get('independent_requests_exact')
                or path.with_suffix('.pt').stat().st_size != item['tensor_bytes']):
            raise ValueError('official readout cache provenance/size differs')
    return dict(worker_sha256=digest(Path(__file__)), core_sha256=CORE_SHA256,
                assets_manifest_sha256=assets['manifest_sha256'], readout_identity=identity_digest,
                frozen_readout_identity=identity, preparation_sha256=digest(preparation),
                archived_recipe_sha256=digest(recipe), cache=str(cache), cached_readouts=len(selected),
                model_loaded=False, tensor_payload_loaded=False, gpu_used=False,
                permitted_scope='archival metadata only; cached training retired in this package cohort')


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def load_plan(path, args):
    """Validate a reviewable long-run recipe before importing Torch/cache data."""
    plan = json.loads(path.read_text())
    from importlib.metadata import version
    runtime = dict(python=sys.version.split()[0],
                   **{name: version(name) for name in ('torch', 'numpy', 'transformers', 'diffusers', 'safetensors')})
    if runtime != plan['runtime_versions']:
        raise ValueError('long-run interpreter/package versions differ')
    expected = dict(preset='bispf_optimized', total_steps=1200, batch_size=2, seed=7,
                    sigma=1.0, response_scale=1.0, sampling='balanced',
                    checkpoint_every=20, replay_start=1180,
                    evaluation_steps=[0, 20, 100, 400, 800, 1200], recipe_overrides={})
    if plan['training'] != expected or args.preset != expected['preset']:
        raise ValueError('long-run recipe must be the fixed reviewed 1200-update public arm')
    if (Path(plan['core']).resolve() != args.core.resolve()
            or Path(plan['overlay']).resolve() != args.overlay.resolve()
            or Path(plan['output']).resolve() != args.output.resolve()):
        raise ValueError('command paths differ from the proposed long-run recipe')
    for name, value in plan['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('process environment differs: ' + name)
    for name, expected_hash in plan['source_sha256'].items():
        if digest(Path(name)) != expected_hash:
            raise ValueError('long-run source/metadata differs: ' + name)
    required = [ROOT / relative for relative in (
        'scripts/verify_bonsai_pg09_cached.py', 'scripts/caption_quality.py',
        'scripts/prepare_text_video_pg09_verification.py',
        'conceptmod/textsliders/pg09_text_video.py', 'conceptmod/textsliders/pg09_backends.py',
        'conceptmod/textsliders/pg09_readout_critic.py',
        'conceptmod/textsliders/bonsai_gguf_particle.py', 'conceptmod/textsliders/caption_recovery.py',
        'conceptmod/textsliders/particle_bridge_gan.py',
        'conceptmod/textsliders/bonsai_gguf_backend.py',
        'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml',
        'evidence/pg09-text-video/assets-prepared.json')]
    required += [args.core.resolve() / 'packages/particle-sliders-core/src/particle_sliders' / name
                 for name in ('pg09.py', 'pg09_continuous_profile.json')]
    if not set(map(str, required)) <= plan['source_sha256'].keys():
        raise ValueError('long-run proposal omits required source/metadata pins')
    if plan['limits'] != dict(cpu_threads=1, interop_threads=1, rss_cap_gib=8,
                             host_reserve_gib=8, nvme_reserve_gib=100,
                             produce_seconds=1800, restore_seconds=180):
        raise ValueError('long-run resource budget differs')
    if (plan['schema'] != 'bonsai-pg09-cached-cpu-longrun-v2'
            or plan['output_filesystem'] != dict(device='/dev/nvme0n1p2', mountpoint='/mnt/nvme')):
        raise ValueError('long-run output requires the reviewed NVMe device/mount; no filesystem fallback')
    return plan


def inspect_output_filesystem(output):
    """Find the actual mount/device of a possibly not-yet-created output."""
    output = Path(output).resolve()
    parent = output
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError('existing output ancestor must be a directory')
    actual_device = parent.stat().st_dev

    def unescape(value):
        for escaped, decoded in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            value = value.replace(escaped, decoded)
        return value

    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        separator = fields.index('-')
        mount = Path(unescape(fields[4]))
        major, minor = map(int, fields[2].split(':'))
        if parent.is_relative_to(mount) and os.makedev(major, minor) == actual_device:
            mounts.append((len(mount.parts), mount, unescape(fields[separator + 2]), fields[2]))
    if not mounts:
        raise OSError('cannot verify actual output filesystem; no fallback')
    _, mount, device, major_minor = max(mounts, key=lambda item: item[0])
    return dict(output=str(output), existing_parent=str(parent), mountpoint=str(mount),
                device=device, major_minor=major_minor,
                free_bytes=shutil.disk_usage(parent).free)


def check_limits(plan, started, mode, *, startup=False):
    if plan is None:
        return
    limits = plan['limits']
    if time.monotonic() - started > limits[mode + '_seconds']:
        raise TimeoutError('admitted process wall-time budget exhausted')
    # Linux ru_maxrss is KiB and includes all native Torch allocations in this
    # process. Check after every update/evaluation; external timeout is separate.
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits['rss_cap_gib'] * 1024**3:
        raise MemoryError('admitted process peak RSS budget exhausted')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    needed = limits['host_reserve_gib'] + (limits['rss_cap_gib'] if startup else 0)
    if available < needed * 1024**3:
        raise MemoryError('host memory reserve unavailable')
    filesystem = inspect_output_filesystem(plan['output'])
    if {key: filesystem[key] for key in ('device', 'mountpoint')} != plan['output_filesystem']:
        raise OSError('output filesystem device/mount differs; no filesystem fallback')
    if filesystem['free_bytes'] < limits['nvme_reserve_gib'] * 1024**3:
        raise OSError('output filesystem reserve unavailable')
    return dict(output_filesystem=filesystem, host_available_bytes=available,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                startup=startup, limits=limits)


def execute(args, receipt):
    # Process execution is admitted by the core owner for an exact command.
    # The script cannot grant itself model/process admission.
    started = time.monotonic()
    plan = getattr(args, 'plan_data', None)
    check_limits(plan, started, args.mode, startup=True)
    training = plan['training'] if plan else dict(total_steps=4, batch_size=2, seed=7,
               sigma=1.0, response_scale=1.0, sampling='balanced', checkpoint_every=2,
               replay_start=2, evaluation_steps=[0, 4])
    total_steps, replay_start = training['total_steps'], training['replay_start']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(args.overlay.resolve()),
                   str(args.core.resolve() / 'packages/particle-sliders-core/src'), str(ROOT), str(ROOT / 'scripts')]
    import torch
    from particle_sliders.pg09 import ConditionalPG09, validate_pg09_domain
    validate_pg09_domain((1, 5120), response_domain='continuous_vector',
                         pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    from conceptmod.textsliders.caption_recovery import FrozenReadouts
    from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
    from conceptmod.textsliders.pg09_text_video import BonsaiReadoutResponseAdapter, BonsaiResponseRow
    from conceptmod.textsliders.pg09_readout_critic import ConditionalReadoutCritic
    from caption_quality import evaluate
    import yaml
    torch.set_num_threads(1)
    if plan:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        import random
        import numpy as np
        random.seed(training['seed'])
        np.random.seed(training['seed'])
    torch.manual_seed(training['seed'])
    cache = FrozenReadouts(None, receipt['cache'], receipt['readout_identity'])
    head = BonsaiParticleHead().float()
    identity = {key: value for key, value in receipt.items()
                if key not in ('permitted_scope', 'model_loaded', 'tensor_payload_loaded', 'gpu_used')}
    identity['evaluation_source_sha256'] = digest(ROOT / 'scripts/caption_quality.py')
    prompts = ROOT / 'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml'
    identity['prompts_sha256'] = digest(prompts)
    if plan:
        identity['long_run_recipe_sha256'] = args.plan_sha256
    adapter = BonsaiReadoutResponseAdapter(cache, head, model_identity=identity,
                    pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    data = yaml.safe_load(prompts.read_text())
    rows = [adapter.prepare(BonsaiResponseRow(r['neutral'], r['positive'])) for r in data]
    conditions = torch.cat([torch.cat((r.neutral.flatten(), torch.tensor([1., 0.])))[None] for r in rows])
    critic = ConditionalReadoutCritic(5120, conditions.shape[1])
    game = ConditionalPG09(adapter, rows, critic, conditions=conditions, particles=adapter.particles,
                           **adapter.domain_options, preset=args.preset, total_steps=total_steps,
                           batch_size=training['batch_size'], seed=training['seed'],
                           sigma=training['sigma'], response_scale=training['response_scale'],
                           sampling=training['sampling'])
    if plan:
        from dataclasses import asdict
        equal(json.loads(json.dumps(asdict(game.recipe))), plan['resolved_public_recipe'])
    initial = {k: v.detach().clone() for k, v in adapter.network.state_dict().items()}
    initial_d = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    initial_bank = head.particles.detach().clone()
    frozen = [t.detach().clone() for r in rows for t in (r.target, r.neutral)]
    context = dict(backend=cache, network=head, fixed=data)

    def sample():
        before, rng = game.state_dict(), torch.get_rng_state().clone()
        result = evaluate(context, 'bonsai')
        result[0]['note'] = ('Shared PG09 neutral-conditioned student; frozen positive teacher. '
                             'Squared errors are held-out diagnostics only; no semantic quality claim.')
        equal(before, game.state_dict())
        assert torch.equal(rng, torch.get_rng_state())
        check_limits(plan, started, args.mode)
        return result

    def step():
        result = game.step()
        assert abs(result['g_loss'] - result['g_adv'] - result['prior_regularization']) < 1e-5
        assert abs(result['d_loss'] - result['d_adv'] - result['d_pen']) < 1e-5
        assert result['d_pen'] > 0
        assert game.penalty.last_stats['input_domain'] == 'continuous'
        check_limits(plan, started, args.mode)
        return result

    def save(value, name):
        path = args.output / name
        pending = path.with_suffix('.pending')
        torch.save(value, pending)
        pending.replace(path)

    def forbidden(*a, **kw):
        raise AssertionError('optimized MSE forbidden in shared GAN canary')
    torch.nn.functional.mse_loss = forbidden
    if args.mode == 'produce':
        args.output.mkdir(parents=True, exist_ok=False)
        initial_sample = sample()
        metrics, evaluations, later = [], {}, []

        def record_evaluation(value):
            if not plan:
                return
            folder = args.output / 'evaluation' / f'step-{game.completed_steps:06d}'
            folder.mkdir(parents=True, exist_ok=False)
            (folder / 'metrics.json').write_text(json.dumps(value[0], indent=2) + '\n')
            save(dict(zero=value[1], plus=value[2]), str(folder.relative_to(args.output) / 'vectors.pt'))
            save(game.export_state_dict(), str(folder.relative_to(args.output) / 'inference-export.pt'))
            evaluations[str(game.completed_steps)] = value[0]

        record_evaluation(initial_sample)
        with (args.output / 'metrics.jsonl').open('x') as stream:
            while game.completed_steps < total_steps:
                result = step()
                metrics.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                if game.completed_steps > replay_start:
                    later.append(result)
                if game.completed_steps == replay_start:
                    save(game.state_dict(), 'checkpoint.pt')
                if plan and game.completed_steps % training['checkpoint_every'] == 0:
                    save(game.state_dict(), f'checkpoint-{game.completed_steps:06d}.pt')
                if plan and game.completed_steps in training['evaluation_steps']:
                    record_evaluation(sample())
                    print(json.dumps(dict(progress_step=game.completed_steps, total_steps=total_steps)), flush=True)
        sample_result = sample()
        export = game.export_state_dict()
        save(export, 'inference-export.pt')
        save(dict(metrics=later, state=game.state_dict(), sample=sample_result), 'expected.pt')
        result = dict(receipt, mode='produce', preset=args.preset, completed_steps=total_steps,
                      metrics=metrics, heldout_initial=initial_sample[0], heldout_final=sample_result[0],
                      adapter_changed=any(not torch.equal(initial[k], v) for k, v in adapter.network.state_dict().items()),
                      critic_changed=any(not torch.equal(initial_d[k], v) for k, v in critic.state_dict().items()),
                      bank_changed=not torch.equal(initial_bank, head.particles))
        assert result['adapter_changed'] and result['critic_changed'] and result['bank_changed']
        if plan:
            result.update(evaluations=evaluations, long_run_recipe_sha256=args.plan_sha256,
                          training_complete=True, checkpoint_every=training['checkpoint_every'])
    else:
        if (args.output / 'restore-receipt.json').exists():
            raise FileExistsError('restore receipt exists; no automatic retry')
        game.load_state_dict(torch.load(args.output / 'checkpoint.pt', weights_only=True))
        assert game.completed_steps == replay_start
        later = [step() for _ in range(total_steps - replay_start)]
        expected = torch.load(args.output / 'expected.pt', weights_only=True)
        assert later == expected['metrics']
        equal(game.state_dict(), expected['state'])
        equal(sample(), expected['sample'])
        game.load_export_state_dict(torch.load(args.output / 'inference-export.pt', weights_only=True))
        rng = torch.get_rng_state().clone()
        equal(evaluate(context, 'bonsai')[1:], expected['sample'][1:])
        assert torch.equal(rng, torch.get_rng_state())
        result = dict(receipt, mode='restore', preset=args.preset, completed_steps=total_steps,
                      exact_full_state=True, exact_losses=True, exact_heldout_vectors=True,
                      exact_export_reload=True)
        if plan:
            result.update(long_run_recipe_sha256=args.plan_sha256, replayed_updates=len(later))
    assert all(torch.equal(saved, t) for saved, t in zip(frozen, [t for r in rows for t in (r.target, r.neutral)]))
    assert not torch.cuda.is_initialized()
    result.update(scope='real official frozen readouts; native torch-side head only',
                  frozen_readouts_unchanged=True, tensor_payload_loaded=True,
                  gguf_weights_loaded=False, server_started=False, gpu_used=False,
                  quality_qualified=False, installed_package_modified=False)
    result.update(elapsed_seconds=time.monotonic() - started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    (args.output / f'{args.mode}-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('mode', 'preset', 'completed_steps', 'scope', 'gpu_used', 'quality_qualified')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'produce', 'restore'], required=True)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=['bispf', 'bispf_optimized'], default='bispf_optimized')
    parser.add_argument('--plan', type=Path, help='source-pinned owner-reviewed 1200-update proposal; no implicit admission')
    args = parser.parse_args()
    if args.mode != 'preflight':
        parser.error('legacy Bonsai cached-readout produce/restore training is retired; '
                     'use an explicitly reviewed pinned installed-package worker/new cohort; '
                     'metadata-only --mode preflight remains available')
    if args.plan:
        args.plan_data = load_plan(args.plan, args)
        args.plan_sha256 = digest(args.plan)
    receipt = preflight(args.core.resolve())
    if args.plan:
        receipt.update(long_run_recipe_sha256=args.plan_sha256,
                       plan_validated_before_tensor_payload=True,
                       permitted_scope='archival source-plan metadata only; no training admission')
    if args.mode == 'preflight':
        if args.plan:
            receipt['resource_preflight'] = check_limits(args.plan_data, time.monotonic(),
                                                         'produce', startup=True)
        assert 'torch' not in sys.modules
        receipt['torch_imported'] = False
        print(json.dumps(receipt, indent=2))
        return
    execute(args, receipt)


if __name__ == '__main__':
    main()
