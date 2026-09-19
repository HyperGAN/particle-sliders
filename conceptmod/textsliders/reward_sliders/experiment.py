"""Durable staged CE experiment. Off wins ties; failed causal probes stop here.

CUDA_VISIBLE_DEVICES=1 python -m conceptmod.textsliders.reward_sliders.experiment run
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import importlib.metadata
import os
from pathlib import Path
import signal
import sys
import time

import torch

from .specs import (DEFAULT_RUN, MODEL, ROOT, WORKSPACE, RewardSpec, CaptureSpec, digest,
                    read_json, write_json, sha, save_tensor, validate_families)
from .fixtures import pilot
from .reward import CEReward
from .render import resolve_styles, ResearchRenderer, studio_snapshot
from .capture import pooled
from .directions import fit_direction, rank_agreement, choose_dev, causal_gate

CAPTURE_MODULES = ('specs.py', 'fixtures.py', 'reward.py', 'capture.py', 'directions.py', 'render.py', 'experiment.py')


def freeze(run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    path = run/'manifest.json'
    if path.exists():
        manifest = read_json(path)
        verify(manifest)
        return manifest
    rows = pilot()
    components = {r['family']: resolve_styles(r['style_multipliers']) for r in rows}
    sys.path.insert(0, str(WORKSPACE))
    from app.rewriter import _artist_name_hit
    from app import sliders
    from diffusers.modular_pipelines.minimax_music3 import encoders, denoise, decoders, modular_pipeline
    import transformers.models.qwen3.modeling_qwen3 as qwen
    installed = [Path(m.__file__) for m in (encoders, denoise, decoders, modular_pipeline, qwen)]
    local = [Path(__file__).parent/name for name in CAPTURE_MODULES]
    local += [WORKSPACE/'app'/name for name in ('generator.py', 'sliders.py', 'lora_runtime.py', 'rewriter.py')]
    sources = {str(p): sha(p) for p in local+installed}
    style_hashes = {}
    for comps in components.values():
        for comp in comps:
            weight = Path(comp['weights'])
            if str(weight) in style_hashes:
                continue
            metadata = read_json(weight.with_suffix('.json'))
            prompt = Path(metadata['prompts_file'])
            if not prompt.is_absolute():
                prompt = ROOT/prompt
            if _artist_name_hit('', prompt.read_text()):
                raise ValueError('Style provenance contains prohibited names; retraining required')
            style_hashes[str(weight)] = sha(weight)
            style_hashes[str(weight.with_suffix('.json'))] = sha(weight.with_suffix('.json'))
            style_hashes[str(prompt)] = sha(prompt)
    print('Hashing exact installed model and scorer weights before collection', flush=True)
    model_paths = sorted(p for p in MODEL.rglob('*') if p.is_file() and p.suffix in ('.json', '.safetensors', '.txt'))
    model_hashes = {str(p): sha(p) for p in model_paths}
    reward = RewardSpec(hashes=CEReward.provenance())
    capture = CaptureSpec(hashes={str(p): sources[str(p)] for p in installed})
    manifest = dict(schema='music-reward-sliders-v1', name='reward-ce-v1',
        reward_spec=asdict(reward), capture_spec=asdict(capture),
        families=rows, style_components=components, style_hashes=style_hashes,
        host_energy_max=sliders.catalog()['energy']['language_model']['max'],
        model_hashes=model_hashes, source_hashes=sources,
        file_stats={p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns)
                    for p in {*model_hashes, *style_hashes, *reward.hashes}},
        packages={name: importlib.metadata.version(name) for name in ('torch','transformers','diffusers','torchaudio')},
        pilot=dict(seeds=[1103, 2207, 3301, 4409], dev_seeds=[1103, 2207],
                   render_cap_seconds=20.4, primary_seconds=20.,
                   cap_interpretation='intentional screen excerpt, 0.4 seconds guard for acoustic chunk rounding; early outputs under 20 seconds fail',
                   baseline_renders=64, dev_new_renders=32, causal_new_renders=48, audit_renders=4,
                   total_short_render_ceiling=148, random_direction_seed=1729,
                   layers=[11,23], strengths=[.01,.03], minimum_causal_ce_gain=.02,
                   causal_rule='all arms valid; positive beats off and random by >=0.02 mean CE; >50% seed wins vs off'),
        sampler=dict(cfg=1.5, semantic_top_k=50, conditional_top_k=50, depth_cfg=1.5,
                     flow_steps=30, generator='device-local torch.Generator, initial seed identical per arm',
                     dtype='bfloat16', style_apply='ordinary full-delta CPU FP32 merge'),
        student=dict(rank=8, alpha=8, host='language_model', target_replace=['Qwen3Attention'],
                     prefix='lora_te', delimiter='-', train_method='full', unit_scale=1.,
                     steps=[300,600,660], batch=4, seed=7,
                     geometry='both CFG rows separately, first 128 generated feedback positions, stride 4 critic features; full shared prefix history',
                     objective='gan_v2 baseline RpGAN/b_cap plus batch-mean FM and neutral EOS margin; no extra KL or distillation',
                     bounded_parameter_step_limit=2., strength_grid=[.1,.3,1.],
                     dev_render_ceiling=72, transfer_families=12, transfer_seeds=2, transfer_render_minimum=48),
        operation=dict(physical_gpu=1, studio=studio_snapshot(), registry_changes=False),
        created_unix=time.time())
    write_json(path, manifest)
    # Archive the exact source bytes used by the collection process.
    for source, expected in sources.items():
        destination = run/'provenance'/Path(source).relative_to('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        assert sha(destination) == expected
    write_json(run/'prompt-audit.json', dict(passed=True, fixtures=len(rows),
               strategy='original sound-only captions and original lyrics, plus project name validator',
               family_sha256={r['family']: digest(r) for r in rows}, style_provenance_checked=True))
    return manifest


def verify(manifest):
    validate_families(manifest['families'])
    for path, expected in manifest['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen collection source changed: {path}')
    for path, expected in manifest['file_stats'].items():
        stat = Path(path).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
            raise ValueError(f'Frozen weight/source file changed: {path}')


def status(run, stage, **extra):
    value = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra)
    write_json(Path(run)/'status.json', value)
    print(f'STAGE {stage}', flush=True)


def run_pilot(run):
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    lock = (run/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'Experiment cancelled by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, cancel)
    manifest = freeze(run)
    scorer = CEReward(RewardSpec(**manifest['reward_spec']))
    renderer = None
    try:
        status(run, 'capture_audit')
        renderer = ResearchRenderer(run, manifest)
        audit = run/'audit/parity.json'
        if not audit.exists():
            renderer.parity_audit(manifest['families'][1])
        if not read_json(audit)['passed']:
            raise ValueError('Capture audit failed')
        status(run, 'matched_baselines', completed=0, total=64)
        observations, features = [], {}
        for family in manifest['families']:
            for seed in manifest['pilot']['seeds']:
                obs = renderer.observe(family, seed, dict(name='off'), scorer, capture=True)
                observations.append(obs)
                if obs['status'] == 'complete':
                    trajectory = torch.load(obs['trajectory'], map_location='cpu', weights_only=True)
                    features[obs['id']] = pooled(trajectory)
                    del trajectory
                status(run, 'matched_baselines', completed=len(observations), total=64)
        save_tensor(run/'pooled.pt', features)
        write_json(run/'baseline-observations.json', observations)
        if any(o['status'] != 'complete' for o in observations):
            decision = dict(decision='failed', stage='baseline_collection',
                            failures=[o['id'] for o in observations if o['status'] != 'complete'],
                            reason='Unresolved baseline output failures; no rerolls and no adapter promotion')
            write_json(run/'decision.json', decision); status(run, 'stopped', **decision)
            return
        verify(manifest)
        teachers = {layer: fit_direction(observations, features, layer) for layer in manifest['pilot']['layers']}
        save_tensor(run/'directions.pt', {str(l): asdict(t) for l,t in teachers.items()})
        write_json(run/'fit.json', {str(l): dict(training_families=t.fitting_families,
                   residual_median_l2=t.residual_norm, norm=float(t.direction.norm()),
                   training=rank_agreement(observations, features, t, 'train'),
                   development=rank_agreement(observations, features, t, 'dev')) for l,t in teachers.items()})
        arms = [dict(name='off')] + [dict(name=f'layer{layer}-s{strength:g}', layer=layer, coefficient=strength)
                for layer in manifest['pilot']['layers'] for strength in manifest['pilot']['strengths']]
        dev = [o for o in observations if o['split'] == 'dev' and o['seed'] in manifest['pilot']['dev_seeds']]
        jobs = [dict(family=f['family'], seed=s, arm=a) for f in manifest['families'] if f['split']=='dev'
                for s in manifest['pilot']['dev_seeds'] for a in arms]
        write_json(run/'dev-arms.json', jobs)
        status(run, 'development_activation_grid', completed=len(dev), total=40)
        for family in [f for f in manifest['families'] if f['split']=='dev']:
            for seed in manifest['pilot']['dev_seeds']:
                for arm in arms[1:]:
                    teacher = replace(teachers[arm['layer']], coefficient=arm['coefficient'])
                    dev.append(renderer.observe(family, seed, arm, scorer, teacher=teacher))
                    status(run, 'development_activation_grid', completed=len(dev), total=40)
        selection = choose_dev(dev, arms)
        write_json(run/'dev-selection.json', selection)
        if selection['selected']['name'] == 'off':
            decision = dict(decision='failed_or_inconclusive', reason='Off selected on development CE; LoRA gate remains closed')
            write_json(run/'decision.json', decision); status(run, 'stopped', **decision)
            return
        selected = selection['selected']
        teacher = replace(teachers[selected['layer']], coefficient=selected['coefficient'])
        rng = torch.Generator().manual_seed(manifest['pilot']['random_direction_seed'])
        random_direction = torch.randn(teacher.direction.shape, generator=rng)
        random_direction /= random_direction.norm()
        control = replace(teacher, direction=random_direction)
        save_tensor(run/'teacher.pt', asdict(teacher))
        save_tensor(run/'random-control.pt', asdict(control))
        test_arms = [dict(name=name, layer=teacher.layer, coefficient=coefficient,
                          direction_sha256=sha(run/('random-control.pt' if name=='random' else 'teacher.pt')))
                     for name,coefficient in [('positive',teacher.coefficient),('reversed',-teacher.coefficient),('random',teacher.coefficient)]]
        write_json(run/'causal-arms.json', [dict(family=f['family'], seed=s, arm=a)
                   for f in manifest['families'] if f['split']=='test' for s in manifest['pilot']['seeds']
                   for a in [dict(name='off')]+test_arms])
        verify(manifest)
        causal = [o for o in observations if o['split']=='test']
        status(run, 'causal_activation_test', completed=len(causal), total=64)
        for family in [f for f in manifest['families'] if f['split']=='test']:
            for seed in manifest['pilot']['seeds']:
                for arm in test_arms:
                    t = control if arm['name']=='random' else replace(teacher, coefficient=arm['coefficient'])
                    causal.append(renderer.observe(family, seed, arm, scorer, teacher=t))
                    status(run, 'causal_activation_test', completed=len(causal), total=64)
        gate = causal_gate(causal, min_gain=manifest['pilot']['minimum_causal_ce_gain'])
        gate.update(manifest_sha256=digest(manifest), teacher_sha256=sha(run/'teacher.pt'))
        write_json(run/'causal-result.json', gate)
        write_json(run/'decision.json', dict(decision=gate['decision'], lora_authorized_by_evidence=gate['passed']))
        status(run, 'teacher_supported' if gate['passed'] else 'stopped', decision=gate['decision'])
    except BaseException as exc:
        status(run, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if renderer is not None:
            renderer.host._merge_sliders(renderer.pipe, renderer.device, [])
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze','run','status'])
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.run_dir)
    elif args.command == 'run':
        run_pilot(args.run_dir)
    else:
        print(__import__('json').dumps(read_json(args.run_dir/'status.json'), indent=2))


if __name__ == '__main__':
    main()
