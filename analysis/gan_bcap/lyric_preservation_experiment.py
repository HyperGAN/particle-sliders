#!/usr/bin/env python3
"""Branch the repaired smoke GAN with an explicit lyric-prefix hold strength."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.gan_bcap.stability_experiment import StabilityHooks, sha, trainer


def lyric_hooks(*, baseline_signature, weight, telemetry_path, diagnostics_every=25):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Lyric hold weight must be finite and nonnegative')
    settings = baseline_signature['settings']
    if not (settings['lm_target'] == 'faithful_plus_neu_lyric' and settings['pole_weight'] == 0):
        raise ValueError('This experiment requires the all-GAN lyric-span recipe')
    hooks = StabilityHooks(baseline_signature=baseline_signature, telemetry_path=telemetry_path,
                           diagnostics_every=diagnostics_every)
    hooks.config['lyric_preservation'] = dict(weight=weight, driver_sha256=sha(__file__),
        definition='Existing neutral-teacher MSE on the prompt lyric hidden states at slider +1; no audio-start anchor added.')
    return hooks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--lyric-hold-weight', type=float, required=True)
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--memory-fraction', type=float)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            p.error('Invalid memory fraction')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    state = torch.load(args.state, map_location='cpu', weights_only=True)
    baseline = state['signature'].get('stability_experiment', {}).get('baseline_signature', state['signature'])
    settings = baseline['settings']
    if not (settings['adv_arch'] == 'tx' and settings['adv_condition'] == 'none'
            and settings['fm_mode'] == 'batch' and settings['fm_weight'] == 1
            and settings['parts'] == 0 and settings['gan_lr_schedule'] == 'constant'):
        raise ValueError('This experiment supports the unchanged smoke GAN card only')
    run = ROOT / 'models/gan-bcap-repair' / args.name
    run.mkdir(exist_ok=True)
    if any((run / f'{args.name}{suffix}').exists() for suffix in ['_train.jsonl', '_last.safetensors', '_telemetry.jsonl']):
        raise ValueError('Use a fresh run name')
    train_args = trainer.parse_args(['--prompts_file', str(ROOT / 'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml')])
    vars(train_args).update(settings)
    train_args.name = args.name
    train_args.save_dir = str(run)
    train_args.steps = args.steps
    train_args.save_every = args.save_every
    train_args.resume_state = str(args.state.resolve())
    train_args.save_training_state = True
    train_args.device = 0
    train_args.lyrichold_weight = args.lyric_hold_weight
    hooks = lyric_hooks(baseline_signature=baseline, weight=args.lyric_hold_weight,
                        telemetry_path=run / f'{args.name}_telemetry.jsonl')
    manifest = dict(source_state=str(args.state.resolve()), source_state_sha256=sha(args.state),
                    source_completed_updates=state['completed_updates'], total_updates=args.steps,
                    intervention=hooks.config, physical_gpu=1, command=sys.argv,
                    reason='User hears some gibberish in the numerically stable 900 comparisons.',
                    caveat='Existing lyric-prefix hold is an indirect constraint, not a guarantee of intelligible singing or preserved voice character.')
    manifest_path = run / 'experiment.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copyfile(__file__, run / 'lyric_preservation_experiment_source.py')
    shutil.copyfile(ROOT / 'analysis/gan_bcap/stability_experiment.py', run / 'stability_experiment_source.py')
    with hooks.installed():
        weights = trainer.train(train_args)
    sidecar = weights.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    meta['stability_experiment'] = hooks.config
    sidecar.write_text(json.dumps(meta, indent=2) + '\n')
    manifest.update(training_exit_code=0, weights=str(weights))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
