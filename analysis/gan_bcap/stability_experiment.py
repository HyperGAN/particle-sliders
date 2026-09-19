#!/usr/bin/env python3
"""Explicit research branches from a full GAN state; production defaults stay fixed.

Run states carry the intervention and this file's fingerprint. A normal trainer
resume cannot silently continue these branches without their experimental hooks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conceptmod.textsliders import lm_gan_state as game
from conceptmod.textsliders import train_lm_slider_music3 as trainer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limit_loss_input_gradient(loss, inputs, input_scale, cap):
    """Keep the loss value; bound its gradient norm in teacher-RMS coordinates."""
    gradient = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
    norm = float(gradient.detach().norm() * input_scale)
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite feature-matching input gradient')
    factor = 1. if cap is None or norm <= cap else cap / norm
    limited = loss if factor == 1. else loss.detach() + factor * (loss - loss.detach())
    return limited, norm, factor


class StabilityHooks:
    def __init__(self, *, baseline_signature, g_lr_scale=1., fm_input_grad_cap=None,
                 telemetry_path, diagnostics_every=25, archive_states=True):
        self.baseline = deepcopy(baseline_signature)
        self.config = dict(schema=1, g_lr_scale=g_lr_scale, fm_input_grad_cap=fm_input_grad_cap,
                           harness_sha256=sha(__file__), baseline_signature=self.baseline)
        self.telemetry_path = Path(telemetry_path)
        self.diagnostics_every = diagnostics_every
        self.archive_states = archive_states
        self.rows = []; self.fm_grads = None; self.step = 0; self.params = []

    @contextmanager
    def installed(self):
        originals = (game.signature, game.restore, game.save,
                     trainer.feature_mean_surrogate, trainer._lm_adv.SpanTransformerD.features)
        signature, restore, save, fm, features = originals
        patched_optimizer = None
        optimizer_step = None

        def signed(args, rows, metadata):
            baseline_args = deepcopy(args)
            vars(baseline_args).update(self.baseline['settings'])
            if signature(baseline_args, rows, metadata) != self.baseline:
                raise ValueError('Baseline sources, settings or loaded prompts changed')
            result = signature(args, rows, metadata)
            result['stability_experiment'] = self.config
            return result

        def restore_branch(path, *, run_signature, modules, optimizers):
            nonlocal patched_optimizer, optimizer_step
            stored = torch.load(path, map_location='cpu', weights_only=True)
            if 'stability_experiment' in stored['signature']:
                history = restore(path, run_signature=run_signature, modules=modules, optimizers=optimizers)
            else:
                if stored['signature'] != self.baseline:
                    raise ValueError('The branch source is not the declared baseline')
                history = restore(path, run_signature=self.baseline, modules=modules, optimizers=optimizers)
            actual = run_signature['settings']
            requested_g_lr = self.baseline['settings']['lr'] * self.config['g_lr_scale']
            base = self.baseline['settings']
            requested_d_lr = base['adv_lr'] if base['adv_lr'] is not None else base['lr'] * trainer._lm_adv.D_LR_MULT
            if actual['lr'] != requested_g_lr:
                raise ValueError('Generator LR does not match the declared intervention')
            for name, rate in [('lora', requested_g_lr), ('critic', requested_d_lr)]:
                for group in optimizers[name].param_groups:
                    group['lr'] = rate
            self.params = [p for p in modules['lora'].parameters() if p.requires_grad]
            self.step = len(history) + 1
            patched_optimizer = optimizers['lora']
            optimizer_step = patched_optimizer.step

            def step_with_telemetry(*args, **kwargs):
                before = [p.detach().clone() for p in self.params]
                grad2 = sum(p.grad.detach().double().square().sum() for p in self.params if p.grad is not None)
                result = optimizer_step(*args, **kwargs)
                delta2 = sum((p.detach().double() - b.double()).square().sum() for p,b in zip(self.params,before))
                weight2 = sum(b.double().square().sum() for b in before)
                record = dict(step=self.step, post_value_clip_gradient_norm=float(grad2.sqrt()),
                              parameter_update_norm=float(delta2.sqrt()),
                              parameter_update_relative_norm=float((delta2 / weight2.clamp_min(1e-30)).sqrt()),
                              g_lr=patched_optimizer.param_groups[0]['lr'], fm_rows=self.rows,
                              fm_cap_active_rows=sum(r['factor'] < 1. for r in self.rows))
                if self.fm_grads is not None:
                    record['raw_fm_parameter_gradient_norm'] = float(self.fm_grads[0].norm())
                    record['limited_fm_parameter_gradient_norm'] = float(self.fm_grads[1].norm())
                with self.telemetry_path.open('a') as handle:
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                self.step += 1; self.rows = []; self.fm_grads = None
                return result

            patched_optimizer.step = step_with_telemetry
            return history

        def tagged_features(critic, seq, mask=None):
            value = features(critic, seq, mask)
            if torch.is_grad_enabled() and seq.requires_grad:
                value._stability_input = seq
                value._stability_scale = critic.input_scale.detach()
            return value

        def controlled_fm(value, fake_mean, real_mean):
            loss = fm(value, fake_mean, real_mean)
            if not hasattr(value, '_stability_input'):
                raise ValueError('Expected an attached span-critic feature tensor')
            limited, norm, factor = limit_loss_input_gradient(
                loss, value._stability_input, value._stability_scale, self.config['fm_input_grad_cap'])
            self.rows.append(dict(input_gradient_norm=norm, factor=factor, limited_input_gradient_norm=norm*factor))
            if self.params and self.diagnostics_every and self.step % self.diagnostics_every == 0:
                grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
                vector = torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten()
                                    for p,g in zip(self.params,grads)])
                # The tested recipe averages all four rows uniformly and uses FM weight one.
                vector = vector / self.baseline['settings']['adv_batch']
                if self.fm_grads is None:
                    self.fm_grads = [vector.clone(), vector * factor]
                else:
                    self.fm_grads[0].add_(vector); self.fm_grads[1].add_(vector, alpha=factor)
            return limited

        def save_branch(path, **kwargs):
            save(path, **kwargs)
            if self.archive_states:
                step = len(kwargs['history'])
                target = Path(path).with_name(Path(path).stem.removesuffix('_state') + f'_step{step}_state.pt')
                if not target.exists():
                    shutil.copyfile(path, target)

        if self.telemetry_path.exists():
            raise ValueError('Use a fresh telemetry path')
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        game.signature = signed; game.restore = restore_branch; game.save = save_branch
        trainer.feature_mean_surrogate = controlled_fm
        trainer._lm_adv.SpanTransformerD.features = tagged_features
        try:
            yield
        finally:
            game.signature, game.restore, game.save, trainer.feature_mean_surrogate, trainer._lm_adv.SpanTransformerD.features = originals
            if patched_optimizer is not None:
                patched_optimizer.step = optimizer_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--g-lr-scale', type=float, default=1.)
    parser.add_argument('--fm-input-grad-cap', type=float)
    parser.add_argument('--memory-fraction', type=float)
    parser.add_argument('--diagnostics-every', type=int, default=25)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.g_lr_scale <= 0 or (args.fm_input_grad_cap is not None and args.fm_input_grad_cap <= 0):
        parser.error('Intervention scales must be positive')
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1: parser.error('Invalid memory fraction')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    state = torch.load(args.state, map_location='cpu', weights_only=True)
    baseline = state['signature'].get('stability_experiment', {}).get('baseline_signature', state['signature'])
    settings = baseline['settings']
    if not (settings['adv_arch']=='tx' and settings['adv_condition']=='none' and settings['fm_mode']=='batch'
            and settings['fm_weight']==1 and settings['parts']==0 and settings['gan_lr_schedule']=='constant'):
        raise ValueError('This bounded experiment supports the unchanged smoke GAN card only')
    prompts = ROOT/'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml'
    train_args = trainer.parse_args(['--prompts_file', str(prompts)])
    vars(train_args).update(settings)
    run = ROOT/'models/gan-bcap-repair'/args.name
    run.mkdir(exist_ok=True)
    if any((run/f'{args.name}{suffix}').exists() for suffix in ['_train.jsonl','_last.safetensors','_telemetry.jsonl']):
        raise ValueError('Use a fresh run name')
    train_args.name=args.name; train_args.save_dir=str(run); train_args.steps=args.steps
    train_args.save_every=args.save_every; train_args.resume_state=str(args.state.resolve())
    train_args.save_training_state=True; train_args.device=0
    train_args.lr=settings['lr']*args.g_lr_scale
    if args.g_lr_scale != 1.:
        train_args.adv_lr=settings['adv_lr'] if settings['adv_lr'] is not None else settings['lr']*trainer._lm_adv.D_LR_MULT
    hooks=StabilityHooks(baseline_signature=baseline,g_lr_scale=args.g_lr_scale,
                         fm_input_grad_cap=args.fm_input_grad_cap,
                         telemetry_path=run/f'{args.name}_telemetry.jsonl', diagnostics_every=args.diagnostics_every)
    manifest=dict(source_state=str(args.state.resolve()),source_state_sha256=sha(args.state),
                  source_completed_updates=state['completed_updates'],total_updates=args.steps,
                  intervention=hooks.config,physical_gpu=1,save_every=args.save_every,
                  caveat='Research intervention; not an exact continuation of the baseline objective/update rule.',
                  command=sys.argv)
    (run/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(__file__,run/'stability_experiment_source.py')
    with hooks.installed():
        weights=trainer.train(train_args)
    sidecar=weights.with_suffix('.json');meta=json.loads(sidecar.read_text())
    meta['stability_experiment']=hooks.config
    sidecar.write_text(json.dumps(meta,indent=2)+'\n')
    manifest['training_exit_code']=0;manifest['weights']=str(weights)
    (run/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
