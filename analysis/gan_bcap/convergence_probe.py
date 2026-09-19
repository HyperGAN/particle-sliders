"""Disposable unilateral-improvement probes using the actual repaired GAN trainer.

The critic is held fixed while G minimizes its complete training objective.
Separately, cloned critics optimize against the initial fixed generator spans.
These finite optimization searches give LOWER bounds on available improvement;
a small result is not a certificate of equilibrium. No source state is mutated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conceptmod.textsliders import lm_gan_state as game
from conceptmod.textsliders import train_lm_slider_music3 as trainer
from conceptmod.textsliders import lm_adv as adv
from analysis.gan_bcap.parameter_step_limit import bound_step


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def improvement(values):
    if not values or not all(torch.isfinite(torch.tensor(v)) for v in values):
        raise ValueError('Expected finite objective measurements')
    initial = values[0]
    best = min(values)
    return dict(initial=initial, best=best, final=values[-1],
                available_improvement=initial-best,
                fractional_improvement=(initial-best)/max(abs(initial), 1e-8),
                best_after_updates=values.index(best))


def tensors_equal(before, after):
    return before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


def discriminator_search(critic, optimizer_state, real, fake, masks, settings,
                         steps, cap_function):
    """Optimize copies only; include regularization in the player's own objective."""
    before = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    results = []
    for fresh in [False, True]:
        for lr_scale in [.25, 1.]:
            model = deepcopy(critic).requires_grad_(True)
            opt = torch.optim.Adam(model.parameters(), lr=1.)
            opt.load_state_dict(deepcopy(optimizer_state))
            if fresh:
                opt.state.clear()
            for group in opt.param_groups:
                group['lr'] *= lr_scale
            trace = []
            for index in range(steps+1):
                opt.zero_grad(set_to_none=True)
                real_logits = model(real, masks[0])
                fake_logits = model(fake, masks[1])
                ranking = adv.rp_d_loss(real_logits, fake_logits)
                penalty, _ = cap_function(model, real, fake,
                    coeff=settings['adv_reg_coeff'], kappa=settings['adv_reg_kappa'],
                    mask_real=masks[0], mask_fake=masks[1])
                loss = ranking + penalty
                trace.append(dict(update=index, loss=float(loss.detach()),
                                  ranking=float(ranking.detach()), penalty=float(penalty.detach())))
                if index < steps:
                    loss.backward()
                    opt.step()
            results.append(dict(optimizer='fresh' if fresh else 'restored', lr_scale=lr_scale,
                                **improvement([r['loss'] for r in trace]), trace=trace))
            del opt, model
    if not tensors_equal(before, critic.state_dict()):
        raise AssertionError('Discriminator probe changed the source critic')
    return results


class FrozenOpponent:
    def __init__(self, state, source_sha, directory, g_lr_scale, d_steps):
        self.state = state
        self.directory = directory
        self.config = dict(version=1, source_state_sha256=source_sha,
            opponent='frozen source critic', g_lr_scale=g_lr_scale,
            generator_step_norm_maximum=2., discriminator_probe_updates=d_steps,
            implementation_sha256=sha(__file__),
            interpretation='Finite unilateral improvement lower bound, not a duality gap certificate.')
        self.g_updates = []
        self.d_results = None
        self.modules = None

    @contextmanager
    def installed(self):
        signature, restore, penalty = game.signature, game.restore, adv.cap_penalty
        originals = []

        def signed(args, rows, metadata):
            actual = signature(args, rows, metadata)
            original = {k: v for k, v in self.state['signature'].items()
                        if k in ['settings', 'sources', 'prompts_sha256']}
            if actual != original:
                raise ValueError('Source code, settings or training prompts changed')
            return dict(actual, convergence_probe=self.config)

        def restored(path, *, run_signature, modules, optimizers):
            if run_signature.get('convergence_probe') != self.config:
                raise ValueError('Missing probe signature')
            history = restore(path, run_signature=self.state['signature'],
                              modules=modules, optimizers=optimizers)
            self.modules = modules
            self.critic_before = {k: v.detach().clone() for k, v in modules['critic'].state_dict().items()}
            self.optimizer_before = deepcopy(optimizers['critic'].state_dict())
            critic_opt, generator_opt = optimizers['critic'], optimizers['lora']
            originals.extend([(critic_opt, critic_opt.step), (generator_opt, generator_opt.step)])
            critic_opt.step = lambda *a, **kw: None
            generator_step = generator_opt.step
            parameters = list(modules['lora'].parameters())
            for group in generator_opt.param_groups:
                group['lr'] *= self.config['g_lr_scale']

            def step(*args, **kwargs):
                before = [p.detach().clone() for p in parameters]
                result = generator_step(*args, **kwargs)
                self.g_updates.append(bound_step(parameters, before, 2.))
                return result

            generator_opt.step = step
            return history

        def measured_penalty(discriminator, real, fake, *args, **kwargs):
            if self.d_results is None and self.modules is not None:
                self.d_results = discriminator_search(discriminator, self.optimizer_before,
                    real.detach(), fake.detach(),
                    (kwargs.get('mask_real'), kwargs.get('mask_fake')),
                    self.state['signature']['settings'],
                    self.config['discriminator_probe_updates'], penalty)
                write(self.directory/'discriminator.json', self.d_results)
                # Useful controls for implementation correctness, not equilibrium labels.
                with torch.no_grad():
                    scores = discriminator(real, kwargs.get('mask_real'))
                    equal_loss = float(adv.rp_d_loss(scores, scores))
                    assert abs(equal_loss - .6931471805599453) < 1e-6
                print('Cloned discriminator searches complete; source critic unchanged.', flush=True)
            return penalty(discriminator, real, fake, *args, **kwargs)

        game.signature, game.restore, adv.cap_penalty = signed, restored, measured_penalty
        try:
            yield self
        finally:
            game.signature, game.restore, adv.cap_penalty = signature, restore, penalty
            for optimizer, original in originals:
                optimizer.step = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--g-updates', type=int, default=20)
    p.add_argument('--g-lr-scale', type=float, default=.25)
    p.add_argument('--d-updates', type=int, default=60)
    args = p.parse_args()
    if args.g_updates < 1 or args.d_updates < 1 or args.g_lr_scale <= 0:
        p.error('Probe budgets and learning-rate scale must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'probe_train.jsonl').exists():
        raise ValueError('Use a fresh probe directory')
    torch.set_num_threads(4)
    state_sha = sha(args.state)
    state = torch.load(args.state, map_location='cpu', weights_only=True)
    settings = state['signature']['settings']
    if not (settings['adv_arch']=='tx' and settings['adv_condition']=='none'
            and settings['adv_batch']==4 and settings['fm_mode']=='batch'
            and settings['parts']==0 and settings['gan_lr_schedule']=='constant'):
        raise ValueError('Probe currently supports the four-row original smoke objective')
    train_args = trainer.parse_args(['--prompts_file', str(ROOT/'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml')])
    vars(train_args).update(settings)
    train_args.name = 'probe'
    train_args.save_dir = str(args.out)
    # Training logs measure BEFORE the update. N+1 records expose the loss after N updates.
    train_args.steps = state['completed_updates'] + args.g_updates + 1
    train_args.save_every = 0
    train_args.resume_state = str(args.state.resolve())
    train_args.save_training_state = True
    train_args.device = 0
    hooks = FrozenOpponent(state, state_sha, args.out, args.g_lr_scale, args.d_updates)
    write(args.out/'manifest.json', dict(source_state=str(args.state.resolve()),
        source_steps=state['completed_updates'], physical_gpu=1, config=hooks.config,
        measured_generator_updates=args.g_updates, actual_generator_updates=args.g_updates+1,
        purpose='Disposable diagnosis; generated weights are not catalog candidates.'))
    with hooks.installed():
        trainer.train(train_args)
    if not tensors_equal(hooks.critic_before, hooks.modules['critic'].state_dict()):
        raise AssertionError('Frozen critic changed during generator updates')
    if sha(args.state) != state_sha:
        raise AssertionError('Source state changed')
    rows = [json.loads(x) for x in (args.out/'probe_train.jsonl').read_text().splitlines()]
    trace = rows[state['completed_updates']:]
    report = dict(status='complete', source_steps=state['completed_updates'],
        source_state=str(args.state.resolve()), source_state_sha256=state_sha,
        config=hooks.config, generator=improvement([r['loss'] for r in trace]),
        generator_trace=trace, generator_update_norms=hooks.g_updates,
        discriminator=hooks.d_results, frozen_critic_exact=True, source_state_unchanged=True,
        limitation='Training-prompt, local optimization diagnostic. A small measured gain may reflect a weak probe or saturation. It cannot establish audio quality or global equilibrium.')
    write(args.out/'result.json', report)
    print(json.dumps(dict(generator=report['generator'],
                         discriminator_best=max(r['available_improvement'] for r in hooks.d_results))), flush=True)


if __name__ == '__main__':
    main()
