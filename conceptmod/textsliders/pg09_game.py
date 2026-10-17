"""Caller-owned paired-output loop using released ParticleGAN public components.

The application owns conditioning, target coordinates and model forwards.
This is not GANTrainer, a reproduction of PG internals, or a claim that an
existing MSE/routed-particle checkpoint has the same objective.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
import hashlib

import torch
from torch import nn
from particlegan import get_recipe, scale_learning_rates
from particlegan.optim.constraint_geometry import constraint_geometry_backward


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class PairedPG09:
    """Fit normalized student outputs to corresponding frozen teacher outputs.

    ``predict(row, phase)`` returns one [1,width] output. Adapter scale stays
    active through backward, including checkpoint recomputation. A task may
    pass its existing routed particle Parameter; the whole bank participates,
    so every active row is explicitly bound to the row-normalized optimizer.
    """
    def __init__(self, network, critic, targets, *, particles=None, preset='bispf_optimized',
                 batch_size=8, seed=7, total_steps=1200, sigma=1., recipe_overrides=None):
        if version('particlegan') != '0.9.0':
            raise RuntimeError('This integration is qualified only with particlegan==0.9.0')
        if targets.ndim != 2 or len(targets) < 2 or not torch.isfinite(targets).all():
            raise ValueError('targets must be finite [rows>=2,width] in fixed task coordinates')
        if batch_size < 2 or sigma <= 0:
            raise ValueError('batch_size>=2 and positive task-reference sigma required')
        self.network, self.critic = network, critic
        first = next(p for p in network.parameters() if p.requires_grad)
        self.device = first.device
        if any(p.dtype != torch.float32 or p.device != self.device
               for m in (network, critic) for p in m.parameters() if p.requires_grad):
            raise ValueError('trainable tensors must share float32 dtype and device')
        self.targets = targets.detach().to(self.device).clone()
        self.batch_size, self.sigma = batch_size, float(sigma)
        self.prior = None
        options = dict(batch_size=batch_size, total_steps=total_steps)
        if particles is not None:
            if not isinstance(particles, nn.Parameter) or particles.ndim != 2:
                raise ValueError('particles must be the existing routed bank Parameter')
            self.prior = nn.Module()
            self.prior.register_parameter('z', particles)
            options.update(num_particles=len(particles), z_dim=particles.shape[1])
        options.update(recipe_overrides or {})
        self.recipe = get_recipe(preset, **options)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(network, critic, self.prior)
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.spread = self.recipe.make_prior_regularizer(prior=self.prior) if self.prior else None
        self.base_lrs = [[g['lr'] for g in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.stream = torch.Generator().manual_seed(seed)
        self.model_cpu_rng = torch.Generator().manual_seed(seed + 1).get_state()
        self.model_cuda_rng = (torch.Generator(device=self.device).manual_seed(seed + 1).get_state()
                               if self.device.type == 'cuda' else None)
        self.completed_steps = 0

    def _batch(self):
        ids = torch.randint(len(self.targets), (self.batch_size,), generator=self.stream)
        noise = torch.randn(self.batch_size, self.targets.shape[1], generator=self.stream)
        return ids, noise.to(self.targets) * self.sigma

    def _predict(self, ids, phase, predict):
        values = {i: predict(i, phase) for i in sorted(set(ids.tolist()))}
        if any(v.shape != (1, self.targets.shape[1]) or not torch.isfinite(v).all()
               for v in values.values()):
            raise ValueError('predict must return finite [1,width] outputs')
        return torch.cat([values[i] for i in ids.tolist()])

    def step(self, predict):
        if self.completed_steps >= self.recipe.total_steps:
            raise RuntimeError('execution budget exhausted')
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.autograd.set_multithreading_enabled(False):
            torch.set_rng_state(self.model_cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self.model_cuda_rng, self.device)
            try:
                return self._step(predict)
            finally:
                self.model_cpu_rng = torch.get_rng_state()
                if devices:
                    self.model_cuda_rng = torch.cuda.get_rng_state(self.device)

    def _step(self, predict):
        scale_learning_rates(self.completed_steps, self.recipe, (self.opt_g, self.opt_d),
                             self.base_lrs, self.prior)
        ids, real = self._batch()
        self.critic.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        context = self.network.scaled(1.) if hasattr(self.network, 'scaled') else nullcontext()
        with torch.no_grad(), context:
            fake = real + self._predict(ids, 'd', predict) - self.targets[ids.to(self.device)]
        d_adv = self.loss.d_loss(self.critic(real), self.critic(fake))
        penalty = self.penalty(self.critic, real, fake)
        d_total = d_adv + penalty
        if not torch.isfinite(d_total):
            raise FloatingPointError('nonfinite D loss')
        d_total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.critic.parameters()):
            raise FloatingPointError('nonfinite D gradients')
        self.opt_d.step()
        self.opt_d.zero_grad(set_to_none=True)
        flags = [p.requires_grad for p in self.critic.parameters()]
        self.critic.requires_grad_(False)
        self.opt_g.zero_grad(set_to_none=True)
        ids, real = self._batch()
        context = self.network.scaled(1.) if hasattr(self.network, 'scaled') else nullcontext()
        try:
            with context:
                fake = real + self._predict(ids, 'g', predict) - self.targets[ids.to(self.device)]
                # Preserve score hooks; do not reshape/detach before passing to PG.
                g_adv = self.loss.g_loss(self.critic(fake), self.critic(real))
                prior_loss = self.spread(self.prior.z) if self.spread else g_adv.new_zeros(())
                total = g_adv + prior_loss  # public factory applies prior weight once
                if not torch.isfinite(total):
                    raise FloatingPointError('nonfinite G loss')
                if self.prior is not None and hasattr(self.opt_g, 'set_sampled_rows'):
                    # Dense routing reads every row, unlike sampled latent G(z).
                    self.opt_g.set_sampled_rows(self.prior.z,
                        torch.arange(len(self.prior.z), device=self.device))
                constraint_geometry_backward(total, self.opt_g, (g_adv,))
            for module in (self.network, self.critic):
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                    raise FloatingPointError('nonfinite gradients')
            self.opt_g.step()
        finally:
            for p, flag in zip(self.critic.parameters(), flags):
                p.requires_grad_(flag)
        self.completed_steps += 1
        return dict(step=self.completed_steps, d_loss=float(d_total.detach()),
                    d_adv=float(d_adv.detach()), d_pen=float(penalty.detach()),
                    g_loss=float(total.detach()), g_adv=float(g_adv.detach()),
                    prior_regularization=float(prior_loss.detach()), **self.penalty.diagnostics())

    def _config(self):
        return dict(pg_version='0.9.0', recipe=asdict(self.recipe),
                    targets_sha256=digest(self.targets), sigma=self.sigma)

    def state_dict(self):
        return deepcopy(dict(schema=1, config=self._config(), completed_steps=self.completed_steps,
            network=self.network.state_dict(), critic=self.critic.state_dict(),
            prior=self.prior.state_dict() if self.prior else None,
            opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(),
            spread=self.spread.state_dict() if self.spread else None,
            stream=self.stream.get_state(), model_cpu_rng=self.model_cpu_rng,
            model_cuda_rng=self.model_cuda_rng))

    def load_state_dict(self, state):
        if state.get('schema') != 1 or state.get('config') != self._config():
            raise ValueError('incompatible PG version, resolved recipe, targets or reference sigma')
        self.network.load_state_dict(state['network'], strict=True)
        self.critic.load_state_dict(state['critic'], strict=True)
        if self.prior:
            self.prior.load_state_dict(state['prior'], strict=True)
            self.spread.load_state_dict(state['spread'], strict=True)
        self.opt_g.load_state_dict(deepcopy(state['opt_g']))
        self.opt_d.load_state_dict(deepcopy(state['opt_d']))
        self.stream.set_state(state['stream'].cpu())
        self.model_cpu_rng = state['model_cpu_rng'].cpu()
        self.model_cuda_rng = state['model_cuda_rng']
        self.completed_steps = state['completed_steps']
