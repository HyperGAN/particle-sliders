"""Recipe-backed BISPF on paired slider errors; the product owns its model.

This is a conditional component loop, not the unconditional GANTrainer API.
Routing reads the entire existing particle bank, so SBNP acts on that bank;
there is no substitution of sampled latent rows or standalone BISPFEngine.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import math

import torch
from torch import nn
from particlegan import get_recipe

UPSTREAM_COMMIT = "b5d2372f2739d7579ab66d6dba6e220d2eb776d9"


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _finite_grad(parameters):
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)


class BISPFGame:
    """One full-batch Sinkhorn D/G update on normalized paired errors.

    ``network.particles`` is the routed trainable [N, latent_dim] bank.
    ``critic.normalize`` maps absolute teacher/student outputs to the same
    coordinates. ``predict(row, phase)`` returns normalized [1, width] student
    output, with a graph in phase ``g``. It must retain the product's routing.
    Every G prediction stays live until the full Sinkhorn loss is evaluated.

    The fixed shared Gaussian is the paired-error task's reference law, not
    an annealing schedule or BISPFEngine's additive thermal noise. Checkpoint
    only between completed updates with state_dict()/load_state_dict().
    """

    def __init__(self, network, critic, targets, *, batch_size=8, seed=0,
                 g_lr=0.0006, d_lr=0.0009, particle_lr=0.006, sigma=1.0,
                 total_steps=1600):
        if type(batch_size) is not int or not 2 <= batch_size <= 512:
            raise ValueError("batch_size must be an integer in [2, 512]")
        if type(seed) is not int or not 0 <= seed < 2**63 - 100:
            raise ValueError("seed must be an integer in [0, 2**63-100)")
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        if any(not math.isfinite(x) or x <= 0 for x in (g_lr, d_lr, particle_lr, sigma)):
            raise ValueError("learning rates and sigma must be finite and positive")
        particles = network.particles
        if (not isinstance(particles, nn.Parameter) or particles.ndim != 2
                or particles.shape[0] < 2 or particles.shape[1] < 1 or not particles.requires_grad):
            raise ValueError("network.particles must be a trainable [N, latent_dim] Parameter")
        self.device = particles.device
        if particles.dtype != torch.float32:
            raise ValueError("BISPF paired game requires float32 trainable weights")
        if (targets.ndim != 2 or len(targets) < 2 or not targets.shape[1]
                or not torch.isfinite(targets).all()):
            raise ValueError("targets must be finite [rows>=2, width>0]")
        for module in (network, critic):
            if any(p.device != self.device or p.dtype != torch.float32
                   for p in module.parameters() if p.requires_grad):
                raise ValueError("all trainable parameters must share device and float32 dtype")
        if hasattr(critic, "score_bound") and critic.score_bound != 0:
            raise ValueError("BISPF requires raw critic logits: set score_bound=0")
        self.network, self.critic = network, critic
        self.targets = critic.normalize(targets.to(device=self.device, dtype=torch.float32)).detach().clone()
        if not torch.isfinite(self.targets).all():
            raise ValueError("normalized targets must be finite")
        self.batch_size, self.sigma = batch_size, float(sigma)
        self.recipe = get_recipe("bispf", num_particles=particles.shape[0],
            z_dim=particles.shape[1], batch_size=batch_size, total_steps=total_steps,
            lr=g_lr, d_lr_mult=d_lr/g_lr, prior_lr_mult=particle_lr/g_lr)
        # Alias, never copy/reinitialize the adapter's existing particle Parameter.
        # Component APIs allow caller-owned priors; A2 is explicitly off in BISPF.
        self.prior = nn.Module()
        self.prior.register_parameter("z", particles)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(network, critic, self.prior)
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.spread = self.recipe.make_prior_regularizer(prior=self.prior)
        self.streams = {name: torch.Generator().manual_seed(seed+offset)
                        for name, offset in (("rows", 10), ("noise", 30))}
        self.model_cpu_rng = torch.Generator().manual_seed(seed+80).get_state()
        self.model_cuda_rng = (torch.Generator(device=self.device).manual_seed(seed+80).get_state()
                               if self.device.type == "cuda" else None)
        self.completed_steps = 0

    def _batch(self):
        rows = torch.randint(len(self.targets), (self.batch_size,), generator=self.streams["rows"])
        noise = torch.randn(self.batch_size, self.targets.shape[1], generator=self.streams["noise"])
        return rows, noise.to(self.device) * self.sigma

    def _predictions(self, rows, phase, predict):
        unique = {}
        for i in sorted(set(rows.tolist())):
            value = predict(i, phase)
            if (value.shape != (1, self.targets.shape[1]) or value.device != self.device
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError("predict must return finite float32 [1, width] on the game device")
            unique[i] = value
        return torch.cat([unique[i] for i in rows.tolist()])

    @contextmanager
    def _model_stream(self):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.model_cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self.model_cuda_rng, self.device)
            try:
                yield
            finally:
                self.model_cpu_rng = torch.get_rng_state()
                if devices:
                    self.model_cuda_rng = torch.cuda.get_rng_state(self.device)

    def step(self, index, predict):
        if type(index) is not int or index != self.completed_steps + 1 or index > self.recipe.total_steps:
            raise ValueError("updates must be contiguous; resume from completed_steps + 1")
        with self._model_stream():
            return self._step(index, predict)

    def _step(self, index, predict):
        d_rows, d_noise = self._batch()
        self.critic.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        context = self.network.scaled(1.) if hasattr(self.network, "scaled") else nullcontext()
        with torch.no_grad(), context:
            fake = d_noise + self._predictions(d_rows, "d", predict) - self.targets[d_rows.to(self.device)]
        # Pass original hook-bearing score tensors: reshape/detach here would
        # discard coordinates and silently couple scalar logits instead.
        d_adv = self.loss.d_loss(self.critic(d_noise), self.critic(fake))
        penalty = self.penalty(self.critic, d_noise, fake)
        d_total = d_adv + penalty
        if not torch.isfinite(d_total):
            raise FloatingPointError("non-finite BISPF critic loss")
        d_total.backward()
        if not _finite_grad(self.critic.parameters()):
            raise FloatingPointError("non-finite BISPF critic gradients")
        self.opt_d.step()
        self.opt_d.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        self.opt_g.zero_grad(set_to_none=True)
        g_rows, g_noise = self._batch()
        with torch.no_grad():
            real_scores = self.critic(g_noise)
        context = self.network.scaled(1.) if hasattr(self.network, "scaled") else nullcontext()
        with context:
            fake = g_noise + self._predictions(g_rows, "g", predict) - self.targets[g_rows.to(self.device)]
            g_adv = self.loss.g_loss(self.critic(fake), real_scores)
            prior_loss = self.spread(self.prior.z)  # recipe weight applied once
            total = g_adv + prior_loss
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite BISPF generator loss")
            total.backward()
        if not _finite_grad(self.network.parameters()):
            raise FloatingPointError("non-finite BISPF generator gradients")
        self.opt_g.step()
        self.completed_steps = index
        return dict(step=index, loss=float(total.detach()), g_adv=float(g_adv.detach()),
            d_loss=float(d_total.detach()), d_adv=float(d_adv.detach()),
            d_pen=float(penalty.detach()), prior_regularization=float(prior_loss.detach()),
            noise_std=self.sigma, d_rows=d_rows.tolist(), g_rows=g_rows.tolist(),
            **self.penalty.diagnostics())

    def _config(self):
        return dict(upstream_commit=UPSTREAM_COMMIT, recipe=self.recipe.to_dict(),
                    sigma=self.sigma, targets_sha256=_digest(self.targets))

    def state_dict(self):
        # Upstream's record serializes EMA residual but recomputes thermal scale
        # approximately on load. Preserve its exact current scale separately,
        # needed by the critic hook before the next penalty recomputes it.
        return deepcopy(dict(schema=1, config=self._config(), completed_steps=self.completed_steps,
            network=self.network.state_dict(), critic=self.critic.state_dict(),
            prior=self.prior.state_dict(), g_optimizer=self.opt_g.state_dict(),
            d_optimizer=self.opt_d.state_dict(),
            thermal_scale=float(getattr(self.opt_d.record, "_thermal_scale", 1.)),
            streams={name: gen.get_state() for name, gen in self.streams.items()},
            model_cpu_rng=self.model_cpu_rng, model_cuda_rng=self.model_cuda_rng))

    def load_state_dict(self, state):
        if state.get("schema") != 1 or state.get("config") != self._config():
            raise ValueError("incompatible BISPF checkpoint: upstream, recipe, sigma or targets differ")
        if type(state["completed_steps"]) is not int or state["completed_steps"] < 0:
            raise ValueError("invalid BISPF completed_steps")
        thermal = state["thermal_scale"]
        if not math.isfinite(thermal) or not 0 < thermal <= 1:
            raise ValueError("invalid BISPF thermal scale")
        self.network.load_state_dict(state["network"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.prior.load_state_dict(state["prior"], strict=True)
        self.opt_g.load_state_dict(deepcopy(state["g_optimizer"]))
        self.opt_d.load_state_dict(deepcopy(state["d_optimizer"]))
        self.opt_d.record._thermal_scale = thermal
        for name, gen in self.streams.items():
            gen.set_state(state["streams"][name].cpu())
        self.model_cpu_rng = state["model_cpu_rng"].cpu()
        self.model_cuda_rng = state["model_cuda_rng"]
        self.completed_steps = state["completed_steps"]
