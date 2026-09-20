"""Reference paired-error game. No reconstruction or perceptual objective."""
import math

import torch

from .vendor.reference import GlobalMixErrorCritic, noise_std, particle_vic, rp_d_loss, rp_g_loss
from .vendor.grad_regularizers import GradRegularizer

RECIPE = dict(rank=8, particles=128, particle_dim=4, router_width=16, branch_width=48,
    critic="gmix", critic_tokens=8, critic_width=48, critic_layers=1, critic_heads=4,
    score_bound=8, ema=.995, batch=8, g_lr=2e-5, d_lr=9e-4, particle_lr=6e-3,
    betas=[0., .999], weight_decay=0, cap_every=4, cap_kappa=1., cap_coeff=1.,
    horizon=1600, noise_hold=1., noise_floor=.03, edit_noise_ratio=.28)


class Sampler:
    def __init__(self, count, seed, batch=8):
        self.count, self.batch = count, batch
        self.generators = {name: torch.Generator().manual_seed(seed + i * 1009)
                           for i, name in enumerate(("d_rows", "g_rows", "d_noise", "g_noise", "vic"))}

    def draw(self, phase, dimension):
        indices = torch.randint(self.count, (self.batch,), generator=self.generators[f"{phase}_rows"]).tolist()
        noise = torch.randn(self.batch, dimension, generator=self.generators[f"{phase}_noise"])
        return indices, noise

    def state_dict(self):
        return dict(count=self.count, batch=self.batch, generators={k: g.get_state() for k, g in self.generators.items()})

    def load_state_dict(self, state):
        if (state["count"], state["batch"]) != (self.count, self.batch):
            raise ValueError("Sampler dimensions changed")
        for name, value in state["generators"].items():
            self.generators[name].set_state(value.cpu())


def grad_norm(parameters):
    norms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.


class Game:
    def __init__(self, adapter, normalization, count, seed=7, microbatch=1):
        if microbatch not in (1, 2, 4, 8):
            raise ValueError("Microbatch must divide effective batch eight")
        self.adapter, self.microbatch = adapter, microbatch
        self.device = adapter.particles.device
        self.scale = normalization["scale"].to(self.device)
        self.edit_rms = normalization["edit_rms"].to(self.device)
        self.noise_starts = [float(r) / .28 for r in normalization["edit_rms"].cpu()]
        dimension = self.scale.shape[-1]
        # The critic receives already normalized paired residuals. Its reference
        # normalization buffers are unused; retain them for reference compatibility.
        self.critic = GlobalMixErrorCritic(torch.stack((torch.zeros(dimension), torch.ones(dimension))),
            tokens=8, width=48, layers=1, heads=4, score_bound=8).to(self.device)
        self.g = torch.optim.Adam([
            dict(params=[p for n, p in adapter.named_parameters() if n != "particles"], lr=2e-5),
            dict(params=[adapter.particles], lr=6e-3)], betas=(0., .999), weight_decay=0.)
        self.d = torch.optim.Adam(self.critic.parameters(), lr=9e-4, betas=(0., .999), weight_decay=0.)
        self.sampler = Sampler(count, seed)
        self.capper = GradRegularizer(arm="b_cap", coeff=1., kappa=1., norm="l2", lazy_k=4)

    def update(self, predict_residual, positions, step, prefetch=None):
        """predict_residual(indices) returns ΔS−ΔT = v_adapted(neutral)−v_frozen(positive).

        Caller holds adapter scales active through this entire method, including
        checkpoint recomputation. Noise and row streams are independent for D/G.
        """
        if step < 1:
            raise ValueError("Updates are numbered starting at one")
        dimension = self.scale.shape[-1]
        values = {}
        self.critic.requires_grad_(True)
        self.d.zero_grad(set_to_none=True)
        di, dn = self.sampler.draw("d", dimension)
        gi, gn = self.sampler.draw("g", dimension)
        if prefetch is not None:
            prefetch(di + gi)
        sigma_by_t = [noise_std(step - 1, start=start, decay_steps=1600, hold=1.)
                      for start in self.noise_starts]

        def noise_and_scale(indices, noise):
            ts = [positions[i] for i in indices]
            sigma = torch.tensor([sigma_by_t[t] for t in ts], device=self.device)
            return noise.to(self.device) * sigma[:, None], self.scale[ts]

        da = cap = 0.
        for start in range(0, len(di), self.microbatch):
            ids = di[start:start + self.microbatch]
            noise, scale = noise_and_scale(ids, dn[start:start + len(ids)])
            with torch.no_grad():
                fake = noise + predict_residual(ids) / scale
            adv = rp_d_loss(self.critic(noise), self.critic(fake))
            penalty, _ = self.capper.penalty(self.critic, noise, fake, step=step, collect_stats=False)
            loss = (adv + penalty) * len(ids) / len(di)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite critic loss")
            loss.backward()
            da += float(adv.detach()) * len(ids) / len(di)
            cap += float(penalty.detach()) * len(ids) / len(di)
        d_norm = grad_norm(self.critic.parameters())
        if not math.isfinite(d_norm):
            raise FloatingPointError("Non-finite critic gradient")
        self.d.step()
        self.d.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        self.g.zero_grad(set_to_none=True)
        ga = 0.
        for start in range(0, len(gi), self.microbatch):
            ids = gi[start:start + self.microbatch]
            noise, scale = noise_and_scale(ids, gn[start:start + len(ids)])
            with torch.no_grad():
                real_score = self.critic(noise)
            fake = noise + predict_residual(ids) / scale
            loss = rp_g_loss(real_score, self.critic(fake)) * len(ids) / len(gi)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite generator loss")
            loss.backward()
            ga += float(loss.detach())
        particle_gan_norm = grad_norm([self.adapter.particles])
        selection = torch.randperm(128, generator=self.sampler.generators["vic"])[:64].to(self.device)
        vic = particle_vic(self.adapter.particles[selection])
        vic.backward()
        g_norm = grad_norm(self.adapter.parameters())
        if not math.isfinite(g_norm):
            raise FloatingPointError("Non-finite generator gradient")
        self.g.step()
        return dict(step=step, d_adv=da, d_penalty=cap, g_adv=ga, vic=float(vic.detach()),
                    d_grad_norm=d_norm, g_grad_norm=g_norm, particle_gan_grad_norm=particle_gan_norm,
                    noise_min=min(sigma_by_t), noise_max=max(sigma_by_t), d_rows=di, g_rows=gi)

    def state_dict(self):
        return dict(critic=self.critic.state_dict(), g=self.g.state_dict(), d=self.d.state_dict(),
                    sampler=self.sampler.state_dict())

    def load_state_dict(self, state):
        self.critic.load_state_dict(state["critic"])
        self.g.load_state_dict(state["g"])
        self.d.load_state_dict(state["d"])
        self.sampler.load_state_dict(state["sampler"])
