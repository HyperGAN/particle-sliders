"""Paired-error ParticleGAN game from model-glue's anneal-routed benchmark.

Reference: model-glue df70ccb, configs/particle-toy-20260917.json;
ParticleGAN 441fdf42. No output reconstruction loss is used.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from analysis.slider2d.adv import rp_d_loss, rp_g_loss, make_grad_regularizer
from conceptmod.textsliders.lm_adv import param_grad_norm

REFERENCE = dict(particles=128, z_dim=4, width=48, router_width=16,
    batch_size=64, g_lr=.0006, d_lr=.0009, particle_lr=.006,
    betas=(0., .999), vic_coeff=1., cap_coeff=1., cap_kappa=1., cap_every=4,
    ema=.995, noise_decay_steps=8000, noise_floor=.03, target_std_floor=1e-4)


def mlp(inputs, outputs, width):
    layers = []
    for n in (inputs, width, width):
        layers.extend((nn.Linear(n, width), nn.LeakyReLU(.2)))
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


class RoutedMLP(nn.Module):
    """Every input uses the same softmax route at train and inference time."""
    def __init__(self, inputs, outputs, z_dim=4, width=48, router_width=16):
        super().__init__()
        self.router = mlp(inputs, z_dim, router_width)
        self.net = mlp(inputs + z_dim, outputs, width)

    def forward(self, x, particles):
        q = self.router(x)
        weights = (q @ particles.T / math.sqrt(particles.shape[1])).softmax(-1)
        z = weights @ particles
        return self.net(torch.cat((x, z), dim=-1))


def particle_vic(z):
    """Pinned upstream VICRegLikeLoss (target_std=1, eps=1e-4).

    Sample standard deviation and off-diagonal covariance, on particles only.
    Source SHA256 d2563bcab93d443cbf2cf2260946a3ba470f1fc584c231b704ec5ac154f668c4.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    variance = F.relu(1. - std).mean()
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / (len(z) - 1)
    dim = z.shape[1]
    if dim == 1:
        return variance
    off = covariance.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    return variance + off.square().sum() / dim


def noise_std(step):
    if step < 0:
        raise ValueError('Step must be nonnegative')
    return REFERENCE['noise_floor'] ** min(step / REFERENCE['noise_decay_steps'], 1.)


class BridgeSampler:
    """CPU random streams, fresh rows/noise for each D and G minibatch."""
    def __init__(self, count, seed):
        if count < 2:
            raise ValueError('At least two training targets are needed for sample std')
        self.count = count
        self.rngs = {name: torch.Generator().manual_seed(seed + offset)
                     for name, offset in [('data', 10), ('noise', 30), ('vic', 40)]}

    def batch(self, dim, device, sigma):
        rows = torch.randint(self.count, (REFERENCE['batch_size'],), generator=self.rngs['data'])
        noise = torch.randn(len(rows), dim, generator=self.rngs['noise']).to(device) * sigma
        return rows, noise

    def vic_rows(self, count, device):
        return torch.randperm(count, generator=self.rngs['vic'])[:64].to(device)

    def state_dict(self):
        return dict(count=self.count, rngs={k: r.get_state() for k, r in self.rngs.items()})

    def load_state_dict(self, state):
        if state['count'] != self.count or state['rngs'].keys() != self.rngs.keys():
            raise ValueError('Incompatible particle-bridge sampler')
        for name, value in state['rngs'].items():
            self.rngs[name].set_state(value)


class ErrorCritic(nn.Module):
    def __init__(self, targets):
        super().__init__()
        if targets.ndim != 2 or len(targets) < 2 or not torch.isfinite(targets).all():
            raise ValueError('Expected finite training targets [rows, width]')
        self.register_buffer('target_mean', targets.mean(0))
        self.register_buffer('target_std', targets.std(0).clamp_min(REFERENCE['target_std_floor']))
        # Scalar for existing diagnostics only; scoring uses target_std exactly once.
        self.register_buffer('input_scale', targets.std(0).square().mean().sqrt())
        self.net = mlp(targets.shape[1], 1, REFERENCE['width'])

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        return self.net(coordinates).squeeze(-1)


def build_game(network, targets):
    critic = ErrorCritic(targets).to(targets.device)
    parameters = [p for name, p in network.named_parameters() if name != 'particles']
    if not parameters or network.particles.shape != (128, 4):
        raise ValueError('Expected generator parameters and one 128x4 particle cloud')
    g = torch.optim.Adam([
        dict(params=parameters, lr=REFERENCE['g_lr'], role='generator'),
        dict(params=[network.particles], lr=REFERENCE['particle_lr'], role='particles'),
    ], betas=REFERENCE['betas'])
    d = torch.optim.Adam(critic.parameters(), lr=REFERENCE['d_lr'], betas=REFERENCE['betas'])
    return critic, g, d


def update(network, critic, g, d, targets, predict, *, sampler, step):
    """predict(row, phase) returns normalized G output, retaining its graph.

    Repeated deterministic rows are forwarded once per phase; their independent
    noise samples still contribute separately to the batch-64 mean and gradient.
    """
    device = critic.target_mean.device
    real_targets = critic.normalize(targets.to(device)).detach()
    sigma = noise_std(step)
    critic.requires_grad_(True)
    di, dn = sampler.batch(targets.shape[1], device, sigma)
    dn = dn.to(real_targets)
    with torch.no_grad():
        unique = {i: predict(i, 'd') for i in sorted(set(di.tolist()))}
        predictions = torch.cat([unique[i] for i in di.tolist()])
        fake = dn + predictions - real_targets[di.to(device)]
    d.zero_grad(set_to_none=True)
    da = rp_d_loss(critic(dn), critic(fake))
    capper = make_grad_regularizer(arm='b_cap', coeff=1., kappa=1., norm='l2',
                                  lazy_k=4, target_anneal='none')
    cap, _ = capper.penalty(critic, dn, fake, step=step, collect_stats=False)
    d_loss = da + cap
    if not torch.isfinite(d_loss):
        raise FloatingPointError('Non-finite particle-bridge D loss')
    d_loss.backward()
    if not math.isfinite(param_grad_norm(critic.parameters())):
        raise FloatingPointError('Non-finite particle-bridge D gradient')
    d.step(); d.zero_grad(set_to_none=True); critic.requires_grad_(False)

    gi, gn = sampler.batch(targets.shape[1], device, sigma)
    gn = gn.to(real_targets)
    with torch.no_grad():
        real_scores = critic(gn)
    g.zero_grad(set_to_none=True)
    ga = 0.
    for i in sorted(set(gi.tolist())):
        mask = (gi == i).to(device)
        # Keep native adapter scale active through checkpointed backward.
        context = network.scaled(1.) if hasattr(network, 'scaled') else _identity_context()
        with context:
            prediction = predict(i, 'g')
            fake = gn[mask] + prediction - real_targets[i:i+1]
            loss = rp_g_loss(real_scores[mask], critic(fake)) * int(mask.sum()) / len(gi)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite particle-bridge G loss')
            loss.backward()
            ga += float(loss.detach())
    particle_gan_norm = param_grad_norm([network.particles])
    pv = particle_vic(network.particles[sampler.vic_rows(len(network.particles), device)])
    if not torch.isfinite(pv):
        raise FloatingPointError('Non-finite particle VIC')
    pv.backward()
    norm = param_grad_norm(network.parameters())
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite particle-bridge G gradient')
    particle_norm = param_grad_norm([network.particles])
    g.step()
    return dict(loss=ga + float(pv.detach()), g_adv=ga, particle_vic=float(pv.detach()),
        d_loss=float(d_loss.detach()), d_adv=float(da.detach()), d_pen=float(cap.detach()),
        grad_norm=norm, particle_grad_norm=particle_norm, particle_gan_grad_norm=particle_gan_norm, noise_std=sigma,
        penalty_center=1., d_rows=di.tolist(), g_rows=gi.tolist())


@contextmanager
def _identity_context():
    yield


def initialize_ema(network):
    return {k: value.detach().clone() for k, value in network.state_dict().items()}


@torch.no_grad()
def update_ema(ema, network):
    parameters = set(dict(network.named_parameters()))
    for name, value in network.state_dict().items():
        if name in parameters:
            ema[name].lerp_(value, 1. - REFERENCE['ema'])
        else:
            ema[name].copy_(value)
