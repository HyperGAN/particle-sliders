"""ParticleGAN / Music-LM adversarial core: RpGAN + one-sided b_cap.

Faithful reconstruction of the 100-Gaussians recipe (ParticlePrior,
Fourier-2 MLP critic, relativistic-pair logistic GAN, one-sided ``b_cap``,
VICReg on particles, EMA G+particles, Adam β1=0, delayed cosine LR) plus
the Music-LM extras that have a 2-D analogue (span samples, end-margin,
optional *normalized* feature matching).

``b_cap`` is the soft steepness cap (ParticleGAN-faithful, κ explicit)::

    0.5 * coeff * (mean(relu(‖∇D(real)‖ − κ)²) + mean(relu(‖∇D(fake)‖ − κ)²))

with ``‖g‖ = √(Σ gᵢ² + 1e-12)``. Free below κ, quadratic above. The penalty is
computed by ParticleGAN's ``GradRegularizer`` (vendored verbatim in
``analysis.slider2d.grad_regularizers``) — not by a reimplementation.
``cap_penalty`` is a thin shim over the same ``GradRegularizer`` phi for
callers that already hold input-gradient tensors; the train paths call
``make_grad_regularizer(...).penalty(D, x_real, x_fake)`` directly.
Feature matching on raw D features is *not* capped by this — keep it off
or L2-normalize the pooled features.

CPU-sized. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.slider2d.grad_regularizers import GradRegularizer


def rp_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic-pair logistic critic loss: prefer D(real) > D(fake)."""
    return F.softplus(-(d_real - d_fake)).mean()


def rp_g_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic-pair logistic generator loss: prefer D(fake) > D(real)."""
    return F.softplus(-(d_fake - d_real)).mean()


def make_grad_regularizer(
    cfg: "AdvConfig | None" = None,
    *,
    arm: str | None = None,
    coeff: float | None = None,
    kappa: float | None = None,
    norm: str | None = None,
    lazy_k: int | None = None,
    target_anneal: str | None = None,
    total_steps: int | None = None,
) -> GradRegularizer:
    """ParticleGAN-faithful ``GradRegularizer`` for the b_cap arm.

    Defaults come from ``AdvConfig`` (arm ``b_cap``, coeff ``b_cap``,
    ``kappa=1.0``, ``norm=l2``); explicit kwargs override. This is the
    Strategy-B entry point — every train path must build its penalty here
    rather than reimplementing the cap.
    """
    cfg = cfg if cfg is not None else AdvConfig()
    arm = cfg.grad_arm if arm is None else arm
    coeff = float(cfg.b_cap) if coeff is None else float(coeff)
    kappa = float(cfg.kappa) if kappa is None else float(kappa)
    norm = cfg.grad_norm if norm is None else norm
    lazy_k = int(cfg.grad_lazy) if lazy_k is None else int(lazy_k)
    target_anneal = cfg.target_anneal if target_anneal is None else target_anneal
    steps = int(cfg.steps) if total_steps is None else int(total_steps)
    return GradRegularizer(
        arm=arm,
        coeff=coeff,
        kappa=kappa,
        lazy_k=lazy_k,
        norm=norm,
        target_anneal=target_anneal,
        total_steps=steps if target_anneal != "none" else 0,
    )


def _l2_norm(g: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 input-gradient norm, ParticleGAN convention (eps 1e-12)."""
    return torch.sqrt(g.flatten(1).pow(2).sum(dim=1) + 1e-12)


def cap_penalty(
    grad_real: torch.Tensor,
    grad_fake: torch.Tensor,
    *,
    coeff: float = 1.0,
    kappa: float = 1.0,
) -> torch.Tensor:
    """One-sided b_cap on input-gradient norms of D.

    ``0.5 * coeff * (mean(relu(‖∇D(x_r)‖−κ)²) + mean(relu(‖∇D(x_f)‖−κ)²))``
    with ``‖g‖ = √(Σgᵢ² + 1e-12)``. Thin shim: the phi is ParticleGAN's
    ``GradRegularizer._phi`` for arm ``b_cap``, not a local reimplementation.
    Matches ``GradRegularizer(arm="b_cap", coeff, kappa)`` with
    ``lazy_k=1`` and ``target_anneal="none"``; annealed or lazy configs must
    go through ``make_grad_regularizer(...).penalty(D, x_real, x_fake)``.
    Prefer ``make_grad_regularizer(...).penalty(D, x_real, x_fake)`` on train
    paths so the autograd norm stays inside the ParticleGAN module too.
    """
    reg = GradRegularizer(arm="b_cap", coeff=coeff, kappa=kappa)
    center = reg.center(0)
    phi_r = reg._phi(_l2_norm(grad_real), center)
    phi_f = reg._phi(_l2_norm(grad_fake), center)
    return (float(coeff) / 2.0) * (phi_r.mean() + phi_f.mean())


def input_grad(critic: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """∇_x D(x) with graph kept so the cap can train D."""
    needed = not x.requires_grad
    if needed:
        x = x.detach().requires_grad_(True)
    score = critic(x)
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).sum(dim=1)
    (grad,) = torch.autograd.grad(score.sum(), x, create_graph=True)
    return grad


def feature_match_loss(
    feat_real: torch.Tensor,
    feat_fake: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Mean-feature L2. Normalize so D cannot dodge b_cap by inflating features."""
    r = feat_real.flatten(1).mean(dim=0)
    f = feat_fake.flatten(1).mean(dim=0)
    if normalize:
        r = F.normalize(r, dim=0)
        f = F.normalize(f, dim=0)
    return (r - f).pow(2).mean()


def vicreg_loss(
    z: torch.Tensor,
    *,
    sim_weight: float = 10.0,
    var_weight: float = 10.0,
    cov_weight: float = 1.0,
    std_target: float = 0.05,
    noise: float = 0.01,
) -> torch.Tensor:
    """VICReg on a particle batch. ``std_target`` is 0.05 on 2-D fixtures (poles are O(1)).

    Fail-closed on fewer than 2 rows: variance and covariance are undefined
    on one particle, and silently returning 0 would disable the spread
    pressure (e.g. ``train_gaussians(n_particles=1)`` collapsing modes with
    no error). Pass at least 2 particles.
    """
    if z.ndim != 2:
        raise ValueError(f"vicreg_loss needs a 2-D particle batch, got ndim={z.ndim}")
    if z.shape[0] < 2:
        raise ValueError(
            f"VICReg needs ≥ 2 particles, got {z.shape[0]}: a single particle "
            "has no variance/covariance, so the regularizer would silently be 0"
        )
    z2 = z + float(noise) * torch.randn_like(z)
    inv = F.mse_loss(z, z2)
    std = z.std(dim=0, unbiased=False)
    var = F.relu(float(std_target) - std).pow(2).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / float(z.shape[0] - 1)
    off = cov.pow(2).sum() - cov.diag().pow(2).sum()
    cov_term = off / float(z.shape[1])
    return (
        float(sim_weight) * inv
        + float(var_weight) * var
        + float(cov_weight) * cov_term
    )


def delayed_cosine(
    step: int,
    *,
    total: int,
    delay: int,
    min_ratio: float = 0.05,
) -> float:
    """1.0 for ``delay`` steps, then cosine down to ``min_ratio``."""
    if step < int(delay):
        return 1.0
    span = max(1, int(total) - int(delay))
    t = min(1.0, float(step - int(delay)) / float(span))
    return float(min_ratio) + 0.5 * (1.0 - float(min_ratio)) * (1.0 + math.cos(math.pi * t))


class ParticlePrior(nn.Module):
    """Learnable particles. Sample = particle + isotropic jitter."""

    def __init__(self, n_particles: int, dim: int, init_std: float = 0.05):
        super().__init__()
        if int(n_particles) < 1:
            raise ValueError(
                f"n_particles must be ≥ 1, got {n_particles!r}: an empty prior "
                "crashes sampling and a missing prior silently drops VICReg"
            )
        self.particles = nn.Parameter(float(init_std) * torch.randn(int(n_particles), int(dim)))

    def sample(self, n: int, *, jitter: float = 0.01) -> torch.Tensor:
        idx = torch.randint(0, self.particles.shape[0], (int(n),), device=self.particles.device)
        out = self.particles[idx]
        if float(jitter) > 0.0:
            out = out + float(jitter) * torch.randn_like(out)
        return out


class Fourier2MLP(nn.Module):
    """Fourier-2 critic: order-2 Fourier features + a 2-hidden-layer MLP.

    Per input dim: ``sin x, cos x, sin 2x, cos 2x``, plus a frozen random
    Fourier bank (Tancik-style, scale 2) so higher-D sheet / exam fields
    still get high-frequency features. Last layer is a scalar logit — no
    sigmoid; RpGAN uses ``softplus`` on the difference.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_rand: int = 16,
        hidden: int = 64,
        fourier_scale: float = 2.0,
        seed: int = 0,
    ):
        super().__init__()
        dim = int(dim)
        gen = torch.Generator().manual_seed(int(seed) + 17)
        bank = float(fourier_scale) * torch.randn(int(n_rand), dim, generator=gen)
        self.register_buffer("bank", bank)
        feat = 4 * dim + 2 * int(n_rand)
        self.net = nn.Sequential(
            nn.Linear(feat, int(hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden), int(hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden), 1),
        )
        self.feat = nn.Sequential(*list(self.net.children())[:-1])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        order1 = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        order2 = torch.cat([torch.sin(2.0 * x), torch.cos(2.0 * x)], dim=-1)
        proj = x @ self.bank.T
        rand = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return torch.cat([order1, order2, rand], dim=-1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(x)).squeeze(-1)


class EMA:
    """Exponential moving average of a parameter list."""

    def __init__(self, params: list[torch.Tensor], decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in params]

    def update(self, params: list[torch.Tensor]) -> None:
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, params: list[torch.Tensor]) -> None:
        for s, p in zip(self.shadow, params):
            p.data.copy_(s)


@dataclass
class AdvConfig:
    """Hyperparameters for the 2-D / sheet / exam adversarial fit."""

    steps: int = 1200
    lr: float = 5.0e-3
    beta1: float = 0.0
    beta2: float = 0.99
    d_lr_mult: float = 1.0
    prior_lr_mult: float = 1.0
    b_cap: float = 1.0
    kappa: float = 1.0
    grad_arm: str = "b_cap"
    grad_norm: str = "l2"
    grad_lazy: int = 1
    target_anneal: str = "none"
    n_particles: int = 12
    batch: int = 32
    cloud_std: float = 0.03
    particle_jitter: float = 0.01
    span_frac: float = 0.40
    end_margin: float = 0.60
    fm_weight: float = 0.0
    fm_normalize: bool = True
    vicreg_weight: float = 0.05
    particle_l2: float = 0.02
    cover_weight: float = 1.5
    d_steps: int = 1
    ema: float = 0.995
    delay: int = 80
    min_lr_ratio: float = 0.05
    critic_hidden: int = 64
    critic_n_rand: int = 16
    seed: int = 0


def toy_lr_triplet(cfg: "AdvConfig") -> tuple[float, float, float]:
    """Per-party LRs from the shared base: ``(g_lr, d_lr, prior_lr)``.

    ``d_lr = lr * d_lr_mult`` (ParticleGAN/Music ``D_LR_MULT = 1.5``),
    ``prior_lr = lr * prior_lr_mult``. Defaults are 1.0, which reproduces
    the locked shared-LR toy exactly. Fail-closed on non-positive or
    non-finite multipliers: a silent 0x LR would freeze a party with no
    error, and a negative LR would ascend.
    """
    base = float(cfg.lr)
    d_mult = float(cfg.d_lr_mult)
    p_mult = float(cfg.prior_lr_mult)
    for name, mult in (("d_lr_mult", d_mult), ("prior_lr_mult", p_mult)):
        if not (0.0 < mult < float("inf")):
            raise ValueError(f"{name} must be positive and finite, got {mult!r}")
    return base, base * d_mult, base * p_mult


def sample_real_cloud(
    poles: torch.Tensor,
    neus: torch.Tensor,
    *,
    n: int,
    cloud_std: float,
    span_frac: float,
    end_margin: float,
) -> torch.Tensor:
    """2-D analogue of lyric-span + last-token pooling.

    ``end_margin`` of the batch sits on the pole (last-token). The rest
    lerp ``neu → pole`` with mass toward the end (span). Tiny isotropic
    noise so D sees a mode, not a Dirac.
    """
    k = poles.shape[0]
    idx = torch.randint(0, k, (int(n),), device=poles.device)
    pole = poles[idx]
    neu = neus[idx]
    n_end = int(round(float(end_margin) * int(n)))
    n_span = int(n) - n_end
    chunks = []
    if n_end:
        chunks.append(pole[:n_end])
    if n_span:
        # Beta(3,1) analogue: u = rand^0.5 puts more mass near the pole.
        u = torch.rand(n_span, 1, device=poles.device).sqrt()
        lo = 1.0 - float(span_frac)
        u = lo + (1.0 - lo) * u
        chunks.append(neu[n_end:] + u * (pole[n_end:] - neu[n_end:]))
    out = torch.cat(chunks, dim=0) if chunks else pole
    if float(cloud_std) > 0.0:
        out = out + float(cloud_std) * torch.randn_like(out)
    return out
