"""Training-only adaptive prompt-row mining.

``ParticleBatch`` retains its historical API name, but it is a row sampler,
not ParticleGAN's movable latent prior. Its logits weight existing prompt
rows using detached discriminator losses. They never enter the generator;
the miners seek weak rows while the generator learns to handle those rows.
In the reference ParticleGAN, latent particles instead pass through G and
minimize the same adversarial objective as G.

A uniform mixture keeps every prompt represented in the generator update.
``balance_loss`` penalizes omitted rows in the sampled probability mixture.
VICReg on logits cannot provide that guarantee: adding a common bias to one
column preserves its variance/covariance while collapsing its softmax.
The separate VICReg helper remains available for actual latent vectors.

Miners are discarded after training; inference remains pure LoRA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLikeLoss(nn.Module):
    """Port of ParticleGAN's VICRegLikeLoss (variance hinge + covariance).

    Spread prior over a particle matrix: hinge each dim's std up to
    ``target_std`` (collapse guard, expansion free) plus off-diagonal
    covariance penalty (decorrelation). Allows arbitrary topology —
    clusters, gaps — unlike a Gaussian-shape prior.
    """

    def __init__(self, target_std: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.target_std = float(target_std)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or min(z.shape) == 0:
            raise ValueError("VICReg expects a nonempty (particles, dimensions) matrix")
        n = z.shape[0]
        # Preserve the reference sample variance for N > 1. A one-particle
        # cloud has no spread, so use its finite zero population variance.
        std_z = torch.sqrt(z.var(dim=0, correction=1 if n > 1 else 0) + self.eps)
        std_loss = torch.mean(F.relu(self.target_std - std_z))
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(n - 1, 1)
        d = z.shape[1]
        if d > 1 and n > 1:
            off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
            cov_loss = off_diag.pow(2).sum() / d
        else:
            cov_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        return std_loss + cov_loss


class ParticleBatch(nn.Module):
    """Learnable row-mining particles: table (M, R) of row-affinity logits.

    Sampling is pure indexing; only sampled table entries receive game or
    balance gradients. ``uniform_mix`` reserves a fraction of each update
    for all prompt rows, regardless of the learned query distribution.
    """

    def __init__(
        self,
        num_particles: int,
        num_rows: int,
        init_std: float = 1.0,
        uniform_mix: float = 0.1,
    ) -> None:
        super().__init__()
        if num_particles <= 0:
            raise ValueError(f"num_particles must be positive, got {num_particles}")
        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")
        if not 0.0 <= float(uniform_mix) <= 1.0:
            raise ValueError("uniform_mix must be finite and between 0 and 1")
        self.uniform_mix = float(uniform_mix)
        self.logits = nn.Parameter(torch.empty(num_particles, num_rows))
        with torch.no_grad():
            self.logits.normal_(mean=0.0, std=init_std)

    @property
    def num_particles(self) -> int:
        return self.logits.shape[0]

    @torch.no_grad()
    def sample_indices(
        self,
        count: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        return torch.randint(
            0, self.num_particles, (count,), device=self.logits.device, generator=generator
        )

    def _log_weights(self, idx: torch.Tensor, temp: float) -> torch.Tensor:
        if not math.isfinite(float(temp)) or temp <= 0.0:
            raise ValueError(f"temp must be positive and finite, got {temp}")
        if idx.ndim != 1 or idx.numel() == 0:
            raise ValueError("idx must be a nonempty vector of particle indices")
        logp = F.log_softmax(self.logits[idx] / float(temp), dim=-1)
        return torch.logsumexp(logp, dim=0) - math.log(idx.numel())

    def weights(
        self,
        idx: torch.Tensor,
        temp: float = 1.0,
        uniform_mix: float | None = None,
    ) -> torch.Tensor:
        """Row distribution with a uniform floor (sums to 1).

        Every row receives at least ``uniform_mix / num_rows``. Pass 0 for
        the historical unconstrained mean-softmax distribution.
        """
        mix = self.uniform_mix if uniform_mix is None else float(uniform_mix)
        if not 0.0 <= mix <= 1.0:
            raise ValueError("uniform_mix must be finite and between 0 and 1")
        mined = self._log_weights(idx, temp).exp()
        return (1.0 - mix) * mined + mix / self.logits.shape[1]

    def balance_loss(self, idx: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """KL(uniform || mined row distribution), before the uniform floor.

        Computed in log space: a shared row-logit shift still receives a
        restoring gradient even when its softmax probabilities underflow.
        This regularizes row coverage directly, unlike VICReg on logits.
        Only the sampled table entries receive gradients.
        """
        logw = self._log_weights(idx, temp)
        return -logw.mean() - math.log(self.logits.shape[1])

    def diagnostics(self, w: torch.Tensor) -> dict:
        ent = float(-(w * (w + 1e-12).log()).sum())
        return {"w_entropy": ent, "w_max": float(w.max())}
