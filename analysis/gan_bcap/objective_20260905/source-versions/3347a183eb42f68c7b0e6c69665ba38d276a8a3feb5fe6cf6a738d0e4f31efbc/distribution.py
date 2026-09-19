"""Fixed-coordinate proper distribution scores for the objective audit.

Research only. These functions do not change any existing training recipe.
Rows are independent draws of ONE conditional distribution. Never pool
different captions/positions and call that conditional matching.
"""
from __future__ import annotations

import math
import torch


def squared_distances(x, y):
    # Direct differences avoid cancellation near exact matches. For the toy
    # this is tiny; high-dimensional singleton targets use paired_energy.
    return (x[:, None] - y[None, :]).square().sum(-1)


def energy_distance(fake, real, *, smoothing=0.01, power=1., unbiased=True):
    """Smoothed energy distance; unbiased draws require >=2 on each side.

    rho(x,y)=sqrt(||x-y||²+epsilon²)-epsilon is of strong negative type
    on Euclidean space. Its V statistic is an empirical distance, whereas
    its U statistic is an unbiased population estimator that may be negative.
    """
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('Smoothing must be finite and positive')
    if not math.isfinite(power) or not 0<power<2:
        raise ValueError('Energy power must be strictly between zero and two')
    if fake.ndim != 2 or real.ndim != 2 or fake.shape[1] != real.shape[1]:
        raise ValueError('Expected matching vector widths')
    if min(len(fake), len(real)) < (2 if unbiased else 1):
        raise ValueError('Too few independent samples for this estimator')
    def distance(x, y):
        squared=squared_distances(x,y)+smoothing**2
        return squared.sqrt()-smoothing if power==1 else squared.pow(power/2)-smoothing**power
    ff, rr, fr = distance(fake, fake), distance(real, real), distance(fake, real)
    if unbiased:
        ff_mean = (ff.sum() - ff.diagonal().sum()) / (len(fake) * (len(fake) - 1))
        rr_mean = (rr.sum() - rr.diagonal().sum()) / (len(real) * (len(real) - 1))
    else:
        ff_mean, rr_mean = ff.mean(), rr.mean()
    return 2 * fr.mean() - ff_mean - rr_mean


def paired_energy(fake, real, *, scale=1., smoothing=0.1):
    """Conditional singleton energy distance in fixed per-coordinate units.

    Every last-axis vector has its OWN condition. At one deterministic
    teacher/student per condition, the within-distribution terms vanish.
    No repulsion is applied between unrelated captions or token positions.
    """
    if fake.shape != real.shape or fake.numel() == 0:
        raise ValueError('Paired tensors must be nonempty and have the same shape')
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('Smoothing must be finite and positive')
    scale = torch.as_tensor(scale, device=fake.device, dtype=fake.dtype)
    if scale.requires_grad or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Scale must be fixed, finite and positive')
    error = (fake - real.detach()) / scale
    squared = error.square().mean(-1)
    # Algebraic rationalization gives exact zero and avoids subtraction near
    # the teacher; the formula is 2*(sqrt(r²+epsilon²)-epsilon).
    return (2 * squared / ((squared + smoothing**2).sqrt() + smoothing)).mean()


def rbf_mmd(fake, real, *, bandwidths=(.03, .1, .3, 1., 3.), unbiased=True):
    """A fixed characteristic multi-scale kernel, with no learned witness."""
    if min(len(fake), len(real)) < (2 if unbiased else 1):
        raise ValueError('Too few samples')
    if not bandwidths or any(not math.isfinite(s) or s <= 0 for s in bandwidths):
        raise ValueError('Invalid kernel bandwidths')
    def kernel(x, y):
        r2 = squared_distances(x, y)
        return sum((-r2 / (2 * s*s)).exp() for s in bandwidths) / len(bandwidths)
    ff, rr, fr = kernel(fake, fake), kernel(real, real), kernel(fake, real)
    if unbiased:
        f = (ff.sum() - ff.diagonal().sum()) / (len(fake) * (len(fake)-1))
        r = (rr.sum() - rr.diagonal().sum()) / (len(real) * (len(real)-1))
    else:
        f, r = ff.mean(), rr.mean()
    return f + r - 2 * fr.mean()
