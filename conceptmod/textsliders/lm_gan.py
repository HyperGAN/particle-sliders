"""Memory-bounded batch feature matching for sequential LM forwards."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math


def delayed_cosine_scale(step: int, steps: int) -> float:
    """Reference GAN schedule: full LR for 60%, then cosine to 5%."""
    progress = step / max(steps - 1, 1)
    fraction = min(1.0, max(0.0, (progress - 0.6) / 0.4))
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * fraction))


def feature_mean_surrogate(
    features: torch.Tensor,
    fake_mean: torch.Tensor,
    real_mean: torch.Tensor,
) -> torch.Tensor:
    """Exact batch-mean FM value and gradient, one attached row at a time.

    The caller computes both means with the current, frozen D and current G,
    then accumulates this expression with the same row weights used in those
    means. Detached means avoid retaining every expensive LM graph. Unlike
    MSE(feature_row, real_mean), this has zero gradient at fake == real and
    does not penalize diversity across prompt rows.
    """
    fake_mean, real_mean = fake_mean.detach(), real_mean.detach()
    difference = fake_mean - real_mean
    linear = 2.0 * (features.float().mean(dim=0) * difference).mean()
    return linear - linear.detach() + F.mse_loss(fake_mean, real_mean)


def weighted_feature_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Feature expectation under the same normalized measure on both sides."""
    return (features.float() * weights.to(features).unsqueeze(-1)).sum(dim=0)
