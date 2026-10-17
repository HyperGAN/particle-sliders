"""Compatibility for historical fixed L2/autograd b_cap callers.

The kernels remain upstream. ParticleGAN 0.8 removed the old top-level
aliases and optional norm/method/anneal arguments. Nondefault retired modes
raise instead of silently changing their meaning.
"""
from particlegan.grad_regularizers import GradientPenalty as UpstreamGradientPenalty


class GradientPenalty(UpstreamGradientPenalty):
    def __init__(self, *args, norm="l2", method="autograd",
                 target_anneal="none", arm="b_cap", **kwargs):
        if (norm, method, target_anneal) != ("l2", "autograd", "none"):
            raise ValueError("ParticleGAN 0.8 compatibility supports L2/autograd/no anneal only")
        super().__init__(*args, arm=arm, **kwargs)


GradRegularizer = GradientPenalty

__all__ = ["GradRegularizer", "GradientPenalty"]
