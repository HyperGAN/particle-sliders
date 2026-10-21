"""Historical constructor/import compatibility; all penalty math stays upstream.

Released ParticleGAN 0.9.1 retains fixed L2/autograd b_cap and a_r1r2. Retired
norms, arms, finite differences and annealing fail before touching a critic.
Use the frozen legacy source pin documented in LEGACY-MIGRATION.md for those.
"""
import math
from particlegan import grad_regularizers as _upstream

_LEGACY_PIN = "17cd484ef00601389256a30f6d52fc79991bcb3d"


class UnsupportedLegacyRegularizer(ValueError):
    """The selected upstream release has no equivalent historical kernel."""


def _unsupported(option):
    return UnsupportedLegacyRegularizer(
        f"ParticleGAN 0.9.1 does not implement legacy {option}; "
        f"use particle-sliders-core at {_LEGACY_PIN} in its separate legacy "
        "environment (LEGACY-MIGRATION.md). No substitute kernel was selected."
    )


def finite_difference_norm(critic, x, eps):
    """Retained import; delegate only if upstream actually provides the kernel."""
    implementation = getattr(_upstream, "finite_difference_norm", None)
    if implementation is None:
        raise _unsupported("finite_difference_norm")
    return implementation(critic, x, eps)


class GradientPenalty(_upstream.GradientPenalty):
    def __init__(self, arm="b_cap", coeff=1.0, kappa=1.0, lazy_k=1,
                 norm="l2", target_anneal="none", total_steps=0,
                 method="autograd", fd_eps=0.05, **kwargs):
        # Preserve the historical positional order and inactive constructor
        # options. Reject removed modes before allocating upstream step state.
        if norm != "l2" or method != "autograd" or target_anneal != "none":
            raise _unsupported(f"norm={norm!r}, method={method!r}, target_anneal={target_anneal!r}")
        if arm not in ("b_cap", "a_r1r2", "k3p"):
            raise _unsupported(f"arm={arm!r}")
        if not math.isfinite(fd_eps) or fd_eps <= 0:
            raise ValueError("fd_eps must be finite and positive")
        total_steps = int(total_steps)
        super().__init__(arm=arm, coeff=coeff, kappa=kappa, lazy_k=lazy_k, **kwargs)
        self.norm, self.method, self.target_anneal = norm, method, target_anneal
        self.total_steps, self.fd_eps = total_steps, float(fd_eps)

    def penalty(self, D, x_real, x_fake, step=1, generator=None,
                collect_stats=True, *, ema_critic=None):
        # Fixed autograd arms never used generator in the legacy implementation.
        return super().penalty(D, x_real, x_fake, step=step,
                               collect_stats=collect_stats, ema_critic=ema_critic)

    def __call__(self, D, x_real, x_fake, step=1, generator=None, *, ema_critic=None):
        return self.penalty(D, x_real, x_fake, step, generator,
                            collect_stats=False, ema_critic=ema_critic)[0]


GradRegularizer = GradientPenalty

__all__ = ["GradRegularizer", "GradientPenalty", "finite_difference_norm",
           "UnsupportedLegacyRegularizer"]
