"""Thin wrapper over ParticleGAN develop gradient penalties.

Products historically imported ``GradRegularizer`` from ``particle_sliders``.
That name remains a compatibility alias of ParticleGAN's ``GradientPenalty``
(``particlegan.grad_regularizers.GradRegularizer``). Prefer calling
``GradientPenalty``, ``get_recipe(...).make_gradient_penalty()``, or
``particlegan.make_b_cap()`` for new code.

This module no longer vendors a ParticleGAN excerpt. The implementation lives
in the ``particlegan`` dependency pinned in ``pyproject.toml``.
"""

from particlegan import GradientPenalty
from particlegan.grad_regularizers import GradRegularizer, finite_difference_norm

# ParticleGAN publishes GradientPenalty as an alias of GradRegularizer.
assert GradRegularizer is GradientPenalty

__all__ = [
    "GradRegularizer",
    "GradientPenalty",
    "finite_difference_norm",
]
