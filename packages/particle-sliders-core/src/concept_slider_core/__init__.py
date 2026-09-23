"""Deprecated import name. Use :mod:`particle_sliders`.

``concept_slider_core`` is the path this package used when it was extracted
from Anima as ``concept-slider-core``. The algorithms now live in
``particle_sliders``. This module re-exports that public API so existing
imports keep working, and warns on import.
"""
import warnings

warnings.warn(
    "concept_slider_core is a deprecated alias of particle_sliders. "
    "Install particle-sliders-core from HyperGAN/particle-sliders and "
    "import particle_sliders. Products train particle_sliders.winning_formulation().",
    DeprecationWarning,
    stacklevel=2,
)

from particle_sliders import *  # noqa: F403
from particle_sliders import __all__ as __all__
from particle_sliders import __version__ as __version__
