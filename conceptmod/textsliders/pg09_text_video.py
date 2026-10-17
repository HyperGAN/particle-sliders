"""Compatibility alias for the installable :mod:`particle_sliders.pg09_text_video`.

Implementation and serialized class identity belong to particle-sliders-core.
"""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("particle_sliders.pg09_text_video")
