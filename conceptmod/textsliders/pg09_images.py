"""Compatibility alias for the installable :mod:`particle_sliders.pg09_images`.

All implementations live in particle-sliders-core. Alias the module itself so
legacy private imports, monkeypatches and serialized class lookups use the same
objects as package imports.
"""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("particle_sliders.pg09_images")
