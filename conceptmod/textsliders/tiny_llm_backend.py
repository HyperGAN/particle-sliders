"""Compatibility alias for installed particle_sliders.tiny_llm_backend."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("particle_sliders.tiny_llm_backend")
