"""Compatibility alias for the packaged historical research loss."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("particle_sliders.textsliders._slider2d.adv")
