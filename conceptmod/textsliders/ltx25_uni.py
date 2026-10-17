"""Compatibility alias for the installed video API; no local implementation."""
import sys
from particle_sliders import ltx25_uni as _implementation
sys.modules[__name__] = _implementation
