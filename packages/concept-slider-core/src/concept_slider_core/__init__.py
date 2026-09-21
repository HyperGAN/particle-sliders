"""Shared routed-particle algorithms. Model integration belongs to consumers."""
from .reference import RoutedMLP, GlobalMixErrorCritic, particle_vic, rp_d_loss, rp_g_loss, noise_std
from .distillation import fit_routed_down

__version__ = "0.1.0"
