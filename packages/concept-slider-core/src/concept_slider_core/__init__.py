"""Shared routed-particle algorithms. Model integration belongs to consumers."""
from .reference import RoutedMLP, GlobalMixErrorCritic, particle_vic, rp_d_loss, rp_g_loss, noise_std
from .distillation import fit_routed_down
from .recipe import SliderRecipe, require_same_critic
from .endpoint_game import EndpointGame, teacher_poles, endpoint_terms

__version__ = "0.2.0"
