"""Shared particle-slider algorithms and the winning formulation stamp.

Model ids, Comfy class names, and model-specific train/infer surfaces belong
to product repositories. The game those products train is
``winning_formulation()``.
"""
from .distillation import fit_routed_down
from .endpoint_game import EndpointGame, endpoint_terms, teacher_poles
from .formulation import (
    WinningFormulation,
    locked_shared_recipe,
    winning_formulation,
)
from .grad_regularizers import GradRegularizer
from .recipe import SliderRecipe, require_same_critic
from .reference import (
    GlobalMixErrorCritic,
    RoutedMLP,
    noise_std,
    particle_vic,
    rp_d_loss,
    rp_g_loss,
)

__version__ = "0.3.0"

__all__ = [
    "EndpointGame",
    "GlobalMixErrorCritic",
    "GradRegularizer",
    "RoutedMLP",
    "SliderRecipe",
    "WinningFormulation",
    "endpoint_terms",
    "fit_routed_down",
    "locked_shared_recipe",
    "noise_std",
    "particle_vic",
    "require_same_critic",
    "rp_d_loss",
    "rp_g_loss",
    "teacher_poles",
    "winning_formulation",
]
