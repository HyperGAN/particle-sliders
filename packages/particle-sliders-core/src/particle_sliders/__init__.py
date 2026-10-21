"""Shared slider package; new image applications use ``particle_sliders.lifecycle``.

The root retains explicitly named historical engines for checkpoint/API
compatibility. Their objectives are not silently substituted for ConditionalPG09.
"""
from .bispf_game import BISPFGame
from .pg09 import ConditionalPG09
from .distillation import fit_routed_down
from .endpoint_game import EndpointGame, endpoint_terms, teacher_poles
from .formulation_game import FormulationGame, dummy_features, run_formulation_game
from .formulation import (
    WinningFormulation,
    gmix_architecture,
    gmix_recipe,
    locked_shared_recipe,
    particle_gmix_1600_v2,
    particlegan_get_recipe,
    particlegan_locked_shared,
    winning_formulation,
)
from .grad_regularizers import GradRegularizer, GradientPenalty
from .recipe import SliderRecipe, require_same_critic
from .reference import (
    GlobalMixErrorCritic,
    RoutedMLP,
    noise_std,
    particle_vic,
    rp_d_loss,
    rp_g_loss,
)

__version__ = "0.8.1"

__all__ = [
    "BISPFGame",
    "ConditionalPG09",
    "EndpointGame",
    "FormulationGame",
    "GlobalMixErrorCritic",
    "GradRegularizer",
    "GradientPenalty",
    "RoutedMLP",
    "SliderRecipe",
    "WinningFormulation",
    "endpoint_terms",
    "dummy_features",
    "fit_routed_down",
    "gmix_architecture",
    "gmix_recipe",
    "locked_shared_recipe",
    "noise_std",
    "particle_gmix_1600_v2",
    "particle_vic",
    "particlegan_get_recipe",
    "particlegan_locked_shared",
    "require_same_critic",
    "rp_d_loss",
    "rp_g_loss",
    "run_formulation_game",
    "teacher_poles",
    "winning_formulation",
]
