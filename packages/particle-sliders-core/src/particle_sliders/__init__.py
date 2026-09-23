"""Shared particle-slider algorithms and the winning formulation stamp.

Model ids, Comfy class names, and model-specific train/infer surfaces belong
to product repositories. Products call ``FormulationGame`` for the stamp
feature-space step; they own feature projection, Hub, and Comfy. The game those products train is
``winning_formulation()``: gmix architecture, formulation parameters from
ParticleGAN #38 when that search crowns a full live-leaderboard winner.
GAN primitives (cap, RpGAN loss, particle VIC) come from the ``particlegan``
develop API; gmix architecture stays in this package.
"""
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

__version__ = "0.3.0"

__all__ = [
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
