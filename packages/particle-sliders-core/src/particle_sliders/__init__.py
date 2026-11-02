"""Shared slider package; new image applications use ``particle_sliders.lifecycle``.

The root retains explicitly named historical engines for checkpoint/API
compatibility. Their objectives are not silently substituted for ConditionalPG09.
"""
from importlib import import_module

_EXPORTS = {
    "BISPFGame": "bispf_game",
    "ConditionalPG09": "pg09",
    "fit_routed_down": "distillation",
    "EndpointGame": "endpoint_game",
    "endpoint_terms": "endpoint_game",
    "teacher_poles": "endpoint_game",
    "FormulationGame": "formulation_game",
    "dummy_features": "formulation_game",
    "run_formulation_game": "formulation_game",
    "WinningFormulation": "formulation",
    "gmix_architecture": "formulation",
    "gmix_recipe": "formulation",
    "locked_shared_recipe": "formulation",
    "particle_gmix_1600_v2": "formulation",
    "particlegan_get_recipe": "formulation",
    "particlegan_locked_shared": "formulation",
    "winning_formulation": "formulation",
    "GradRegularizer": "grad_regularizers",
    "GradientPenalty": "grad_regularizers",
    "SliderRecipe": "recipe",
    "require_same_critic": "recipe",
    "GlobalMixErrorCritic": "reference",
    "RoutedMLP": "reference",
    "noise_std": "reference",
    "particle_vic": "reference",
    "rp_d_loss": "reference",
    "rp_g_loss": "reference",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__version__ = "0.12.0"

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
