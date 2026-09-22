"""Product winning-formulation stamp.

Import this module from anima-particle-sliders, krea2-particle-sliders,
supra-concept-sliders, and any later ``*-particle-sliders`` product.
Do not copy it into a product repo.

The stamp is allowed to change in HyperGAN/particle-sliders. A new winner
lands here (new ``STAMP_ID`` and knobs). Products pick it up by bumping the
git pin of ``particle-sliders-core`` and calling ``winning_formulation()``
again. Restating the loss, the critic, the noise schedule, or
``GradRegularizer`` in a product is a fork.

Today's stamp is the released routed-particle game ``particle-gmix-1600-v2``
(recipe ``anneal-routed-particle-error-yue2-v1``). The numbers are the Hub
golden record mirrored by ``analysis/slider2d/yue2_gmix_v2_exam.V2_SPEC``.
``tests/test_particle_sliders_formulation.py`` fails if the two diverge.
``propose_only`` on that exam arm only keeps the arm from flipping the Music
trainer default. It does not exempt products from this stamp.

``locked_shared`` (``SliderRecipe.require_locked_shared``, Music Arm B,
bipolar leaderboard teacher ``faithful_guard_e``) stays importable through
``locked_shared_recipe``. It is a different, older endpoint-MSE stamp.
It is not ``winning_formulation()``.
"""
from types import MappingProxyType

from .grad_regularizers import GradRegularizer
from .recipe import SliderRecipe
from .reference import (
    GlobalMixErrorCritic,
    RoutedMLP,
    noise_std,
    particle_vic,
    rp_d_loss,
    rp_g_loss,
)

STAMP_ID = "particle-gmix-1600-v2"
FAMILY = "anneal-routed-particle-error"
# The 2D exam records this arm as propose_only so it cannot retarget Music.
# Products still train the stamp below.
RESEARCH_EXAM_IS_PROPOSE_ONLY = True

# Golden knobs. Keep in lockstep with V2_SPEC (except propose_only).
_SPEC = {
    "recipe_name": "anneal-routed-particle-error-yue2-v1",
    "config_sha256": "1ef39a623505691b8710cd37cb768452cd79f6666d109614af297e9d270d88bb",
    "model_glue_reference": "df70ccb2ca8f532bdcc07a343fd12bec77362523",
    "generator_objective": "paired_error_rpgan_plus_particle_vic",
    "critic": "gmix",
    "critic_tokens": 8,
    "critic_width": 48,
    "critic_layers": 1,
    "critic_heads": 4,
    "critic_score_bound": 8.0,
    "g_lr": 0.0006,
    "d_lr": 0.0009,
    "particle_lr": 0.006,
    "betas": (0.0, 0.999),
    "schedule": "constant",
    "ema": 0.995,
    "parts": 128,
    "particle_dim": 4,
    "particle_vic_batch": 64,
    "particle_vic_target_std": 1.0,
    "particle_vic_eps": 1e-4,
    "vicreg_weight": 1.0,
    "adv_b_cap": 1.0,
    "adv_reg_kappa": 1.0,
    "penalty_lazy_k": 4,
    "penalty_method": "autograd",
    "penalty_anneal": "none",
    "cap_coordinates": "normalized_paired_error_plus_shared_gaussian",
    "target_normalization": "paired_edit_per_coordinate_std_median_rms_gain",
    "edit_rms_target": 1.0,
    "edit_noise_ratio": 0.28,
    "noise_start": "edit_rms/edit_noise_ratio",
    "noise_floor": 0.03,
    "noise_decay_steps": 1600,
    "noise_hold": "edit_rms*noise_hold_ratio",
    "noise_hold_ratio": 1.3,
    "adv_batch": 8,
    "sample_seeds": 128,
    "history_tokens": 32,
    "seedbank_sources": 512,
    "polarity": "unipolar",
    "lm_target": "faithful_plus_neu",
    "trained_scales": (1.0,),
    "recommended_range": (0.0, 1.0),
    "adapter_rank": 8,
    "adapter_alpha": 8.0,
    "adapter_width": 48,
    "router_width": 16,
    "adv_weight": 1.0,
    "aux_weights": {
        "anchor_weight": 0.0,
        "cover_weight": 0.0,
        "fm_weight": 0.0,
        "end_weight": 0.0,
        "lyrichold_weight": 0.0,
        "plan_weight": 0.0,
        "pole_weight": 0.0,
    },
}

# Game identity. A product that changes any of these has forked the stamp.
FORMULATION_KEYS = frozenset({
    "generator_objective",
    "critic",
    "critic_tokens",
    "critic_width",
    "critic_layers",
    "critic_heads",
    "critic_score_bound",
    "betas",
    "schedule",
    "ema",
    "parts",
    "particle_dim",
    "particle_vic_batch",
    "particle_vic_target_std",
    "particle_vic_eps",
    "vicreg_weight",
    "adv_b_cap",
    "adv_reg_kappa",
    "penalty_lazy_k",
    "penalty_method",
    "penalty_anneal",
    "cap_coordinates",
    "target_normalization",
    "edit_rms_target",
    "edit_noise_ratio",
    "noise_start",
    "noise_floor",
    "noise_decay_steps",
    "noise_hold",
    "noise_hold_ratio",
    "polarity",
    "trained_scales",
    "recommended_range",
    "adapter_rank",
    "adapter_alpha",
    "adapter_width",
    "router_width",
    "adv_weight",
    "aux_weights",
})

# Backbone step size, batch, and data-budget fields. Products may override
# these through require(); they may not re-declare the game to do it.
MODEL_SURFACE_KEYS = frozenset({
    "g_lr",
    "d_lr",
    "particle_lr",
    "adv_batch",
    "sample_seeds",
    "history_tokens",
    "seedbank_sources",
})

# Identity of the golden record. Omitted keys are fine. Present keys must match.
PROVENANCE_KEYS = frozenset({
    "recipe_name",
    "config_sha256",
    "model_glue_reference",
    "lm_target",
})


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple) and value and isinstance(value[0], (tuple, dict, MappingProxyType)):
        return tuple(_thaw(item) for item in value)
    return value


class WinningFormulation:
    """The formulation products train. Construct via ``winning_formulation()``."""

    def __init__(self):
        overlap = FORMULATION_KEYS & MODEL_SURFACE_KEYS | FORMULATION_KEYS & PROVENANCE_KEYS | MODEL_SURFACE_KEYS & PROVENANCE_KEYS
        if overlap:
            raise RuntimeError(f"stamp key classified twice: {sorted(overlap)}")
        covered = FORMULATION_KEYS | MODEL_SURFACE_KEYS | PROVENANCE_KEYS
        if covered != set(_SPEC):
            raise RuntimeError(
                f"stamp keys drifted from the classifier: missing={sorted(set(_SPEC) - covered)} "
                f"extra={sorted(covered - set(_SPEC))}"
            )
        self.id = STAMP_ID
        self.family = FAMILY
        self.spec = _freeze(_SPEC)
        self.formulation_keys = FORMULATION_KEYS
        self.model_surface_keys = MODEL_SURFACE_KEYS
        self.provenance_keys = PROVENANCE_KEYS

    def as_dict(self):
        """Plain copy of the golden record, including reference model-surface values."""
        return {key: _thaw(self.spec[key]) for key in self.spec}

    def require(self, actual):
        """Reject a product-local restatement that drifts from this stamp.

        Formulation keys are required and must match. Model-surface keys may
        differ. Provenance keys may be omitted; if present they must match.
        Any other key is a fork and raises.
        """
        if not isinstance(actual, dict):
            raise TypeError("winning formulation require() expects a dict of knobs")
        missing = sorted(self.formulation_keys - set(actual))
        if missing:
            raise ValueError(f"winning formulation {self.id} incomplete: {missing}")
        differences = {}
        for key in sorted(self.formulation_keys | (self.provenance_keys & set(actual))):
            if actual[key] != self.spec[key]:
                differences[key] = {"expected": _thaw(self.spec[key]), "actual": actual[key]}
        if differences:
            raise ValueError(
                f"winning formulation drift ({self.id}): {differences}. "
                "Change the stamp in HyperGAN/particle-sliders; do not fork it in a product repo."
            )
        unknown = sorted(set(actual) - self.formulation_keys - self.model_surface_keys - self.provenance_keys)
        if unknown:
            raise ValueError(f"unknown formulation knobs (possible fork): {unknown}")
        return self

    def regularizer(self):
        """ParticleGAN ``b_cap`` arm at this stamp's coefficient, kappa, and lazy interval."""
        return GradRegularizer(
            arm="b_cap",
            coeff=float(self.spec["adv_b_cap"]),
            kappa=float(self.spec["adv_reg_kappa"]),
            norm="l2",
            lazy_k=int(self.spec["penalty_lazy_k"]),
            method=str(self.spec["penalty_method"]),
            target_anneal=str(self.spec["penalty_anneal"]),
        )

    def bridge(self):
        """Routed particle MLP at the stamp's rank, width, and particle dimension."""
        rank = int(self.spec["adapter_rank"])
        return RoutedMLP(
            rank,
            rank,
            z_dim=int(self.spec["particle_dim"]),
            width=int(self.spec["adapter_width"]),
            router_width=int(self.spec["router_width"]),
        )

    def critic(self, targets, neutrals=None):
        """Global-mix critic at the stamp's token, width, depth, and score bound."""
        if self.spec["critic"] != "gmix":
            raise ValueError(f"stamp critic {self.spec['critic']!r} has no builder")
        return GlobalMixErrorCritic(
            targets,
            neutrals=neutrals,
            tokens=int(self.spec["critic_tokens"]),
            width=int(self.spec["critic_width"]),
            layers=int(self.spec["critic_layers"]),
            heads=int(self.spec["critic_heads"]),
            score_bound=float(self.spec["critic_score_bound"]),
        )

    def noise_std_at(self, step, edit_rms):
        """Shared geometric noise curve for this stamp's horizon and hold."""
        edit_rms = float(edit_rms)
        start = max(edit_rms / float(self.spec["edit_noise_ratio"]), float(self.spec["noise_floor"]))
        hold = edit_rms * float(self.spec["noise_hold_ratio"])
        return noise_std(step, start=start, decay_steps=self.spec["noise_decay_steps"], hold=hold)

    def losses(self):
        """Relativistic paired losses and the particle VIC term. Do not reimplement."""
        return rp_d_loss, rp_g_loss, particle_vic


_WINNING = WinningFormulation()


def winning_formulation():
    """Return the current product stamp. This object is the entry point to import."""
    return _WINNING


def locked_shared_recipe():
    """Bipolar endpoint stamp (Music Arm B / ``require_locked_shared``).

    Kept so products do not reimplement it. It is not the winning formulation:
    VIC, noise, and feature matching are off, and the objective is endpoint MSE
    with teacher ``faithful_guard_e``.
    """
    recipe = SliderRecipe()
    recipe.require_locked_shared()
    return recipe
