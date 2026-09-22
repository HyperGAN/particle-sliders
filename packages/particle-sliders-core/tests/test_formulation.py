"""CPU smoke for the shared package and the winning-formulation stamp."""
import warnings

import pytest
import torch

from particle_sliders import (
    GradRegularizer,
    RoutedMLP,
    locked_shared_recipe,
    winning_formulation,
)


def test_package_exports_the_stamp_and_regularizer():
    stamp = winning_formulation()
    assert stamp.id == "particle-gmix-1600-v2"
    assert stamp.family == "anneal-routed-particle-error"
    assert stamp is winning_formulation()
    regularizer = stamp.regularizer()
    assert isinstance(regularizer, GradRegularizer)
    assert regularizer.arm == "b_cap"
    assert regularizer.lazy_k == 4


def test_require_locks_the_game_and_allows_model_surfaces():
    stamp = winning_formulation()
    declared = stamp.as_dict()
    declared["g_lr"] = 2e-5
    declared["adv_batch"] = 4
    stamp.require(declared)
    forked = stamp.as_dict()
    forked["aux_weights"] = dict(forked["aux_weights"], cover_weight=1.0)
    with pytest.raises(ValueError, match="drift"):
        stamp.require(forked)
    with pytest.raises(ValueError, match="incomplete"):
        stamp.require({"critic": "gmix"})
    with pytest.raises(ValueError, match="unknown"):
        stamp.require({**stamp.as_dict(), "local_mse": 1.0})


def test_builders_run_on_cpu():
    torch.manual_seed(7)
    stamp = winning_formulation()
    targets = torch.randn(4, 6)
    neutrals = torch.randn(4, 6)
    critic = stamp.critic(targets, neutrals=neutrals)
    scores = critic(torch.randn(3, 6))
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    bridge = stamp.bridge()
    assert isinstance(bridge, RoutedMLP)
    particles = torch.randn(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]))
    out = bridge(torch.randn(2, int(stamp.spec["adapter_rank"])), particles)
    assert out.shape == (2, int(stamp.spec["adapter_rank"]))
    assert torch.isfinite(out).all()
    sigma = stamp.noise_std_at(0, edit_rms=1.0)
    assert sigma > stamp.noise_std_at(int(stamp.spec["noise_decay_steps"]), edit_rms=1.0)
    d_loss, g_loss, vic = stamp.losses()
    real, fake = torch.ones(4), -torch.ones(4)
    assert torch.isfinite(d_loss(real, fake))
    assert torch.isfinite(g_loss(real, fake))
    assert torch.isfinite(vic(particles[: int(stamp.spec["particle_vic_batch"])]))


def test_locked_shared_stays_importable_and_is_not_the_winner():
    recipe = locked_shared_recipe()
    recipe.require_locked_shared()
    stamp = winning_formulation()
    assert stamp.spec["aux_weights"]["cover_weight"] == 0.0
    assert recipe.cover_weight == 1.0
    assert recipe.teacher == "faithful_guard_e"
    assert stamp.spec["vicreg_weight"] == 1.0
    assert recipe.vicreg_weight == 0.0


def test_deprecated_import_alias_matches_and_warns():
    import sys

    sys.modules.pop("concept_slider_core", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import concept_slider_core
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert concept_slider_core.winning_formulation() is winning_formulation()
    assert concept_slider_core.RoutedMLP is RoutedMLP
