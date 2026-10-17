"""CPU smoke for FormulationGame against winning_formulation()."""
from __future__ import annotations

import torch
import pytest

from particle_sliders import FormulationGame, dummy_features, run_formulation_game, winning_formulation


def test_formulation_game_dummy_steps_finite():
    stamp = winning_formulation()
    declared = {**stamp.as_dict(), "g_lr": 2e-5, "adv_batch": 4}
    stamp.require(declared)
    game = FormulationGame(stamp, declared, seed=7, device="cpu")
    batch = int(declared["adv_batch"])
    history = run_formulation_game(game, steps=3, features_for_step=dummy_features(game.rank, batch, 7))
    assert len(history) == 3
    assert all(torch.isfinite(torch.tensor(row["g_loss"])) for row in history)
    assert history[-1]["step"] == 3.0


def test_frozen_critic_parameters_and_phase_flags_survive_success_and_failure():
    stamp = winning_formulation()
    declared = stamp.as_dict()
    data = torch.Generator().manual_seed(17)
    targets = torch.randn(6, stamp.spec["adapter_rank"], generator=data)
    critic = stamp.critic(targets)
    frozen = critic.mix.weight
    frozen.requires_grad_(False)
    game = FormulationGame(stamp, declared, seed=7, critic=critic)
    before = frozen.detach().clone()
    flags = [p.requires_grad for p in critic.parameters()]
    assert not any(p is frozen for group in game.opt_d.param_groups for p in group["params"])
    features = dummy_features(game.rank, 4, 7)(1)
    game.step(1, features)
    assert torch.equal(before, frozen)
    assert [p.requires_grad for p in critic.parameters()] == flags

    def fail_g_loss(*args):
        raise RuntimeError("injected generator failure")

    game.g_loss = fail_g_loss
    with pytest.raises(RuntimeError, match="injected generator failure"):
        game.step(2, features)
    assert torch.equal(before, frozen)
    assert [p.requires_grad for p in critic.parameters()] == flags
