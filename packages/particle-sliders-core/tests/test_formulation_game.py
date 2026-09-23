"""CPU smoke for FormulationGame against winning_formulation()."""
from __future__ import annotations

import torch

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
