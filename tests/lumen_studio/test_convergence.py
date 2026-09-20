import copy

import pytest
import torch

from lumen_studio.convergence import movement, objective_rows, paired_improvement, response_step
from lumen_studio.game import Game
from lumen_studio.particles import ParticleAdapter


def fixture(tiny):
    adapter = ParticleAdapter(tiny.transformer)
    tiny.mixer.add("candlelit", adapter)
    normalization = {"scale": torch.ones(10, 16), "edit_rms": torch.ones(10)}
    game = Game(adapter, normalization, 20, seed=29)
    rows = [tiny.noise(i, 512, 512) for i in range(20)]
    embed = tiny.encode("neutral")
    with torch.no_grad():
        positive = [tiny.predict(row, 1000-(i % 10)*100, tiny.encode("positive")) for i, row in enumerate(rows)]

    def predict(ids):
        return torch.cat([(tiny.predict(rows[i], 1000-(i % 10)*100, embed)-positive[i]).flatten(1) for i in ids])

    return adapter, game, predict


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("step", [3, 4])
@pytest.mark.parametrize("side", ["d", "g"])
def test_response_matches_production_phase_and_freezes_opponent(tiny, step, side):
    adapter, game, predict = fixture(tiny)
    source = copy.deepcopy(dict(adapter=adapter.state_dict(), game=game.state_dict()))
    frozen_base = copy.deepcopy(tiny.transformer.state_dict())
    original_step = game.d.step
    if side == "g":
        # Production G phase with the D optimizer frozen is the G-only reference.
        game.d.step = lambda *a, **k: None
    with tiny.mixer.scales({"candlelit": 1.}):
        game.update(predict, list(range(10))*2, step)
    game.d.step = original_step
    expected = copy.deepcopy(game.critic.state_dict() if side == "d" else adapter.state_dict())
    expected_optimizer = copy.deepcopy((game.d if side == "d" else game.g).state_dict())
    adapter.load_state_dict(source["adapter"])
    game.load_state_dict(source["game"])
    opponent = adapter if side == "d" else game.critic
    frozen_opponent = copy.deepcopy(opponent.state_dict())
    with tiny.mixer.scales({"candlelit": 1.}):
        result = response_step(game, predict, list(range(10))*2, step, side)
    assert equal_state(expected, game.critic.state_dict() if side == "d" else adapter.state_dict())
    assert equal_state(expected_optimizer, (game.d if side == "d" else game.g).state_dict())
    assert equal_state(frozen_opponent, opponent.state_dict())
    assert equal_state(frozen_base, tiny.transformer.state_dict())
    assert result["gradient_norm"] > 0


def test_function_movement_detects_rotation_with_unchanged_magnitude():
    before, after = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])
    records = [dict(position=0, row=dict(definition="a", character="b", shared=dict(framing="portrait")))]
    result = movement(before, after, before, torch.ones_like(before), records)
    assert result["previous_edit_rms"] == result["current_edit_rms"]
    assert result["relative_to_previous_edit"] == pytest.approx(2**.5)
    assert result["direction_cosine"] == 0
    unchanged = movement(before, before, before, torch.ones_like(before), records)
    assert unchanged["change_rms"] == 0
    zero = movement(torch.zeros_like(before), after, before, torch.ones_like(before), records)
    assert zero["relative_to_previous_edit"] is None


def test_bootstrap_clusters_correlated_timesteps():
    a = paired_improvement([3., 5.], [2., 3.], ["a", "b"])
    b = paired_improvement([3.]*20+[5.]*20, [2.]*20+[3.]*20, ["a"]*20+["b"]*20)
    assert a["improvement"] == b["improvement"] == 1.5
    assert a["ci95"] == b["ci95"]
    assert b["prompt_rows"] == 2


def test_fixed_objective_is_repeatable_and_does_not_update_models(tiny):
    adapter, game, predict = fixture(tiny)
    state = copy.deepcopy(game.state_dict())
    weights = copy.deepcopy(adapter.state_dict())
    with tiny.mixer.scales({"candlelit": 1.}), torch.no_grad():
        residuals = predict([0, 1, 2]).detach()
    first = objective_rows(game, residuals, [0, 1, 2])
    second = objective_rows(game, residuals, [0, 1, 2])
    assert first == second
    assert equal_state(state, game.state_dict())
    assert equal_state(weights, adapter.state_dict())
    assert all(a+b == t for a, b, t in zip(first["d_adversarial"], first["d_cap"], first["d_total"]))
