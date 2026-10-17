"""Reject corrupt Adam histories through the packaged LTX full-resume path.

These CPU tests exercise the CLI restore seam and its live receiver. Fresh
process continuation remains covered by verify_ltx25_sourcefree.py.
"""
from copy import deepcopy

import pytest
import torch

from particle_sliders import ltx25_cli as cli


def options(path, *extra):
    return cli.parse_args([
        "--dummy", "--steps", "4", "--rank", "2", "--alpha", "2",
        "--te_last_n", "1", "--seed", "21", "--preset", "bispf",
        "--pg_source_profile", "alz-continuous-v1",
        "--penalty_input_domain", "continuous", "--no_sample",
        "--save_dir", str(path), *extra,
    ])


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            equal_tree(value, reference)
    else:
        assert actual == expected


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("optimizer,corruption,error", [
    (optimizer, corruption, error)
    for optimizer in ("opt_g", "opt_d")
    for corruption, error in (
        ("negative_second_moment", "second moment is negative"),
        ("fractional_clock", "optimizer step count"),
        ("missing_first_moment", "Adam history fields differ"),
    )
] + [
    # Every generator leaf is graph-connected before an admitted update, so
    # Adam cannot legitimately lack its history after two completed updates.
    ("opt_g", "absent_history", "optimizer|Adam"),
    ("opt_g", "empty_history", "optimizer|Adam"),
])
def test_cli_resume_rejects_corrupt_adam_before_live_mutation(
    optimizer, corruption, error, tmp_path, monkeypatch,
):
    from particlegan.bispf import _LAST_THERMAL_SCALE

    source = tmp_path / "source"
    cli.train(options(source, "--stop_after", "2"))
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = checkpoint["game"][optimizer]
    key = history["param_groups"][0]["params"][0]
    slots = history["state"][key]
    assert {"step", "exp_avg", "exp_avg_sq"} <= slots.keys()
    if corruption == "negative_second_moment":
        slots["exp_avg_sq"].fill_(-1.)
    elif corruption == "fractional_clock":
        slots["step"] = torch.tensor(.5)
    elif corruption == "absent_history":
        del history["state"][key]
    elif corruption == "empty_history":
        history["state"][key] = {}
    else:
        del slots["exp_avg"]
    # A rejected candidate must not publish its optimizer's device-level
    # critic thermal state, including through a temporary shadow optimizer.
    checkpoint["game"]["opt_d"]["regularizer"]["record"]["thermal_scale"] = .123
    corrupt = tmp_path / "corrupt.pt"
    torch.save(checkpoint, corrupt)
    snapshot = deepcopy(checkpoint)
    captured = {}
    create_session = cli.create_session

    def capture_receiver(*args, **kwargs):
        session = create_session(*args, **kwargs)
        game = session[2]
        assert isinstance(game.opt_g, torch.optim.Adam)
        assert isinstance(game.opt_d, torch.optim.Adam)
        # Critic optimizer construction can reset the device cache. Establish
        # the receiver's sentinel after construction, directly before restore.
        _LAST_THERMAL_SCALE[game.device] = .789
        captured.update(game=game, before=game.state_dict(),
                        thermal_before=dict(_LAST_THERMAL_SCALE),
                        parameter_ids=[id(p) for root in (game.network, game.critic)
                                       for p in root.parameters()])
        return session

    monkeypatch.setattr(cli, "create_session", capture_receiver)
    output = tmp_path / "must-not-exist"
    ambient_rng = torch.get_rng_state().clone()
    published_before = dict(_LAST_THERMAL_SCALE)
    try:
        with pytest.raises(ValueError, match=error):
            cli.train(options(output, "--resume", str(corrupt)))
        assert captured, "the CLI must reach and validate its full-checkpoint receiver"
        game = captured["game"]
        equal_tree(game.state_dict(), captured["before"])
        assert [id(p) for root in (game.network, game.critic)
                for p in root.parameters()] == captured["parameter_ids"]
        assert _LAST_THERMAL_SCALE == captured["thermal_before"]
        assert torch.equal(ambient_rng, torch.get_rng_state())
        assert not output.exists()
        equal_tree(torch.load(corrupt, map_location="cpu", weights_only=True), snapshot)
        assert not torch.cuda.is_initialized()
    finally:
        _LAST_THERMAL_SCALE.clear()
        _LAST_THERMAL_SCALE.update(published_before)
