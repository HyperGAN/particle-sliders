"""CLONE ARM 1/5 vs locked_shared: `pg_big_particles` (propose_only).

Only delta: n_particles 12 -> 64 (feasible_max 256). Everything else is
`locked_cfg`. Production argv and live trainer defaults untouched.
"""

from __future__ import annotations

import pytest

from analysis.gan_bcap.gaussian_repro import train_gaussians
from analysis.slider2d import locked_baseline_defaults as locked
from analysis.slider2d.adv import AdvConfig
from analysis.slider2d.gan import DEFAULT_TEACHER, default_cfg, score_field2d
from analysis.slider2d.run_lm_adv import build_parser
from conceptmod.textsliders.train_lm_slider_music3 import parse_args

ARM = "pg_big_particles"


def test_locked_matches_advconfig_defaults():
    cfg = AdvConfig()
    for key, value in locked.LOCKED.items():
        assert getattr(cfg, key) == value, key
    assert locked.LOCKED_TEACHER == DEFAULT_TEACHER == "faithful_guard_e"


def test_locked_matches_default_cfg_factory():
    cfg = default_cfg()
    for key, value in locked.LOCKED.items():
        assert getattr(cfg, key) == value, key


def test_production_argv_untouched():
    got = vars(build_parser().parse_args([]))
    got.pop("out")
    assert got == locked.PRODUCTION_ARGV_DEFAULTS


def test_live_music_default_still_v9():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_locked_cfg_rejects_formulation_overrides():
    with pytest.raises(ValueError):
        locked.locked_cfg(n_particles=64)
    with pytest.raises(ValueError):
        locked.locked_cfg(cover_weight=0.0)
    assert locked.locked_cfg(steps=200).steps == 200
    assert locked.locked_cfg(seed=3).seed == 3


def test_arm_is_only_delta_vs_locked():
    cfg = locked.arm_cfg(ARM)
    assert cfg.n_particles == 64
    locked.assert_only_delta(cfg, ARM)
    with pytest.raises(ValueError):
        locked.arm_cfg("no_such_arm")
    with pytest.raises(ValueError):
        locked.arm_cfg(ARM, n_particles=256)


def test_gaussian_smoke_arm_vs_locked():
    """Same 8-mode smoke, locked n vs arm n: both cover, both keep D flat."""
    base = train_gaussians(n_modes=8, steps=1200, seed=1234, b_cap=1.0)
    arm = train_gaussians(n_modes=8, steps=1200, seed=1234, b_cap=1.0, n_particles=64)
    for row in (base, arm):
        assert row["modes"] >= 6
        assert row["hq"] >= 0.70
        assert row["grad_peak_med"] <= 2.0
    assert arm["modes"] >= base["modes"] - 2  # honesty: no regression, no win claimed


def test_field2d_gates_arm_vs_locked():
    """Cheap polarity comparison: locked n=12 vs arm n=64/256 at 200 steps."""
    ref = score_field2d(locked.locked_cfg(steps=200, seed=0))
    arm = locked.arm_cfg(ARM, steps=200, seed=0)
    locked.assert_only_delta(arm, ARM, budget={"steps": 200, "seed": 0})
    mid = score_field2d(arm)
    big = score_field2d(default_cfg(steps=200, seed=0, n_particles=256))
    for row in (ref, mid, big):
        assert row["pass"]
        assert row["cos_slider_plus"] >= 0.90
        assert abs(row["leak_ratio"]) <= 0.20
        assert row["cos_plus_minus"] <= -0.85
        assert row["leak_frac"] <= -0.85
