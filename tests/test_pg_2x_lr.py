"""Clone arm 3/5 (``pg_2x_lr``): propose-only ParticleGAN-champion 2xLR.

The arm doubles the toy's shared ``5e-3`` base to ``1e-2`` and applies the
PG-like critic ratio (``d_lr_mult=1.5``, the Music ``D_LR_MULT`` convention),
keeping ``b_cap``/``kappa``/Music extras locked and the prior ratio at 1.0
(HOLD — see ``analysis/slider2d/pg_2x_lr.py``).

CPU only. No Music GPU train. Does not change the live trainer default or
the Music ``ARM_B`` argv.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from analysis.slider2d import pg_2x_lr as arm
from analysis.slider2d.adv import AdvConfig, toy_lr_triplet
from analysis.slider2d.gan import default_cfg, score_field2d
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B, parse_args

SMOKE_STEPS = 250


def test_propose_only_marker():
    assert arm.PROPOSE_ONLY is True
    assert arm.ARM_NAME == "pg_2x_lr"


def test_locked_defaults_unchanged():
    cfg = AdvConfig()
    assert cfg.lr == pytest.approx(5.0e-3)
    assert cfg.d_lr_mult == pytest.approx(1.0)
    assert cfg.prior_lr_mult == pytest.approx(1.0)
    assert cfg.b_cap == pytest.approx(1.0)
    assert cfg.kappa == pytest.approx(1.0)
    assert cfg.fm_weight == pytest.approx(0.0)
    assert cfg.cover_weight == pytest.approx(1.5)
    assert cfg.beta1 == pytest.approx(0.0)
    assert cfg.delay == 80
    assert cfg.steps == 1200
    assert cfg.seed == 0
    g, d, p = toy_lr_triplet(cfg)
    assert (g, d, p) == pytest.approx((5.0e-3, 5.0e-3, 5.0e-3))


def test_arm_lr_values():
    cfg = arm.pg_2x_lr_cfg()
    assert cfg.lr == pytest.approx(1.0e-2)
    assert cfg.lr == pytest.approx(2.0 * arm.BASE_LR)
    assert cfg.d_lr_mult == pytest.approx(1.5)
    assert cfg.prior_lr_mult == pytest.approx(1.0)
    g, d, p = arm.arm_lr_triplet()
    assert g == pytest.approx(1.0e-2)
    assert d == pytest.approx(1.5e-2)
    assert p == pytest.approx(1.0e-2)


def test_arm_deltas_are_lr_only():
    """Single-variable clone: only lr + d_lr_mult differ from locked."""
    locked = AdvConfig()
    got = arm.pg_2x_lr_cfg()
    diff = {
        f.name
        for f in dataclasses.fields(AdvConfig)
        if getattr(locked, f.name) != getattr(got, f.name)
    }
    assert diff == {"lr", "d_lr_mult"}
    assert got.b_cap == pytest.approx(locked.b_cap)
    assert got.kappa == pytest.approx(locked.kappa)
    assert got.fm_weight == pytest.approx(locked.fm_weight)
    assert got.cover_weight == pytest.approx(locked.cover_weight)
    assert got.vicreg_weight == pytest.approx(locked.vicreg_weight)
    assert got.particle_l2 == pytest.approx(locked.particle_l2)
    assert got.prior_lr_mult == pytest.approx(1.0)


def test_arm_lr_overrides_are_refused():
    with pytest.raises(ValueError):
        arm.pg_2x_lr_cfg(lr=5.0e-3)
    with pytest.raises(ValueError):
        arm.pg_2x_lr_cfg(d_lr_mult=1.0)
    with pytest.raises(ValueError):
        arm.pg_2x_lr_cfg(prior_lr_mult=10.0)
    # Non-LR overrides (e.g. CPU-budget steps) are allowed.
    assert arm.pg_2x_lr_cfg(steps=SMOKE_STEPS).steps == SMOKE_STEPS


def test_lr_mults_are_fail_closed():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            toy_lr_triplet(default_cfg(d_lr_mult=bad))
        with pytest.raises(ValueError):
            toy_lr_triplet(default_cfg(prior_lr_mult=bad))


def test_music_argv_untouched():
    """No LR keys leak into the production Music ARM_B row or its defaults."""
    assert "lr" not in ARM_B
    assert "d_lr_mult" not in ARM_B
    assert "prior_lr_mult" not in ARM_B
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.adv_preset == "none"
    assert float(args.lr) == pytest.approx(5e-4)


def test_smoke_locked_vs_2x_field2d():
    """CPU smoke: both configs train to finite losses; report arm-vs-locked."""
    locked = arm.locked_cfg(steps=SMOKE_STEPS, seed=0)
    champ = arm.pg_2x_lr_cfg(steps=SMOKE_STEPS, seed=0)
    base = score_field2d(locked)
    trial = score_field2d(champ)
    for row, name in ((base, "locked"), (trial, "pg_2x_lr")):
        assert row["pass"] is not None
        for key in ("cos_slider_plus", "leak_ratio", "cos_plus_minus", "leak_frac"):
            assert torch.isfinite(torch.tensor(float(row[key]))), (name, key)
    print(
        f"\nlocked   slider={base['cos_slider_plus']:+.3f} "
        f"leak={base['leak_ratio']:+.3f} pm={base['cos_plus_minus']:+.3f} "
        f"lf={base['leak_frac']:+.3f} pass={base['pass']}"
    )
    print(
        f"pg_2x_lr slider={trial['cos_slider_plus']:+.3f} "
        f"leak={trial['leak_ratio']:+.3f} pm={trial['cos_plus_minus']:+.3f} "
        f"lf={trial['leak_frac']:+.3f} pass={trial['pass']}"
    )
