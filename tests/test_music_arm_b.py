"""Music Arm B recipe adapter + smoke — interface-first, no weights.

Torch-free on purpose: the recipe module and the smoke must validate the
Arm B argv shape without loading Music weights (or torch at all), so this
file runs in CI environments without the minimax-music3 env.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

from conceptmod.textsliders.music_arm_b import (
    apply_arm_b_defaults,
    get_music_arm_b_recipe,
    honored_trainer_overrides,
    to_argv,
    to_dict,
    validate_music_arm_b_argv,
)


def test_recipe_matches_handoff_table():
    d = to_dict()
    assert d["adv_loss"] == "rpgan_logistic"
    assert d["grad_reg"] == "b_cap"
    assert d["grad_coeff"] == 1.0
    assert d["adv_reg_kappa"] == 1.0
    assert d["grad_norm"] == "l2"
    assert d["b_cap"] == 1.0
    assert d["fm_weight"] == 0.0
    assert d["lm_target"] == "faithful_guard_e"
    assert d["adv_arch"] == "mlp"
    # Music transfer prefers 1.0 (Field3D demo uses 1.5).
    assert d["cover_weight"] == 1.0
    assert d["pole_weight"] == 1.0
    assert d["parts"] == 0
    assert d["vicreg_weight"] == 0.0
    assert d["eval_scales"] == [-1.0, 0.0, 0.5, 1.0]


def test_canonical_argv_validates_clean():
    assert validate_music_arm_b_argv(to_dict()) == []


def test_argv_lists_every_table_row():
    argv = " ".join(to_argv())
    for key in (
        "--adv_loss",
        "--grad_reg",
        "--b_cap",
        "--adv_reg_kappa",
        "--fm_weight",
        "--lm_target",
        "--adv_arch",
        "--cover_weight",
        "--pole_weight",
        "--parts",
        "--vicreg_weight",
        "--eval_scales",
    ):
        assert key in argv


def test_each_single_knob_drift_is_rejected():
    base = to_dict()
    drifts = {
        "fm_weight": 0.1,
        "lm_target": "faithful_raw",
        "adv_arch": "tx",
        "b_cap": 2.0,
        "adv_reg_kappa": 0.5,
        "grad_coeff": 2.0,
        "grad_reg": "r1",
        "adv_loss": "hinge",
        "cover_weight": 1.5,
        "pole_weight": 1.5,
        "parts": 4,
        "vicreg_weight": 0.05,
        "eval_scales": [-1.0, 1.0],
        "student_arm": "lowrank_k3",
    }
    for key, value in drifts.items():
        drifted = copy.deepcopy(base)
        drifted[key] = value
        assert validate_music_arm_b_argv(drifted), f"{key}={value} not rejected"


def test_honored_subset_is_trainer_only_subset():
    assert honored_trainer_overrides() == {"lm_target": "faithful_guard_e"}


def test_apply_arm_b_defaults_sets_teacher_and_rejects_conflict(capsys):
    # Bare --arm_b (trainer default v9, not explicit) is overridden.
    args = argparse.Namespace(lm_target="v9", arm_b=True)
    apply_arm_b_defaults(args, explicit_lm_target=False)
    assert args.lm_target == "faithful_guard_e"
    assert "faithful_guard_e" in capsys.readouterr().out

    # Explicit conflicting --lm_target fails closed.
    with pytest.raises(ValueError, match="drifted recipe"):
        apply_arm_b_defaults(
            argparse.Namespace(lm_target="v9", arm_b=True),
            explicit_lm_target=True,
        )

    # Matching explicit --lm_target is accepted.
    ok_args = argparse.Namespace(lm_target="faithful_guard_e", arm_b=True)
    apply_arm_b_defaults(ok_args, explicit_lm_target=True)
    assert ok_args.lm_target == "faithful_guard_e"


def test_smoke_script_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "smoke_music_arm_b_argv.py")],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SMOKE PASSED" in proc.stdout


def test_recipe_is_stable_single_source():
    assert to_dict(get_music_arm_b_recipe()) == to_dict(get_music_arm_b_recipe())
