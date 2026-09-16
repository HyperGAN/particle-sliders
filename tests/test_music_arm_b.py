"""Strategy C: Music Arm B single source + fail-closed trainer gate.

CPU only. No model, no Hub, no GPU train. These tests fail if anyone changes
an Arm B default in either ``music_arm_b_defaults`` or
``train_lm_slider_music3.parse_args`` without updating the other.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from analysis.slider2d.adv import AdvConfig, cap_penalty
from conceptmod.textsliders import music_arm_b_defaults as arm_b
from conceptmod.textsliders.music_arm_b_defaults import (
    FIELD_COVER_WEIGHT,
    MUSIC_ARM_B_ADV_ARCH,
    MUSIC_ARM_B_B_CAP,
    MUSIC_ARM_B_COVER_WEIGHT,
    MUSIC_ARM_B_FM_WEIGHT,
    MUSIC_ARM_B_GRAD_NORM,
    MUSIC_ARM_B_KAPPA,
    MUSIC_ARM_B_PARTS,
    MUSIC_ARM_B_POLE_WEIGHT,
    MUSIC_ARM_B_TEACHER,
    MUSIC_ARM_B_VICREG_WEIGHT,
    assert_music_arm_b_argv,
    check_music_arm_b_argv,
    music_b_cap_penalty,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


def _arm_b_argv(**overrides: object) -> list[str]:
    base = ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_guard_e", "--music_arm_b"]
    for key, value in overrides.items():
        base += [f"--{key}", str(value)]
    return base


def test_single_source_shape_is_arm_b():
    assert arm_b.MUSIC_ARM_B_GRAD_ARM == "b_cap"
    assert arm_b.MUSIC_ARM_B_B_CAP == pytest.approx(1.0)
    assert arm_b.MUSIC_ARM_B_KAPPA == pytest.approx(1.0)
    assert arm_b.MUSIC_ARM_B_GRAD_NORM == "l2"
    assert arm_b.MUSIC_ARM_B_FM_WEIGHT == pytest.approx(0.0)
    assert arm_b.MUSIC_ARM_B_TEACHER == "faithful_guard_e"
    assert arm_b.MUSIC_ARM_B_ADV_ARCH == "mlp"
    assert arm_b.MUSIC_ARM_B_PARTS == 0
    assert arm_b.MUSIC_ARM_B_COVER_WEIGHT == pytest.approx(1.0)
    assert arm_b.MUSIC_ARM_B_POLE_WEIGHT == pytest.approx(1.0)
    assert arm_b.MUSIC_ARM_B_VICREG_WEIGHT == pytest.approx(0.0)
    # The Field3D demo pin is deliberately NOT the Music value.
    assert FIELD_COVER_WEIGHT == pytest.approx(1.5)
    assert MUSIC_ARM_B_COVER_WEIGHT != pytest.approx(FIELD_COVER_WEIGHT)


def test_trainer_adv_defaults_read_from_single_source_only():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.adv_arch == MUSIC_ARM_B_ADV_ARCH
    assert args.b_cap == pytest.approx(MUSIC_ARM_B_B_CAP)
    assert args.adv_reg_kappa == pytest.approx(MUSIC_ARM_B_KAPPA)
    assert args.adv_reg_norm == MUSIC_ARM_B_GRAD_NORM
    assert args.fm_weight == pytest.approx(MUSIC_ARM_B_FM_WEIGHT)
    assert args.cover_weight == pytest.approx(MUSIC_ARM_B_COVER_WEIGHT)
    assert args.pole_weight == pytest.approx(MUSIC_ARM_B_POLE_WEIGHT)
    assert args.parts == MUSIC_ARM_B_PARTS
    assert args.vicreg_weight == pytest.approx(MUSIC_ARM_B_VICREG_WEIGHT)
    # Opt-in switch stays off by default: legacy recipes (v9) are unchanged.
    assert args.music_arm_b is False
    assert args.lm_target == "v9"
    # Defaults must reference the module, not hardcoded literals.
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    for const in (
        "MUSIC_ARM_B_ADV_ARCH",
        "MUSIC_ARM_B_B_CAP",
        "MUSIC_ARM_B_KAPPA",
        "MUSIC_ARM_B_GRAD_NORM",
        "MUSIC_ARM_B_FM_WEIGHT",
        "MUSIC_ARM_B_COVER_WEIGHT",
        "MUSIC_ARM_B_POLE_WEIGHT",
        "MUSIC_ARM_B_PARTS",
        "MUSIC_ARM_B_VICREG_WEIGHT",
    ):
        assert const in src, f"trainer default must read {const} from single source"
    assert "from conceptmod.textsliders.music_arm_b_defaults import" in src
    assert "assert_music_arm_b_argv" in src


def test_arm_b_argv_passes_on_locked_defaults():
    args = parse_args(_arm_b_argv())
    assert check_music_arm_b_argv(args) == []
    assert_music_arm_b_argv(args)  # must not raise
    assert args.lm_target == MUSIC_ARM_B_TEACHER


def test_arm_b_gate_rejects_each_single_knob_drift():
    drifts: dict[str, object] = {
        "lm_target": "v9",
        "adv_arch": "tx",
        "b_cap": 2.0,
        "adv_reg_kappa": 0.5,
        "adv_reg_norm": "linf",
        "fm_weight": 0.5,
        "cover_weight": 1.5,  # Field3D demo value must fail for Music
        "pole_weight": 0.5,
        "parts": 4,
        "vicreg_weight": 0.05,
    }
    for key, value in drifts.items():
        args = parse_args(_arm_b_argv(**{key: value}))
        bad = check_music_arm_b_argv(args)
        assert bad, f"--{key} {value} should drift"
        with pytest.raises(AssertionError, match="Music Arm B argv drifted"):
            assert_music_arm_b_argv(args)


def test_arm_b_gate_forbids_tx_with_faithful_guard_e():
    args = parse_args(_arm_b_argv(adv_arch="tx"))
    with pytest.raises(AssertionError, match="dual-arm incompatible"):
        assert_music_arm_b_argv(args)


def test_arm_b_gate_is_fail_closed_on_missing_keys():
    with pytest.raises(AssertionError, match="missing"):
        assert_music_arm_b_argv({})
    bad = check_music_arm_b_argv({"lm_target": "faithful_guard_e"})
    assert any("missing" in row for row in bad)


def test_arm_b_gate_accepts_plain_mapping():
    assert check_music_arm_b_argv(arm_b.music_arm_b_dict()) == []
    drifted = dict(arm_b.music_arm_b_dict(), b_cap=3.0)
    assert check_music_arm_b_argv(drifted) != []


def test_b_cap_penalty_wires_to_particle_gan_faithful_path():
    gen = torch.Generator().manual_seed(0)
    grad_real = torch.randn(4, 3, generator=gen)
    grad_fake = torch.randn(4, 3, generator=gen)
    got = music_b_cap_penalty(grad_real, grad_fake)
    want = cap_penalty(grad_real, grad_fake, coeff=1.0)
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    # Below-kappa gradients are free; above-kappa are quadratic (shared path).
    low = torch.ones(4, 2) * 0.4
    assert float(music_b_cap_penalty(low, low)) == pytest.approx(0.0)
    high = torch.ones(4, 2)
    assert float(music_b_cap_penalty(high, high)) == pytest.approx(
        (2.0**0.5 - 1.0) ** 2, rel=1e-5
    )
    # No thinned reimplementation: the wrapper delegates to cap_penalty.
    src = inspect.getsource(music_b_cap_penalty)
    assert "cap_penalty" in src
    assert "return cap_penalty(" in src
    assert "F.relu" not in src and ".pow(2)" not in src


def test_adv_config_translation_keeps_music_cover_not_demo_cover():
    kwargs = arm_b.music_arm_b_adv_kwargs()
    assert kwargs["b_cap"] == pytest.approx(1.0)
    assert kwargs["fm_weight"] == pytest.approx(0.0)
    assert kwargs["cover_weight"] == pytest.approx(1.0)
    cfg = AdvConfig(**{k: v for k, v in kwargs.items() if k in AdvConfig.__dataclass_fields__})
    assert arm_b.check_adv_config_arm_b_shape(cfg) == []
    demo = AdvConfig(cover_weight=FIELD_COVER_WEIGHT)
    assert arm_b.check_adv_config_arm_b_shape(demo) != []


def test_train_entry_calls_fail_closed_gate_before_spend():
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    gate = src.index("assert_music_arm_b_argv(args)")
    assert 'getattr(args, "music_arm_b"' in src[max(0, gate - 600) : gate]
    assert "cuda" in src[gate : gate + 2000]
