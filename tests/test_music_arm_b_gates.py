"""Strategy E: Music Arm B fail-closed gates (CPU-only, no GPU train).

Tests-first port of the Arm B winning shape from the Slider CPU / Field3D
selection handoff: Arm B = #94 ParticleGAN-faithful RpGAN + b_cap,
leftover-gated.

    fm_weight=0, b_cap=1, adv_reg_kappa=1, teacher faithful_guard_e,
    critic mlp (never tx + guard), parts=0, pole_weight=1, cover_weight=1.

The gates below are written against literal expected values (not against
the single-source dict) so drift in either the trainer defaults or the
validator fails. The single-source contract itself is locked by
``test_music_arm_b_single_source_matches_literals``.

Live-default posture (deliberate, minimum-change): the trainer keeps
``--lm_target v9`` / ``--pole_mode hidden`` as its default — that is
locked by ``test_live_default_is_still_v9`` and the analysis
"does not change the live trainer default" rule. Arm B is one explicit
flag away (``--lm_target faithful_guard_e``) and fail-closed: anything
else is reported by ``check_music_arm_b`` instead of silently training
a drifted recipe. ``--require_arm_b`` turns the report into a refusal
at parse time, before any GPU work.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.adv import AdvConfig, cap_penalty
from analysis.slider2d.gan import DEFAULT_TEACHER
from conceptmod.textsliders.music_arm_b import (
    MUSIC_ARM_B,
    assert_music_arm_b,
    check_music_arm_b,
    format_arm_b_argv,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


# Literal Arm B shape. Every gate below reads from here so a drift in the
# trainer defaults, the validator, or the single-source dict fails loudly.
WANT = {
    "lm_target": "faithful_guard_e",
    "adv_arch": "mlp",
    "fm_weight": 0.0,
    "b_cap": 1.0,
    "adv_reg_kappa": 1.0,
    "parts": 0,
    "pole_weight": 1.0,
    "cover_weight": 1.0,
    "vicreg_weight": 0.0,
}

ARM_B_ARGV = [
    "--prompts", "x.yaml",
    "--lm_target", "faithful_guard_e",
    "--adv_arch", "mlp",
    "--fm_weight", "0",
    "--b_cap", "1",
    "--adv_reg_kappa", "1",
    "--parts", "0",
    "--pole_weight", "1",
    "--cover_weight", "1",
    "--vicreg_weight", "0",
]


# -- single source --------------------------------------------------------


def test_music_arm_b_single_source_matches_literals():
    for key, want in WANT.items():
        assert key in MUSIC_ARM_B, f"MUSIC_ARM_B missing {key}"
        assert MUSIC_ARM_B[key] == pytest.approx(want), f"MUSIC_ARM_B[{key}] drifted"


# -- trainer defaults -----------------------------------------------------


def test_adv_knob_defaults_are_arm_b():
    """Every new adv knob defaults to the Arm B value (only the teacher is explicit)."""
    args = parse_args(["--prompts", "x.yaml"])
    for key in (
        "adv_arch",
        "fm_weight",
        "b_cap",
        "adv_reg_kappa",
        "parts",
        "pole_weight",
        "cover_weight",
        "vicreg_weight",
    ):
        assert getattr(args, key) == pytest.approx(WANT[key]), f"--{key} default drifted"


def test_live_default_is_still_v9_not_silent_arm_b():
    """The live default stays v9/hidden; Arm B must be asked for, never silent."""
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    bad = check_music_arm_b(args)
    assert bad, "default argv must NOT pass the Arm B gate (teacher is v9, not guard_e)"
    assert any("faithful_guard_e" in m for m in bad)


# -- explicit Arm B passes -------------------------------------------------


def test_explicit_arm_b_argv_passes():
    args = parse_args(ARM_B_ARGV)
    assert check_music_arm_b(args) == []
    assert_music_arm_b(args)  # must not raise


def test_arm_b_argv_printer_covers_every_handoff_row():
    shown = format_arm_b_argv(parse_args(ARM_B_ARGV))
    for token in (
        "faithful_guard_e",
        "mlp",
        "fm_weight=0",
        "b_cap=1",
        "kappa=1",
        "parts=0",
        "pole_weight=1",
        "cover_weight=1",
        "vicreg_weight=0",
    ):
        assert token in shown, f"adv argv printout missing {token!r}: {shown}"


# -- drift fails closed ----------------------------------------------------


def test_tx_plus_guard_is_rejected():
    args = parse_args(
        ARM_B_ARGV + ["--adv_arch", "tx"],
    )
    bad = check_music_arm_b(args)
    assert bad, "tx + faithful_guard_e must fail (dual-arm incompatible)"
    assert any("tx" in m and "faithful_guard_e" in m for m in bad)
    with pytest.raises(AssertionError):
        assert_music_arm_b(args)


def test_fm_on_under_b_cap_is_rejected():
    args = parse_args(["--prompts", "x.yaml", "--fm_weight", "0.1"])
    bad = check_music_arm_b(args)
    assert any("fm_weight" in m for m in bad)


def test_fm_on_flagged_even_on_arm_b_teacher():
    args = parse_args(ARM_B_ARGV + ["--fm_weight", "0.05"])
    bad = check_music_arm_b(args)
    assert bad == [m for m in bad if "fm_weight" in m] and bad


def test_b_cap_drift_is_rejected():
    for argv in (["--b_cap", "0.5"], ["--b_cap", "0"]):
        args = parse_args(["--prompts", "x.yaml"] + argv)
        assert any("b_cap" in m for m in check_music_arm_b(args)), argv


def test_kappa_drift_is_rejected():
    args = parse_args(["--prompts", "x.yaml", "--adv_reg_kappa", "2"])
    assert any("kappa" in m for m in check_music_arm_b(args))


def test_parts_drift_is_rejected():
    args = parse_args(["--prompts", "x.yaml", "--parts", "4"])
    assert any("parts" in m for m in check_music_arm_b(args))


def test_pole_cover_drift_is_rejected():
    # cover 1.5 is the Field3D demo pin, not the Music transfer (1.0).
    args = parse_args(["--prompts", "x.yaml", "--cover_weight", "1.5"])
    assert any("cover_weight" in m for m in check_music_arm_b(args))
    args = parse_args(["--prompts", "x.yaml", "--pole_weight", "0.5"])
    assert any("pole_weight" in m for m in check_music_arm_b(args))


def test_vicreg_on_with_parts0_is_rejected():
    args = parse_args(["--prompts", "x.yaml", "--vicreg_weight", "0.05"])
    assert any("vicreg" in m for m in check_music_arm_b(args))


def test_wrong_teacher_is_rejected_by_name():
    args = parse_args(["--prompts", "x.yaml", "--lm_target", "v9"])
    bad = check_music_arm_b(args)
    assert any("lm_target" in m for m in bad)


# -- parse-time refusal ----------------------------------------------------


def test_require_arm_b_refuses_drift_before_gpu():
    with pytest.raises(SystemExit):
        parse_args(["--prompts", "x.yaml", "--require_arm_b"])
    with pytest.raises(SystemExit):
        parse_args(ARM_B_ARGV + ["--adv_arch", "tx", "--require_arm_b"])
    ok = parse_args(ARM_B_ARGV + ["--require_arm_b"])
    assert check_music_arm_b(ok) == []


def test_require_arm_b_defaults_off():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.require_arm_b is False


# -- analysis-side shape (shared with the demo) ----------------------------


def test_analysis_demo_agrees_with_arm_b_shape():
    assert DEFAULT_TEACHER == "faithful_guard_e"
    cfg = AdvConfig()
    assert cfg.b_cap == pytest.approx(1.0)
    assert cfg.fm_weight == pytest.approx(0.0)


def test_b_cap_math_is_particlegan_shaped():
    """Present-form b_cap: (coeff/2)(E_r+E_f) relu(||grad||-1)^2.

    Zero below 1, quadratic above, linear in coeff. The full
    ParticleGAN GradRegularizer (explicit kappa / eps / norms) is not
    ported in this checkout — that stays a ranked residual diff, and
    this gate pins the formula that IS present so it cannot silently
    change shape.
    """
    import torch

    low = torch.ones(4, 2) * 0.4
    assert float(cap_penalty(low, low, coeff=1.0)) == pytest.approx(0.0)
    high = torch.ones(4, 2)
    want = (2.0**0.5 - 1.0) ** 2
    assert float(cap_penalty(high, high, coeff=1.0)) == pytest.approx(want, rel=1e-5)
    assert float(cap_penalty(high, high, coeff=2.0)) == pytest.approx(2.0 * want, rel=1e-5)
    # One-sided: real and fake sides contribute independently.
    mixed = float(cap_penalty(high, low, coeff=1.0))
    assert mixed == pytest.approx(want / 2.0, rel=1e-5)
