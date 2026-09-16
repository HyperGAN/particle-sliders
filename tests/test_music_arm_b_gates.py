"""Music Arm B fail-closed shape gates on top of #101 (CPU-only, no GPU train).

Port of the unique value from Strategy E (#100) onto the Strategy B (#101)
foundation:

- Formulation (ParticleGAN-faithful ``GradRegularizer`` b_cap, kappa explicit,
  ``--adv_preset arm_b``) is owned by #101 and covered by
  ``tests/test_music_arm_b.py`` — this file does not reimplement it.
- This file ports E's strongest layer: fail-closed argv-shape gates over
  literal expected values, plus the ``--require_arm_b`` parse-time refusal
  and the ``--vicreg_weight`` 0-at-parts-0 gate, neither of which #101 had.

Arm B = #94 ParticleGAN-faithful RpGAN + ``b_cap``, leftover-gated::

    lm_target=faithful_guard_e · adv_arch=mlp · adv_norm=l2 · fm_weight=0
    adv_b_cap=1 · adv_reg_kappa=1 · parts=0
    pole_weight=1 · cover_weight=1 · vicreg_weight=0

Live-default posture (deliberate): the trainer keeps ``--lm_target v9`` /
``--pole_mode hidden`` as its default — Arm B is one explicit flag away
(``--lm_target faithful_guard_e`` + ``--adv_preset arm_b`` or
``--require_arm_b``) and fail-closed: anything else is refused at parse
time instead of silently training a drifted recipe.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.adv import AdvConfig, cap_penalty
from analysis.slider2d.gan import DEFAULT_TEACHER
from conceptmod.textsliders.train_lm_slider_music3 import (
    ARM_B,
    make_music_grad_regularizer,
    parse_args,
)


# Literal Arm B shape. Gates read from here so drift in the trainer defaults
# or the ARM_B row fails loudly. Flag names follow #101 (``--adv_b_cap``,
# not E's ``--b_cap``; plus ``--adv_norm`` and ``--vicreg_weight``).
WANT = {
    "lm_target": "faithful_guard_e",
    "adv_arch": "mlp",
    "adv_norm": "l2",
    "fm_weight": 0.0,
    "parts": 0,
    "pole_weight": 1.0,
    "cover_weight": 1.0,
    "adv_reg_kappa": 1.0,
    "adv_b_cap": 1.0,
    "vicreg_weight": 0.0,
}

BASE_ARGV = ["--prompts_file", "x.yaml"]
ARM_B_PRESET_ARGV = [
    "--prompts_file", "x.yaml",
    "--lm_target", "faithful_guard_e",
    "--adv_preset", "arm_b",
]
ARM_B_REQUIRE_ARGV = [
    "--prompts_file", "x.yaml",
    "--lm_target", "faithful_guard_e",
    "--require_arm_b",
]


# -- single source --------------------------------------------------------


def test_arm_b_row_matches_literals():
    for key, want in WANT.items():
        assert key in ARM_B, f"ARM_B missing {key}"
        assert ARM_B[key] == pytest.approx(want), f"ARM_B[{key}] drifted"


# -- trainer defaults -----------------------------------------------------


def test_adv_knob_defaults_are_arm_b():
    """Every adv knob defaults to the Arm B value (only the teacher is explicit)."""
    args = parse_args(BASE_ARGV)
    for key in (
        "adv_arch",
        "adv_norm",
        "fm_weight",
        "adv_b_cap",
        "adv_reg_kappa",
        "parts",
        "pole_weight",
        "cover_weight",
        "vicreg_weight",
    ):
        assert getattr(args, key) == pytest.approx(WANT[key]), f"--{key} default drifted"


def test_live_default_is_still_v9_not_silent_arm_b():
    """The live default stays v9/hidden/none; Arm B must be asked for, never silent."""
    args = parse_args(BASE_ARGV)
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.adv_preset == "none"
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--require_arm_b"])


# -- explicit Arm B passes -------------------------------------------------


def test_explicit_preset_argv_passes():
    args = parse_args(ARM_B_PRESET_ARGV)
    for key, want in WANT.items():
        got = getattr(args, key)
        if isinstance(want, float):
            assert float(got) == pytest.approx(want), key
        else:
            assert got == want, key
    reg = make_music_grad_regularizer(args)
    assert reg.arm == "b_cap"
    assert reg.coeff == pytest.approx(1.0)
    assert reg.kappa == pytest.approx(1.0)
    assert reg.norm == "l2"


def test_require_arm_b_passes_on_full_row_without_preset():
    args = parse_args(ARM_B_REQUIRE_ARGV)
    assert args.adv_preset == "none"
    reg = make_music_grad_regularizer(args)
    assert (reg.arm, reg.norm) == ("b_cap", "l2")


def test_require_arm_b_defaults_off():
    assert parse_args(BASE_ARGV).require_arm_b is False


def _argv_tokens(argv: list[str]) -> set[str]:
    """Exact `--key=value` tokens for the Arm B row (no substring matching)."""
    args = parse_args(argv)
    return {f"--{key}={getattr(args, key)}" for key in WANT}


def test_handoff_verify_tokens_present_on_arm_b_argv():
    got = _argv_tokens(ARM_B_PRESET_ARGV)
    want = {f"--{key}={value}" for key, value in WANT.items()}
    assert got == want, f"adv argv drifted: {sorted(got ^ want)}"
    # Prefix-extension drifts the old substring check blessed: fm 0.0->0.1,
    # cover 1.0->1.5 and vicreg 0.0->0.05 all still contain "fm_weight=0" /
    # "cover_weight=1" / "vicreg_weight=0" as substrings. Exact tokens catch
    # every one of them.
    for drifted in (
        {"fm_weight": 0.1},
        {"cover_weight": 1.5},
        {"vicreg_weight": 0.05},
        {"pole_weight": 0.5},
        {"adv_b_cap": 0.5},
        {"adv_reg_kappa": 2.0},
    ):
        key, bad = next(iter(drifted.items()))
        shown = " ".join(sorted(got))
        assert f"--{key}={WANT[key]}" in shown  # sanity: good row has the token
        assert f"--{key}={bad}" not in shown  # ... and not the drifted one
        assert (got - {f"--{key}={WANT[key]}"}) | {f"--{key}={bad}"} != want


# -- drift fails closed ----------------------------------------------------


def test_tx_plus_guard_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--lm_target", "faithful_guard_e", "--adv_arch", "tx"])
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--adv_arch", "tx"])


def test_fm_on_under_b_cap_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--fm_weight", "0.1"])
    with pytest.raises(SystemExit):
        parse_args(ARM_B_REQUIRE_ARGV + ["--fm_weight", "0.05"])


def test_b_cap_drift_is_rejected():
    for argv in (["--adv_b_cap", "0.5"], ["--adv_b_cap", "0"]):
        with pytest.raises(SystemExit):
            parse_args(ARM_B_PRESET_ARGV + argv)


def test_kappa_drift_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--adv_reg_kappa", "2"])


def test_parts_drift_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--parts", "4"])


def test_pole_cover_drift_is_rejected():
    # cover 1.5 is the Field3D demo pin, not the Music transfer (1.0).
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--cover_weight", "1.5"])
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--pole_weight", "0.5"])


def test_vicreg_on_with_parts0_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--vicreg_weight", "0.05"])
    with pytest.raises(SystemExit):
        parse_args(ARM_B_REQUIRE_ARGV + ["--vicreg_weight", "0.05"])


def test_adv_norm_drift_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(ARM_B_PRESET_ARGV + ["--adv_norm", "l1"])


def test_wrong_teacher_is_rejected_under_preset():
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--adv_preset", "arm_b"])


def test_require_arm_b_refuses_drift_before_gpu():
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--require_arm_b"])
    with pytest.raises(SystemExit):
        parse_args(ARM_B_REQUIRE_ARGV + ["--adv_arch", "tx"])


# -- analysis-side shape (shared with the demo) ----------------------------


def test_analysis_demo_agrees_with_arm_b_shape():
    assert DEFAULT_TEACHER == "faithful_guard_e"
    cfg = AdvConfig()
    assert cfg.b_cap == pytest.approx(1.0)
    assert cfg.fm_weight == pytest.approx(0.0)
    assert cfg.kappa == pytest.approx(1.0)


def test_b_cap_math_is_particlegan_shaped_through_grad_regularizer():
    """Faithful b_cap: (coeff/2)(E_r+E_f) relu(n-kappa)^2, n=sqrt(sum g^2+1e-12).

    ``cap_penalty`` routes its phi through ParticleGAN's
    ``GradRegularizer._phi`` (#101) — zero below kappa, quadratic above,
    linear in coeff, one-sided. This pins the shape so it cannot silently
    change back to a thinned reimplementation.
    """
    import torch

    low = torch.ones(4, 2) * 0.4
    assert float(cap_penalty(low, low, coeff=1.0, kappa=1.0)) == pytest.approx(0.0)
    high = torch.ones(4, 2)
    want = (2.0**0.5 - 1.0) ** 2
    assert float(cap_penalty(high, high, coeff=1.0, kappa=1.0)) == pytest.approx(
        want, rel=1e-5
    )
    assert float(cap_penalty(high, high, coeff=2.0, kappa=1.0)) == pytest.approx(
        2.0 * want, rel=1e-5
    )
    # One-sided: real and fake sides contribute independently.
    mixed = float(cap_penalty(high, low, coeff=1.0, kappa=1.0))
    assert mixed == pytest.approx(want / 2.0, rel=1e-5)
    # kappa explicit, not hardcoded: kappa=0.2 also caps the 0.4 row.
    assert float(cap_penalty(low, low, coeff=1.0, kappa=0.2)) > 0.0
