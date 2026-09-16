"""Music Arm B CLI-defaults smoke test (Strategy A).

Asserts `train_lm_slider_music3.parse_args` default argv matches the Arm B
winning config (ParticleGAN-faithful RpGAN + b_cap, leftover-gated), and that
the Arm B teacher is available explicitly. CPU-only: parses argv, trains
nothing.
"""

from conceptmod.textsliders.train_lm_slider_music3 import parse_args


def _bare():
    return parse_args(["--prompts_file", "prompts.yaml"])


def test_arm_b_adv_defaults():
    args = _bare()
    assert args.adv_loss == "rpgan"
    assert args.adv_arch == "mlp"
    assert args.b_cap == 1.0
    assert args.adv_reg_kappa == 1.0
    assert args.adv_reg_norm == "l2"
    assert args.fm_weight == 0.0
    assert args.cover_weight == 1.0
    assert args.pole_weight == 1.0
    assert args.parts == 0
    assert args.vicreg_weight == 0.0


def test_arm_b_teacher_explicit_lm_target_default_stays_v9():
    # --lm_target default stays v9 for gender; Arm B Music runs pass
    # --lm_target faithful_guard_e explicitly (mlp critic, never tx+guard).
    assert _bare().lm_target == "v9"
    arm_b = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_guard_e"]
    )
    assert arm_b.lm_target == "faithful_guard_e"
    assert arm_b.adv_arch == "mlp"
    assert arm_b.fm_weight == 0.0


def test_arm_b_adv_overrides_still_parse():
    args = parse_args(
        [
            "--prompts_file",
            "prompts.yaml",
            "--adv_loss",
            "off",
            "--b_cap",
            "2.0",
            "--parts",
            "4",
        ]
    )
    assert args.adv_loss == "off"
    assert args.b_cap == 2.0
    assert args.parts == 4
