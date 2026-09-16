"""UniPG-B unipolar GAN toy exam: gates, GAN-only, determinism, Music untouched.

CPU only. Keeps runtimes small (<=600 steps, single cells/seeds) so the
suite stays fast; the full budget ladder lives in the scoreboard doc.
"""

from __future__ import annotations

import torch

from analysis.slider2d import yue2_gan_exam as exam
from analysis.slider2d.exam import close_field, divergent_field


def test_gate_contract_and_eval_scales():
    assert exam.COVER_MIN == 0.85
    assert exam.LEAK_MAX == 0.05
    assert exam.NEU_HOLD_MIN == 0.85
    assert list(exam.EVAL_SCALES) == [0.0, 0.5, 1.0]
    assert tuple(exam.REQUIRED_CELLS) == ("divergent", "close")
    assert exam.PROPOSE_ONLY is True
    assert exam.MERGE_TO_TRAINER is False


def test_module_is_gan_only_no_supervised_mse():
    src = __import__("pathlib").Path(exam.__file__).read_text()
    for token in (
        "MSELoss",
        "mse_loss",
        "lm_plus_loss",
        "lm_plus_neu_loss",
        "lm_slider_loss",
        "feature_match_loss",
    ):
        assert token not in src, token
    # The AdvConfig mirror must pin the supervised extras to 0.0 (never
    # wired into G: fit_uni_gan reads no cover/FM field at all).
    fit_src = src.split("def fit_uni_gan", 1)[1].split("def score_uni_gan", 1)[0]
    assert "cover_weight" not in fit_src
    assert "fm_weight" not in fit_src
    mirror = src.split("def adv_config", 1)[1].split("def assert_gan_only", 1)[0]
    assert "cover_weight=0.0" in mirror and "fm_weight=0.0" in mirror
    assert "rp_g_loss" in src and "rp_d_loss" in src
    assert "make_grad_regularizer" in src


def test_champion_regularizer_pinned_and_gan_only_gate():
    cfg = exam.UniGanConfig()
    exam.assert_gan_only(cfg)
    assert (cfg.grad_arm, cfg.b_cap, cfg.kappa, cfg.grad_norm) == ("b_cap", 1.0, 1.0, "l2")
    assert cfg.grad_lazy == 1 and cfg.target_anneal == "none"


def test_arm_registry_is_propose_only_with_identities():
    assert len(exam.ARMS) >= 7
    propose = [n for n, a in exam.ARMS.items() if a["status"] == "propose_only"]
    assert len(propose) >= 6
    assert "baseline_learned12" in exam.ARMS
    for name, arm in exam.ARMS.items():
        assert arm["identity"] and arm["match"] in ("MATCH", "PARTIAL", "DRIFT")
    cfg = exam.arm_cfg("unipg_b_frozen64", steps=600, seed=0)
    assert cfg.freeze_prior is True and cfg.n_particles == 64
    assert cfg.vicreg_weight == 0.0 and cfg.particle_l2 == 0.0
    faithful = exam.arm_cfg("unipg_b_center_vicregfaithful1", steps=600, seed=0)
    assert faithful.vicreg_form == "faithful" and faithful.vicreg_weight == 1.0
    assert faithful.center_particles is True


def test_scale_zero_holds_by_construction():
    field = divergent_field(seed=0)
    cfg = exam.arm_cfg("unipg_b_frozen32", steps=50, seed=0)
    residual = exam.fit_uni_gan(field, cfg=cfg)
    assert float(residual.delta(0.0).norm()) == 0.0
    row = exam.score_uni_gan(field, residual)
    assert row["neu_hold"] >= exam.NEU_HOLD_MIN
    assert row["canary"]["scored"] is False
    assert row["antipodal_consulted"] is False
    assert row["eval_scales"] == [0.0, 0.5, 1.0]


def test_fit_is_deterministic_at_fixed_seed():
    field = divergent_field(seed=0)
    rows = [
        exam.score_uni_gan(
            field, exam.fit_uni_gan(field, cfg=exam.arm_cfg("unipg_b_frozen32", steps=120, seed=0))
        )
        for _ in range(2)
    ]
    assert rows[0]["cover"] == rows[1]["cover"]
    assert rows[0]["leak"] == rows[1]["leak"]


def test_baseline_drift_fails_divergent_cover_at_600():
    field = divergent_field(seed=0)
    row = exam.score_uni_gan(
        field,
        exam.fit_uni_gan(field, cfg=exam.arm_cfg("baseline_learned12", steps=600, seed=0)),
        name="baseline_learned12",
    )
    assert row["hit"] is False
    assert row["cover"] < exam.COVER_MIN


def test_frozen_arm_passes_close_at_600():
    field = close_field(seed=0)
    row = exam.score_uni_gan(
        field,
        exam.fit_uni_gan(field, cfg=exam.arm_cfg("unipg_b_frozen64", steps=600, seed=0)),
        name="unipg_b_frozen64",
    )
    assert row["cover"] >= exam.COVER_MIN
    assert row["leak"] <= exam.LEAK_MAX
    assert row["neu_hold"] >= exam.NEU_HOLD_MIN
    assert row["hit"] is True


def test_music_bipolar_and_live_defaults_untouched():
    from conceptmod.textsliders.train_lm_slider_music3 import ARM_B
    from conceptmod.textsliders import yue2_arm_b as arm_b

    assert ARM_B["lm_target"] == "faithful_guard_e"
    assert ARM_B["adv_b_cap"] == 1.0 and ARM_B["adv_reg_kappa"] == 1.0
    assert arm_b.RECIPE["generator_objective"] == "positive_rpgan_only"
    assert arm_b.RECIPE["trained_scales"] == [1.0]
    assert arm_b.RECIPE["recommended_range"] == [0.0, 1.0]
