"""Unipolar GAN-only toy exam (UniPG-A): GradRegularizer sweep, propose-only.

CPU-only. Pins that the exam is GAN-only (no supervised G loss), unipolar
raw-positive (no minus branch, no leak-axis teacher), scale-0 exact zero,
deterministic per seed, and propose-only (AdvConfig / ARM_B / live
defaults untouched). Keeps the sweep honest: >=6 arms, single-knob
GradRegularizer deltas only.
"""

from __future__ import annotations

import torch

from analysis.slider2d import yue2_gan_exam as exam
from analysis.slider2d.adv import AdvConfig, make_grad_regularizer
from analysis.slider2d.exam import divergent_field
from analysis.slider2d import locked_baseline_defaults as locked
from analysis.slider2d.run_lm_adv import build_parser
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B


def test_propose_only_markers():
    assert exam.PROPOSE_ONLY is True
    assert exam.MERGE_TO_TRAINER is False


def test_arm_registry_is_gradreg_only_and_sized():
    assert len(exam.UNIPG_A_ARMS) >= 6
    for name, arm in exam.UNIPG_A_ARMS.items():
        assert set(arm["delta"]) <= exam.UNIPG_A_KNOBS - {"steps", "seed"}, name
        assert arm["blurb"], name


def test_uni_gan_cfg_is_gan_only_and_locked_elsewhere():
    cfg = exam.uni_gan_cfg()
    assert cfg.cover_weight == 0.0
    assert cfg.fm_weight == 0.0
    base = AdvConfig()
    for key in ("lr", "beta1", "beta2", "n_particles", "batch", "vicreg_weight",
                "particle_l2", "critic_hidden", "critic_n_rand", "d_steps",
                "ema", "delay", "min_lr_ratio", "d_lr_mult", "prior_lr_mult"):
        assert getattr(cfg, key) == getattr(base, key), key


def test_uni_gan_cfg_refuses_non_unipg_a_knobs():
    for bad in ({"lr": 1e-2}, {"vicreg_weight": 1.0}, {"n_particles": 64},
                {"cover_weight": 1.5}, {"fm_weight": 0.1}, {"teacher": "x"}):
        try:
            exam.uni_gan_cfg(**bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"uni_gan_cfg accepted {bad}")


def test_fit_refuses_supervised_g_losses():
    field = divergent_field(seed=0)
    cfg = AdvConfig(steps=8, seed=0, cover_weight=1.5)
    try:
        exam.fit_uni_gan(field, cfg=cfg)
    except ValueError:
        return
    raise AssertionError("fit_uni_gan accepted cover_weight=1.5")


def test_teacher_is_raw_positive_and_leak_blind():
    field = divergent_field(seed=0)
    for row in range(int(field.rows)):
        pos, _neg, _neu = field.poles(row)
        assert torch.allclose(exam.uni_teacher_plus(field, row), pos)
    cfg = exam.uni_gan_cfg(steps=8, seed=0)
    r_plain, _ = exam.fit_uni_gan(field, cfg=cfg)
    r_leak, _ = exam.fit_uni_gan(field, cfg=cfg, leak_dir=field.declared_e())
    assert torch.allclose(r_plain.delta(1.0), r_leak.delta(1.0))


def test_scale_zero_is_exact_zero_and_hold_measured():
    field = divergent_field(seed=0)
    cfg = exam.uni_gan_cfg(steps=8, seed=0)
    residual, _ = exam.fit_uni_gan(field, cfg=cfg)
    assert torch.allclose(residual.delta(0.0), torch.zeros_like(residual.delta(0.0)))
    row = exam.score_uni_residual("probe", field, residual)
    assert row["neu_hold"] == 1.0
    assert row["half_scale"]["half_cover"] is not None
    assert "dangerous" in row["canary"]


def test_deterministic_per_seed():
    field = divergent_field(seed=0)
    cfg = exam.uni_gan_cfg(steps=16, seed=0)
    r1, _ = exam.fit_uni_gan(field, cfg=cfg)
    r2, _ = exam.fit_uni_gan(field, cfg=cfg)
    assert torch.allclose(r1.delta(1.0), r2.delta(1.0))


def test_smoke_gates_are_computed_not_asserted():
    field = divergent_field(seed=0)
    cfg = exam.uni_gan_cfg(steps=16, seed=0)
    row = exam.score_uni_gan("smoke", field, cfg=cfg)
    for key in ("cover", "off_caption", "neu_hold", "hit", "tag", "grad_arm", "prior"):
        assert key in row, key
    assert row["prior"] == "learned"
    assert row["tag"] == "DRIFT"  # learned particle branch drifts from YuE2 unipolar
    assert 0.0 <= row["cover"] <= 1.0
    assert 0.0 <= row["off_caption"] <= 1.0


def test_jitter_prior_is_match_and_unsticks_cover():
    field = divergent_field(seed=0)
    cfg = exam.uni_gan_cfg(steps=600, seed=0)
    learned = exam.score_uni_gan("learned", field, cfg=cfg, freeze_prior=False)
    jitter = exam.score_uni_gan("jitter", field, cfg=cfg, freeze_prior=True)
    assert jitter["prior"] == "jitter"
    assert jitter["tag"] == "MATCH"
    assert jitter["cover"] > learned["cover"] + 0.15


def test_gradreg_arms_thread_into_penalty():
    for arm in ("unipg_a_baseline", "unipg_a_ginterp", "unipg_a_delayed",
                "unipg_a_lazy2", "unipg_a_kappa20"):
        cfg = exam.arm_cfg(arm, steps=200, seed=0)
        reg = make_grad_regularizer(cfg)
        assert reg.arm == cfg.grad_arm
        assert reg.center(1) >= 0.0


def test_locked_and_music_untouched():
    locked.assert_advconfig_defaults_match_locked()
    got = vars(build_parser().parse_args([]))
    got.pop("out")
    assert got == locked.PRODUCTION_ARGV_DEFAULTS
    assert ARM_B["lm_target"] == "faithful_guard_e"
