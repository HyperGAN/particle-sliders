"""Unipolar GAN-only toy exam: gates, determinism, propose_only pins."""
import pytest
import torch

from analysis.slider2d import yue2_gan_exam as exam
from analysis.slider2d.adv import AdvConfig
from analysis.slider2d.locked_baseline_defaults import LOCKED


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_every_arm_is_gan_only_propose_only():
    """No arm may carry supervised G knobs; registry cards stay honest."""
    assert exam.PROPOSE_ONLY is True
    for name in exam.ARM_NAMES:
        cfg, vfn = exam.arm_cfg(name, steps=50, seed=0)
        assert float(cfg.cover_weight) == 0.0, name
        assert float(cfg.fm_weight) == 0.0, name
        assert vfn in ("locked", "faithful"), name
        meta = exam.ARMS[name]
        assert meta["hook"] and meta["card"], name
        assert meta["match"] in ("MATCH", "DRIFT"), name


def test_fit_refuses_supervised_g():
    """cover / FM on G raise instead of silently supervising."""
    from dataclasses import replace

    field = exam.UNI_CELLS["divergent"](seed=0)
    with pytest.raises(ValueError, match="GAN-only"):
        exam.fit_uni_gan(field, cfg=replace(AdvConfig(), cover_weight=1.5))
    with pytest.raises(ValueError, match="GAN-only"):
        exam.fit_uni_gan(
            field, cfg=replace(AdvConfig(), cover_weight=0.0, fm_weight=0.5)
        )


def test_teacher_is_raw_pos_and_zero_is_exact():
    """lm_faithful_plus_neu contract: teacher IS pos; scale 0 is base."""
    field = exam.UNI_CELLS["close"](seed=0)
    poles_p, neus = exam.uni_teacher_points(field)
    for row in range(int(field.rows)):
        pos, _neg, neu = field.poles(row)
        assert torch.equal(poles_p[row], pos.flatten())
        assert torch.equal(neus[row], neu.flatten())
    cfg, _ = exam.arm_cfg("uni_baseline", steps=5, seed=0)
    res, stats = exam.fit_uni_gan(field, cfg=cfg)
    assert torch.equal(res.delta(0.0), torch.zeros_like(res.delta(0.0)))
    assert stats["gan_only"] is True
    row = exam.score_uni_cell("probe", field, res)
    assert row["neu_hold"] == pytest.approx(1.0)
    assert set(row) >= {"cover", "off_caption", "neu_hold", "half_cover", "hit", "canary"}
    assert row["canary"]["scored"] is False


def test_fits_are_deterministic():
    """Same seed twice => identical gates (isolated RNG, no order effects)."""
    outs = []
    for _ in range(2):
        cfg, _ = exam.arm_cfg("uni_prior01_ginterp", steps=150, seed=1)
        field = exam.UNI_CELLS["close"](seed=1)
        res, _ = exam.fit_uni_gan(field, cfg=cfg)
        row = exam.score_uni_cell("x", field, res)
        outs.append((round(row["cover"], 6), round(row["off_caption"], 6)))
    assert outs[0] == outs[1]


def test_arms_never_mutate_locked_defaults():
    """Building every sweep arm leaves AdvConfig() + LOCKED untouched."""
    before = AdvConfig()
    for name in exam.ARM_NAMES:
        exam.arm_cfg(name, steps=50, seed=0)
    after = AdvConfig()
    assert vars(before) == vars(after)
    assert LOCKED["cover_weight"] == 1.5
    assert LOCKED["b_cap"] == 1.0 and LOCKED["kappa"] == 1.0


def test_unknown_arm_fails_closed():
    with pytest.raises(ValueError, match="unknown uni arm"):
        exam.arm_cfg("nope", steps=10, seed=0)
    with pytest.raises(ValueError, match="cell must be one of"):
        exam.score_uni_arm("uni_baseline", "nope", steps=10, seed=0)


def test_ginterp_passes_the_seed_that_kills_bcap():
    """Train-seed 1 dead-games b_cap arms; the interp-path cap holds it.

    Pinned budget: 1200 steps, both required cells. This is the sweep's
    PASS claim in miniature (~15s CPU).
    """
    for cell in ("divergent", "close"):
        row = exam.score_uni_arm(
            "uni_prior01_ginterp", cell, steps=1200, seed=1
        )
        assert row["hit"], (cell, row)
        assert row["cover"] >= exam.UNI_COVER_MIN
        assert row["off_caption"] <= exam.UNI_OFF_MAX
        assert row["neu_hold"] >= exam.UNI_HOLD_MIN
