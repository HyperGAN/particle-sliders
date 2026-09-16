"""UniPG-C schedule/optimizer arms: GAN-only, Music-untouched, parity-pinned."""
import torch

from analysis.slider2d import yue2_unipg_c as unipg
from analysis.slider2d.yue2_gan_exam import run_cell as ref_run_cell
from analysis.slider2d.adv import AdvConfig
from conceptmod.textsliders import yue2_arm_b as game


def test_propose_only_and_no_trainer_merge():
    assert unipg.PROPOSE_ONLY is True
    assert unipg.MERGE_TO_TRAINER is False


def test_production_recipe_untouched():
    assert game.RECIPE["name"] == "unipolar-rpgan-bcap-yue2-v3"
    assert game.RECIPE["polarity"] == "unipolar"
    assert game.RECIPE["cover_weight"] == 0.0
    assert game.RECIPE["trained_scales"] == [1.0]
    assert game.RECIPE["schedule"] == "constant"
    assert game.RECIPE["betas"] == [0.0, 0.999]


def test_locked_toy_defaults_untouched():
    assert AdvConfig().beta2 == 0.99
    assert AdvConfig().ema == 0.995
    assert AdvConfig().delay == 80


def test_every_arm_keeps_bcap_spine_and_unipolar_gan_only():
    for name, arm in unipg.ARMS.items():
        assert arm["g_lr"] > 0 and arm["d_lr"] > 0
        assert arm["beta2"] in (0.99, 0.999)
        assert arm["schedule"] in ("constant", "delayed_abs", "hold60")
        assert arm["critic"] in ("thick256", "fourier64")
        assert float(arm["ema"]) in (0.0, 0.995)
    propose = [a for a in unipg.ARMS if unipg.ARMS[a]["propose_only"]]
    assert len(propose) >= 6
    # Prior x10 has no arm: the YuE2 port has no ParticlePrior (tagged HOLD).
    assert all("prior" not in a for a in propose)


def test_reference_arm_matches_production_parity():
    torch.set_num_threads(1)
    ref = ref_run_cell("divergent", steps=(10,), seed=0)["checkpoints"][0]
    got = unipg.run_cell("divergent", "c0_production", steps=(10,), seed=0)["checkpoints"][0]
    assert got["cover"] == ref["cover"]
    assert got["off_caption"] == ref["off_caption"]
    assert got["neu_hold"] == ref["neu_hold"]


def test_schedule_shapes():
    c3 = unipg.ARMS["c3_delay80"]
    c4 = unipg.ARMS["c4_hold60"]
    assert unipg.lr_scale(1, c3, 3400) == 1.0
    assert unipg.lr_scale(81, c3, 3400) < 1.0
    assert unipg.lr_scale(1, c4, 3400) == 1.0
    assert unipg.lr_scale(int(0.6 * 3400) - 1, c4, 3400) == 1.0
    assert unipg.lr_scale(3400, c4, 3400) == 0.05
    assert unipg.lr_scale(999, unipg.ARMS["c0_production"], 3400) == 1.0


def test_ema_and_fourier_smokes_stay_gan_only():
    torch.set_num_threads(1)
    for arm in ("c2_ema_on", "c7_sched_clone", "c8_fourier_sched"):
        out = unipg.run_cell("close", arm, steps=(20,), seed=0)
        assert out["control"]["hit"]
        assert out["checkpoints"][0]["neu_hold"] == 1.0
        assert all(h["loss"] == h["g_adv"] for h in out["history"])
