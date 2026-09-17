"""UniPG-E unipolar GAN-only sweep: honesty pins + fast smokes.

- GAN-only is fail-closed (cover/FM on G raise).
- Scale 0 is exact base by construction; -1 is canary-only.
- New recipes are propose_only; Music bipolar / locked defaults untouched.
- Scoreboard schema (when the sweep artifacts are committed).
"""

import json

import pytest
import torch

torch.set_num_threads(1)

from analysis.slider2d import run_unipolar_gan_sweep as sweep
from analysis.slider2d import unipolar_gan as ug
from analysis.slider2d.adv import AdvConfig
from analysis.slider2d.exam import close_field, divergent_field


def test_matrix_has_enough_gan_only_arms():
    assert len(sweep.ARMS) >= 12
    for name, arm in sweep.ARMS.items():
        assert arm["pg_tag"] in ("MATCH", "DRIFT"), name
        assert arm["vicreg_fn"] in ("locked", "faithful"), name
        assert arm["card"], name
        delta = arm["delta"]
        assert float(delta.get("cover_weight", 0.0)) == 0.0, name
        assert float(delta.get("fm_weight", 0.0)) == 0.0, name
        cfg = ug.uni_cfg(steps=50, seed=0, **delta)
        assert float(cfg.cover_weight) == 0.0
        assert float(cfg.fm_weight) == 0.0


def test_gan_only_fail_closed():
    field = divergent_field(seed=0)
    with pytest.raises(ValueError, match="GAN-only"):
        ug.fit_uni_adv(field, cfg=ug.uni_cfg(steps=5, seed=0, cover_weight=1.5))
    with pytest.raises(ValueError, match="GAN-only"):
        ug.fit_uni_adv(field, cfg=ug.uni_cfg(steps=5, seed=0, fm_weight=1.0))


def test_zero_at_zero_and_canary_unscored():
    cfg = ug.uni_cfg(steps=10, seed=0)
    field = divergent_field(seed=0)
    residual, _ = ug.fit_uni_adv(field, cfg=cfg)
    assert float(residual.delta(0.0).norm()) == 0.0
    row = ug.score_uni_residual("probe", field, residual)
    assert row["neu_hold"] == pytest.approx(1.0)
    assert row["canary"]["scored"] is False
    assert 0.0 <= row["cover"] <= 1.0
    assert 0.0 <= row["off_caption"] <= 1.0
    assert 0.0 <= row["half_cover"] <= 1.0


def test_propose_only_and_locked_defaults_untouched():
    assert ug.PROPOSE_ONLY is True
    assert ug.MERGE_TO_TRAINER is False
    assert sweep.PROPOSE_ONLY is True
    # Bare AdvConfig defaults still the locked #94 shape (no drift by us).
    from analysis.slider2d.locked_baseline_defaults import (
        assert_advconfig_defaults_match_locked,
    )

    assert_advconfig_defaults_match_locked(AdvConfig())


def test_music_bipolar_untouched():
    from analysis.slider2d.run_lm_adv import build_parser

    args = build_parser().parse_args([])
    assert args.lm_target if hasattr(args, "lm_target") else True
    assert args.teacher == "faithful_guard_e"
    assert float(args.cover_weight) == 1.5
    assert float(args.lr) == 5.0e-3
    assert args.grad_arm == "b_cap"
    assert args.target_anneal == "none"


def test_run_one_smoke_both_cells():
    for cell in ("divergent", "close"):
        out = sweep.run_one({"arm": "uni_locked", "steps": 50, "seed": 0, "cell": cell})
        assert out["arm"] == "uni_locked"
        assert out["pg_tag"] == "MATCH"
        assert 0.0 <= out["cover"] <= 1.0
        assert 0.0 <= out["off_caption"] <= 1.0
        assert out["neu_hold"] == pytest.approx(1.0)
        assert isinstance(out["hit"], bool)


def test_summarize_pass_needs_six_of_six():
    rows = []
    for cell in ("divergent", "close"):
        for seed in (0, 1, 7):
            rows.append(
                {
                    "arm": "probe",
                    "steps": 600,
                    "seed": seed,
                    "cell": cell,
                    "cover": 0.9,
                    "off_caption": 0.0,
                    "neu_hold": 1.0,
                    "overlap_pos": 1.0,
                    "blend_toward_mid": 0.1,
                    "half_cover": 0.5,
                    "hit": not (cell == "close" and seed == 7),
                    "pg_tag": "MATCH",
                    "d_loss": 0.5,
                    "g_loss": 0.7,
                    "canary_landed": "neu",
                    "canary_off": 0.0,
                    "canary_dangerous": False,
                }
            )
    import copy

    arms = copy.deepcopy(sweep.ARMS)
    arms["probe"] = {"delta": {}, "vicreg_fn": "locked", "pg_tag": "MATCH", "card": "t"}
    real_arms = sweep.ARMS
    sweep.ARMS = arms
    try:
        summary = sweep.summarize(rows)
    finally:
        sweep.ARMS = real_arms
    assert summary["probe"]["budgets"]["600"]["hits"] == "5/6"
    assert summary["probe"]["budgets"]["600"]["pass"] is False
    assert summary["probe"]["earliest_pass"] is None


def test_committed_scoreboard_schema():
    from pathlib import Path

    path = Path("docs/unipolar-gan-sweep/scoreboard.json")
    if not path.exists():
        pytest.skip("sweep artifacts not committed yet")
    blob = json.loads(path.read_text())
    assert blob["propose_only"] is True
    assert blob["merge_to_trainer"] is False
    assert blob["music_bipolar_untouched"] is True
    assert set(blob["budgets"]) == {600, 1200, 2400, 3400}
    assert set(blob["seeds"]) == {0, 1, 7}
    assert len(blob["arms"]) >= 12
    assert len(blob["runs"]) == len(blob["arms"]) * 4 * 3 * 2
    for arm, s in blob["arms"].items():
        assert s["pg_tag"] in ("MATCH", "DRIFT")
        for b in ("600", "1200", "2400", "3400"):
            bb = s["budgets"][b]
            assert bb["min_hold"] >= 0.85, (arm, b)
