"""Formulation leaderboards: unipolar sibling next to bipolar.

CPU-only. Pins that the unipolar board emits with plus honesty gates and
never requires antipodal collapse, that the bipolar board still emits,
that one method can top one board and fail the other (the split is the
point), and that the live v9 default / ARM_B row are untouched (no new
trainer arm is added here — everything is propose-side analysis).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.slider2d.formulation_leaderboard import (
    BIPOLAR_EVAL_SCALES,
    BI_FORMULATION_ARMS,
    POLARITY_BI,
    POLARITY_NOTE,
    POLARITY_UNI,
    REQUIRED_CELLS,
    UNIPOLAR_EVAL_SCALES,
    bipolar_hit,
    collect_bipolar_board,
    collect_boards,
    collect_unipolar_board,
    unipolar_hit,
)
from analysis.slider2d.plus_exam import PLUS_COVER_MIN, PLUS_OFF_MAX
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN
from conceptmod.textsliders.train_lm_slider_music3 import (
    ARM_B,
    parse_args,
)


UNI_STEPS = 60
BI_STEPS = 60
_CACHE: dict[str, dict] = {}


def uni() -> dict:
    if "uni" not in _CACHE:
        _CACHE["uni"] = collect_unipolar_board(steps=UNI_STEPS, seed=0)
    return _CACHE["uni"]


def bi() -> dict:
    if "bi" not in _CACHE:
        _CACHE["bi"] = collect_bipolar_board(steps=BI_STEPS, seed=0)
    return _CACHE["bi"]


def uni_rows(cell: str) -> dict[str, dict]:
    return {r["name"]: r for r in uni()["cells"][cell]}


def bi_rows(cell: str) -> dict[str, dict]:
    return {r["name"]: r for r in bi()["cells"][cell]}


# -- unipolar board emits --------------------------------------------------


def test_unipolar_board_emits_with_polarity_and_scales():
    board = uni()
    assert board["polarity"] == POLARITY_UNI
    assert board["eval_scales"] == [0.0, 0.5, 1.0]
    assert list(UNIPOLAR_EVAL_SCALES) == [0.0, 0.5, 1.0]
    assert board["scored"] == ["cover", "off_caption", "neu_hold"]
    assert board["minus_is_diagnostic_only"] is True
    assert board["gates"] == {
        "cover_min": PLUS_COVER_MIN,
        "off_caption_max": PLUS_OFF_MAX,
        "neu_hold_min": PLUS_NEU_HOLD_MIN,
    }
    for cell in ("divergent", "close", "unused_e"):
        assert cell in board["cells"]
    names = [r["name"] for r in board["cells"]["divergent"]]
    assert names == [
        "faithful_plus_neu",
        "faithful_plus",
        "leftover_gate_bipolar",
        "faithful_even_blend",
        "pair_odd_midpoint",
        "rpgan_bcap_plus_neu",
    ]
    for cell, rows in board["cells"].items():
        for row in rows:
            assert row["polarity"] == "uni"
            assert row["eval_scales"] == [0.0, 0.5, 1.0]
            assert row["hit"] == unipolar_hit(row)
            assert row["canary"]["scored"] is False
            assert row["antipodal_consulted"] is False
            assert row["half_scale"]["scored"] is False
            assert row["half_scale"]["scale"] == 0.5
            assert 0.0 <= row["half_scale"]["half_cover"] <= 1.0


def test_unipolar_hit_never_requires_antipodal_collapse():
    base = {"cover": 0.93, "off_caption": 0.0, "neu_hold": 1.0}
    assert unipolar_hit(base) is True
    for collapse in (-1.0, -0.5, 0.0, 0.5, 1.0, None):
        row = {**base, "collapse": collapse, "antipodal_collapse": collapse}
        assert unipolar_hit(row) is True, collapse
    miss = {**base, "cover": 0.5}
    assert unipolar_hit(miss) is False
    assert unipolar_hit({**base, "off_caption": 0.5}) is False
    assert unipolar_hit({**base, "neu_hold": 0.5}) is False
    board = uni()
    assert "antipodal_collapse" in board["not_scored"]
    assert "minus_continuation" in board["not_scored"]


def test_unipolar_minus_is_diagnostic_even_when_dangerous():
    row = uni_rows("divergent")["faithful_plus_neu"]
    assert row["hit"] is True
    assert row["canary"]["dangerous"] is True
    # The dangerous -1 canary does not flip the + hit.
    assert unipolar_hit(row) is True


# -- bipolar board still emits ---------------------------------------------


def test_bipolar_board_emits_with_both_pole_readouts():
    board = bi()
    assert board["polarity"] == POLARITY_BI
    assert board["eval_scales"] == [-1.0, 1.0]
    assert list(BIPOLAR_EVAL_SCALES) == [-1.0, 1.0]
    assert board["antipodal_is_diagnostic_only"] is True
    assert "antipodal_collapse" in board["not_scored"]
    for cell in ("divergent", "close", "unused_e"):
        names = [r["name"] for r in board["cells"][cell]]
        for arm in BI_FORMULATION_ARMS:
            if arm in ("faithful_raw", "faithful_sub_e_if_unused",
                       "faithful_guard_e", "pair_odd_midpoint"):
                assert arm in names, (cell, arm)
        for row in board["cells"][cell]:
            assert row["polarity"] == "bi"
            assert row["hit"] == bipolar_hit(row) == bool(row["hit"])
            assert "sings_both_poles" in row
            assert "antipodal_collapse" in row
            assert row["antipodal_consulted"] is False


def test_bipolar_hit_is_the_both_pole_continuation_not_the_lock():
    assert bipolar_hit({"pass": True, "collapse": 0.0}) is True
    assert bipolar_hit({"pass": False, "collapse": -1.0}) is False


def test_compiled_bipolar_board_source_is_untouched():
    src = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "faithful_plus" not in src
    assert "plus_neu_exam" not in src
    assert "formulation_leaderboard" not in src
    plus = Path("analysis/slider2d/plus_exam.py").read_text()
    assert "from analysis.slider2d.scoreboard" not in plus


# -- the split: one method tops one board and fails the other --------------


def test_pair_odd_is_bipolar_good_but_unipolar_weak():
    assert bi_rows("close")["pair_odd_midpoint"]["hit"] is True
    for cell in REQUIRED_CELLS:
        assert uni_rows(cell)["pair_odd_midpoint"]["hit"] is False


def test_uni_is_unipolar_good_but_has_no_bipolar_pair_fit():
    for cell in REQUIRED_CELLS:
        assert uni_rows(cell)["faithful_plus_neu"]["hit"] is True
    for cell in ("divergent", "close", "unused_e"):
        assert "faithful_plus_neu" not in bi_rows(cell)
    assert uni()["rank"][0]["name"] == "faithful_plus_neu"
    assert uni()["rank"][0]["in_box"] is True


def test_combined_rows_carry_a_polarity_column():
    both = collect_boards(steps=UNI_STEPS, seed=0)
    pols = {r["polarity"] for r in both["rows"]}
    assert pols == {"uni", "bi"}
    assert both["uni"]["polarity"] == "uni"
    assert both["bi"]["polarity"] == "bi"


# -- live default / ARM_B untouched (no new trainer arm here) --------------


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.adv_preset == "none"
    assert args.require_arm_b is False


def test_arm_b_row_is_untouched():
    assert ARM_B["lm_target"] == "faithful_guard_e"
    assert ARM_B["adv_arch"] == "mlp"
    assert ARM_B["adv_b_cap"] == pytest.approx(1.0)
    assert ARM_B["adv_reg_kappa"] == pytest.approx(1.0)
    assert ARM_B["fm_weight"] == pytest.approx(0.0)
    assert ARM_B["vicreg_weight"] == pytest.approx(0.0)
    assert ARM_B["parts"] == 0


# -- runner: --polarity flag -------------------------------------------------


def test_runner_polarity_flag_selects_boards(tmp_path, monkeypatch):
    import analysis.slider2d.run_formulation_leaderboard as runner

    monkeypatch.setattr(runner, "UNI_MD", tmp_path / "UNI.md")
    monkeypatch.setattr(runner, "BI_MD", tmp_path / "BI.md")
    out = tmp_path / "metrics"
    assert runner.main(
        ["--polarity", "uni", "--out", str(out), "--steps", "20", "--seed", "0"]
    ) == 0
    assert (out / "metrics_uni.json").exists()
    assert (tmp_path / "UNI.md").exists()
    assert not (out / "metrics_bi.json").exists()
    assert not (tmp_path / "BI.md").exists()

    assert runner.main(
        ["--polarity", "bi", "--out", str(out), "--steps", "20", "--seed", "0"]
    ) == 0
    assert (out / "metrics_bi.json").exists()
    assert (tmp_path / "BI.md").exists()

    blob = json.loads((out / "metrics_uni.json").read_text())
    assert blob["polarity"] == "uni"
    blob = json.loads((out / "metrics_bi.json").read_text())
    assert blob["polarity"] == "bi"


def test_runner_both_writes_combined_polarity_column(tmp_path, monkeypatch):
    import analysis.slider2d.run_formulation_leaderboard as runner

    monkeypatch.setattr(runner, "UNI_MD", tmp_path / "UNI.md")
    monkeypatch.setattr(runner, "BI_MD", tmp_path / "BI.md")
    out = tmp_path / "metrics"
    assert runner.main(
        ["--polarity", "both", "--out", str(out), "--steps", "20", "--seed", "0"]
    ) == 0
    both = json.loads((out / "metrics_both.json").read_text())
    pols = {r["polarity"] for r in both["rows"]}
    assert pols == {"uni", "bi"}


def test_polarity_note_says_when_to_read_which():
    assert "signed concept axis" in POLARITY_NOTE
    assert "+ end is trained" in POLARITY_NOTE
    assert "never" in POLARITY_NOTE or "neither" in POLARITY_NOTE
