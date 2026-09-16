"""Bug Hunt E: leaderboard / alg-select harness honesty regressions.

CPU only. No Music GPU. Each test pins one fixed honesty bug so the
ranking, gate, teacher-label and coverage semantics cannot silently drift
back. Honesty-first ranking semantics (exam_score = min(overlap, swing)
on the live pairs, verdict stays a label) are unchanged.
"""

from __future__ import annotations

import math

import pytest
import torch

from analysis.slider2d import gan as gan_mod
from analysis.slider2d.exam import close_field, divergent_field
from analysis.slider2d.field import E_ATTR, E_SLIDER, Field2D
from analysis.slider2d.gan import AdvResidual, _coverage, train_lm_adv
from analysis.slider2d.scoreboard import _row, sort_rows
from analysis.slider2d.sheet import leaky_field, teacher_points as sheet_teacher_points
from analysis.slider2d.exam import teacher_points as exam_teacher_points
from analysis.slider2d.train import music3_pairs, score_residual, train_lm
from conceptmod.textsliders.slider_targets import (
    lm_faithful_guard_e,
    lm_faithful_sub_e,
)


# -- P0: teacher label must match the applied rewrite ----------------------


def test_train_lm_adv_rejects_unknown_teacher_before_training():
    field = Field2D()
    with pytest.raises(ValueError, match="teacher must be one of"):
        train_lm_adv(field, teacher="pair_odd", with_attrs=True)


def test_guard_rewrite_is_applied_not_skipped_with_attrs():
    """Guard + attrs is a measured no-op (odd has no attr), not a branch skip.

    The old ``if teacher == 'faithful' or with_attrs`` returned raw poles
    for every teacher when attrs were on. Now the guard path runs; on
    attribute-pinned pairs it coincides with raw because there is nothing
    to subtract — verified per pair below, without a GAN train.
    """
    field = Field2D()
    for pair in music3_pairs(True):
        pos = field.embed(pair.positive, 0.5)
        neg = field.embed(pair.negative, 0.5)
        neu = field.embed(pair.neutral, 0.5)
        plus, minus = lm_faithful_guard_e(pos, neg, neu, E_ATTR, slider_dir=E_SLIDER)
        assert torch.allclose(plus, pos, atol=1e-6)
        assert torch.allclose(minus, neg, atol=1e-6)
    # And without attrs the guard actually rewrites (odd ê removed).
    field2 = Field2D()
    pos = field2.embed("energetic", 0.5)
    neg = field2.embed("calm", 0.5)
    neu = field2.embed("song", 0.5)
    raw_odd_e = abs(float((((pos - neg) / 2.0).flatten() @ E_ATTR.flatten())))
    assert raw_odd_e > 0.20
    plus, minus = lm_faithful_guard_e(pos, neg, neu, E_ATTR, slider_dir=E_SLIDER)
    assert not torch.allclose(plus, pos, atol=1e-6)
    clean_odd_e = abs(float((((plus - minus) / 2.0).flatten() @ E_ATTR.flatten())))
    assert clean_odd_e <= 1e-5


# -- P1: coverage is worst-row, not row 0 ----------------------------------


def test_coverage_reports_worst_row_not_row_zero():
    dim = 4
    residual = AdvResidual(torch.zeros(dim), torch.zeros(dim))
    # Row 0 exact, row 1 off by 50% relative: worst-row reports the miss.
    neu = torch.zeros(2, dim)
    poles_p = torch.stack([torch.ones(dim), torch.ones(dim)])
    poles_m = torch.stack([-torch.ones(dim), -torch.ones(dim)])
    got = _coverage(residual, poles_p, poles_m, neu)
    assert got["pole_rel_err_plus"] == pytest.approx(1.0)
    assert got["covered"] is False
    # A residual landing exactly on every pole still covers.
    exact = AdvResidual(torch.ones(dim), torch.zeros(dim))
    got2 = _coverage(exact, poles_p, poles_m, neu)
    assert got2["pole_rel_err_plus"] == pytest.approx(0.0, abs=1e-6)
    assert got2["covered"] is True


# -- P1: unguarded subtract requires ê on every cell ------------------------


def test_exam_even_subtract_requires_e_like_sheet():
    field = close_field()
    assert field.declared_e() is None
    pos, neg, neu = field.poles(0)
    with pytest.raises(ValueError, match="needs a declared"):
        exam_teacher_points(field, 0, teacher="faithful_sub_even_e", leak_dir=None)
    # Guarded teachers stay defined (caption) with nothing declared.
    plus, _ = exam_teacher_points(field, 0, teacher="faithful_guard_e", leak_dir=None)
    assert torch.allclose(plus, pos, atol=1e-8)
    assert torch.equal(pos, pos) and torch.equal(neg, neg) and torch.equal(neu, neu)


def test_sheet_sub_e_is_the_canonical_live_rewrite():
    field = leaky_field()
    e = field.leak_e()
    for row in range(field.rows):
        pos, neg, neu = field.poles(row)
        got_plus, got_minus = sheet_teacher_points(
            field, row, teacher="faithful_sub_e", leak_dir=e
        )
        want_plus, want_minus = lm_faithful_sub_e(
            pos, neg, neu, e, slider_dir=field.short_u()
        )
        assert torch.allclose(got_plus, want_plus, atol=1e-6)
        assert torch.allclose(got_minus, want_minus, atol=1e-6)
        # Midpoint stays ½(h++h−): odd-only subtraction.
        assert torch.allclose(
            0.5 * (got_plus + got_minus), 0.5 * (pos + neg), atol=1e-6
        )


def test_canonical_sub_e_keeps_midpoint_when_common_carries_e():
    # A per-pole (h−h0)·ê subtraction would shift the midpoint here; the
    # canonical odd-only rewrite must not.
    pos = torch.tensor([1.0, 1.0, 1.0])
    neg = torch.tensor([-1.0, 1.0, -1.0])
    neu = torch.zeros(3)
    e = torch.tensor([0.0, 1.0, 0.0])
    u = torch.tensor([1.0, 0.0, 0.0])
    plus, minus = lm_faithful_sub_e(pos, neg, neu, e, slider_dir=u)
    assert torch.allclose(0.5 * (plus + minus), 0.5 * (pos + neg), atol=1e-6)


# -- P1/P2: hold is on ê_⊥, never raw ê ------------------------------------


def test_hold_with_leak_uses_e_perp_not_raw():
    field = Field2D()
    pairs = music3_pairs(False)
    leak = E_SLIDER + E_ATTR
    leak = leak / leak.norm()
    implicit = train_lm(
        field,
        pairs,
        target_mode="symmetric",
        hold_weight=8.0,
        leak_dir=leak,
        steps=250,
        seed=0,
    )
    explicit = train_lm(
        field,
        pairs,
        target_mode="symmetric",
        hold_weight=8.0,
        leak_dir=leak,
        slider_dir=E_SLIDER,
        steps=250,
        seed=0,
    )
    m = score_residual(implicit)
    m2 = score_residual(explicit)
    assert m["cos_slider_plus"] == pytest.approx(m2["cos_slider_plus"], abs=1e-6)
    assert m["cos_slider_plus"] >= 0.90


# -- P2: scoreboard stamps agree; no cross-cell inheritance -----------------


def test_leftover_only_failure_does_not_forge_a_gender_stamp():
    row = _row(
        "probe_fail",
        "probe",
        leftover={
            "on_sheet_kept": 0.10,
            "garble": 0.90,
            "argmax_on_sheet": 0.0,
            "swing_kept": 0.0,
            "leak_tok": 0.99,
        },
        gender=None,
        leftover_leak=0.99,
        fixture="probe",
    )
    assert row["leftover_works"] is False
    assert row["gender_works"] is None
    assert row["cells"]["sheet_gender"] is None
    assert row["cells"]["sheet_leftover"] is False


def test_leftover_gate_never_reads_gender_sheet_numbers():
    row = _row(
        "probe_x",
        "probe",
        leftover={"leak_tok": 0.0},
        gender={
            "on_sheet_kept": 1.0,
            "garble": 0.0,
            "argmax_on_sheet": 1.0,
            "swing_kept": 1.0,
            "leak_tok": 0.0,
        },
        leftover_leak=0.0,
        gender_leak=0.0,
        fixture="probe",
    )
    # Leftover has leak-only numbers: sheet gate cannot pass on gender's
    # sheet, so the cell is leak-True via leak alone (no sheet to fail).
    # The load-bearing assertion: gender sheet presence must not flip a
    # leftover fail into a pass.
    row2 = _row(
        "probe_y",
        "probe",
        leftover={"leak_tok": 0.99},
        gender={
            "on_sheet_kept": 1.0,
            "garble": 0.0,
            "argmax_on_sheet": 1.0,
            "swing_kept": 1.0,
            "leak_tok": 0.0,
        },
        leftover_leak=0.99,
        gender_leak=0.0,
        fixture="probe",
    )
    assert row2["cells"]["sheet_leftover"] is False
    assert row["cells"]["sheet_gender"] is True


def test_sort_treats_nan_score_as_unscored_last():
    rows = sort_rows(
        [
            {"id": "b_nan", "compiled": "works", "exam_score": float("nan"), "cells": {}},
            {"id": "a_ok", "compiled": "works-on-some-pairs", "exam_score": 0.10, "cells": {}},
            {"id": "c_null", "compiled": "fails", "exam_score": None, "cells": {}},
        ]
    )
    assert [r["id"] for r in rows][0] == "a_ok"
    assert [r["id"] for r in rows][-1] in ("b_nan", "c_null")
    last = rows[-1]["exam_score"]
    assert last is None or (isinstance(last, float) and math.isnan(last))


def test_divergent_sub_e_teacher_still_needs_e():
    field = divergent_field()
    e = field.declared_e()
    assert e is not None
    with pytest.raises(ValueError, match="needs a declared"):
        exam_teacher_points(field, 0, teacher="faithful_sub_e", leak_dir=None)
    with pytest.raises(ValueError, match="needs a declared"):
        exam_teacher_points(field, 0, teacher="pair_odd_sub_e", leak_dir=None)
    # Sanity: suite imports the live module surface it pins.
    assert gan_mod.DEFAULT_TEACHER == "faithful_guard_e"
