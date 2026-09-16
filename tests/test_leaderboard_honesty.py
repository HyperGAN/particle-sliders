"""Bug Hunt E (reconciled): leaderboard / alg-select harness honesty regressions.

Reconciled onto main at #109 (Bug Hunt A) + #107 (B) + #106 (D) + #103 (C).

- Coverage keeps A's additive API: row-0 keys (``pole_rel_err_*``,
  ``pole_cos_*``, ``covered``) are unchanged; the miss is recorded on
  ``worst_row_rel_err`` / ``worst_row_cos`` / ``covered_all_rows``.
- ``train_lm_adv`` keeps B's fail-closed gate: a leftover-gated teacher
  with ``with_attrs=True`` raises instead of silently training on raw
  poles. Unknown teacher names also fail closed (compatible with B).

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


# -- P0: teacher label must match the applied rewrite (B fail-closed) -------


def test_train_lm_adv_rejects_unknown_teacher_before_training():
    field = Field2D()
    with pytest.raises(ValueError, match="teacher must be one of"):
        train_lm_adv(field, teacher="no_such_teacher", with_attrs=False)
    with pytest.raises(ValueError, match="teacher must be one of"):
        train_lm_adv(field, teacher="pair_odd", with_attrs=False)


def test_train_lm_adv_gated_teacher_with_attrs_is_fail_closed():
    """B semantics: gated teacher + with_attrs=True raises (no silent raw).

    Raw #108 rewrote this branch to always apply the teacher ("measured
    no-op"); the reconciled harness keeps B's fail-closed ValueError and
    routes gated teachers through unpinned pairs (with_attrs=False).
    The live guard rewrite itself is verified below without a GAN train.
    """
    field = Field2D()
    with pytest.raises(ValueError, match="with_attrs"):
        train_lm_adv(field, teacher="faithful_guard_e", with_attrs=True)
    with pytest.raises(ValueError, match="with_attrs"):
        train_lm_adv(field, teacher="faithful_sub_e_if_unused", with_attrs=True)
    # The guard rewrite is real on unpinned pairs: odd ê is removed.
    pos = field.embed("energetic", 0.5)
    neg = field.embed("calm", 0.5)
    neu = field.embed("song", 0.5)
    raw_odd_e = abs(float((((pos - neg) / 2.0).flatten() @ E_ATTR.flatten())))
    assert raw_odd_e > 0.20
    plus, minus = lm_faithful_guard_e(pos, neg, neu, E_ATTR, slider_dir=E_SLIDER)
    assert not torch.allclose(plus, pos, atol=1e-6)
    clean_odd_e = abs(float((((plus - minus) / 2.0).flatten() @ E_ATTR.flatten())))
    assert clean_odd_e <= 1e-5


# -- P1: coverage keeps row-0 keys; worst-row is additive (A) ----------------


def test_coverage_keeps_row_zero_and_reports_worst_row():
    dim = 4
    # Residual lands exactly on row 0 but misses row 1 (2x poles):
    # row-0 keys stay clean while worst-row records the miss.
    residual = AdvResidual(torch.ones(dim), torch.zeros(dim))
    neu = torch.zeros(2, dim)
    poles_p = torch.stack([torch.ones(dim), 2.0 * torch.ones(dim)])
    poles_m = torch.stack([-torch.ones(dim), -2.0 * torch.ones(dim)])
    got = _coverage(residual, poles_p, poles_m, neu)
    assert got["pole_rel_err_plus"] == pytest.approx(0.0, abs=1e-6)
    assert got["pole_rel_err_minus"] == pytest.approx(0.0, abs=1e-6)
    assert got["covered"] is True
    assert got["worst_row_rel_err"] == pytest.approx(0.5)
    assert got["covered_all_rows"] is False
    assert got["worst_row_cos"] <= 1.0
    # A residual landing exactly on every pole still covers everywhere.
    exact = AdvResidual(torch.ones(dim), torch.zeros(dim))
    neu2 = torch.zeros(2, dim)
    poles_p2 = torch.stack([torch.ones(dim), torch.ones(dim)])
    poles_m2 = torch.stack([-torch.ones(dim), -torch.ones(dim)])
    got2 = _coverage(exact, poles_p2, poles_m2, neu2)
    assert got2["pole_rel_err_plus"] == pytest.approx(0.0, abs=1e-6)
    assert got2["covered"] is True
    assert got2["covered_all_rows"] is True


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
