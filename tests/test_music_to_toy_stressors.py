"""Optional pytest: Music→toy stressor smoke (CPU, short seeds)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path("/workspace/sliders-conceptmod")
if not (REPO / "analysis/slider2d/field3d.py").exists():
    # pop-os / alternate checkout
    for cand in (Path(__file__).resolve().parents[1], Path.cwd()):
        if (cand / "analysis/slider2d/field3d.py").exists():
            REPO = cand
            break
    else:
        pytest.skip("field3d.py not present", allow_module_level=True)

sys.path.insert(0, str(REPO))
# apply hooks
hooks = Path(__file__).resolve().parent / "field3d_music_stressor_hooks.py"
if hooks.exists():
    sys.path.insert(0, str(hooks.parent))
    import field3d_music_stressor_hooks as h

    h.apply()

from analysis.slider2d.field3d import (  # noqa: E402
    lyric_span_entangle_field3d,
    close_live_noise_field3d,
    close_field3d,
    dual_arm_leftover_geom_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
    summarize_music_close_posture,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402


def _cfg(seed=0, cover=1.5, n=1):
    return default_cfg(
        steps=400,  # short smoke
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n,
        particle_l2=0.02,
    )


def test_lyric_span_entangle_constructs_and_runs():
    f = lyric_span_entangle_field3d(seed=0)
    row = score_adv_field3d_exam(f, teacher="faithful_guard_e", cfg=_cfg(), name="pytest_lyric")
    assert "u_kept" in row or "exam_score" in row
    # lyric is NOT close-family — Music posture warn must stay clear @ n=1
    assert row.get("music_close_posture_warn") is None


def test_close_live_noise_constructs_and_runs():
    f = close_live_noise_field3d(seed=0)
    row = score_adv_field3d_exam(f, teacher="faithful_guard_e", cfg=_cfg(), name="pytest_close_noise")
    assert "u_kept" in row or "exam_score" in row
    # Fire #23: dig/runners must surface Music parts0 close warn at n=1
    assert row.get("music_close_posture_warn")
    assert "n_particles>=2" in row["music_close_posture_warn"]


def test_close_live_noise_warn_clears_at_n2():
    f = close_live_noise_field3d(seed=0)
    row = score_adv_field3d_exam(
        f, teacher="faithful_guard_e", cfg=_cfg(n=2), name="pytest_close_noise_n2"
    )
    assert row.get("music_close_posture_warn") is None


def test_dual_arm_geom_constructs():
    f = dual_arm_leftover_geom_field3d()
    row = score_adv_field3d(f, teacher="faithful_guard_e", cfg=_cfg(cover=0.0), name="pytest_dual")
    assert "pass" in row
    assert row.get("music_close_posture_warn") is None


def test_summarize_music_close_posture_aggregates_dig_rows():
    rows = [
        score_adv_field3d_exam(
            close_field3d(), teacher="faithful_guard_e", cfg=_cfg(n=1), name="c_n1"
        ),
        score_adv_field3d_exam(
            close_field3d(), teacher="faithful_guard_e", cfg=_cfg(n=2), name="c_n2"
        ),
        score_adv_field3d_exam(
            lyric_span_entangle_field3d(seed=0),
            teacher="faithful_guard_e",
            cfg=_cfg(n=1),
            name="lyric_n1",
        ),
    ]
    summary = summarize_music_close_posture(rows)
    assert summary["n_rows"] == 3
    assert summary["n_warn"] == 1
    assert summary["n_clear"] == 2
    assert summary["n_missing_key"] == 0
    assert summary["any_warn"] is True
    assert summary["warns"][0]["name"] == "c_n1"
