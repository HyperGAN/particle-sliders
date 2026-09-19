"""Optional pytest: Music→toy stressor smoke (CPU, short seeds)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]  # sliders-conceptmod root (box or pop-os)
if not (REPO / "analysis/slider2d/field3d.py").exists():
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
    dual_arm_leftover_geom_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
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


def test_close_live_noise_constructs_and_runs():
    f = close_live_noise_field3d(seed=0)
    row = score_adv_field3d_exam(f, teacher="faithful_guard_e", cfg=_cfg(), name="pytest_close_noise")
    assert "u_kept" in row or "exam_score" in row


def test_dual_arm_geom_constructs():
    f = dual_arm_leftover_geom_field3d()
    row = score_adv_field3d(f, teacher="faithful_guard_e", cfg=_cfg(cover=0.0), name="pytest_dual")
    assert "pass" in row
