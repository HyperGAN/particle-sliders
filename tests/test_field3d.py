"""Unit tests for Field3D scaffold (geometry + score wiring)."""
from __future__ import annotations

import torch

from analysis.slider2d.field3d import (
    CELLS_3D,
    Field3D,
    MUSIC_CLOSE_POSTURE_KINDS,
    music_close_posture_warn,
    close_field3d,
    close_live_noise_field3d,
    close_with_leak_field3d,
    cross_axis_rows_field3d,
    divergent_field3d,
    dual_arm_leftover_geom_field3d,
    e_on_u_declare_lie_field3d,
    grit_content_dom_field3d,
    leftover_field3d,
    lyric_span_entangle_field3d,
    amp_lie_leftover_declare_field3d,
    hold_e_lyric_mix_field3d,
    stagger_mild_cross_field3d,
    multipair_corr_seed_field3d,
    content_leak_flip_rows_field3d,
    lyric_neu_heavy_gate_field3d,
    declare_split_three_field3d,
    scale_descent_homo_field3d,
    guard_refuse_hot_eoc_field3d,
    content_cascade_rows_field3d,
    eoc_threshold_edge_field3d,
    leftover_hot_eoc_declare_field3d,
    prefix_shared_proxy_field3d,
    roles_split_proxy_field3d,
    scale_stagger_homo_field3d,
    tiny_slider_dom_field3d,
    unused_e_field3d,
)


def test_leftover_basis_orthonormalish():
    f = leftover_field3d()
    u, c, e = f.short_u(), f.content_dir(), f.leak_e()
    assert torch.allclose(u.norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(c.norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(e.norm(), torch.tensor(1.0), atol=1e-5)
    assert abs(float(u @ c)) < 1e-4
    assert abs(float(u @ e)) < 1e-4
    assert abs(float(c @ e)) < 1e-4


def test_odd_scales_with_row_scales():
    f = leftover_field3d(rows=2, row_scales=(1.0, 2.0))
    assert torch.allclose(f.odd(1), 2.0 * f.odd(0), atol=1e-5)


def test_close_and_divergent_construct():
    assert close_field3d().kind == "close"
    assert divergent_field3d().kind == "divergent"
    assert unused_e_field3d().e_unused > 0


def test_declared_e_unused_default():
    f = leftover_field3d()
    # default leftover: e unused => declared points near unused e
    de = f.declared_e()
    assert de.shape == f.short_u().shape


def test_cells_registry():
    assert "close" in CELLS_3D
    assert "divergent" in CELLS_3D
    assert "unused_e" in CELLS_3D


def test_row_amps_cross_axis():
    f = cross_axis_rows_field3d()
    assert f.row_amps is not None
    o0, o1 = f.odd(0), f.odd(1)
    # different axis mixes => not parallel
    cos = float((o0 @ o1) / (o0.norm() * o1.norm()))
    assert abs(cos) < 0.99


def test_row_amps_length_guard():
    try:
        Field3D(rows=2, row_amps=((1.0, 0.0, 0.0),))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_score_adv_field3d_importable():
    from analysis.slider2d.field3d import score_adv_field3d, score_adv_field3d_exam
    assert callable(score_adv_field3d)
    assert callable(score_adv_field3d_exam)


def test_gan_accepts_field3d():
    from analysis.slider2d.gan import _teacher_pair, _field_leak_dir
    f = leftover_field3d()
    t_plus, t_minus, neu = _teacher_pair(f, 0, teacher="faithful_guard_e", leak_dir=f.declared_e())
    assert t_plus.shape == f.short_u().shape
    assert _field_leak_dir(f) is not None


def test_music_stressor_ctors():
    f = lyric_span_entangle_field3d(seed=0)
    assert f.kind == "lyric_span_entangle"
    assert f.rows == 5
    assert f.e_on_content > 0
    assert f.row_amps is not None
    n = close_live_noise_field3d(seed=0)
    assert n.kind == "close_live_noise"
    d = dual_arm_leftover_geom_field3d()
    assert d.kind == "dual_arm_leftover_geom"


def test_music_stressors_in_cells_registry():
    for k in ("lyric_span_entangle", "close_live_noise", "dual_arm_leftover_geom"):
        assert k in CELLS_3D
        assert CELLS_3D[k]().kind == k


def test_m13_m19_ctors_and_registry():
    ctors = {
        "tiny_slider_dom": tiny_slider_dom_field3d,
        "e_on_u_declare_lie": e_on_u_declare_lie_field3d,
        "prefix_shared_proxy": prefix_shared_proxy_field3d,
        "scale_stagger_homo": scale_stagger_homo_field3d,
        "roles_split_proxy": roles_split_proxy_field3d,
        "close_with_leak": close_with_leak_field3d,
        "grit_content_dom": grit_content_dom_field3d,
    }
    for k, fn in ctors.items():
        assert k in CELLS_3D
        f = fn()
        assert f.kind == k
        assert CELLS_3D[k]().kind == k


def test_batch2_music_stressors():
    f = amp_lie_leftover_declare_field3d()
    assert f.kind == "amp_lie_leftover_declare"
    assert f.e_on_content > 1.0
    h = hold_e_lyric_mix_field3d()
    assert h.kind == "hold_e_lyric_mix"
    assert h.content > 0.7 and h.e_on_content > 0
    s = stagger_mild_cross_field3d()
    assert s.kind == "stagger_mild_cross"
    assert s.row_amps is not None and s.rows == 4
    m0 = multipair_corr_seed_field3d(seed=0)
    m7 = multipair_corr_seed_field3d(seed=7)
    assert m0.kind == "multipair_corr_seed"
    assert m0.row_amps != m7.row_amps
    for k in ("amp_lie_leftover_declare", "hold_e_lyric_mix", "stagger_mild_cross", "multipair_corr_seed"):
        assert k in CELLS_3D
        assert CELLS_3D[k](seed=0).kind == k if k == "multipair_corr_seed" else CELLS_3D[k]().kind == k


def test_batch3_music_stressors():
    f = content_leak_flip_rows_field3d()
    assert f.kind == "content_leak_flip_rows"
    assert f.row_amps is not None and f.rows == 4
    # content↔leak flip: row0 content>leak, row1 leak>content
    assert f.row_amps[0][1] > f.row_amps[0][2]
    assert f.row_amps[1][2] > f.row_amps[1][1]
    n = lyric_neu_heavy_gate_field3d()
    assert n.kind == "lyric_neu_heavy_gate"
    assert n.lyric > 1.5 and n.e_unused > 0
    d = declare_split_three_field3d()
    assert d.kind == "declare_split_three"
    assert d.e_on_u > 0 and d.e_on_content > 0 and d.e_unused > 0
    s = scale_descent_homo_field3d()
    assert s.kind == "scale_descent_homo"
    assert s.row_scales[0] > s.row_scales[-1]
    for k in (
        "content_leak_flip_rows",
        "lyric_neu_heavy_gate",
        "declare_split_three",
        "scale_descent_homo",
    ):
        assert k in CELLS_3D
        assert CELLS_3D[k]().kind == k





def test_batch4_music_stressors():
    g = guard_refuse_hot_eoc_field3d()
    assert g.kind == "guard_refuse_hot_eoc"
    assert g.e_on_content >= 0.8 and g.content > 0.7
    c = content_cascade_rows_field3d()
    assert c.kind == "content_cascade_rows"
    assert c.row_amps is not None and c.rows == 4
    assert c.row_amps[0][1] < c.row_amps[-1][1]  # ascending content
    e = eoc_threshold_edge_field3d()
    assert e.kind == "eoc_threshold_edge"
    assert 0.30 <= e.e_on_content <= 0.34
    h = leftover_hot_eoc_declare_field3d()
    assert h.kind == "leftover_hot_eoc_declare"
    assert h.e_on_content > 1.0 and h.e_unused > 0.3
    for k in (
        "guard_refuse_hot_eoc",
        "content_cascade_rows",
        "eoc_threshold_edge",
        "leftover_hot_eoc_declare",
    ):
        assert k in CELLS_3D
        assert CELLS_3D[k]().kind == k


def test_music_close_posture_warn_n1_close_family():
    """Fire #22: Music parts0 (n=1) warns on close-family; silent elsewhere."""
    from analysis.slider2d.adv import AdvConfig

    w = music_close_posture_warn(1, cell_kind="close")
    assert w and "n_particles=1" in w and "n_particles>=2" in w
    assert "multi-seed" in w.lower() or "multi-seed" in w
    assert "vicreg_weight=0" in w

    assert music_close_posture_warn(2, cell_kind="close") is None
    assert music_close_posture_warn(12, cell_kind="close") is None
    assert music_close_posture_warn(1, cell_kind="leftover") is None
    assert music_close_posture_warn(1, cell_kind="leftover_field3d") is None
    assert music_close_posture_warn(1, cell_kind="divergent") is None

    for kind in ("close", "close_live_noise", "tiny_slider_dom"):
        assert kind in MUSIC_CLOSE_POSTURE_KINDS
        assert music_close_posture_warn(1, cell_kind=kind)
        assert music_close_posture_warn(2, cell_kind=kind) is None

    cfg_n1 = AdvConfig(n_particles=1)
    cfg_n2 = AdvConfig(n_particles=2)
    assert music_close_posture_warn(cfg_n1, cell_kind="close")
    assert music_close_posture_warn(cfg_n2, cell_kind="close") is None
    # leftover_field3d kind should not warn even at n=1
    assert leftover_field3d().kind == "leftover"
    assert music_close_posture_warn(1, cell_kind=leftover_field3d().kind) is None


def test_score_adv_field3d_emits_music_close_posture_warn_key():
    """score path surfaces the warn string without changing train defaults."""
    from analysis.slider2d.field3d import score_adv_field3d
    from analysis.slider2d.gan import default_cfg

    cfg = default_cfg(
        steps=50,
        seed=0,
        b_cap=1.0,
        cover_weight=1.5,
        fm_weight=0.0,
        n_particles=1,
        particle_l2=0.02,
    )
    row = score_adv_field3d(close_field3d(), cfg=cfg, name="pytest_close_warn")
    assert row.get("music_close_posture_warn")
    assert "n_particles>=2" in row["music_close_posture_warn"]

    cfg2 = default_cfg(
        steps=50,
        seed=0,
        b_cap=1.0,
        cover_weight=1.5,
        fm_weight=0.0,
        n_particles=2,
        particle_l2=0.02,
    )
    row2 = score_adv_field3d(close_field3d(), cfg=cfg2, name="pytest_close_n2")
    assert row2.get("music_close_posture_warn") is None

    row_l = score_adv_field3d(leftover_field3d(), cfg=cfg, name="pytest_leftover_n1")
    assert row_l.get("music_close_posture_warn") is None
