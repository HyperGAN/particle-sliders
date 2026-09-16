"""CLONE ARM 2/5 — VICReg-faithful propose-only arm (CPU only, no GPU train).

Covers the ``pg_vicreg_faithful`` arm without flipping anything locked:

- formulation: :func:`vicreg_faithful_loss` matches upstream ParticleGAN
  ``VICRegLikeLoss`` (linear-hinge var + cov, no sim term) on synthetic
  batches, and differs from the locked demo ``vicreg_loss`` exactly where
  the gap note says (weight 1 vs 0.05, std 1 vs 0.05, var+cov vs sim+var+cov);
- locked defaults: ``AdvConfig()``, ``ARM_B`` and the Music trainer argv
  are unchanged; the faithful cfg moves only ``vicreg_weight``;
- gate/smoke vs locked_shared: small-step Field2D + sheet fits run for both
  recipes and report finite geometry (the arm is a proposal, not a gate).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from analysis.slider2d import pg_vicreg_faithful as pg
from analysis.slider2d.adv import AdvConfig, vicreg_loss
from analysis.slider2d.field import Field2D
from analysis.slider2d.gan import fit_adv, train_lm_adv
from analysis.slider2d.sheet import leaky_field
from analysis.slider2d.train import score_residual
from conceptmod.textsliders.slider_targets import leftover_bipolar
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B, parse_args

STEPS = 200


def _upstream_reference(z: torch.Tensor, *, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Inline transcription of upstream ``VICRegLikeLoss.forward`` (particlegan 0.2.0).

    Kept in the test (not imported) so the arm is checked against the
    formulation text, not against itself.
    """
    std_z = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(target_std - std_z))
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    d = z.shape[1]
    if d > 1:
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        cov_loss = off_diag.pow(2).sum() / d
    else:
        cov_loss = z.new_zeros(())
    return std_loss + cov_loss


# -- propose-only marker ----------------------------------------------------


def test_arm_is_marked_propose_only():
    assert pg.PROPOSE_ONLY is True


# -- locked defaults untouched ----------------------------------------------


def test_locked_demo_defaults_unchanged():
    cfg = AdvConfig()
    assert cfg.vicreg_weight == pytest.approx(0.05)
    assert cfg.b_cap == pytest.approx(1.0)
    assert cfg.kappa == pytest.approx(1.0)
    assert cfg.fm_weight == pytest.approx(0.0)
    assert cfg.cover_weight == pytest.approx(1.5)


def test_locked_music_row_and_argv_unchanged():
    assert ARM_B["vicreg_weight"] == pytest.approx(0.0)
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.vicreg_weight == pytest.approx(0.0)
    assert args.lm_target == "v9"
    assert args.adv_preset == "none"


def test_faithful_cfg_moves_only_vicreg_weight():
    faithful = pg.pg_vicreg_faithful_cfg()
    assert faithful.vicreg_weight == pytest.approx(pg.PG_VICREG_WEIGHT)
    assert pg.PG_VICREG_WEIGHT == pytest.approx(1.0)
    locked = dataclasses.asdict(AdvConfig())
    moved = dataclasses.asdict(faithful)
    assert set(moved) == set(locked)
    for key in locked:
        if key == "vicreg_weight":
            continue
        assert moved[key] == locked[key], f"locked field {key} drifted"


def test_constructing_arm_does_not_mutate_locked_defaults():
    before = dataclasses.asdict(AdvConfig())
    pg.pg_vicreg_faithful_cfg()
    pg.pg_vicreg_faithful_cfg(AdvConfig(steps=50, seed=1))
    assert dataclasses.asdict(AdvConfig()) == before


# -- formulation: upstream-faithful ------------------------------------------


def test_faithful_loss_matches_upstream_on_synthetic_batches():
    torch.manual_seed(0)
    for shape in ((2, 2), (5, 2), (24, 2), (9, 4)):
        z = torch.randn(*shape)
        assert float(pg.vicreg_faithful_loss(z)) == pytest.approx(
            float(_upstream_reference(z)), rel=1e-5
        )
    # Non-default std target threads through identically.
    z = torch.randn(12, 3)
    assert float(pg.vicreg_faithful_loss(z, target_std=0.5)) == pytest.approx(
        float(_upstream_reference(z, target_std=0.5)), rel=1e-5
    )


def test_faithful_loss_has_no_invariance_term():
    # Upstream never augments the particles: same input, same loss, regardless
    # of global RNG draws. The demo sim term (noise=0.01) jitters instead.
    torch.manual_seed(123)
    z = torch.randn(16, 2)
    first = float(pg.vicreg_faithful_loss(z))
    torch.randn(1000, 8)  # burn global draws
    assert float(pg.vicreg_faithful_loss(z)) == pytest.approx(first, rel=1e-9)
    torch.manual_seed(7)
    demo_a = float(vicreg_loss(z))
    torch.manual_seed(999)
    demo_b = float(vicreg_loss(z))
    assert demo_a != pytest.approx(demo_b)


def test_faithful_var_is_linear_hinge_not_squared():
    # Collapsed batch (std ~ 0): upstream owes mean(relu(t - sqrt(eps))),
    # the demo's squared hinge owes mean(relu(t - s)^2). At t=0.5 they differ.
    z = torch.zeros(8, 2)
    got = float(pg.vicreg_faithful_loss(z, target_std=0.5))
    want = float(F.relu(torch.tensor(0.5 - (pg.PG_EPS) ** 0.5)))
    assert got == pytest.approx(want, rel=1e-4)
    assert got != pytest.approx(want**2)
    demo = float(vicreg_loss(z, std_target=0.5, sim_weight=0.0, noise=0.0))
    # Demo std has no eps (unbiased=False std of zeros is exactly 0).
    assert demo == pytest.approx(10.0 * 0.5**2, rel=1e-4)  # default var_weight=10


def test_faithful_loss_is_fail_closed():
    with pytest.raises(ValueError):
        pg.vicreg_faithful_loss(torch.randn(1, 4))
    with pytest.raises(ValueError):
        pg.vicreg_faithful_loss(torch.randn(4))
    assert float(pg.vicreg_faithful_loss(torch.randn(3, 4))) > 0.0


# -- threading: locked path byte-identical ----------------------------------


def test_vicreg_fn_defaults_to_locked_demo_loss():
    # Same seed twice through the default path is bit-identical, with or
    # without passing vicreg_fn=None explicitly.
    field = Field2D()
    cfg = AdvConfig(steps=60, seed=0)
    a = train_lm_adv(field, cfg=cfg).delta(1.0)
    b = train_lm_adv(field, cfg=AdvConfig(steps=60, seed=0), vicreg_fn=None).delta(1.0)
    assert torch.equal(a, b)


# -- gate/smoke vs locked_shared ---------------------------------------------


def _field2d_row(cfg: AdvConfig, vicreg_fn=None) -> dict:
    residual = train_lm_adv(Field2D(), cfg=cfg, vicreg_fn=vicreg_fn)
    row = dict(score_residual(residual))
    row.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    return row


def test_smoke_field2d_locked_vs_faithful():
    locked = _field2d_row(AdvConfig(steps=STEPS, seed=0))
    arm_cfg = pg.pg_vicreg_faithful_cfg(AdvConfig(steps=STEPS, seed=0))
    faithful = _field2d_row(arm_cfg, vicreg_fn=pg.vicreg_faithful_loss)
    for row in (locked, faithful):
        for key in ("cos_slider_plus", "leak_ratio", "cos_plus_minus", "leak_frac"):
            assert torch.isfinite(torch.tensor(float(row[key]))), (key, row)
    # The smoke reports geometry, it does not gate: record both rows so the
    # note can quote them, and fail loudly only if a path diverged to NaN.
    assert isinstance(faithful["cos_slider_plus"], float)


def test_smoke_sheet_fit_locked_vs_faithful():
    field = leaky_field()
    locked_cfg = AdvConfig(steps=STEPS, seed=0)
    arm_cfg = pg.pg_vicreg_faithful_cfg(AdvConfig(steps=STEPS, seed=0))
    _, locked_stats = fit_adv(field, cfg=locked_cfg)
    _, faithful_stats = fit_adv(field, cfg=arm_cfg, vicreg_fn=pg.vicreg_faithful_loss)
    assert torch.isfinite(torch.tensor(float(locked_stats["g_loss"])))
    assert torch.isfinite(torch.tensor(float(faithful_stats["g_loss"])))
    assert faithful_stats["b_cap"] == pytest.approx(1.0)  # KEEP rides along
    assert locked_stats["b_cap"] == pytest.approx(1.0)


# -- propose-only boundary: never Music argv ----------------------------------


def test_faithful_weight_is_refused_under_music_arm_b():
    # Weight 1.0 at --parts 0 must fail the Arm B preset/require gates: the
    # arm is toy-only by design, never a silent Music retrain.
    preset = ["--prompts_file", "x.yaml", "--lm_target", "faithful_guard_e", "--adv_preset", "arm_b"]
    require = ["--prompts_file", "x.yaml", "--lm_target", "faithful_guard_e", "--require_arm_b"]
    with pytest.raises(SystemExit):
        parse_args(preset + ["--vicreg_weight", "1.0"])
    with pytest.raises(SystemExit):
        parse_args(require + ["--vicreg_weight", "1.0"])


def test_disposition_marks_music_and_live_as_drop():
    note = pg.disposition()
    assert note["b_cap"]["verdict"] == "KEEP"
    assert note["music_extras"]["verdict"] == "KEEP"
    assert note["vicreg_weight"]["verdict"] == "HOLD"
    assert note["music_trainer_argv"]["verdict"] == "DROP"
    assert note["live_defaults"]["verdict"] == "DROP"
