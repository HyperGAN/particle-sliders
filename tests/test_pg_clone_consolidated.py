"""Consolidated ParticleGAN clone salvage (propose-only, CPU toys).

Pins locked defaults (AdvConfig / production argv / ARM_B / live v9) and
covers the salvaged hooks: per-party LR mults + triplet, vicreg faithful
loss + vicreg_fn threading, grad-arm/anneal CLI + math, thin full-clone
preset. All five arms HOLD; locked KEEP.

CPU only. No Music GPU, no audio claims.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

from analysis.slider2d import locked_baseline_defaults as locked
from analysis.slider2d import pg_clone_propose as propose
from analysis.slider2d.adv import AdvConfig, make_grad_regularizer, toy_lr_triplet
from analysis.slider2d.gan import (
    DEFAULT_TEACHER,
    default_cfg,
    fit_adv,
    score_field2d,
    train_lm_adv,
)
from analysis.slider2d.grad_regularizers import GradRegularizer
from analysis.slider2d.run_lm_adv import build_parser
from analysis.slider2d.sheet import leaky_field
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B, parse_args


# -- propose-only markers -------------------------------------------------


def test_propose_only_markers():
    assert propose.PROPOSE_ONLY is True
    assert propose.MERGE_TO_TRAINER is False
    assert locked.FORMULATION_ARMS["pg_big_particles"]["status"] == "propose_only"


def test_disposition_all_hold_locked_keep():
    d = propose.disposition()
    assert d["locked"]["verdict"] == "KEEP"
    for arm in ("pg_big_particles", "pg_2x_lr", "pg_vicreg_faithful", "anneal_ginterp", "pg_full_clone"):
        assert d[arm]["verdict"] == "HOLD", arm


# -- locked pins -----------------------------------------------------------


def test_locked_matches_advconfig_defaults():
    cfg = AdvConfig()
    for key, value in locked.LOCKED.items():
        assert getattr(cfg, key) == value, key
    assert locked.LOCKED_TEACHER == DEFAULT_TEACHER == "faithful_guard_e"


def test_locked_matches_default_cfg_factory():
    cfg = default_cfg()
    for key, value in locked.LOCKED.items():
        assert getattr(cfg, key) == value, key


def test_production_argv_untouched():
    got = vars(build_parser().parse_args([]))
    got.pop("out")
    assert got == locked.PRODUCTION_ARGV_DEFAULTS


def test_live_music_row_untouched():
    assert ARM_B == {
        "lm_target": "faithful_guard_e",
        "adv_arch": "mlp",
        "adv_norm": "l2",
        "fm_weight": 0.0,
        "parts": 0,
        "pole_weight": 1.0,
        "cover_weight": 1.0,
        "adv_reg_kappa": 1.0,
        "adv_b_cap": 1.0,
        "vicreg_weight": 0.0,
    }
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.adv_preset == "none"
    assert "pg_full_clone" not in str(args.adv_preset)


def test_locked_cfg_rejects_formulation_overrides():
    with pytest.raises(ValueError):
        locked.locked_cfg(n_particles=64)
    with pytest.raises(ValueError):
        locked.locked_cfg(cover_weight=0.0)
    with pytest.raises(ValueError):
        locked.locked_cfg(lr=1e-2)
    assert locked.locked_cfg(steps=200).steps == 200
    assert locked.locked_cfg(seed=3).seed == 3


def test_arm_is_only_delta_vs_locked():
    cfg = locked.arm_cfg("pg_big_particles")
    assert cfg.n_particles == 64
    locked.assert_only_delta(cfg, "pg_big_particles")
    with pytest.raises(ValueError):
        locked.arm_cfg("no_such_arm")
    with pytest.raises(ValueError):
        locked.arm_cfg("pg_big_particles", n_particles=256)


# -- per-party LR (#112 salvage) -------------------------------------------


def test_locked_lr_defaults():
    cfg = AdvConfig()
    assert cfg.lr == pytest.approx(5.0e-3)
    assert cfg.d_lr_mult == pytest.approx(1.0)
    assert cfg.prior_lr_mult == pytest.approx(1.0)
    assert toy_lr_triplet(cfg) == pytest.approx((5.0e-3, 5.0e-3, 5.0e-3))


def test_2x_lr_values():
    cfg = propose.pg_2x_lr_cfg()
    assert cfg.lr == pytest.approx(1.0e-2)
    assert cfg.d_lr_mult == pytest.approx(1.5)
    assert cfg.prior_lr_mult == pytest.approx(1.0)
    assert propose.arm_lr_triplet() == pytest.approx((1.0e-2, 1.5e-2, 1.0e-2))


def test_2x_lr_deltas_are_lr_only():
    diff = {
        f.name
        for f in dataclasses.fields(AdvConfig)
        if getattr(AdvConfig(), f.name) != getattr(propose.pg_2x_lr_cfg(), f.name)
    }
    assert diff == {"lr", "d_lr_mult"}


def test_2x_lr_overrides_refused():
    with pytest.raises(ValueError):
        propose.pg_2x_lr_cfg(lr=5.0e-3)
    with pytest.raises(ValueError):
        propose.pg_2x_lr_cfg(d_lr_mult=1.0)
    with pytest.raises(ValueError):
        propose.pg_2x_lr_cfg(prior_lr_mult=10.0)
    assert propose.pg_2x_lr_cfg(steps=250).steps == 250


def test_lr_mults_fail_closed():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            toy_lr_triplet(default_cfg(d_lr_mult=bad))
        with pytest.raises(ValueError):
            toy_lr_triplet(default_cfg(prior_lr_mult=bad))


# -- VicReg faithful (#113 salvage) ----------------------------------------


def test_vicreg_faithful_matches_upstream_math():
    torch.manual_seed(0)
    z = torch.randn(16, 4)
    got = float(propose.vicreg_faithful_loss(z).detach())
    # Inline upstream transcription: linear-hinge var + cov, no sim.
    std_z = torch.sqrt(z.var(dim=0) + 1e-4)
    var = torch.relu(1.0 - std_z).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / float(z.shape[0] - 1)
    off = cov.pow(2).sum() - cov.diag().pow(2).sum()
    want = float((var + off / float(z.shape[1])).detach())
    assert got == pytest.approx(want, rel=1e-5, abs=1e-7)


def test_vicreg_faithful_fail_closed():
    with pytest.raises(ValueError):
        propose.vicreg_faithful_loss(torch.randn(1, 4))
    with pytest.raises(ValueError):
        propose.vicreg_faithful_loss(torch.randn(8, 4, 2))


def test_vicreg_weight_cfg_only_moves_weight():
    cfg = propose.pg_vicreg_faithful_cfg()
    assert cfg.vicreg_weight == pytest.approx(1.0)
    diff = {
        f.name
        for f in dataclasses.fields(AdvConfig)
        if getattr(AdvConfig(), f.name) != getattr(cfg, f.name)
    }
    assert diff == {"vicreg_weight"}


def test_vicreg_fn_none_is_locked_path():
    # The default None path uses the locked demo loss: explicit vicreg_loss
    # through the hook must match the None default run exactly (same seed).
    from analysis.slider2d.adv import vicreg_loss
    from analysis.slider2d.field import Field2D

    field = Field2D()
    a = train_lm_adv(field, cfg=default_cfg(steps=50, seed=0))
    b = train_lm_adv(field, cfg=default_cfg(steps=50, seed=0), vicreg_fn=vicreg_loss)
    assert torch.equal(a.delta(1.0), b.delta(1.0))
    torch.manual_seed(0)
    z = torch.randn(12, 2)
    assert float(propose.vicreg_faithful_loss(z)) != pytest.approx(float(vicreg_loss(z)))


def test_vicreg_fn_threading_deterministic():
    field_seed_cfg = default_cfg(steps=50, seed=0)
    from analysis.slider2d.field import Field2D

    field = Field2D()
    a = train_lm_adv(field, cfg=default_cfg(steps=50, seed=0), vicreg_fn=propose.vicreg_faithful_loss)
    b = train_lm_adv(field, cfg=default_cfg(steps=50, seed=0), vicreg_fn=propose.vicreg_faithful_loss)
    assert torch.equal(a.delta(1.0), b.delta(1.0))
    assert field_seed_cfg.vicreg_weight == pytest.approx(0.05)


# -- anneal / g_interp (#115 salvage: CLI + math pins) ----------------------


def test_adv_config_anneal_defaults_locked():
    cfg = AdvConfig()
    assert cfg.grad_arm == "b_cap"
    assert cfg.target_anneal == "none"
    reg = make_grad_regularizer()
    assert (reg.arm, reg.target_anneal) == ("b_cap", "none")


def test_demo_cli_anneal_defaults_locked():
    args = build_parser().parse_args([])
    assert args.grad_arm == "b_cap"
    assert args.target_anneal == "none"
    assert args.lr == pytest.approx(5.0e-3)
    assert args.d_lr_mult == pytest.approx(1.0)
    assert args.prior_lr_mult == pytest.approx(1.0)


def test_anneal_center_math():
    reg = GradRegularizer(arm="b_cap", coeff=1.0, kappa=1.0, target_anneal="linear", total_steps=100)
    assert reg.center(0) == pytest.approx(1.0)
    assert reg.center(50) == pytest.approx(0.5)
    assert reg.center(100) == pytest.approx(0.0)
    reg_d = GradRegularizer(arm="b_cap", coeff=1.0, kappa=2.0, target_anneal="delayed", total_steps=100)
    assert reg_d.center(0) == pytest.approx(2.0)
    assert reg_d.center(59) == pytest.approx(2.0)
    assert reg_d.center(80) == pytest.approx(1.0)
    assert reg_d.center(100) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        GradRegularizer(arm="b_cap", target_anneal="linear", total_steps=0)
    with pytest.raises(ValueError):
        GradRegularizer(arm="g_interp_cap", target_anneal="delayed", total_steps=0)


class _LinearCritic(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(x.shape[0], -1) @ self.w.reshape(-1)) + self.b


class _SquareCritic(nn.Module):
    def __init__(self, v: torch.Tensor):
        super().__init__()
        self.v = nn.Parameter(v.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(x.shape[0], -1) @ self.v.reshape(-1)) ** 2


def test_g_interp_cap_closed_form():
    d = _LinearCritic(torch.tensor([3.0, 4.0]))
    torch.manual_seed(0)
    xr = torch.randn(8, 2)
    xf = torch.randn(8, 2)
    reg = GradRegularizer(arm="g_interp_cap", coeff=1.0, kappa=1.0)
    assert float(reg(d, xr, xf).detach()) == pytest.approx(16.0, rel=1e-4)
    reg2 = GradRegularizer(arm="g_interp_cap", coeff=1.0, kappa=2.5)
    assert float(reg2(d, xr, xf).detach()) == pytest.approx(6.25, rel=1e-4)
    d_small = _LinearCritic(torch.tensor([0.3, 0.4]))
    assert float(reg(d_small, xr, xf).detach()) == pytest.approx(0.0, abs=1e-9)


def test_g_interp_differs_on_split_poles():
    d = _SquareCritic(torch.tensor([1.0, 0.0]))
    b_cap = GradRegularizer(arm="b_cap", coeff=1.0, kappa=1.0)
    g_interp = GradRegularizer(arm="g_interp_cap", coeff=1.0, kappa=1.0)
    same = torch.ones(16, 2)
    assert float(b_cap(d, same, same).detach()) == pytest.approx(1.0, rel=1e-4)
    assert float(g_interp(d, same, same).detach()) == pytest.approx(1.0, rel=1e-4)
    plus = torch.ones(16, 2)
    minus = -torch.ones(16, 2)
    assert float(b_cap(d, plus, minus).detach()) == pytest.approx(1.0, rel=1e-4)
    assert float(g_interp(d, plus, minus).detach()) != pytest.approx(1.0, rel=1e-2)


# -- full clone thin preset (#114 salvage) ----------------------------------


def test_full_clone_preset_values():
    cfg = propose.pg_full_clone_cfg()
    for key, want in propose.PG_FULL_CLONE.items():
        assert getattr(cfg, key) == want, key
    assert cfg.beta2 == pytest.approx(0.999)
    assert cfg.n_particles == 32
    assert cfg.vicreg_weight == pytest.approx(1.0)
    assert cfg.particle_l2 == pytest.approx(0.0)


def test_full_clone_ledger_covers_drifts():
    text = " ".join(" ".join(row) for row in propose.PG_FULL_CLONE_LEDGER)
    for knob in ("particles", "VICReg", "beta2", "LR hold", "EMA"):
        assert knob in text, knob


def test_full_clone_did_not_move_locked():
    cfg = AdvConfig()
    assert cfg.beta2 == pytest.approx(0.99)
    assert cfg.n_particles == 12
    assert cfg.vicreg_weight == pytest.approx(0.05)
    assert cfg.particle_l2 == pytest.approx(0.02)


# -- CPU smokes (thin, field2d 200-250 steps) --------------------------------


def _finite_field2d(row: dict) -> None:
    for key in ("cos_slider_plus", "leak_ratio", "cos_plus_minus", "leak_frac"):
        assert torch.isfinite(torch.tensor(float(row[key]))), key


def test_smoke_locked_vs_arms_field2d():
    base = score_field2d(locked.locked_cfg(steps=200, seed=0))
    lr_arm = score_field2d(propose.pg_2x_lr_cfg(steps=200, seed=0))
    vr_cfg = propose.pg_vicreg_faithful_cfg(default_cfg(steps=200, seed=0))
    from analysis.slider2d.field import Field2D

    field = Field2D()
    vr_res = train_lm_adv(field, cfg=vr_cfg, vicreg_fn=propose.vicreg_faithful_loss)
    from analysis.slider2d.train import score_residual

    vr_row = score_residual(vr_res)
    full = score_field2d(propose.pg_full_clone_cfg(steps=200, seed=0))
    _finite_field2d(base)
    _finite_field2d(lr_arm)
    _finite_field2d(full)
    assert base["pass"] is True
    assert lr_arm["pass"] is True
    assert full["pass"] is True
    for key in ("cos_slider_plus", "leak_ratio", "cos_plus_minus", "leak_frac"):
        assert torch.isfinite(torch.tensor(float(vr_row[key]))), key
    print(
        f"\nlocked slider={base['cos_slider_plus']:+.3f} leak={base['leak_ratio']:+.3f} "
        f"2x slider={lr_arm['cos_slider_plus']:+.3f} leak={lr_arm['leak_ratio']:+.3f} "
        f"full slider={full['cos_slider_plus']:+.3f} leak={full['leak_ratio']:+.3f}"
    )


def test_smoke_anneal_cards_finite():
    for grad_arm in ("b_cap", "g_interp_cap"):
        for anneal in ("none", "delayed"):
            cfg = default_cfg(steps=200, seed=0, grad_arm=grad_arm, target_anneal=anneal)
            row = score_field2d(cfg)
            _finite_field2d(row)


def test_collect_threads_propose_flags():
    from analysis.slider2d.run_lm_adv import collect

    blob = collect(
        steps=200,
        exam_steps=200,
        seed=0,
        teacher="faithful",
        b_cap=1.0,
        kappa=1.0,
        fm_weight=0.0,
        baseline_steps=50,
        cover_weight=1.5,
        lr=1e-2,
        d_lr_mult=1.5,
        prior_lr_mult=1.0,
        grad_arm="g_interp_cap",
        target_anneal="delayed",
    )
    assert blob["cfg"]["lr"] == pytest.approx(1e-2)
    assert blob["cfg"]["d_lr_mult"] == pytest.approx(1.5)
    assert blob["cfg"]["grad_arm"] == "g_interp_cap"
    assert blob["cfg"]["target_anneal"] == "delayed"


def test_fit_adv_vicreg_fn_smoke():
    field = leaky_field()
    _, stats_locked = fit_adv(field, cfg=default_cfg(steps=50, seed=0))
    _, stats_faithful = fit_adv(
        field, cfg=default_cfg(steps=50, seed=0), vicreg_fn=propose.vicreg_faithful_loss
    )
    assert stats_locked["steps"] == 50
    assert stats_faithful["steps"] == 50
