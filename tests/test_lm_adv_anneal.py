"""Clone arms 4/5 (propose-only): anneal + g_interp_cap CPU tests.

Locks the contract the task requires:

- locked defaults do NOT move: ``AdvConfig`` stays sample-point ``b_cap`` /
  anneal ``none``; the demo CLI (``run_lm_adv``) defaults to the same; the
  Music trainer argv (``ARM_B``) and the YuE2 locked recipe stay inert
  (``b_cap`` / ``none``).
- ParticleGAN-faithful math both propose arms need: the ``linear`` /
  ``delayed`` center schedule (incl. ``total_steps`` required) and the
  ``g_interp_cap`` interp-path geometry (kappa explicit, distinct from the
  sample-point cap).
- Honesty smoke vs locked on CPU toy fixtures: every propose card runs
  finite and reports pass/fail by the same gates — never silently.

CPU only. No Music GPU, no audio claims.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from analysis.slider2d.adv import AdvConfig, make_grad_regularizer
from analysis.slider2d.gan import default_cfg, score_adv_sheet, score_field2d
from analysis.slider2d.grad_regularizers import GradRegularizer
from analysis.slider2d.propose_anneal import CARDS, PROPOSE_ONLY
from analysis.slider2d.sheet import leaky_field


class _LinearCritic(nn.Module):
    """D(x) = w.x + b: input gradient is exactly w at every sample."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(x.shape[0], -1) @ self.w.reshape(-1)) + self.b


class _SquareCritic(nn.Module):
    """D(x) = (x.v)^2: gradient norm 2|x.v|*||v|| — flat at v-orthogonal points."""

    def __init__(self, v: torch.Tensor):
        super().__init__()
        self.v = nn.Parameter(v.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.reshape(x.shape[0], -1) @ self.v.reshape(-1)) ** 2)


# -- locked defaults do not move ------------------------------------------


def test_propose_module_is_marked_propose_only():
    assert PROPOSE_ONLY is True
    names = [c["name"] for c in CARDS]
    assert names[0] == "locked_b_cap_none"
    assert {c["grad_arm"] for c in CARDS} == {"b_cap", "g_interp_cap"}
    assert {c["target_anneal"] for c in CARDS} == {"none", "delayed"}


def test_adv_config_defaults_stay_locked():
    cfg = AdvConfig()
    assert cfg.grad_arm == "b_cap"
    assert cfg.target_anneal == "none"
    reg = make_grad_regularizer()
    assert (reg.arm, reg.target_anneal) == ("b_cap", "none")


def test_demo_cli_defaults_stay_locked():
    from analysis.slider2d import run_lm_adv

    args = run_lm_adv.build_parser().parse_args([])
    assert args.grad_arm == "b_cap"
    assert args.target_anneal == "none"
    assert args.teacher == run_lm_adv.DEFAULT_TEACHER
    assert run_lm_adv.collect.__kwdefaults__ == {
        "cover_weight": 1.5,
        "grad_arm": "b_cap",
        "target_anneal": "none",
    }


def test_music_trainer_argv_and_yue2_recipe_stay_locked():
    from conceptmod.textsliders.train_lm_slider_music3 import (
        ARM_B,
        make_music_grad_regularizer,
        parse_args,
    )
    from conceptmod.textsliders import yue2_arm_b as arm

    assert "target_anneal" not in ARM_B
    assert "grad_arm" not in ARM_B
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.adv_preset == "none"
    reg = make_music_grad_regularizer(args)
    assert (reg.arm, reg.target_anneal) == ("b_cap", "none")
    yue2_reg = arm.make_regularizer()
    assert (yue2_reg.arm, yue2_reg.target_anneal) == ("b_cap", "none")
    for key, want in ARM_B.items():
        assert arm.RECIPE[key] == want


# -- anneal schedule math ---------------------------------------------------


def test_linear_anneal_ramps_center_to_zero():
    reg = GradRegularizer(arm="b_cap", coeff=1.0, kappa=1.0, target_anneal="linear", total_steps=100)
    assert reg.center(0) == pytest.approx(1.0)
    assert reg.center(50) == pytest.approx(0.5)
    assert reg.center(100) == pytest.approx(0.0)
    assert reg.center(200) == pytest.approx(0.0)


def test_delayed_anneal_holds_sixty_percent_then_ramps():
    reg = GradRegularizer(arm="b_cap", coeff=1.0, kappa=2.0, target_anneal="delayed", total_steps=100)
    assert reg.center(0) == pytest.approx(2.0)
    assert reg.center(59) == pytest.approx(2.0)
    assert reg.center(60) == pytest.approx(2.0)
    assert reg.center(80) == pytest.approx(1.0)
    assert reg.center(100) == pytest.approx(0.0)


def test_anneal_requires_total_steps():
    with pytest.raises(ValueError):
        GradRegularizer(arm="b_cap", target_anneal="linear", total_steps=0)
    with pytest.raises(ValueError):
        GradRegularizer(arm="g_interp_cap", target_anneal="delayed", total_steps=0)


def test_make_grad_regularizer_threads_anneal_and_steps():
    cfg = default_cfg(steps=500, grad_arm="g_interp_cap", target_anneal="delayed")
    reg = make_grad_regularizer(cfg)
    assert reg.arm == "g_interp_cap"
    assert reg.target_anneal == "delayed"
    assert reg.total_steps == 500
    assert reg.center(0) == pytest.approx(1.0)
    assert reg.center(500) == pytest.approx(0.0)


# -- g_interp_cap geometry ----------------------------------------------------


def test_g_interp_cap_closed_form_on_linear_critic():
    # ||g|| = 5 everywhere (w = (3, 4)); kappa=1, coeff=1: relu(5-1)^2 = 16.
    d = _LinearCritic(torch.tensor([3.0, 4.0]))
    torch.manual_seed(0)
    xr = torch.randn(8, 2)
    xf = torch.randn(8, 2)
    reg = GradRegularizer(arm="g_interp_cap", coeff=1.0, kappa=1.0)
    assert float(reg(d, xr, xf).detach()) == pytest.approx(16.0, rel=1e-4)
    # kappa explicit, not hardcoded: kappa=2.5 -> relu(5-2.5)^2 = 6.25.
    reg2 = GradRegularizer(arm="g_interp_cap", coeff=1.0, kappa=2.5)
    assert float(reg2(d, xr, xf).detach()) == pytest.approx(6.25, rel=1e-4)
    # Free below kappa.
    d_small = _LinearCritic(torch.tensor([0.3, 0.4]))  # ||g|| = 0.5
    assert float(reg(d_small, xr, xf).detach()) == pytest.approx(0.0, abs=1e-9)


def test_g_interp_cap_matches_b_cap_on_coincident_poles_but_not_split_ones():
    # D(x) = (x.v)^2 with v = (1, 0): ||grad|| = 2|x0|.
    d = _SquareCritic(torch.tensor([1.0, 0.0]))
    b_cap = GradRegularizer(arm="b_cap", coeff=1.0, kappa=1.0)
    g_interp = GradRegularizer(arm="g_interp_cap", coeff=1.0, kappa=1.0)
    # Coincident poles at x0 = +1: every interp is also x0 = +1, so the
    # interp cap must equal the sample-point cap (both see ||g|| = 2).
    same = torch.ones(16, 2)
    assert float(b_cap(d, same, same).detach()) == pytest.approx(1.0, rel=1e-4)
    assert float(g_interp(d, same, same).detach()) == pytest.approx(1.0, rel=1e-4)
    # Split poles (+1 vs -1): samples still see ||g|| = 2, but the interp
    # path crosses x0 = 0 where D is flat — the interp penalty must differ.
    plus = torch.ones(16, 2)
    minus = -torch.ones(16, 2)
    b_split = float(b_cap(d, plus, minus).detach())
    g_split = float(g_interp(d, plus, minus).detach())
    assert b_split == pytest.approx(1.0, rel=1e-4)
    assert g_split != pytest.approx(b_split, rel=1e-2)


# -- honesty smoke vs locked (CPU toy, reduced steps) -------------------------


def _finite(row: dict, *keys: str) -> None:
    for key in keys:
        value = row.get(key)
        assert value is not None, key
        assert torch.isfinite(torch.tensor(float(value))), (key, value)


@pytest.mark.parametrize("card", CARDS, ids=[c["name"] for c in CARDS])
def test_every_propose_card_runs_finite_and_reports(card):
    cfg = default_cfg(
        steps=200, seed=0, grad_arm=card["grad_arm"], target_anneal=card["target_anneal"]
    )
    f2 = score_field2d(cfg)
    _finite(f2, "cos_slider_plus", "leak_ratio", "cos_plus_minus")
    assert "pass" in f2 and isinstance(f2["pass"], bool)
    sheet = score_adv_sheet(leaky_field(), cfg=cfg)
    _finite(sheet, "leak_tok", "on_sheet_kept")
    assert "pass" in sheet and isinstance(sheet["pass"], bool)
