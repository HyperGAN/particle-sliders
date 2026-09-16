"""Music Arm B (Strategy B): ParticleGAN-faithful b_cap + trainer flags.

The Music trainer's gradient penalty must call ParticleGAN's real
``GradRegularizer`` (vendored verbatim at
``analysis/slider2d/grad_regularizers.py``) with kappa explicit, L2
``n = sqrt(sum g^2 + 1e-12)``, ``(coeff/2)(E_r + E_f)`` — not a
CLI-defaults tweak of the old thinned cap (kappa hardcoded, no
``GradRegularizer``, no epsilon).

CPU only. No Music GPU train: these tests cover the penalty math on
synthetic grads and the trainer flag contract.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from analysis.slider2d import grad_regularizers as gr
from analysis.slider2d.adv import AdvConfig, cap_penalty, make_grad_regularizer
from analysis.slider2d.grad_regularizers import GradRegularizer
from conceptmod.textsliders.train_lm_slider_music3 import (
    ARM_B,
    make_music_grad_regularizer,
    parse_args,
)


class _LinearCritic(nn.Module):
    """D(x) = w.x + b: input gradient is exactly w at every sample."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(x.shape[0], -1) @ self.w.reshape(-1)) + self.b


def _arm_b_argv(*extra: str) -> list[str]:
    return [
        "--prompts_file",
        "x.yaml",
        "--lm_target",
        "faithful_guard_e",
        "--adv_preset",
        "arm_b",
        *extra,
    ]


# -- penalty math calls ParticleGAN's b_cap ---------------------------------


def test_b_cap_matches_particlegan_phi_on_synthetic_grads():
    # ||g|| = 5 everywhere (w = (3, 4)); kappa=1, coeff=1:
    # (1/2)(relu(5-1)^2 + relu(5-1)^2) = 16.
    d = _LinearCritic(torch.tensor([3.0, 4.0]))
    xr = torch.randn(6, 2)
    xf = torch.randn(6, 2)
    reg = GradRegularizer(arm="b_cap", coeff=1.0, kappa=1.0)
    pen, stats = reg.penalty(d, xr, xf)
    assert float(pen.detach()) == pytest.approx(16.0, rel=1e-5)
    assert stats["applied"] is True
    assert stats["center"] == pytest.approx(1.0)

    # Non-default kappa/coeff prove kappa is explicit, not hardcoded:
    # (0.5/2)(relu(5-2.5)^2 + relu(5-2.5)^2) = 3.125.
    reg2 = GradRegularizer(arm="b_cap", coeff=0.5, kappa=2.5)
    assert float(reg2(d, xr, xf).detach()) == pytest.approx(3.125, rel=1e-5)


def test_b_cap_shim_agrees_with_closed_form_on_synthetic_grad_tensors():
    # Rows with ||g|| = 0.4 (free) and 1.6 (capped): with kappa=1, coeff=1,
    # mean phi per side = (0 + 0.6^2)/2 = 0.18, penalty = (1/2)(0.18+0.18).
    low = torch.tensor([[0.4, 0.0]])
    high = torch.tensor([[1.6, 0.0]])
    grads = torch.cat([low, high], dim=0)
    assert float(cap_penalty(grads, grads, coeff=1.0, kappa=1.0)) == pytest.approx(
        0.18, rel=1e-5
    )
    # Zero grads: n = sqrt(1e-12) ~ 1e-6 < kappa -> exactly free.
    zero = torch.zeros(4, 3)
    assert float(cap_penalty(zero, zero)) == pytest.approx(0.0, abs=1e-9)
    # kappa explicit: kappa=0.2 also caps the 0.4 row.
    got = float(cap_penalty(grads, grads, coeff=1.0, kappa=0.2))
    want = 0.5 * (
        ((0.4 - 0.2) ** 2 + (1.6 - 0.2) ** 2) / 2.0
        + ((0.4 - 0.2) ** 2 + (1.6 - 0.2) ** 2) / 2.0
    )
    assert got == pytest.approx(want, rel=1e-5)


def test_cap_penalty_routes_through_grad_regularizer_phi():
    torch.manual_seed(0)
    d = _LinearCritic(torch.randn(4))
    xr = torch.randn(8, 4)
    xf = torch.randn(8, 4)
    for coeff, kappa in ((1.0, 1.0), (1.5, 0.7), (2.0, 2.5)):
        reg = GradRegularizer(arm="b_cap", coeff=coeff, kappa=kappa)
        want = float(reg(d, xr, xf).detach())
        with torch.enable_grad():
            xrd = xr.detach().requires_grad_(True)
            xfd = xf.detach().requires_grad_(True)
            gr_r = torch.autograd.grad(d(xrd).sum(), xrd, create_graph=False)[0]
            gr_f = torch.autograd.grad(d(xfd).sum(), xfd, create_graph=False)[0]
        got = float(cap_penalty(gr_r.detach(), gr_f.detach(), coeff=coeff, kappa=kappa))
        assert got == pytest.approx(want, rel=1e-5)


def test_make_grad_regularizer_defaults_to_arm_b_shape():
    reg = make_grad_regularizer()
    assert reg.arm == "b_cap"
    assert reg.coeff == pytest.approx(1.0)
    assert reg.kappa == pytest.approx(1.0)
    assert reg.norm == "l2"
    cfg = AdvConfig(kappa=0.5, b_cap=2.0)
    reg2 = make_grad_regularizer(cfg)
    assert reg2.kappa == pytest.approx(0.5)
    assert reg2.coeff == pytest.approx(2.0)


def test_grad_regularizer_source_pinned():
    src = inspect.getsource(gr)
    assert "https://github.com/255BITS/ParticleGAN" in src
    assert "particlegan 0.2.0" in src
    assert "b_cap" in GradRegularizer.ARMS
    sig = inspect.signature(GradRegularizer.__init__)
    assert "kappa" in sig.parameters
    assert sig.parameters["kappa"].default == 1.0
    assert sig.parameters["norm"].default == "l2"
    assert gr.GradientPenalty is GradRegularizer


# -- Music trainer flag contract --------------------------------------------


def test_arm_b_preset_carries_winning_config():
    args = parse_args(_arm_b_argv())
    for key, want in ARM_B.items():
        got = getattr(args, key)
        if isinstance(want, float):
            assert float(got) == pytest.approx(want), key
        else:
            assert got == want, key
    reg = make_music_grad_regularizer(args)
    assert reg.arm == "b_cap"
    assert reg.coeff == pytest.approx(1.0)
    assert reg.kappa == pytest.approx(1.0)
    assert reg.norm == "l2"


def test_music_grad_regularizer_matches_closed_form():
    args = parse_args(_arm_b_argv())
    reg = make_music_grad_regularizer(args)
    d = _LinearCritic(torch.tensor([3.0, 4.0]))
    xr = torch.randn(5, 2)
    xf = torch.randn(5, 2)
    assert float(reg(d, xr, xf).detach()) == pytest.approx(16.0, rel=1e-5)


def test_adv_arch_none_is_inert():
    args = parse_args(
        ["--prompts_file", "x.yaml", "--adv_arch", "none"]
    )
    reg = make_music_grad_regularizer(args)
    d = _LinearCritic(torch.tensor([3.0, 4.0]))
    xr = torch.randn(5, 2).requires_grad_(True)
    xf = torch.randn(5, 2).requires_grad_(True)
    assert float(reg(d, xr, xf).detach()) == pytest.approx(0.0, abs=1e-9)


def test_tx_with_guard_is_refused():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prompts_file",
                "x.yaml",
                "--lm_target",
                "faithful_guard_e",
                "--adv_arch",
                "tx",
            ]
        )


def test_arm_b_preset_reports_drift_instead_of_silent_retrain():
    with pytest.raises(SystemExit):
        parse_args(_arm_b_argv("--fm_weight", "0.5"))
    with pytest.raises(SystemExit):
        parse_args(_arm_b_argv("--adv_reg_kappa", "0.5"))


def test_live_defaults_unchanged():
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.adv_preset == "none"
    assert args.fm_weight == pytest.approx(0.0)
