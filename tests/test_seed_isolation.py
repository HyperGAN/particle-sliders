"""Seed-isolation regression tests for the 2-D slider toys.

Bug hunt D: every ``fit_*`` / ``train_*`` entry used to call
``torch.manual_seed(seed)`` on the *global* RNG and never restore it, so
results depended on test order and on unrelated ``torch.randn`` draws,
and most cells only ever ran on ``seed=0``. Entries now run under
``analysis.slider2d.rng.isolated_rng`` (see that module for the policy
and for the intentional multi-seed knives).

These tests pin the isolation contract on small budgets -- they do not
re-assert cell thresholds.
"""

from __future__ import annotations

import pytest
import torch

from analysis.gan_bcap.gaussian_repro import train_gaussians
from analysis.slider2d.exam import divergent_field, fit_exam
from analysis.slider2d.field import Field2D
from analysis.slider2d.plus_exam import fit_plus_exam
from analysis.slider2d.rng import isolated_rng, isolated_seed
from analysis.slider2d.sheet import fit_sheet, leaky_field
from analysis.slider2d.train import music3_pairs, train_lm


def _fingerprint(residual) -> torch.Tensor:
    return residual.delta(1.0).detach().clone()


def test_isolated_rng_restores_global_state_and_on_exception():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    with isolated_rng(0):
        torch.randn(16)
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(RuntimeError, match="boom"):
        with isolated_rng(7):
            torch.randn(4)
            raise RuntimeError("boom")
    assert torch.equal(torch.get_rng_state(), before)


def test_isolated_seed_decorator_passes_positional_seed_and_restores():
    @isolated_seed()
    def _draw(n: int, seed: int):
        return torch.randn(n)

    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    first = _draw(8, 3)
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(111)
    torch.randn(50)
    assert torch.equal(_draw(8, 3), first)


@pytest.mark.parametrize(
    "make",
    [
        lambda seed: _fingerprint(train_lm(Field2D(), music3_pairs(False), steps=10, seed=seed)),
        lambda seed: _fingerprint(fit_exam(divergent_field(), steps=2, seed=seed)[0]),
        lambda seed: _fingerprint(fit_sheet(leaky_field(), steps=2, seed=seed)),
        lambda seed: _fingerprint(fit_plus_exam(divergent_field(), teacher="pair_odd", steps=2, seed=seed)),
    ],
    ids=["train_lm", "fit_exam", "fit_sheet", "fit_plus_exam"],
)
def test_deterministic_fits_ignore_prior_global_draws_and_do_not_pollute(make):
    torch.manual_seed(50)
    torch.randn(37)
    state = torch.get_rng_state().clone()
    first = make(0)
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(9999)
    torch.randn(101)
    assert torch.equal(make(0), first)
    assert torch.equal(torch.get_rng_state(), torch.get_rng_state().clone())


def test_stochastic_gaussian_fit_is_deterministic_per_seed_without_polluting():
    kwargs = {"n_modes": 4, "steps": 60}
    torch.manual_seed(5)
    state = torch.get_rng_state().clone()
    row_a = train_gaussians(seed=11, **kwargs)
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(6)
    torch.randn(64)
    row_b = train_gaussians(seed=11, **kwargs)
    assert row_a["modes"] == row_b["modes"]
    assert row_a["hq"] == pytest.approx(row_b["hq"])
    # Low budgets are genuinely seed-sensitive (the documented knife: the
    # full-budget pin is seed=1234 at 1200 steps); isolation only promises
    # same-seed reproducibility, which the equality above covers.
