"""The production GAN must face continuation gates, not just equation checks.

A green test of rejection is not a passing training recipe: the 600-step
production port currently fails the exam. The CLI reports this with exit 1.
"""
from copy import deepcopy

import pytest
import torch

from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_CELLS, fit_plus_neu_exam, score_plus_neu_exam, score_plus_neu_residual,
)
from analysis.slider2d.yue2_gan_exam import accepted, run_cell, ToySlider


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('cell', ['divergent', 'close'])
def test_extracted_scorer_preserves_existing_exam(cell):
    field = PLUS_NEU_CELLS[cell](seed=0)
    residual = fit_plus_neu_exam(field, teacher='faithful_plus_neu', plus_neu=True, steps=60)
    got = score_plus_neu_residual('control', field, residual,
        teacher='faithful_plus_neu', plus_neu=True)
    expected = score_plus_neu_exam('control', field,
        teacher='faithful_plus_neu', plus_neu=True, steps=60)
    assert got == expected
    assert got['hit']


def test_zero_is_exact_and_negative_scale_is_not_trained():
    slider = ToySlider(9)
    with torch.no_grad():
        slider.odd.fill_(2.)
        slider.even.fill_(3.)
    assert torch.equal(slider.delta(0.), torch.zeros(9))
    assert torch.equal(slider.snapshot().delta(1.), slider.delta(1.))
    with pytest.raises(ValueError, match='unipolar'):
        with slider.scaled(-1.):
            pass


def test_gate_requires_both_cells_every_seed_and_every_requested_budget():
    good = dict(cover=.93, off_caption=0., neu_hold=1.)
    results = [dict(seed=seed, cell=cell, control=dict(good),
                    checkpoints=[dict(step=600, **good), dict(step=3400, **good)])
               for seed in (0, 1, 7) for cell in ('divergent', 'close')]
    assert accepted(results, steps=[600, 3400])
    assert not accepted(results[:-1], steps=[600])
    assert not accepted(results, steps=[1200])
    assert not accepted(results, steps=[])
    assert not accepted([], steps=[600])
    bad = deepcopy(results)
    bad[-1]['checkpoints'][0].update(cover=.5, hit=True, pole_cos=1.)
    assert not accepted(bad, steps=[600, 3400])  # Later success cannot hide an earlier failure.
    assert accepted(bad, steps=[3400])
    bad = deepcopy(results)
    bad[-1]['control']['cover'] = .5
    assert not accepted(bad, steps=[600])


def test_current_600_step_production_update_is_rejected():
    before = torch.get_rng_state().clone()
    results = [run_cell(cell, steps=(600,), seed=0) for cell in ('divergent', 'close')]
    assert torch.equal(before, torch.get_rng_state())
    assert all(r['control']['hit'] for r in results)
    assert all(r['checkpoints'][0]['neu_hold'] == 1. for r in results)
    assert all(r['checkpoints'][0]['cover'] < .85 for r in results)
    assert not accepted(results, steps=[600], seeds=[0])
    assert all(h['loss'] == h['g_adv'] for r in results for h in r['history'])
