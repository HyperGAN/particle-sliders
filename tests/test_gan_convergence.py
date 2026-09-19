from copy import deepcopy

import pytest
import torch

from analysis.gan_bcap.convergence_probe import discriminator_search, improvement, tensors_equal
from analysis.gan_bcap.convergence_criteria import paired_gain_bound, stopping_decision, game_evidence
from conceptmod.textsliders import lm_adv as adv


def test_gain_includes_initial_checkpoint_and_reports_best_not_last():
    row = improvement([2., 1., 3.])
    assert row['available_improvement'] == 1.
    assert row['best_after_updates'] == 1
    assert improvement([2., 3.])['available_improvement'] == 0.
    with pytest.raises(ValueError):
        improvement([float('nan')])


def test_discriminator_search_learns_without_changing_source():
    torch.manual_seed(7)
    critic = adv.SpanTransformerD(4, width=8, n_layers=1, n_heads=2,
                                 in_mode='scaled', readout='mean_last')
    opt = torch.optim.Adam(critic.parameters(), lr=.003, betas=(0., .999))
    before = deepcopy(critic.state_dict())
    real = torch.randn(4, 3, 4) + 2.
    fake = torch.randn(4, 3, 4) - 2.
    results = discriminator_search(critic, opt.state_dict(), real, fake, (None, None),
                                  dict(adv_reg_coeff=0., adv_reg_kappa=1.), 12, adv.cap_penalty)
    assert len(results) == 4
    assert max(r['available_improvement'] for r in results) > .01
    assert tensors_equal(before, critic.state_dict())
    assert not opt.state
    assert all(len(r['trace']) == 13 for r in results)


def test_identical_inputs_have_equal_relativistic_scores():
    scores = torch.tensor([-10., .3, 15.])
    assert float(adv.rp_d_loss(scores, scores)) == pytest.approx(.69314718)


def fixture_data(gains, seeds=4):
    groups = {(i,s):i for i in range(len(gains)) for s in range(seeds)}
    candidate = {key:gains[key[0]] for key in groups}
    baseline = {key:0. for key in groups}
    return candidate, baseline, groups


def test_seeds_cannot_substitute_for_prompt_coverage():
    r = paired_gain_bound(*fixture_data([0., 0.], seeds=100))
    assert r['decision']=='evaluate_more'
    assert r['upper_gain_bound'] is None


def test_no_significant_gain_is_not_evidence_of_no_worthwhile_gain():
    r = paired_gain_bound(*fixture_data([-.5, .5]*4))
    assert r['mean_gain']==0.
    assert r['upper_gain_bound'] > .05
    assert r['decision']=='evaluate_more'


def test_clear_plateau_and_clear_gain():
    assert paired_gain_bound(*fixture_data([0.]*8))['decision']=='no_worthwhile_gain'
    assert paired_gain_bound(*fixture_data([.1]*8))['decision']=='worthwhile_gain'


def test_more_comparisons_require_more_evidence():
    data = fixture_data([.00,.01,.02,.03,.04,.05,.06,.07])
    a = paired_gain_bound(*data, planned_comparisons=1)
    b = paired_gain_bound(*data, planned_comparisons=12)
    assert b['upper_gain_bound'] > a['upper_gain_bound']
    with pytest.raises(ValueError):
        paired_gain_bound({'a':1.}, {'b':1.}, {'a':'prompt'})


def test_quality_failures_and_budget_stops_are_never_convergence():
    plateau = [dict(decision='no_worthwhile_gain')]*3
    assert stopping_decision(plateau, quality='fail')=='stalled_change_recipe'
    assert stopping_decision(plateau, quality='unvalidated')=='validate_judge_and_quality'
    assert stopping_decision(plateau, quality='pass')=='run_reserved_confirmation'
    assert stopping_decision(plateau, quality='pass', independent_confirmation=True)=='stop_keep_best'
    assert stopping_decision([], quality='pass', budget_exhausted=True)=='budget_stopped_not_converged'
    assert stopping_decision(plateau, quality='pass', unstable=True)=='stop_unstable_keep_previous'


def test_small_unilateral_gain_is_not_a_certificate():
    assert game_evidence([{'fractional_improvement':0.}], [{'available_improvement':0.}])==\
        'small_observed_gain_requires_stronger_probes'
    assert game_evidence([{'fractional_improvement':.9}], [{'available_improvement':0.}])==\
        'not_stationary_under_tested_tolerances'
