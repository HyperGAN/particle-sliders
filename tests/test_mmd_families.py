import pytest

from analysis.gan_bcap.mmd_game_20260905.family_bookkeeping import round_result, validate_budget


def entry(name, family, score, eligible=True):
    return dict(id=name, family=family, round='round-0004', score=score, eligible=eligible)


def test_non_mmd_win_does_not_claim_mmd_record():
    result = round_result({'new': entry('new', 'calibration', 1.1)}, 'round-0004',
        dict(mmd=.75, overall=.8), dict(mmd='old', overall='new'))
    assert result['outcome'] == 'new overall lead'
    assert result['gain_vs_previous_mmd'] is None
    assert result['gain_vs_previous_overall'] == pytest.approx(.3)
    assert result['candidates'] == ['new']


def test_non_mmd_beating_mmd_only_is_not_a_mmd_record():
    result = round_result({'new': entry('new', 'distillation', .78)}, 'round-0004',
        dict(mmd=.75, overall=.8), dict(mmd='old', overall='leader'))
    assert result['outcome'] == 'no new record'
    assert result['gain_vs_previous_mmd'] is None


def test_old_reference_is_not_the_rounds_new_experiment():
    result = round_result({'ref': entry('ref', 'reference', 1.2)}, 'round-0004',
        dict(mmd=.75, overall=.8), dict(mmd='old', overall='ref'))
    assert result['outcome'] == 'failed experiment'
    assert result['candidates'] == []


def test_ineligible_new_candidate_cannot_win():
    result = round_result({'new': entry('new', 'hybrid', 2., False)}, 'round-0004',
        dict(mmd=.75, overall=.8), dict(mmd='old', overall='leader'))
    assert result['outcome'] == 'ineligible candidate'


def test_zero_update_budget_requires_a_construction_family():
    validate_budget('calibration', 0)
    validate_budget('hybrid', 0)
    validate_budget('gan', 150)
    for family in ('mmd', 'gan', 'distillation'):
        with pytest.raises(ValueError, match='positive'):
            validate_budget(family, 0)
