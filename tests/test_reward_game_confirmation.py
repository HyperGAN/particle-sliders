import pytest

from conceptmod.textsliders.reward_game.confirmation_stats import family_interval, summarize


def bank():
    return [dict(id=f'f{f}-s{s}',family=f'f{f}',seed=s) for f in range(8) for s in (1,2)]


def test_uncertainty_resamples_whole_families():
    # Perfectly correlated seeds within a family must not create more sampling units.
    cases=bank()
    rows=[dict(c,valid=True,off=7.,original=7.,candidate=8. if int(c['family'][1:])<4 else 6.) for c in cases]
    card=summarize(rows,cases)
    interval=card['comparisons']['off']['interval']
    assert interval==family_interval({f'f{i}':1. if i<4 else -1. for i in range(8)})
    assert interval['low']==pytest.approx(-.75)
    assert interval['high']==pytest.approx(.75)
    assert summarize(rows[::-1],cases)['comparisons']==card['comparisons']
    assert not card['batch_pass']


def test_missing_or_invalid_case_never_gets_confirmation_interval():
    cases=bank();rows=[dict(c,valid=True,off=7.,original=7.,candidate=7.3) for c in cases]
    card=summarize(rows,cases)
    assert card['batch_pass'] and not card['research_complete']
    incomplete=summarize(rows[:-1],cases)
    assert not incomplete['batch_pass'] and incomplete['comparisons']['off']['interval'] is None
    rows[-1]['candidate']=float('nan')
    assert not summarize(rows,cases)['batch_pass']
    with pytest.raises(ValueError,match='duplicate'):
        summarize(rows+[rows[0]],cases)
