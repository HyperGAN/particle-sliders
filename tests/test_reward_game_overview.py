"""Combined results must preserve cohort, stage and confirmation boundaries."""
from copy import deepcopy
from pathlib import Path

import pytest

from conceptmod.textsliders.reward_game.core import read, write, digest, scorecard, DEFAULT_SPEC
from conceptmod.textsliders.reward_game.overview import snapshot, render


@pytest.fixture
def campaign(tmp_path):
    game = read(DEFAULT_SPEC)
    for case in game['cases']:
        for control in case['controls'].values():
            control['ce'] = 2.
    def add_home(name=None, version=None):
        home = tmp_path/name if name else tmp_path
        current = deepcopy(version or game); spec = home/'spec.json'
        write(spec, current)
        record = dict(spec=str(spec), benchmark_sha256=digest(current))
        if name:
            record['parent_home'] = str(tmp_path)
        write(home/'game.json', record)
        write(home/'audit/off-pcm-v1/result.json', dict(passed=True))
        return home, current
    add_home()
    def add_card(home, current, run, deltas, stage):
        rows = {c['id']:dict(status='complete', reward=dict(valid=True, scalar=2.+d))
                for c,d in zip(current['cases'],deltas)}
        card = scorecard(current, rows, stage)
        card.update(benchmark_sha256=digest(current), run_id=run)
        write(home/'evaluations'/run/'scorecard.json', card)
        write(home/'evaluations'/run/'candidate.json', dict(path=f'/weights/{run}.safetensors',
                                                          multiplier=1., weights_sha256=run))
        return card
    return tmp_path, game, add_home, add_card


def test_consistency_precedes_mean_and_stages_stay_separate(campaign):
    root, game, add_home, add_card = campaign
    add_card(root, game, 'larger-mean', [.5]*13+[-.1]*3, 16)
    add_card(root, game, 'early', [10.]*4, 4)
    add_card(root, game, 'incomplete', [20.]*2, 4)
    child, child_game = add_home('acoustic-v1')
    add_card(child, child_game, 'consistent', [.2]*16, 16)
    result = snapshot(root)
    assert [r['label'] for r in result['full_development']] == ['consistent','larger-mean']
    assert [r['label'] for r in result['early_development']] == ['early']
    assert len(result['incomplete']) == 1
    assert result['incomplete'][0]['card']['valid_cases'] == 2
    assert not result['confirmed']
    assert not result['excluded']


@pytest.mark.parametrize('mismatch', ['control','gpu','reward','missing_off'])
def test_incompatible_child_is_not_combined(campaign, mismatch):
    root, game, add_home, add_card = campaign
    changed = deepcopy(game)
    if mismatch == 'control':
        changed['cases'][0]['controls']['off']['ce'] += .1
    elif mismatch == 'gpu':
        changed['cases'][0]['physical_gpu'] = 1-changed['cases'][0]['physical_gpu']
    elif mismatch == 'reward':
        changed['reward_spec']['version'] = 'changed'
    child, cg = add_home('acoustic-v1', changed)
    add_card(child, cg, 'child', [.2]*16, 16)
    if mismatch == 'missing_off':
        (child/'audit/off-pcm-v1/result.json').unlink()
    result = snapshot(root)
    assert not result['full_development']
    assert len(result['excluded']) == 1


def test_changed_scorecard_and_passing_fresh_do_not_promote(campaign):
    root, game, _, add_card = campaign
    card = add_card(root, game, 'tampered', [.2]*16, 16)
    card['comparisons']['off']['wins'] = 999
    write(root/'evaluations/tampered/scorecard.json', card)
    protocol = dict(candidate=dict(weights_sha256='fresh-candidate'))
    folder = root/'confirmation/first'
    write(folder/'protocol.json', protocol)
    write(folder/'scorecard.json', dict(protocol_sha256=digest(protocol), batch_pass=True))
    result = snapshot(root)
    assert not result['full_development']
    assert len(result['excluded']) == 1
    assert result['fresh'][0]['card']['batch_pass']
    assert not result['confirmed']


def test_page_escapes_labels_and_keeps_actual_denominators(campaign, tmp_path):
    root, game, _, add_card = campaign
    add_card(root, game, 'early', [.2]*4, 4)
    result = snapshot(root)
    result['early_development'][0]['label'] = '<script>alert(1)</script>'
    body = render(result, tmp_path)
    assert '&lt;script&gt;' in body and '<script>' not in body
    assert '>4/4<' in body
    assert 'No confirmed improving incumbent' in body
