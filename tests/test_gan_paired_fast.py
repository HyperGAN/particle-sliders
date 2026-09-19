from copy import deepcopy

import pytest

from analysis.gan_bcap.paired_fast import lyric_alarm


def examples():
    good = dict(lyric_diagnostics=dict(precision=.9, phrase_accuracy=.9, recall=.9))
    return [dict(seed=s, candidate=deepcopy(good), baseline=deepcopy(good),
                 positive_reference=deepcopy(good)) for s in (7, 23)]


def test_fast_check_does_not_stop_for_slow_coverage_or_one_failed_sample():
    rows = examples()
    for row in rows: row['candidate']['lyric_diagnostics']['recall'] = .1
    assert not lyric_alarm(rows)['stop']
    rows[0]['candidate']['lyric_diagnostics'].update(precision=.1, phrase_accuracy=.1)
    assert not lyric_alarm(rows)['stop']


def test_both_severe_samples_require_good_controls():
    rows = examples()
    for row in rows: row['candidate']['lyric_diagnostics'].update(precision=.1, phrase_accuracy=.1)
    assert lyric_alarm(rows)['stop']
    rows[1]['positive_reference']['lyric_diagnostics']['precision'] = .2
    assert not lyric_alarm(rows)['stop']


def test_duplicate_seed_is_not_two_observations():
    rows = examples(); rows[1]['seed'] = rows[0]['seed']
    with pytest.raises(ValueError, match='distinct'):
        lyric_alarm(rows)
