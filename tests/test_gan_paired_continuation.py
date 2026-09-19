"""Operational stops must not turn subjective ASR flags into collapse claims."""
from copy import deepcopy

import pytest

from analysis.gan_bcap.paired_continuation import severe_audio_alarm


def observed():
    return dict(diversity={str(i): dict(status='no_collapse_detected') for i in range(8)},
        clips=[dict(diagnostics=dict(failures=['lyric_proxy_regression'])) for _ in range(32)])


def test_lyric_proxy_alone_does_not_stop_training():
    assert not severe_audio_alarm(observed())['stop']


def test_sparse_alarm_stays_visible_without_stopping():
    data = observed(); data['diversity']['0']['status'] = 'collapse_suspected'
    for row in data['clips'][:3]: row['diagnostics']['failures'].append('near_silence')
    assert not severe_audio_alarm(data)['stop']
    two_prompts = deepcopy(data); two_prompts['diversity']['1']['status'] = 'collapse_suspected'
    assert severe_audio_alarm(two_prompts)['stop']
    four_clips = deepcopy(data); four_clips['clips'][3]['diagnostics']['failures'].append('near_silence')
    assert severe_audio_alarm(four_clips)['stop']


def test_partial_screen_cannot_trigger_the_full_stop_rule():
    data = observed(); data['clips'].pop()
    with pytest.raises(ValueError, match='all eight'):
        severe_audio_alarm(data)


def lyric_examples():
    data = observed(); records = []
    for i, clip in enumerate(data['clips']):
        clip.update(fixture=str(i), prompt=str(i // 4))
        good = dict(lyric_diagnostics=dict(precision=.9, phrase_accuracy=.9, recall=.9))
        records.append(dict(fixture=str(i), candidate=deepcopy(good), baseline=deepcopy(good),
            positive_reference=deepcopy(good)))
    return data, records


def test_slower_lyric_coverage_does_not_trigger_gibberish_stop():
    data, records = lyric_examples()
    for row in records: row['candidate']['lyric_diagnostics']['recall'] = .1
    assert not severe_audio_alarm(data, records)['stop']


def test_widespread_wrong_words_with_good_references_stop_branch():
    data, records = lyric_examples()
    # Eight clips over three prompts; both word and phrase matching fail.
    for i in (0, 1, 2, 4, 5, 6, 8, 9):
        records[i]['candidate']['lyric_diagnostics'].update(precision=.2, phrase_accuracy=.1)
    result = severe_audio_alarm(data, records)
    assert result['stop'] and result['severe_lyric_mismatch_prompts'] == 3
    # ASR that also fails on the control cannot identify a generator failure.
    for row in records: row['baseline']['lyric_diagnostics']['precision'] = .2
    assert not severe_audio_alarm(data, records)['stop']
