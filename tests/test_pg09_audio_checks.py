import numpy as np
import pytest

from conceptmod.textsliders.pg09_audio_checks import matched_waveform_checks


def waves():
    value = np.sin(np.linspace(0, 20, 800)).astype(np.float32) * .1
    return dict(base=value, off=value.copy(), half=value * 1.1,
                on=value * 1.2, positive_caption=value * 1.3)


def test_matched_outputs_preserve_zero_and_report_effect_without_quality_claim():
    result = matched_waveform_checks(waves(), sample_rate=8000, maximum_seconds=.1,
                                     truncated={'on': True})
    assert result['zero_replay_exact'] and result['adapter_changes_waveform']
    assert result['off_on_rms_difference'] > 0
    assert result['outputs']['on']['seconds'] == .1
    assert result['truncation_flags']['on']
    assert not result['semantic_quality_qualified']
    assert result['requires_listening_review']


def test_different_lengths_are_reported_and_bounds_are_enforced():
    values = waves()
    values['on'] = values['on'][:-1]
    result = matched_waveform_checks(values, sample_rate=8000, maximum_seconds=.1)
    assert not result['same_output_shape'] and result['off_on_rms_difference'] is None
    with pytest.raises(ValueError, match='duration bound'):
        matched_waveform_checks(values, sample_rate=8000, maximum_seconds=.05)


@pytest.mark.parametrize('failure', ['changed_off', 'nonfinite', 'empty', 'channels'])
def test_invalid_matched_output_fails_closed(failure):
    values = waves()
    if failure == 'changed_off':
        values['off'][0] += .01
    elif failure == 'nonfinite':
        values['on'][0] = np.nan
    elif failure == 'empty':
        values['on'] = np.empty((0, 2))
    else:
        values['on'] = np.ones((800, 3))
    with pytest.raises(ValueError):
        matched_waveform_checks(values, sample_rate=8000)
