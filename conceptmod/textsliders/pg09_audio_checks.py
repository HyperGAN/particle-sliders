"""Deterministic waveform checks for predeclared matched audio comparisons.

These measure software output integrity, not musical or semantic quality.
"""
import numpy as np


def _wave(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if (value.ndim != 2 or value.shape[1] not in (1, 2) or not value.shape[0]
            or not np.isfinite(value).all()):
        raise ValueError('expected nonempty finite [samples, mono/stereo] waveform')
    return value


def matched_waveform_checks(waves, *, sample_rate, maximum_seconds=None, truncated=None):
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('positive integer sample rate required')
    if set(waves) != {'base', 'off', 'half', 'on', 'positive_caption'}:
        raise ValueError('predeclared base/off/half/on/positive-caption comparison required')
    arrays = {name: _wave(value) for name, value in waves.items()}
    if not np.array_equal(arrays['base'], arrays['off']):
        raise ValueError('adapter-off failed exact base waveform replay')
    if maximum_seconds is not None and (not np.isfinite(maximum_seconds) or maximum_seconds <= 0):
        raise ValueError('positive finite duration bound required')
    checks = {}
    for name, value in arrays.items():
        seconds = len(value) / sample_rate
        if maximum_seconds is not None and seconds > maximum_seconds + 1 / sample_rate:
            raise ValueError('waveform exceeds admitted duration bound: ' + name)
        rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))
        checks[name] = dict(samples=len(value), channels=value.shape[1], seconds=seconds,
            rms=rms, peak=float(np.max(np.abs(value))),
            clipping_fraction=float(np.mean(np.abs(value) >= 1.)),
            effectively_silent=rms < 1e-6)
    same_shape = arrays['base'].shape == arrays['on'].shape
    difference = (float(np.sqrt(np.mean((arrays['on'].astype(np.float64)
                    - arrays['base'].astype(np.float64)) ** 2))) if same_shape else None)
    return dict(zero_replay_exact=True,
        adapter_changes_waveform=not np.array_equal(arrays['base'], arrays['on']),
        same_output_shape=same_shape, off_on_rms_difference=difference,
        sample_rate=sample_rate, maximum_seconds=maximum_seconds,
        truncation_flags=dict(truncated or {}), outputs=checks,
        semantic_quality_qualified=False, requires_listening_review=True)
