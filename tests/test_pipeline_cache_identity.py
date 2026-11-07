"""Cache adoption binds measured audio bytes and the measuring implementation."""
import json
import math
import os
from pathlib import Path
import struct
import wave

import pytest

from slider_pipeline import stages


def write_clip(path, gain=0.3):
    sr = 16000
    frames = b''.join(struct.pack('<hh', int(gain * 32767 * math.sin(2 * math.pi * 440 * n / sr)),
                                 int(gain * 32767 * math.sin(2 * math.pi * 442 * n / sr)))
                      for n in range(sr))
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sr)
        stream.writeframes(frames)


@pytest.fixture
def measured_folder(tmp_path, monkeypatch):
    write_clip(tmp_path/'ladder_zero.wav')
    write_clip(tmp_path/'ladder_plus1.wav', 0.4)
    measured = []
    original = stages.scan_folder
    def scan(*args, **kwargs):
        measured.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(stages, 'scan_folder', scan)
    return tmp_path, measured


def test_unchanged_audio_hits_cache_but_preserved_mtime_content_change_does_not(measured_folder):
    folder, measured = measured_folder
    first = stages._folder_to_cells(folder, 1.0)
    assert stages._folder_to_cells(folder, 1.0) == first
    assert len(measured) == 1
    clip = folder/'ladder_plus1.wav'
    stat = clip.stat()
    write_clip(clip, 0.2)
    os.utime(clip, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert clip.stat().st_size == stat.st_size
    assert clip.stat().st_mtime_ns == stat.st_mtime_ns
    changed = stages._folder_to_cells(folder, 1.0)
    assert len(measured) == 2
    assert changed[1.0].feats['rms_pc'] < first[1.0].feats['rms_pc']


@pytest.mark.parametrize('source', ['cells', 'features', 'instrument'])
def test_measurement_source_content_change_invalidates_cache(measured_folder, monkeypatch, source):
    folder, measured = measured_folder
    # Copy one implementation source without mutating any captured source.
    modules = {'cells': stages, 'features': stages.features, 'instrument': stages.features.bm}
    module = modules[source]
    copied = folder/'measurement.py'
    copied.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, '__file__', str(copied))
    stages._folder_to_cells(folder, 1.0)
    before = stages._measurement_identity()
    stat = copied.stat()
    with copied.open('a') as stream:
        stream.write('\n# changed measurement implementation\n')
    os.utime(copied, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    stages._folder_to_cells(folder, 1.0)
    assert len(measured) == 2
    assert stages._measurement_identity()['source_sha256'][source] != before['source_sha256'][source]


def test_numerical_runtime_and_requested_duration_invalidate_cache(measured_folder, monkeypatch):
    folder, measured = measured_folder
    stages._folder_to_cells(folder, 1.0)
    monkeypatch.setattr(stages.features.np, '__version__', 'different-numerical-runtime')
    stages._folder_to_cells(folder, 1.0)
    stages._folder_to_cells(folder, 1.05)
    assert len(measured) == 3


@pytest.mark.parametrize('damage', ['legacy', 'cells', 'list', 'json'])
def test_legacy_or_malformed_cache_is_measured_again(measured_folder, damage):
    folder, measured = measured_folder
    stages._folder_to_cells(folder, 1.0)
    cache = folder/'.pipeline_features.json'
    payload = json.loads(cache.read_text())
    if damage == 'legacy':
        payload.pop('measurement')
        payload.pop('schema')
    elif damage == 'cells':
        payload['cells'] = None
    elif damage == 'list':
        payload = []
    cache.write_text('{' if damage == 'json' else json.dumps(payload))
    stages._folder_to_cells(folder, 1.0)
    assert len(measured) == 2
    assert json.loads(cache.read_text())['schema'] == stages.FEATURE_CACHE_SCHEMA


def test_audio_changed_during_scan_is_not_adopted_or_cached(measured_folder, monkeypatch):
    folder, measured = measured_folder
    scan = stages.scan_folder
    def changing(*args, **kwargs):
        clips = scan(*args, **kwargs)
        write_clip(folder/'ladder_plus1.wav', 0.2)
        return clips
    monkeypatch.setattr(stages, 'scan_folder', changing)
    with pytest.raises(RuntimeError, match='audio changed during measurement'):
        stages._folder_to_cells(folder, 1.0)
    assert not (folder/'.pipeline_features.json').exists()


def test_missing_source_identity_and_cache_write_failure_leave_measurement_usable(measured_folder, monkeypatch):
    folder, measured = measured_folder
    write = Path.write_text
    def unavailable_cache(path, *args, **kwargs):
        if path.name == '.pipeline_features.json':
            raise OSError('read-only cache')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', unavailable_cache)
    result = stages._folder_to_cells(folder, 1.0)
    assert result[1.0].feats['rms_pc'] > 0
    assert not (folder/'.pipeline_features.json').exists()
    monkeypatch.setattr(stages.features.bm, '__file__', str(folder/'missing-instrument.py'))
    assert stages._folder_to_cells(folder, 1.0) == result
    assert len(measured) == 2
    assert not (folder/'.pipeline_features.json').exists()
