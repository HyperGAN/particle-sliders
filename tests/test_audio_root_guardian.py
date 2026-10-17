import json
from pathlib import Path

import pytest

import guarded_audio_pg09 as guardian


def test_bound_audio_card_rejects_source_and_native_environment_tamper():
    root=Path(__file__).resolve().parents[1]
    card=json.loads((root/'qualification/audio-reruns/yue2-smoke.json').read_text())
    assert guardian.validate(card)==Path(card['output_directory'])
    shadow=json.loads(json.dumps(card))
    shadow['environment']['PYTHONPATH']+=':/other/site-packages'
    with pytest.raises(ValueError,match='environment'):
        guardian.validate(shadow)
    shadow=json.loads(json.dumps(card))
    shadow['source_pins'][guardian.WORKER]='0'*64
    with pytest.raises(ValueError,match='source changed'):
        guardian.validate(shadow)


def test_only_one_cuda_child_within_own_process_group_is_permitted(monkeypatch):
    monkeypatch.setattr(guardian.os,'getpgid',lambda pid:{11:123,12:123,99:9}[pid])
    assert guardian.owned_gpu_processes({3196,11},123)==[dict(pid=11,pgid=123)]
    with pytest.raises(RuntimeError,match='unowned'):
        guardian.owned_gpu_processes({3196,99},123)
    with pytest.raises(RuntimeError,match='more than one'):
        guardian.owned_gpu_processes({3196,11,12},123)
