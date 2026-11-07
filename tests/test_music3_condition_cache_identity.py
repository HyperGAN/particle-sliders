"""Historical cache consumer over the shared helper; tiny byte files and CPU tensors."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from particle_sliders.textsliders import train_lora_music3 as training


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model_index.json').write_text('{"architecture":"fixture"}')
    for component in training.AR_COMPONENTS:
        folder = model / component
        folder.mkdir()
        (folder / 'config.json').write_text('{"width":2}')
        (folder / 'weights.bin').write_bytes(b'fixture-weights')
    calls = []
    monkeypatch.setattr(training, '_load_ar_pipeline', lambda *args: calls.append('load') or object())
    monkeypatch.setattr(training, '_attach_lm_slider', lambda *args: calls.append('attach') or object())
    def encode(*args, **kwargs):
        calls.append('encode')
        return torch.full((1, 3, 2), float(args[3]) + kwargs['lm_scale'])
    monkeypatch.setattr(training, '_encode_condition', encode)
    monkeypatch.setattr(training, '_unload_ar', lambda *args: None)
    monkeypatch.setattr(training, '_flush', lambda: None)
    prompt = SimpleNamespace(target='caption', positive='caption', negative='caption', neutral='caption', lyrics='words')
    def build(*, skip=False, duration=1.0001, adapter=None):
        return training.build_conditions([prompt], tmp_path / 'cache', duration, [7],
                    torch.device('cpu'), model, skip, False, lm_weights=adapter)
    return SimpleNamespace(model=model, cache=tmp_path / 'cache', calls=calls, build=build,
                           prompt=prompt, root=tmp_path)


def adapter_file(setup):
    path = setup.root / 'adapter.safetensors'
    save_file({'weight': torch.ones(1)}, path)
    path.with_suffix('.json').write_text(json.dumps({'rank': 8, 'weights_kind': 'raw'}))
    return path


def test_cache_hit_uses_same_identity_without_loading_the_host_or_adapter(setup):
    adapter = adapter_file(setup)
    first = setup.build(adapter=adapter)
    assert setup.calls.count('load') == 1 and setup.calls.count('attach') == 1
    assert setup.calls.count('encode') == 3
    setup.calls.clear()
    restored = setup.build(skip=True, adapter=adapter)
    assert setup.calls == []
    for name in ('target', 'positive', 'negative', 'neutral'):
        assert torch.equal(first[0][1][name], restored[0][1][name])
    assert len(list(setup.cache.glob('identity-v1-*.pt'))) == 3
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change', ['weight', 'architecture', 'setting', 'exact_duration'])
def test_changed_model_bytes_architecture_or_generation_settings_refuse_stale_hit(setup, monkeypatch, change):
    setup.build()
    captured = {path: path.read_bytes() for path in setup.cache.iterdir()}
    setup.calls.clear()
    duration = 1.0001
    if change == 'weight':
        (setup.model / 'language_model/weights.bin').write_bytes(b'changed-weights')
    elif change == 'architecture':
        (setup.model / 'language_model/config.json').write_text('{"width":3}')
    elif change == 'setting':
        monkeypatch.setattr(training, 'AR_RETRY_LIMIT', training.AR_RETRY_LIMIT + 1)
    else:
        duration = 1.0002  # The old rounded key aliased these two durations.
    with pytest.raises(FileNotFoundError, match='skip_ar'):
        setup.build(skip=True, duration=duration)
    assert setup.calls == []
    assert all(path.read_bytes() == value for path, value in captured.items())


@pytest.mark.parametrize('change', ['weights', 'architecture'])
def test_changed_adapter_content_refuses_stale_hit_even_at_the_same_path(setup, change):
    adapter = adapter_file(setup)
    setup.build(adapter=adapter)
    setup.calls.clear()
    if change == 'weights':
        save_file({'weight': torch.zeros(1)}, adapter)
    else:
        adapter.with_suffix('.json').write_text(json.dumps({'rank': 9, 'weights_kind': 'raw'}))
    with pytest.raises(FileNotFoundError, match='skip_ar'):
        setup.build(skip=True, adapter=adapter)
    assert setup.calls == []


def test_unidentified_historical_cache_is_retained_and_never_admitted(setup):
    setup.cache.mkdir()
    payload = 'caption\n---\nwords\n---\n1.000\n---\n7'
    old = setup.cache / (hashlib.sha256(payload.encode()).hexdigest()[:16] + '.pt')
    torch.save({'condition': torch.full((1, 3, 2), 99.)}, old)
    captured = old.read_bytes()
    with pytest.raises(FileNotFoundError, match='skip_ar'):
        setup.build(skip=True)
    assert old.read_bytes() == captured and setup.calls == []
    setup.build()
    assert old.read_bytes() == captured
    assert len(list(setup.cache.glob('identity-v1-*.pt'))) == 1


def test_mismatched_payload_at_identity_path_is_refused_without_loading_host(setup):
    setup.build()
    path, = setup.cache.glob('identity-v1-*.pt')
    payload = torch.load(path, weights_only=True)
    payload['identity']['settings']['retry_limit'] += 1
    torch.save(payload, path)
    captured = path.read_bytes()
    setup.calls.clear()
    with pytest.raises(ValueError, match='identity differs'):
        setup.build(skip=True)
    assert setup.calls == [] and path.read_bytes() == captured
