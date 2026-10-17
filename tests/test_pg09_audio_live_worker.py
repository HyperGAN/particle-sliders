from pathlib import Path
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_pg09_audio import equal


def test_bounded_worker_exact_fresh_process_resume(tmp_path):
    driver = Path(__file__).resolve().parents[1] / 'scripts/qualify_audio_pg09_live.py'
    full, first, resumed = [tmp_path / name for name in ('full', 'first', 'resumed')]
    base = [sys.executable, str(driver), '--family', 'music3', '--fixture',
            '--device', 'cpu', '--total-steps', '4']
    for extra in (['--until', '3', '--out', str(full)],
                  ['--until', '1', '--out', str(first)],
                  ['--until', '3', '--out', str(resumed), '--resume', str(first / 'checkpoint.pt')]):
        result = subprocess.run([*base, *extra], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    equal(torch.load(full / 'qualification.pt', weights_only=True),
          torch.load(resumed / 'qualification.pt', weights_only=True))
    for directory in (full, first, resumed):
        progress = json.loads((directory / 'progress.json').read_text())
        assert progress['actual_bank_weights_changed'] is True
        assert progress['actual_bank_delta_rms'] > 0
    # Reconstruct from an imported module, as a separate renderer would do.
    from conceptmod.textsliders.pg09_audio import AudioResponseCritic
    from pg09_audio_cpu_fixtures import make_audio
    adapter, _, fixed, game = make_audio('music3', checkpointing=True,
                                       critic_class=AudioResponseCritic)
    game.load_export_state_dict(torch.load(full / 'adapter-export.pt', weights_only=True))
    packet = torch.load(full / 'qualification.pt', weights_only=True)
    from dataclasses import replace
    for prepared, expected in zip(fixed, packet['probes']):
        heldout = adapter.prepare(replace(prepared.row,
            neutral_ids=(61,) + prepared.row.neutral_ids[1:]))
        assert torch.equal(adapter.sample(heldout), expected)


def test_native_worker_without_admission_stops_before_model_load(tmp_path):
    driver = Path(__file__).resolve().parents[1] / 'scripts/qualify_audio_pg09_live.py'
    output = tmp_path / 'not-admitted'
    result = subprocess.run([sys.executable, str(driver), '--family', 'music3',
        '--device', 'cuda:0', '--until', '3', '--out', str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'explicit owner admission' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('failure', ['directory', 'files', 'loaded_identity', 'constructor_arguments'])
def test_yue2_manifest_is_checked_and_native_model_id_is_preserved(tmp_path, monkeypatch, failure):
    # Stop before any model construction; exercise the live loader's provenance boundary.
    import qualify_audio_pg09_live as worker
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    expected = {'files': {'model.safetensors': {'sha256': 'expected', 'bytes': 2}},
                'config_sha256': 'expected-config'}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(model_identity=expected, cache_dir=str(tmp_path),
        records={'model': dict(model='m-a-p/YuE2-3B', revision='pinned', path=str(model_dir))})))
    calls = []
    class BeforeWeights(Exception):
        pass
    def constructor(model_id, **kwargs):
        calls.append((model_id, kwargs))
        assert model_id == 'm-a-p/YuE2-3B'
        assert kwargs == dict(device='cuda:0', allow_hub=False,
                              revision='pinned', cache_dir=str(tmp_path))
        if failure == 'loaded_identity':
            return SimpleNamespace(identity={'wrong': True})
        raise BeforeWeights()
    modules = {name: ModuleType(name) for name in ('yue2', 'yue2.backend', 'yue2.slider', 'yue2.storage')}
    modules['yue2'].__path__ = []
    modules['yue2.backend'].YuE2Backend = constructor
    modules['yue2.slider'].ParticleSlider = None
    modules['yue2.storage'].resolve_model = lambda *a, **k: model_dir
    modules['yue2.storage'].model_identity = lambda p: ({'wrong': True} if failure == 'files' else expected)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(worker, 'bootstrap_yue2_product', lambda p: None)
    args = SimpleNamespace(family='yue2', product=tmp_path, manifest=manifest,
        model_dir=(tmp_path / 'different' if failure == 'directory' else model_dir),
        device='cuda:0')
    if failure == 'constructor_arguments':
        with pytest.raises(BeforeWeights):
            worker.load_native(args, {}, lambda: None)
        assert calls
    else:
        with pytest.raises(ValueError, match='YuE2'):
            worker.load_native(args, {}, lambda: None)
        assert bool(calls) == (failure == 'loaded_identity')
