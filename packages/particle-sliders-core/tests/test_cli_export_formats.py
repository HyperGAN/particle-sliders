"""Explicit artifact selection; authored pending an admitted CPU slot."""
from dataclasses import replace
import json

import pytest
import torch

from particle_sliders import lifecycle
from particle_sliders.inference import INFERENCE_V2
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from particle_sliders.session_cli import main
from test_compact_inference import plugin  # shared local staged fixture


@pytest.mark.parametrize('format,names', [
    ('v1', {'adapter.pt'}), ('v2', {'adapter-v2.pt'}), ('both', {'adapter.pt', 'adapter-v2.pt'}),
])
def test_explicit_cli_formats(tmp_path, plugin, monkeypatch, format, names):
    monkeypatch.setattr(lifecycle, 'backend', lambda name: plugin.TRAINING)
    config = SessionConfig('compact-fixture', 'cli-new-source-cohort',
        model_file=str(tmp_path / 'model.pt'), training_file=str(tmp_path / 'training.pt'), ema_decay=.9)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.as_dict()))
    output = tmp_path / 'run'
    main(['train', '--config', str(path), '--out', str(output), '--stop-after', '2',
          '--export-format', format, '--selection', 'ema'])
    assert {p.name for p in output.glob('adapter*.pt')} == names
    manifest = json.loads((output / 'manifest.json').read_text())
    assert set(manifest['files']) == names | {'training.pt', 'metrics.jsonl'}
    if format != 'v1':
        artifact = torch.load(output / 'adapter-v2.pt', weights_only=True)
        assert artifact['schema'] == INFERENCE_V2 and artifact['selection'] == 'ema'


def test_missing_v2_spec_fails_before_update_or_output(tmp_path, plugin, monkeypatch):
    prepare = plugin.TRAINING.prepare
    def without_spec(options):
        binding = prepare(options)
        binding.inference = None
        binding.game.step = lambda: pytest.fail('update before v2 specification admission')
        return binding
    product = replace(plugin.TRAINING, prepare=without_spec)
    monkeypatch.setattr(lifecycle, 'backend', lambda name: product)
    config = SessionConfig('compact-fixture', 'cli-new-source-cohort',
        model_file=str(tmp_path / 'model.pt'), training_file=str(tmp_path / 'training.pt'))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.as_dict()))
    output = tmp_path / 'run'
    with pytest.raises(UnsupportedOperation, match='InferenceSpec'):
        main(['train', '--config', str(path), '--out', str(output), '--export-format', 'v2'])
    assert not output.exists()
