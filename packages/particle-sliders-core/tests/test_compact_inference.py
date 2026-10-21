"""Native-only inference: no training data, game, critic or optimizer in fresh Python."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from particle_sliders.inference import INFERENCE_V2, open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, read_packet
from particle_sliders.pg09 import _hash


@pytest.fixture(autouse=True)
def cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def plugin(tmp_path):
    source = Path(__file__).parent / 'fixtures/compact_backend.py'
    module_spec = importlib.util.spec_from_file_location('compact_fixture', source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules['compact_fixture'] = module
    module_spec.loader.exec_module(module)
    with torch.random.fork_rng():
        torch.manual_seed(14)
        model = torch.nn.Conv2d(2, 2, 1, bias=False)
        torch.save(model.state_dict(), tmp_path / 'model.pt')
        torch.save([torch.rand(1, 2, 10, 10) for _ in range(4)], tmp_path / 'training.pt')
    return module


@pytest.fixture
def exported(tmp_path, plugin):
    config = SessionConfig('compact-fixture', 'new-compact-v2', model_file=str(tmp_path / 'model.pt'),
                           training_file=str(tmp_path / 'training.pt'), ema_decay=.9)
    with plugin.TRAINING.open(config) as session:
        session.train(stop_after=2)
        before = session.checkpoint()
        raw = session.export_inference(tmp_path / 'raw.pt')
        ema = session.export_inference(tmp_path / 'ema.pt', selection='ema')
        assert before['sha256'] == session.checkpoint()['sha256']
        raw_expected = session.game.backend.response(strength=.5).detach()
        weights = deepcopy(session.game.network.state_dict())
        session.game.network.load_state_dict(session.ema.values)
        ema_expected = session.game.backend.response(strength=.5).detach()
        session.game.network.load_state_dict(weights)
    return raw, ema, raw_expected, ema_expected


def test_export_has_no_training_payload_and_resume_remains_separate(tmp_path, plugin, exported):
    raw, ema, _, _ = exported
    assert raw['schema'] == ema['schema'] == INFERENCE_V2
    assert raw['selection'] == 'raw' and ema['selection'] == 'ema'
    text = json.dumps({key: value for key, value in ema.items() if key != 'network'})
    for forbidden in ('training_file', 'training.pt', 'rows_sha256', 'game_signature', 'optimizer', 'critic', 'recipe'):
        assert forbidden not in text
    assert not torch.equal(raw['network']['weights.0'], ema['network']['weights.0'])
    with pytest.raises(CheckpointError, match='version'):
        read_packet(tmp_path / 'ema.pt')
    with open_inference(tmp_path / 'ema.pt', options={'model_file': str(tmp_path / 'model.pt')},
                        backend=plugin.INFERENCE) as inference:
        with pytest.raises(UnsupportedOperation, match='resume'):
            inference.train()
        assert not hasattr(inference, 'game')


def test_fresh_installed_plugin_without_training_data(tmp_path, plugin, exported):
    raw, ema, raw_expected, ema_expected = exported
    (tmp_path / 'training.pt').unlink()
    torch.save({'raw': raw_expected, 'ema': ema_expected}, tmp_path / 'expected.pt')
    # A tiny installed wheel-style plugin with both entry-point groups. Source
    # code is copied into the isolated child installation, not imported from tests.
    installed = tmp_path / 'installed'
    installed.mkdir()
    archive = tmp_path / 'compact_fixture-1-py3-none-any.whl'
    with zipfile.ZipFile(archive, 'w') as wheel:
        wheel.write(Path(__file__).parent / 'fixtures/compact_backend.py', 'compact_fixture.py')
        wheel.writestr('compact_fixture-1.dist-info/METADATA', 'Metadata-Version: 2.1\nName: compact-fixture\nVersion: 1\n')
        wheel.writestr('compact_fixture-1.dist-info/entry_points.txt',
            '[particle_sliders.image_backends]\ncompact-fixture=compact_fixture:TRAINING\n'
            '[particle_sliders.inference_backends]\ncompact-fixture=compact_fixture:INFERENCE\n')
    with zipfile.ZipFile(archive) as wheel:
        wheel.extractall(installed)
    import particle_sliders, particlegan
    code = '''
import sys, json
from pathlib import Path
sys.path[:0] = json.loads(sys.argv[1])
import torch
from particle_sliders.inference import open_inference
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.field_critic import FieldCritic
import compact_fixture
root = Path(sys.argv[2])
assert not (root / 'training.pt').exists()
def forbidden(*args, **kwargs):
    raise AssertionError('training/critic/optimizer construction during inference')
ConditionalPG09.__init__ = forbidden
FieldCritic.__init__ = forbidden
torch.optim.Optimizer.__init__ = forbidden
compact_fixture.TRAINING.prepare.__code__ = forbidden.__code__
expected = torch.load(root / 'expected.pt', weights_only=True)
before = torch.get_rng_state().clone()
for selection in ('raw', 'ema'):
    with open_inference(root / (selection + '.pt'), options={'model_file': str(root / 'model.pt')}) as session:
        assert session.selection == selection
        assert torch.equal(session.render(strength=.5), expected[selection])
        assert torch.equal(session.response(strength=.5), expected[selection])
        for strength in (-.5, 0., .5, 1.):
            assert torch.isfinite(session.render(strength=strength)).all()
        assert not any(hasattr(session, name) for name in ('game', 'rows', 'critic', 'optimizer'))
assert torch.equal(before, torch.get_rng_state())
assert not torch.cuda.is_initialized()
from particle_sliders.session_cli import main
(root / 'local.json').write_text(json.dumps({'model_file': str(root / 'model.pt')}))
main(['infer', '--artifact', str(root / 'ema.pt'), '--runtime', str(root / 'local.json'),
      '--mode', 'response', '--strength', '.5', '--out', str(root / 'response.pt')])
result = torch.load(root / 'response.pt', weights_only=True)
assert result['selection'] == 'ema' and torch.equal(result['value'], expected['ema'])
assert (root / 'response.pt.json').is_file()
print('fresh native-only raw/EMA inference without training data passed')
'''
    paths = [str(installed), str(Path(particle_sliders.__file__).parent.parent), str(Path(particlegan.__file__).parent.parent)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH='')
    result = subprocess.run([sys.executable, '-I', '-c', code, json.dumps(paths), str(tmp_path)],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('damage', ['checksum', 'model', 'architecture', 'protocol', 'source', 'tensor', 'selection'])
def test_invalid_inference_is_atomic_and_cleans_up(tmp_path, plugin, exported, damage):
    packet = deepcopy(exported[0])
    closed_before = len(plugin.CLOSED)
    if damage == 'checksum':
        packet['sha256'] = 'wrong'
    elif damage == 'selection':
        packet['selection'] = 'unknown'
    elif damage == 'source':
        packet['specification']['factory_sources']['prepare'] = 'wrong'
    elif damage == 'tensor':
        packet['network']['weights.0'].fill_(float('nan'))
    elif damage == 'model':
        packet['specification']['identity']['model']['sha256'] = 'wrong'
    elif damage == 'architecture':
        packet['specification']['identity']['network_schema']['weights.0'][0] = [999]
    else:
        packet['specification']['identity']['protocol'] = 'other-v1'
    if damage not in ('checksum', 'tensor'):
        packet['sha256'] = _hash({key: value for key, value in packet.items() if key != 'sha256'})
    with pytest.raises((CheckpointError, ValueError)):
        open_inference(packet, options={'model_file': str(tmp_path / 'model.pt')}, backend=plugin.INFERENCE)
    if damage in ('model', 'architecture', 'protocol'):
        assert len(plugin.CLOSED) == closed_before + 1


def test_preflight_unsupported_before_native_allocation(tmp_path, plugin, exported):
    before = len(plugin.CLOSED)
    with pytest.raises(UnsupportedOperation):
        open_inference(exported[0], options={}, backend=plugin.INFERENCE, output_type='pil')
    assert len(plugin.CLOSED) == before
    with pytest.raises(CheckpointError, match='v2'):
        open_inference({'schema': 'particle-sliders-image-session-v1-inference'})


def test_tensor_schema_failure_closes_without_applying_values(tmp_path, plugin, exported):
    packet = deepcopy(exported[0])
    packet['network']['weights.0'] = torch.zeros(999)
    packet['sha256'] = _hash({key: value for key, value in packet.items() if key != 'sha256'})
    before = len(plugin.CLOSED)
    with pytest.raises(ValueError):
        open_inference(packet, options={'model_file': str(tmp_path / 'model.pt')}, backend=plugin.INFERENCE)
    assert len(plugin.CLOSED) == before + 1
