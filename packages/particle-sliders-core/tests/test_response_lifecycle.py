"""Response-session conformance authored for an admitted CPU slot; no media claim."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from particle_sliders import lifecycle, pg09, response
from particle_sliders.inference import InferenceSpec, open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, _hash
from particle_sliders.response import ResponseCapabilities
from test_image_lifecycle import equal


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def vector(tmp_path, monkeypatch):
    source = Path(__file__).parent / 'fixtures/response_backend.py'
    spec = importlib.util.spec_from_file_location('vector_fixture', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(response, 'response_backend', lambda name: module.TRAINING)
    (tmp_path / 'base.txt').write_text('.3')
    (tmp_path / 'rows.txt').write_text('4')
    return module


def config(tmp_path):
    return SessionConfig('vector-fixture', 'new-response-source-cohort',
        model_file=str(tmp_path / 'base.txt'), training_file=str(tmp_path / 'rows.txt'), ema_decay=.9)


def test_shared_response_resume_ema_and_observation(tmp_path, vector):
    options = config(tmp_path)
    with response.open_response(options) as full, response.open_response(options) as split:
        full.train()
        split.train(stop_after=2)
        split.checkpoint(tmp_path / 'training.pt')
        assert split.manifest['api'] == response.API
        assert split.checkpoint()['schema'] == response.TRAINING
        with response.open_response(options, resume=tmp_path / 'training.pt') as resumed:
            before, rng = resumed.checkpoint(), torch.get_rng_state().clone()
            for strength in (0., .5, 1.):
                assert torch.equal(resumed.evaluate(strength=strength), resumed.evaluate(strength=strength))
                equal(before, resumed.checkpoint())
            assert torch.equal(resumed.evaluate(strength=0), resumed.game.rows[0].neutral)
            assert torch.equal(rng, torch.get_rng_state())
            resumed.train()
            equal(full.checkpoint(), resumed.checkpoint())
            artifact = resumed.export()
            assert artifact['schema'] == response.INFERENCE
            resumed.load_inference(artifact)
            with pytest.raises(UnsupportedOperation, match='inference'):
                resumed.train()
    assert not torch.cuda.is_initialized()


def test_response_failures_are_atomic_and_cleanup(tmp_path, vector):
    with response.open_response(config(tmp_path)) as session:
        before = session.checkpoint()
        invalid = deepcopy(before)
        invalid['manifest']['config']['cohort'] = 'foreign'
        invalid['sha256'] = _hash({k: v for k, v in invalid.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            session.restore(invalid)
        equal(before, session.checkpoint())
        closed = len(vector.CLOSED)
        session.game.step = lambda: (_ for _ in ()).throw(RuntimeError('update failure'))
        with pytest.raises(RuntimeError, match='update failure'):
            session.step()
        assert len(vector.CLOSED) == closed + 1


def test_wrong_packet_family_rejected_before_factory_resolution(tmp_path, vector, monkeypatch):
    with response.open_response(config(tmp_path)) as session:
        packet = session.checkpoint(tmp_path / 'response.pt')
    monkeypatch.setattr(lifecycle, 'backend', lambda name: pytest.fail('wrong image family allocated'))
    with pytest.raises(CheckpointError, match='packet kind'):
        lifecycle.open_session(config(tmp_path), resume=tmp_path / 'response.pt')
    packet['schema'] = lifecycle.TRAINING
    packet['sha256'] = _hash({k: v for k, v in packet.items() if k != 'sha256'})
    lifecycle.atomic_save(tmp_path / 'image.pt', packet)
    monkeypatch.setattr(response, 'response_backend', lambda name: pytest.fail('wrong response family allocated'))
    with pytest.raises(CheckpointError, match='packet kind'):
        response.open_response(config(tmp_path), resume=tmp_path / 'image.pt')
    assert lifecycle.API == 'particle-sliders-image-session-v1'


def test_separate_registry_without_image_fallback(vector, monkeypatch):
    calls = []
    class Entry:
        def load(self):
            return vector.TRAINING
    def entries(**kwargs):
        calls.append(kwargs)
        return [Entry()]
    # Undo fixture's resolver patch and exercise installed discovery function directly.
    function = response.response_backend
    monkeypatch.undo()
    monkeypatch.setattr(response, 'entry_points', entries)
    assert response.response_backend('vector-fixture') is vector.TRAINING
    assert calls == [dict(group='particle_sliders.response_backends', name='vector-fixture')]
    monkeypatch.setattr(lifecycle, 'entry_points', entries)
    with pytest.raises(lifecycle.BackendUnavailable):
        lifecycle.backend('vector-fixture')


def test_preflight_domain_profile_and_unsupported_media_before_prepare(tmp_path, vector, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('allocation before native/domain admission')
    backend = replace(vector.TRAINING, prepare=forbidden)
    for request in (dict(operation='render'), dict(strength=-.5), dict(references=True)):
        with pytest.raises(UnsupportedOperation):
            backend.open(config(tmp_path), **request)
    wide = replace(backend, capabilities=replace(vector.CAPS, response_shape=(1, 2048)))
    with pytest.raises(ValueError, match='width >=170'):
        wide.open(config(tmp_path))
    wrong_version = replace(backend, capabilities=replace(vector.CAPS, pg_version='0.0.0'))
    with pytest.raises(lifecycle.BackendUnavailable, match='exact'):
        wrong_version.open(config(tmp_path))
    overlay = replace(vector.CAPS, response_shape=(1, 2048),
        pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    spy = []
    monkeypatch.setattr(pg09, '_released_sources', lambda profile: spy.append(profile) or 'source-test-double')
    gated = replace(vector.TRAINING, capabilities=overlay)
    if pg09.PG_VERSION == '0.9.1':
        # Admission control-flow test only, never evidence of installed-overlay update parity.
        gated._check_pg({}, overlay)
        assert spy == ['alz-continuous-v1']
    else:
        with pytest.raises(ValueError, match='qualified pair'):
            gated._check_pg({}, overlay)
        assert spy == []


def test_actual_game_must_match_declared_shape_and_cleanup(tmp_path, vector):
    backend = replace(vector.TRAINING, capabilities=replace(vector.CAPS, response_shape=(1, 16)))
    closed = len(vector.CLOSED)
    with pytest.raises(ValueError, match='contradicts'):
        backend.open(config(tmp_path))
    assert len(vector.CLOSED) == closed + 1


def test_response_descriptor_is_explicit_without_image_fields(vector):
    data = asdict(vector.CAPS)
    restored = ResponseCapabilities.from_dict(data)
    assert restored == vector.CAPS
    assert not {'reference_images', 'render_outputs', 'render'} & data.keys()
    assert lifecycle.capabilities_from_dict(data) == vector.CAPS
    with pytest.raises(ValueError, match='version'):
        ResponseCapabilities.from_dict(dict(data, schema='unknown'))
    unavailable = replace(vector.CAPS, operations=(), base_precisions=(),
        unavailable_reasons=(('train', 'native loader is not implemented'),))
    with pytest.raises(UnsupportedOperation, match='native loader is not implemented'):
        unavailable.check()


def test_response_native_only_fresh_process_without_training_data(tmp_path, vector):
    with response.open_response(config(tmp_path)) as session:
        session.train(stop_after=2)
        expected = {}
        for selection in ('raw', 'ema'):
            packet = session.export_inference(tmp_path / (selection + '.pt'), selection=selection)
            InferenceSpec(packet['specification'])
            with open_inference(packet, options={'model_file': str(tmp_path / 'base.txt')},
                                backend=vector.INFERENCE, operation='response') as native:
                expected[selection] = native.response(strength=.5)
    (tmp_path / 'rows.txt').unlink()
    torch.save(expected, tmp_path / 'expected.pt')
    installed = tmp_path / 'installed'
    installed.mkdir()
    (installed / 'vector_fixture.py').write_bytes((Path(__file__).parent / 'fixtures/response_backend.py').read_bytes())
    dist = installed / 'vector_fixture-1.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: vector-fixture\nVersion: 1\n')
    (dist / 'entry_points.txt').write_text(
        '[particle_sliders.response_backends]\nvector-fixture=vector_fixture:TRAINING\n'
        '[particle_sliders.inference_backends]\nvector-fixture=vector_fixture:INFERENCE\n')
    import particle_sliders, particlegan
    code = '''
import sys, json
from pathlib import Path
sys.path[:0] = json.loads(sys.argv[1])
import torch, vector_fixture
from particle_sliders.inference import open_inference
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import AudioResponseCritic
root = Path(sys.argv[2])
assert not (root / 'rows.txt').exists()
def forbidden(*args, **kwargs):
    raise AssertionError('training/teacher/critic/optimizer construction during native response loading')
ConditionalPG09.__init__ = forbidden
AudioResponseCritic.__init__ = forbidden
torch.optim.Optimizer.__init__ = forbidden
vector_fixture.TRAINING.prepare.__code__ = forbidden.__code__
expected = torch.load(root / 'expected.pt', weights_only=True)
for selection in ('raw', 'ema'):
    with open_inference(root / (selection + '.pt'), options={'model_file': str(root / 'base.txt')},
                        operation='response') as session:
        assert session.selection == selection
        assert torch.equal(session.response(strength=.5), expected[selection])
assert not torch.cuda.is_initialized()
print('fresh native-only vector raw/EMA response passed')
'''
    paths = [str(installed), str(Path(particle_sliders.__file__).parent.parent), str(Path(particlegan.__file__).parent.parent)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH='')
    result = subprocess.run([sys.executable, '-I', '-c', code, json.dumps(paths), str(tmp_path)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_response_capabilities_and_outputs(tmp_path, vector, capsys):
    from particle_sliders.session_cli import main
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config(tmp_path).as_dict()))
    main(['capabilities', '--interface', 'response', '--config', str(path)])
    assert json.loads(capsys.readouterr().out)['response_domain'] == 'continuous_vector'
    main(['train', '--interface', 'response', '--config', str(path), '--out', str(tmp_path / 'run'),
          '--stop-after', '2', '--export-format', 'both'])
    main(['evaluate', '--interface', 'response', '--config', str(path),
          '--resume', str(tmp_path / 'run/training.pt'), '--out', str(tmp_path / 'probe.pt')])
    probe = torch.load(tmp_path / 'probe.pt', weights_only=True)
    assert probe['kind'] == 'native_final_hidden_fixture' and not probe['quality_validated']
