"""Identity/readout software controls: real PG/HG/HNDL, synthetic readouts."""
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

# Explicit installed-fixture lane; default repository collection must work
# without this optional distribution, its runtime, or a reference checkout.
if os.environ.get('BONSAI_INSTALLED_TESTS') != '1':
    pytest.skip('opt-in installed Bonsai fixture: set BONSAI_INSTALLED_TESTS=1', allow_module_level=True)

import torch
from torch import nn

from bonsai_readout_shared import binding as product
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, _hash
from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic
from particle_sliders.response import open_response, response_backend


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    path = tmp_path / 'rows.json'
    path.write_text(json.dumps([
        dict(positive='a bright room', neutral='a room'),
        dict(positive='a cheerful person', neutral='a person'),
        dict(positive='a sunny day', neutral='a day')]))
    return SessionConfig('bonsai-readout-fixture', 'bonsai-identity-readout-software-fixture-v1',
        rows_file=str(path), steps=4, batch=2, seed=7, ema_decay=.9)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_installed_package_and_registry():
    assert version('bonsai-readout-shared') == '0.1.0'
    assert version('particle-sliders-core') == '0.12.0'
    assert version('particlegan') == '0.9.4'
    assert version('hndl') == '0.10.0'
    assert Path(product.__file__).resolve().is_relative_to(Path(distribution('bonsai-readout-shared').locate_file('')).resolve())
    assert response_backend('bonsai-readout-fixture') is product.FIXTURE_TRAINING
    assert response_backend('bonsai-gguf-shared') is product.TRAINING
    metadata = distribution('bonsai-readout-shared').metadata
    assert product.CORE_COMMIT in metadata['Requires-Dist']
    assert 'hypergan-hndl' in metadata['Requires-Dist']
    _, identity = product.verify_runtime()
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert not torch.cuda.is_initialized()


def test_installed_import_has_no_legacy_game_or_native_side_effects(tmp_path):
    result = subprocess.run([sys.executable, '-c', '''
import json, sys, torch
from bonsai_readout_shared import binding
assert not torch.cuda.is_initialized()
assert 'conceptmod.textsliders.particle_bridge_gan' not in sys.modules
assert 'conceptmod.textsliders.bonsai_gguf_backend' not in sys.modules
assert 'hypergan' not in sys.modules and 'hndl' not in sys.modules
print(json.dumps({'source': binding.__file__, 'profile': binding.PROFILE}))
'''], cwd=tmp_path, text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    assert 'site-packages' in receipt['source'] and receipt['profile'] == product.PROFILE


def test_real_hndl_critic_matches_outputs_vjps_and_penalty_second_derivatives():
    torch.manual_seed(2)
    actual = product.HNDLReadoutCritic(16, 16).double()
    expected = ConditionalReadoutCritic(16, 16).double()
    assert isinstance(actual.graph['hidden'], nn.Linear)
    assert isinstance(actual.graph['score'], nn.Linear)
    with torch.no_grad():
        for old, new in [(expected.net[0], actual.graph['hidden']), (expected.net[2], actual.graph['score'])]:
            old.load_state_dict(new.state_dict())
    inputs = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    condition = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    actual_score, expected_score = actual(inputs, condition), expected(inputs, condition)
    torch.testing.assert_close(actual_score, expected_score, rtol=1e-12, atol=1e-12)
    left = torch.autograd.grad(actual_score.sum(), (inputs, condition), create_graph=True)
    right = torch.autograd.grad(expected_score.sum(), (inputs, condition), create_graph=True)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    # A nonzero derivative-norm penalty differentiates through the actual critic
    # w.r.t. both response inputs and critic weights (including mixed derivatives).
    a_penalty, b_penalty = left[0].square().sum(), right[0].square().sum()
    a = torch.autograd.grad(a_penalty, (inputs, *actual.parameters()), allow_unused=True)
    b = torch.autograd.grad(b_penalty, (inputs, *expected.parameters()), allow_unused=True)
    for x, y in zip(a, b):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-12)
    identity = json.loads(actual.extra_repr())
    assert identity['graph_sha256'] == product._sha(product.CRITIC_SOURCE.encode())
    assert identity['input_shapes']['response'] == ['B', 16]

    # Invoke the real public PG penalty as well; the product defines none.
    from particlegan import get_recipe
    recipe = get_recipe('bispf_optimized', batch_size=2, total_steps=4)
    penalties = []
    for critic in (actual, expected):
        _, optimizer = recipe.make_optimizers(nn.Linear(16, 16).double(), critic)
        penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
        torch.manual_seed(10)
        penalties.append(penalty(critic, inputs.detach(), -inputs.detach(), condition.detach()))
    torch.testing.assert_close(*penalties, rtol=1e-10, atol=1e-12)
    gradients = [torch.autograd.grad(loss, tuple(critic.parameters()), allow_unused=True)
                 for loss, critic in zip(penalties, (actual, expected))]
    for x, y in zip(*gradients):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-11)


def test_shared_head_matches_exact_historical_architecture_and_vjps():
    # Use the existing implementation as the reference; no architecture or GAN
    # equations copied into this product. Width5120 matches native head geometry.
    # The unmodified historical reference is test input only. The product was
    # imported from the installed wheel above, before adding the reference path.
    if not os.environ.get('BONSAI_REFERENCE_ROOT'):
        pytest.skip('optional historical architecture comparison needs BONSAI_REFERENCE_ROOT')
    reference = Path(os.environ['BONSAI_REFERENCE_ROOT']).resolve()
    sys.path.insert(0, str(reference))
    try:
        from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
    finally:
        sys.path.remove(str(reference))
    torch.manual_seed(3)
    actual = product.SharedReadoutHead(5120).double()
    expected = BonsaiParticleHead().double()
    graph = actual.projection.graph
    assert isinstance(graph['down'], nn.Linear) and isinstance(graph['up'], nn.Linear)
    with torch.no_grad():
        graph['up'].weight.normal_(0, .02)
        expected.particles.copy_(actual.particles)
        expected.down.load_state_dict(graph['down'].state_dict())
        expected.up.load_state_dict(graph['up'].state_dict())
        for index in (0, 2, 4, 6):
            expected.bridge.router[index].load_state_dict(graph['router' + str(index)].state_dict())
            expected.bridge.net[index].load_state_dict(graph['bridge' + str(index)].state_dict())
    pairs = [(actual.particles, expected.particles)]
    for left, right in [(graph['down'], expected.down), (graph['up'], expected.up)] + [
            pair for index in (0, 2, 4, 6) for pair in (
                (graph['router' + str(index)], expected.bridge.router[index]),
                (graph['bridge' + str(index)], expected.bridge.net[index]))]:
        pairs.extend(zip(left.parameters(), right.parameters()))
    values = torch.randn(2, 5120, dtype=torch.float64, requires_grad=True)
    with actual.scaled(1.), expected.scaled(1.):
        a, b = actual(values), expected(values)
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    vjp = torch.randn_like(a)
    da = torch.autograd.grad(a, (values, *(p[0] for p in pairs)), vjp)
    db = torch.autograd.grad(b, (values, *(p[1] for p in pairs)), vjp)
    for x, y in zip(da, db):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-12)
    assert da[1].abs().sum() > 0
    assert [name for name, p in actual.named_parameters() if p is actual.particles] == ['particles']
    assert sum(p is actual.particles for p in actual.parameters()) == 1


def test_identity_readout_head_zero_init_strength_and_gradient_ownership():
    runtime = product.FixtureReadout()
    adapter = product.BonsaiReadoutAdapter(runtime)
    network = adapter.network
    x = runtime.hidden('test')
    with network.scaled(1.):
        assert torch.equal(network(x), x)
    with torch.no_grad():
        network.projection.lora_up.weight.normal_(0, .02)
    before = network(x)
    with network.scaled(1.):
        y = network(x)
        with network.scaled(0.):
            assert network(x) is x
        assert network.scale == 1.
    assert torch.equal(before, x) and network.scale == 0.
    with network.scaled(.5):
        torch.testing.assert_close(network(x), x + (y - x) * .5)
    rows = adapter.prepare([product.ReadoutRow('positive', 'neutral')])
    output = adapter.generated(rows[0])
    gradients = torch.autograd.grad(output.sum(), tuple(network.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert not rows[0].target.requires_grad and not rows[0].neutral.requires_grad
    assert not x.requires_grad
    with pytest.raises(RuntimeError):
        with network.scaled(1.):
            raise RuntimeError('restore scale')
    assert network.scale == 0.


def test_zero_up_positive_input_identity_control_has_zero_paired_error_before_training(config):
    """Characterize the existing task; this is not concept-learning evidence."""
    with open_response(config) as session:
        game, network = session.game, session.game.network
        assert game.completed_steps == 0
        assert torch.count_nonzero(network.projection.lora_up.weight) == 0
        assert torch.equal(game.conditions, torch.cat([row.neutral for row in game.rows]))
        assert session.capabilities.response_kind == 'positive_input_identity_readout_fixture'
        assert game.backend.metadata()['neutral_to_positive_training'] is False
        observed = []
        handle = network.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        try:
            for prepared in game.rows:
                assert not torch.equal(prepared.neutral, prepared.target)
                for strength in (0., .5, 1.):
                    row = replace(prepared, row=replace(prepared.row, scale=strength))
                    generated = game.backend.generated(row)
                    # Observe the actual executed input rather than assuming
                    # that neutral critic conditioning reaches the generator.
                    assert torch.equal(observed[-1], prepared.target)
                    assert not torch.equal(observed[-1], prepared.neutral)
                    assert torch.equal(generated, prepared.target)
                    assert torch.count_nonzero(generated - prepared.target) == 0
        finally:
            handle.remove()
        assert game.completed_steps == 0


def test_identity_readout_game_software_update_resume_ema_and_inference(config, tmp_path, monkeypatch):
    with open_response(config) as full, open_response(config) as split:
        original = deepcopy(full.game.network.state_dict())
        critic = deepcopy(full.game.critic.state_dict())
        full.train()
        assert any(not torch.equal(original[k], v) for k, v in full.game.network.state_dict().items())
        assert any(not torch.equal(critic[k], v) for k, v in full.game.critic.state_dict().items())
        owned = [p for group in full.game.opt_g.param_groups for p in group['params']]
        assert sum(p is full.game.network.particles for p in owned) == 1
        assert set(map(id, owned)) == set(map(id, full.game.network.parameters()))
        split.train(stop_after=2)
        split.checkpoint(tmp_path / 'resume.pt')
        with open_response(config, resume=tmp_path / 'resume.pt') as resumed:
            before = resumed.checkpoint()
            for strength in (0., .5, 1.):
                a = resumed.evaluate(strength=strength)
                b = resumed.evaluate(strength=strength)
                assert torch.equal(a, b)
                equal(before, resumed.checkpoint())
            # Positive-input head convention is explicit, including scale0.
            assert torch.equal(resumed.evaluate(strength=0.), resumed.game.rows[0].target)
            resumed.train()
            equal(full.checkpoint(), resumed.checkpoint())
            resumed.export_inference(tmp_path / 'ema.pt', selection='ema')
            resumed.export_inference(tmp_path / 'raw.pt', selection='raw')
            with resumed._observation(0), resumed.game.network.scaled(1.):
                with torch.no_grad():
                    for key, value in resumed.game.network.state_dict().items():
                        value.copy_(resumed.ema.values[key])
                expected = resumed.game.backend.response(text='unseen caption', strength=.5).detach().clone()
    Path(config.options['rows_file']).unlink()
    def forbidden(*args, **kwargs):
        pytest.fail('native inference constructed a game or accessed training rows')
    monkeypatch.setattr(product, 'ConditionalPG094', forbidden)
    monkeypatch.setattr(product, '_read_rows', forbidden)
    with open_inference(tmp_path / 'ema.pt', operation='response') as inference:
        assert not hasattr(inference, 'game') and not hasattr(inference, 'critic')
        assert inference.selection == 'ema'
        torch.testing.assert_close(inference.response(text='unseen caption', strength=.5), expected, rtol=0, atol=0)
        assert torch.equal(inference.response(text='unseen caption', strength=0.),
                           product.FixtureReadout().hidden('unseen caption'))
    with open_inference(tmp_path / 'raw.pt', operation='response') as inference:
        assert inference.selection == 'raw'


@pytest.mark.parametrize('field,value', [
    ('network_profile', 'legacy'), ('pg_source_profile', 'alz-continuous-v1'),
    ('penalty_input_domain', 'continuous'), ('pg_version', '0.9.3'),
    ('base_precision', 'bfloat16'), ('width', 16), ('model_identity', 'foreign')])
def test_changed_profile_or_model_options_rejected_before_allocation(config, monkeypatch, field, value):
    monkeypatch.setattr(product, 'SharedReadoutHead', lambda *a: pytest.fail('allocated before admission'))
    altered = config.options | {field: value}
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(SessionConfig(config.backend, config.cohort, **altered))


def test_native_width_refusal_before_readout_server_or_graph(config, monkeypatch):
    from bonsai_readout_shared import _native as legacy
    monkeypatch.setattr(legacy, '_get_json', lambda *a: pytest.fail('server contacted'))
    monkeypatch.setattr(product, 'SharedReadoutHead', lambda *a: pytest.fail('graph allocated'))
    native = SessionConfig('bonsai-gguf-shared', 'fresh-blocked-native-cohort',
        **(config.options | dict(weights=legacy.FILENAME, server_url='http://127.0.0.1:8080', model_sha256='0' * 64)))
    with pytest.raises(ValueError, match='width >=170'):
        open_response(native)
    class Wide:
        width = 5120
        def hidden(self, text):
            pytest.fail('readout fetched')
    with pytest.raises(ValueError, match='width >=170'):
        product.BonsaiReadoutAdapter(Wide())


@pytest.mark.parametrize('change', ['source', 'shape', 'model', 'optimizer', 'ema'])
def test_corrupted_checkpoint_rejected_without_mutation(config, change):
    with open_response(config) as session:
        session.train(stop_after=2)
        before = session.checkpoint()
        packet = deepcopy(before)
        signature = packet['manifest']['game_signature']
        if change == 'source':
            signature['pg_source_profile'] = 'foreign'
        elif change == 'shape':
            signature['response_shape'] = [5120]
        elif change == 'model':
            signature['backend']['model_identity']['kind'] = 'foreign'
        elif change == 'optimizer':
            packet['payload']['game']['opt_g']['param_groups'][0]['lr'] = -1.
        else:
            packet['payload']['ema']['decay'] = .5
        packet['sha256'] = _hash({k: v for k, v in packet.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            session.restore(packet)
        equal(before, session.checkpoint())


def test_builder_source_tamper_rejected_before_head(config, monkeypatch, tmp_path):
    import inspect
    import particle_sliders.fibo_hndl as common
    from hypergan.hndl_networks import build_network
    original = inspect.getsourcefile
    source_root = Path(original(build_network)).parent
    manifest = json.loads((common.ROOT / 'fibo_hndl_010_sources.json').read_text())
    modified = tmp_path / 'actual-hypergan-closure'
    modified.mkdir()
    for name in [*manifest['hypergan'], '_build_provenance.json']:
        destination = modified / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, destination)
    path = modified / 'hndl_networks.py'
    path.write_bytes(path.read_bytes() + b'\n# actual source-byte tamper\n')
    # Redirect the gate to a private copy; hash real changed bytes. No tampered
    # code is executed and no shared/read-only runtime is modified.
    monkeypatch.setattr(inspect, 'getsourcefile', lambda owner: str(path) if owner is build_network else original(owner))
    with pytest.raises(ValueError, match='builder closure'):
        open_response(config)


@pytest.mark.parametrize('file,reason', [('pg094.py', 'core sources differ'),
    ('fibo_projection.hndl', 'graph/source manifests differ'),
    ('pg094_release.json', 'graph/source manifests differ')])
def test_core_source_tamper_rejected(config, monkeypatch, tmp_path, file, reason):
    import particle_sliders
    root = Path(particle_sliders.__file__).parent
    modified = tmp_path / 'actual-core-closure'
    shutil.copytree(root, modified, ignore=shutil.ignore_patterns('__pycache__'))
    source = modified / file
    source.write_bytes(source.read_bytes() + b'\n# actual source-byte tamper\n')
    monkeypatch.setattr(particle_sliders, '__file__', str(modified / '__init__.py'))
    with pytest.raises(ValueError, match=reason + '|preloaded import origin differs'):
        open_response(config)


def test_inference_model_change_rejected_before_head(config, monkeypatch):
    with open_response(config) as session:
        artifact = session.export_inference()
    artifact['specification']['identity']['model']['kind'] = 'foreign'
    artifact['sha256'] = _hash({k: v for k, v in artifact.items() if k != 'sha256'})
    monkeypatch.setattr(product, 'SharedReadoutHead', lambda *a: pytest.fail('allocated before model identity'))
    with pytest.raises(ValueError, match='model identity'):
        open_inference(artifact, operation='response')
