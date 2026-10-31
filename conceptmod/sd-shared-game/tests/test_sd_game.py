"""Installed-package CPU tests with real Diffusers/HG/HNDL/PG; no model weights."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from sd_slider_game import adapter as sd
from particle_sliders.field_critic import FieldCritic
from particle_sliders.fibo_hndl import HNDLFieldCritic
from particle_sliders.inference import export_inference, open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.pg09 import _hash
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.response import open_response
from particle_sliders.routed_projection import TinyParticleProjection


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(previous)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def snapshot(model):
    return dict(state=deepcopy(model.state_dict()), modes=[m.training for m in model.modules()],
                flags=[p.requires_grad for p in model.parameters()], grads=[p.grad for p in model.parameters()],
                forwards=[m.forward for m in model.modules()])


def unchanged(model, before):
    equal(before['state'], model.state_dict())
    assert before['modes'] == [m.training for m in model.modules()]
    assert before['flags'] == [p.requires_grad for p in model.parameters()]
    assert all(a is p.grad for a, p in zip(before['grads'], model.parameters()))
    assert before['forwards'] == [m.forward for m in model.modules()]


def affine(destination, source):
    with torch.no_grad():
        destination.weight.copy_(source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)


@pytest.mark.parametrize('prediction_type', ['epsilon', 'v_prediction'])
@pytest.mark.parametrize('scheduler_class', ['DDIMScheduler', 'DDPMScheduler'])
def test_native_discrete_prediction_cfg_and_borrowed_state(prediction_type, scheduler_class):
    import diffusers
    model, _ = sd.fixture_native()
    scheduler = getattr(diffusers, scheduler_class)(prediction_type=prediction_type)
    row = replace(sd.fixture_rows()[0], guidance=3.)
    model.train()
    model.down_blocks[0].eval()
    first = next(model.parameters())
    first.requires_grad_(False)
    first.grad = torch.ones_like(first)
    before = snapshot(model)
    with torch.no_grad():
        model.eval()
        latent = scheduler.scale_model_input(row.latent, row.timestep)
        positive = model(latent, row.timestep, encoder_hidden_states=row.neutral['embeds']).sample
        negative = model(latent, row.timestep, encoder_hidden_states=row.negative['embeds']).sample
        expected = negative + row.guidance * (positive - negative)
    for module, mode in zip(model.modules(), before['modes']):
        module.training = mode
    response, _, close = sd.attach(model, scheduler)
    assert response.metadata()['native_scheduler']['prediction_type'] == prediction_type
    with torch.no_grad():
        actual = response._response(row, scale=0.)
    assert torch.equal(actual, expected)
    assert all(not p.requires_grad for p in model.parameters())
    close(); close()
    unchanged(model, before)


def test_real_native_unet_matched_routed_outputs_and_all_vjps():
    model, scheduler = sd.fixture_native()
    original = deepcopy(model).eval().requires_grad_(False)
    response, _, close = sd.attach(model, scheduler)
    network = response.network
    bank = nn.Parameter(network.particles.detach().clone())
    old_adapters = []
    for name, host in sd.targets(original).items():
        old = TinyParticleProjection(name, host, bank, 8, 8)
        new = network.adapters[name.replace('.', '-')]
        nn.init.normal_(old.lora_up.weight, std=.02)
        affine(new.graph['down'], old.lora_down)
        affine(new.graph['up'], old.lora_up)
        for family, sequence in [('router', old.bridge.router), ('bridge', old.bridge.net)]:
            for index in (0, 2, 4, 6):
                affine(new.graph[family + str(index)], sequence[index])
        old.multiplier = .7
        old_adapters.append(old)
        assert type(new.graph).__module__ == 'hndl.torch'
    row = sd.fixture_rows()[0]
    value = row.latent.clone().requires_grad_()
    twin = value.detach().clone().requires_grad_()
    embeds = row.neutral['embeds'].clone().requires_grad_()
    twin_embeds = embeds.detach().clone().requires_grad_()
    expected = original(value, row.timestep, encoder_hidden_states=embeds).sample
    with network.scaled(.7):
        actual = model(twin, row.timestep, encoder_hidden_states=twin_embeds).sample
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    old_parameters = [value, embeds, bank]
    new_parameters = [twin, twin_embeds, network.particles]
    for old, new in zip(old_adapters, network.adapters.values()):
        for key in ['down', 'up'] + [f'{kind}{index}' for kind in ('router', 'bridge') for index in (0, 2, 4, 6)]:
            if key == 'down':
                module = old.lora_down
            elif key == 'up':
                module = old.lora_up
            else:
                sequence = old.bridge.router if key.startswith('router') else old.bridge.net
                module = sequence[int(key[-1])]
            old_parameters.extend(module.parameters())
            new_parameters.extend(new.graph[key].parameters())
    vjp0 = torch.autograd.grad(expected.square().sum(), old_parameters)
    vjp1 = torch.autograd.grad(actual.square().sum(), new_parameters)
    for a, b in zip(vjp1, vjp0):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=5e-4)
    assert vjp1[2].abs().sum() > 0
    assert sum(p is network.particles for p in network.parameters()) == 1
    assert len(list(network.parameters())) == len({id(p) for p in network.parameters()})
    close()


def test_real_hndl_critic_outputs_input_parameter_and_penalty_second_derivatives():
    torch.manual_seed(42)
    old, new = FieldCritic(4, 3, width=4), HNDLFieldCritic(4, 3, (8, 8), width=4)
    names = [('field0', old.field[0]), ('field2', old.field[2]),
             ('condition', old.condition), ('score0', old.score[0]), ('score2', old.score[2])]
    for name, module in names:
        affine(new.graph[name], module)
    assert type(new.graph).__module__ == 'hndl.torch'
    value = torch.randn(2, 4, 8, 8, requires_grad=True)
    twin = value.detach().clone().requires_grad_()
    condition = torch.randn(2, 3)
    expected, actual = old(value, condition), new(twin, condition)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    parameters0 = [p for _, module in names for p in module.parameters()]
    parameters1 = [p for name, _ in names for p in new.graph[name].parameters()]
    grad0 = torch.autograd.grad(expected.sum(), [value] + parameters0, create_graph=True)
    grad1 = torch.autograd.grad(actual.sum(), [twin] + parameters1, create_graph=True)
    for a, b in zip(grad1, grad0):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
    # Actual released public BISPF penalty, including its critic-weight second derivatives.
    from particlegan import get_recipe
    penalties, derivatives = [], []
    for critic, response, parameters in [(old, value, parameters0), (new, twin, parameters1)]:
        recipe = get_recipe('bispf_optimized', batch_size=2, total_steps=4)
        optimizer = recipe.make_critic_optimizer(critic)
        penalty = recipe.make_critic_penalty(optimizer)
        amount = penalty(critic, response, response.detach() + .1, condition)
        penalties.append(amount)
        derivatives.append(torch.autograd.grad(amount, parameters, allow_unused=True))
    torch.testing.assert_close(penalties[0], penalties[1], atol=2e-7, rtol=3e-5)
    for a, b in zip(*derivatives):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=3e-7, rtol=5e-5)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_actual_shared_updates_exact_resume_ema_strength_native_only(preset, monkeypatch, tmp_path):
    options = dict(fixture=True, steps=4, preset=preset, ema_decay=.9)
    rng = torch.get_rng_state().clone()
    with open_response(sd.config(**options)) as full, open_response(sd.config(**options)) as split:
        initial = deepcopy(full.game.network.state_dict())
        full.train()
        assert any(not torch.equal(value, full.game.network.state_dict()[key]) for key, value in initial.items())
        assert not torch.equal(initial['particles'], full.game.network.particles)
        split.train(stop_after=2)
        split.checkpoint(tmp_path / 'split.pt')
        with open_response(sd.config(**options), resume=tmp_path / 'split.pt') as resumed:
            resumed.train()
            equal(full.checkpoint(), resumed.checkpoint())
            state = resumed.checkpoint()
            for strength in (0., .5, 1.):
                actual = resumed.evaluate(strength=strength)
                assert actual.shape == (1, 4, 8, 8)
                assert torch.equal(actual, resumed.evaluate(strength=strength))
                equal(state, resumed.checkpoint())
            assert torch.equal(resumed.evaluate(strength=0.), resumed.game.rows[0].neutral)
            exports = []
            raw_values = deepcopy(resumed.game.network.state_dict())
            for selection in ('raw', 'ema'):
                packet = export_inference(resumed, selection=selection)
                if selection == 'ema':
                    resumed.game.network.load_state_dict(resumed.ema.values)
                expected = resumed.evaluate(strength=.5, seed=0)
                resumed.game.network.load_state_dict(raw_values)
                exports.append((selection, packet, expected))
            equal(state, resumed.checkpoint())
            row = deepcopy(resumed.game.rows[0].row)
    assert torch.equal(rng, torch.get_rng_state())
    # Removing training constructors/data cannot affect native-only loading.
    def forbidden(*args, **kwargs):
        raise AssertionError('training factory or rows used during native inference')
    monkeypatch.setattr(sd, 'fixture_rows', forbidden)
    monkeypatch.setattr(sd, 'prepare', forbidden)
    monkeypatch.setattr(sd, 'ConditionalPG094', forbidden)
    for selection, packet, expected in exports:
        assert set(packet) == {'schema', 'specification', 'origin', 'selection', 'network', 'sha256'}
        with open_inference(packet, operation='response') as native:
            assert native.selection == selection
            assert not hasattr(native, 'game')
            result = native.response(row=row, strength=.5)
            assert torch.equal(result, expected)
            assert result.shape == (1, 4, 8, 8)
            assert torch.equal(result, native.response(row=row, strength=.5))
            for invalid in (-.1, 1.1, float('nan')):
                with pytest.raises(Exception):
                    native.response(row=row, strength=invalid)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('stage', ['teacher', 'critic', 'game', 'spec', 'update'])
def test_every_failure_restores_borrowed_host(stage, monkeypatch):
    model, scheduler = sd.fixture_native()
    model.train()
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    before = snapshot(model)
    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + stage)
    if stage == 'teacher':
        monkeypatch.setattr(sd.SDResponse, 'prepare', fail)
    elif stage == 'critic':
        monkeypatch.setattr(sd, 'HNDLFieldCritic', fail)
    elif stage == 'game':
        monkeypatch.setattr(sd, 'ConditionalPG094', fail)
    elif stage == 'spec':
        monkeypatch.setattr(type(sd.INFERENCE), 'specification', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with sd.BACKEND.open(sd.config(fixture=True), model=model, scheduler=scheduler,
                             rows=sd.fixture_rows()) as session:
            assert stage == 'update'
            monkeypatch.setattr(session.game.backend, 'generated', fail)
            session.step()
    unchanged(model, before)


def test_mismatch_tamper_and_old_packet_rejection_before_native_mutation(monkeypatch):
    model, scheduler = sd.fixture_native()
    before = snapshot(model)
    bad_rows = [replace(row, timestep=torch.tensor(1000)) for row in sd.fixture_rows()]
    with pytest.raises(ValueError, match='timestep'):
        sd.BACKEND.open(sd.config(), model=model, scheduler=scheduler, rows=bad_rows)
    unchanged(model, before)
    identity = sd.model_identity(model, scheduler)
    identity['weights'][next(iter(identity['weights']))]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='before attachment'):
        sd.attach(model, scheduler, expected=identity)
    unchanged(model, before)
    import particle_sliders.fibo_hndl as hndl
    digest = hndl._sha
    monkeypatch.setattr(hndl, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else digest(p))
    with pytest.raises(ValueError, match='closure'):
        sd.attach(model, scheduler)
    unchanged(model, before)
    monkeypatch.undo()
    with pytest.raises(ValueError, match='exact cohort'):
        sd.BACKEND.open(SessionConfig(sd.NAME, 'historical-mse'), model=model,
                        scheduler=scheduler, rows=sd.fixture_rows())
    unchanged(model, before)
    with open_response(sd.config(fixture=True, ema_decay=.9)) as session:
        initial = session.checkpoint()
        packet = deepcopy(initial)
        packet['manifest']['config']['cohort'] = 'historical-mse'
        packet['sha256'] = _hash({k: v for k, v in packet.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            session.restore(packet)
        equal(initial, session.checkpoint())
        with pytest.raises(CheckpointError):
            session.restore({'lora_unet': torch.zeros(1)})
        equal(initial, session.checkpoint())


def test_nested_strength_and_attachment_rollback(monkeypatch):
    model, scheduler = sd.fixture_native()
    before = snapshot(model)
    response, _, close = sd.attach(model, scheduler)
    network = response.network
    modules = list(network.adapters.values())
    modules[0].multiplier = .3
    with pytest.raises(RuntimeError):
        with network.scaled(.7):
            assert all(m.multiplier == .7 for m in modules)
            with network.scaled(0.):
                assert all(m.multiplier == 0 for m in modules)
            assert all(m.multiplier == .7 for m in modules)
            raise RuntimeError('scoped')
    assert modules[0].multiplier == .3
    assert all(m.multiplier == 0 for m in modules[1:])
    close()
    unchanged(model, before)
    original = sd.SDResponse.__init__
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('post-attach')
    monkeypatch.setattr(sd.SDResponse, '__init__', fail)
    with pytest.raises(RuntimeError, match='post-attach'):
        sd.attach(model, scheduler)
    unchanged(model, before)


def test_installed_identity_no_namespace_collision():
    import importlib.metadata as metadata
    assert 'site-packages' in str(Path(sd.__file__).resolve())
    assert metadata.version('conceptmod-sd-shared-game') == '0.1.0'
    assert metadata.version('particlegan') == '0.9.4'
    assert metadata.version('hndl') == '0.10.0'
    for group in ('particle_sliders.response_backends', 'particle_sliders.inference_backends'):
        assert len(list(metadata.entry_points(group=group, name=sd.NAME))) == 1
    from particle_sliders.fibo_hndl import verified_builder
    _, identity = verified_builder()
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert identity['profile'] == 'hypergan-hndl-v2'


def test_restore_invalid_source_tensor_and_ema_reject_before_load_hooks():
    with open_response(sd.config(fixture=True, ema_decay=.9)) as session:
        before = session.checkpoint()
        calls = []
        hook = session.game.network.register_load_state_dict_pre_hook(lambda *args: calls.append(True))
        for damage in ('source', 'tensor', 'ema'):
            packet = deepcopy(before)
            if damage == 'source':
                packet['manifest']['game_signature']['pg_source_profile'] = 'unqualified'
            elif damage == 'tensor':
                next(iter(packet['payload']['game']['network'].values())).fill_(float('nan'))
            else:
                next(iter(packet['payload']['ema']['values'].values())).fill_(float('nan'))
            # Nonfinite values are refused by the common checksum gate, before load hooks.
            if damage == 'source':
                packet['sha256'] = _hash({k: v for k, v in packet.items() if k != 'sha256'})
            with pytest.raises(CheckpointError):
                session.restore(packet)
            assert not calls
            equal(before, session.checkpoint())
        hook.remove()


def test_cleanup_attempts_remaining_actions_and_preserves_original(monkeypatch):
    model, scheduler = sd.fixture_native()
    before = snapshot(model)
    response, _, close = sd.attach(model, scheduler)
    original = response.network.close
    def broken_close():
        original()
        raise RuntimeError('cleanup failure')
    monkeypatch.setattr(response.network, 'close', broken_close)
    with pytest.raises(ValueError, match='original') as error:
        try:
            raise ValueError('original')
        finally:
            close()
    assert any('cleanup failure' in note for note in error.value.__notes__)
    unchanged(model, before)


def test_preflight_geometry_source_and_checkpointing_refuse_before_forward_mutation(monkeypatch):
    model, scheduler = sd.fixture_native()
    before = snapshot(model)
    for options in (dict(response_shape=[1, 256]), dict(base_precision='bfloat16'),
                    dict(penalty_input_domain='continuous'), dict(pg_source_profile='unreviewed')):
        with pytest.raises(Exception):
            sd.BACKEND.open(sd.config(**options), model=model, scheduler=scheduler, rows=sd.fixture_rows())
        unchanged(model, before)
    model.enable_gradient_checkpointing()
    with pytest.raises(ValueError, match='checkpointing disabled'):
        sd.attach(model, scheduler)
    model.disable_gradient_checkpointing()
    unchanged(model, before)
    original_sha = sd.sha
    monkeypatch.setattr(sd, 'sha', lambda p: '0' * 64 if str(p).endswith('lifecycle.py') else original_sha(p))
    with pytest.raises(ValueError, match='shared core source closure'):
        sd.attach(model, scheduler)
    unchanged(model, before)


def test_local_safetensors_loader_and_native_inference_after_data_deletion(tmp_path, monkeypatch):
    model, scheduler = sd.fixture_native()
    model_path, scheduler_path = tmp_path / 'unet', tmp_path / 'scheduler'
    model.save_pretrained(model_path, safe_serialization=True)
    scheduler.save_pretrained(scheduler_path)
    from dataclasses import asdict
    rows_path = tmp_path / 'rows.pt'
    torch.save([asdict(row) for row in sd.fixture_rows()], rows_path)
    options = dict(model_path=str(model_path), scheduler_path=str(scheduler_path),
                   rows_path=str(rows_path), steps=1, ema_decay=.9)
    with open_response(sd.config(**options)) as session:
        session.train()
        packet = export_inference(session, selection='ema')
        row = deepcopy(session.game.rows[0].row)
        session.game.network.load_state_dict(session.ema.values)
        expected = session.evaluate(strength=.5)
    rows_path.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('training reconstructed')
    monkeypatch.setattr(sd, 'prepare', forbidden)
    monkeypatch.setattr(sd, 'fixture_rows', forbidden)
    with open_inference(packet, operation='response',
                        options=dict(model_path=str(model_path), scheduler_path=str(scheduler_path))) as native:
        assert torch.equal(expected, native.response(row=row, strength=.5))
    bad = deepcopy(packet)
    bad['specification']['configuration']['response_shape'] = [1, 4, 16, 16]
    bad['specification']['capabilities']['response_shape'] = [1, 4, 16, 16]
    bad['sha256'] = _hash({key: value for key, value in bad.items() if key != 'sha256'})
    with pytest.raises(ValueError, match='shape differs'):
        open_inference(bad, operation='response',
                       options=dict(model_path=str(model_path), scheduler_path=str(scheduler_path)))


def test_fresh_process_exports_with_different_python_hash_seed(tmp_path):
    with open_response(sd.config(fixture=True, steps=2, ema_decay=.9)) as session:
        session.train()
        raw = deepcopy(session.game.network.state_dict())
        for selection in ('raw', 'ema'):
            export_inference(session, tmp_path / (selection + '.pt'), selection=selection)
            if selection == 'ema':
                session.game.network.load_state_dict(session.ema.values)
            torch.save(session.evaluate(strength=.5), tmp_path / (selection + '-expected.pt'))
            session.game.network.load_state_dict(raw)
    code = '''
from pathlib import Path
import torch
from particle_sliders.inference import open_inference
from sd_slider_game import adapter as sd
root=Path(__import__('sys').argv[1]); row=sd.fixture_rows()[0]
def forbidden(*args, **kwargs): raise AssertionError('training reconstructed')
sd.fixture_rows=sd.prepare=sd.ConditionalPG094=forbidden
for selection in ('raw','ema'):
 with open_inference(root/(selection+'.pt'),operation='response') as native:
  expected=torch.load(root/(selection+'-expected.pt'),weights_only=True)
  assert torch.equal(native.response(row=row,strength=.5),expected)
  assert not hasattr(native,'game')
assert not torch.cuda.is_initialized()
print('fresh raw/EMA native-only outputs exact; differing Python hash seed')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
        env=dict(os.environ, OMP_NUM_THREADS='1', PYTHONHASHSEED='237'),
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
