"""Installed wheel conformance with a tiny actual Diffusers SD3; no model downloads."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusers import SD3Transformer2DModel
from particle_sliders.lifecycle import SessionConfig, CheckpointError, UnsupportedOperation, open_session
from particle_sliders.inference import open_inference
from particle_sliders.pg09 import _hash
from particle_sliders.field_critic import FieldCritic
from particle_sliders.fibo_hndl import HNDLFieldCritic
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders_sd3_shared.backend import BACKEND, NAME, check_model, bind
from particle_sliders_sd3_shared import runtime


@pytest.fixture(autouse=True)
def cpu_only():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)
    assert not torch.cuda.is_initialized()


def model():
    with torch.random.fork_rng():
        torch.manual_seed(15)
        return SD3Transformer2DModel(sample_size=8, patch_size=2, in_channels=2,
            num_layers=2, attention_head_dim=4, num_attention_heads=2,
            joint_attention_dim=8, caption_projection_dim=8,
            pooled_projection_dim=6, out_channels=2, pos_embed_max_size=4)


def rows():
    g = torch.Generator().manual_seed(123)
    def condition():
        return dict(embeds=torch.randn(1, 3, 8, generator=g), pooled=torch.randn(1, 6, generator=g))
    return [ResponseRow(condition(), condition(), condition(),
        torch.randn(1, 2, 8, 6, generator=g), timestep=torch.tensor(500. + i)) for i in range(3)]


def config(base=None, **options):
    base = model() if base is None else base
    return SessionConfig(NAME, 'sd3-shared-tiny-native-hndl010-v1',
        **dict(base_sha256=runtime.model_identity(base)['base_sha256'], rank=2, alpha=2.,
               critic_width=4, steps=3, batch=2, seed=7, ema_decay=.9, **options))


def session():
    base = model()
    return BACKEND.open(config(base), model=base, rows=rows())


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def seal(packet):
    packet['sha256'] = _hash({k: v for k, v in packet.items() if k != 'sha256'})
    return packet


def snapshot(base):
    return ([(m, m.training) for m in base.modules()],
            [(p, p.requires_grad) for p in base.parameters()],
            [(m, dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in base.modules()],
            [(m, {k: getattr(m, k) for k in ('gradient_checkpointing', '_gradient_checkpointing_func') if hasattr(m, k)}) for m in base.modules()])


def restored(state):
    modes, flags, hooks, checkpoints = state
    assert all(m.training == flag for m, flag in modes)
    assert all(p.requires_grad == flag for p, flag in flags)
    assert all(dict(m._forward_hooks) == f and dict(m._forward_pre_hooks) == p for m, f, p in hooks)
    assert all({k: getattr(m, k) for k in ('gradient_checkpointing', '_gradient_checkpointing_func') if hasattr(m, k)} == settings for m, settings in checkpoints)


def test_projection_outputs_and_vjps():
    builder, identity = runtime.admitted_builder()
    layer = runtime.Projection(nn.Linear(8, 5), 3, 4., builder, identity).double()
    assert builder.__module__ == 'hypergan.hndl_networks'
    assert type(layer.graph).__module__ == 'hndl.torch'
    with torch.no_grad():
        layer.graph['up'].weight.normal_(0, .2)
    layer.multiplier = .4
    x = torch.randn(2, 3, 4, 8, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    a, b = layer.graph['down'].weight, layer.graph['up'].weight
    ref = F.linear(F.linear(x, a), b) * (.4 * 4 / 3)
    torch.testing.assert_close(y, ref, atol=1e-14, rtol=1e-14)
    direction = torch.randn_like(y)
    left = torch.autograd.grad((y * direction).sum(), (x, a, b), retain_graph=True)
    right = torch.autograd.grad((ref * direction).sum(), (x, a, b))
    for l, r in zip(left, right):
        torch.testing.assert_close(l, r, atol=1e-13, rtol=1e-13)


def test_field_critic_output_vjp_and_second_derivatives():
    native, graph = FieldCritic(2, 5, 4).double(), HNDLFieldCritic(2, 5, (8, 6), 4).double()
    names = {'field0': native.field[0], 'field2': native.field[2], 'condition': native.condition,
             'score0': native.score[0], 'score2': native.score[2]}
    for name, layer in names.items():
        graph.graph[name].load_state_dict(layer.state_dict())
    x = torch.randn(2, 2, 8, 6, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    outputs = [critic(x, c) for critic in (native, graph)]
    torch.testing.assert_close(*outputs, atol=1e-13, rtol=1e-13)
    gradients = [torch.autograd.grad(o.sum(), (x, c), create_graph=True) for o in outputs]
    for l, r in zip(*gradients):
        torch.testing.assert_close(l, r, atol=1e-13, rtol=1e-13)
    second = []
    for critic, grad in zip((native, graph), gradients):
        params = [p for layer in (names.values() if critic is native else [graph.graph[n] for n in names]) for p in layer.parameters()]
        second.append(torch.autograd.grad(sum(g.square().sum() for g in grad), params, allow_unused=True))
    for l, r in zip(*second):
        assert (l is None) == (r is None)
        if l is not None:
            torch.testing.assert_close(l, r, atol=1e-13, rtol=1e-13)


def test_actual_sd3_native_output_and_vjp_preserve_joint_conditioning():
    base, row = model(), rows()[0]
    identity, selected = check_model(base, config(base).options)
    before = snapshot(base)
    original = deepcopy(base)
    response, close = bind(base, config(base).options, identity, selected)
    try:
        # Explicit matched Torch residual hooks on an independent native transformer.
        handles = []
        for (_, module), layer in zip(runtime.targets(original), response.adapter.layers):
            with torch.no_grad():
                layer.graph['up'].weight.normal_(0, .03)
            def hook(module, arguments, output, layer=layer):
                return output + F.linear(F.linear(arguments[0], layer.graph['down'].weight), layer.graph['up'].weight) * .5
            handles.append(module.register_forward_hook(hook))
        x = row.latent.clone().requires_grad_()
        actual = response.velocity(row.neutral, x, row.timestep, scale=.5)
        expected = original(hidden_states=x, timestep=row.timestep.reshape(1),
            encoder_hidden_states=row.neutral['embeds'], pooled_projections=row.neutral['pooled'], return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        weights = tuple(response.network.parameters())
        for left, right in zip(torch.autograd.grad(actual.sum(), (x, *weights), retain_graph=True),
                               torch.autograd.grad(expected.sum(), (x, *weights))):
            torch.testing.assert_close(left, right, rtol=2e-4, atol=3e-6)
        for handle in handles:
            handle.remove()
        zero = response.velocity(row.neutral, x, row.timestep, scale=0.)
        raw = original(hidden_states=x, timestep=row.timestep.reshape(1),
            encoder_hidden_states=row.neutral['embeds'], pooled_projections=row.neutral['pooled'], return_dict=False)[0]
        assert torch.equal(zero, raw)
        assert actual.shape == (1, 2, 8, 6)
        assert any('add_q_proj' in name for name in response.adapter.names)
        assert all(layer.multiplier == 0 for layer in response.adapter.layers)
    finally:
        close()
    restored(before)


def test_real_updates_exact_resume_ema_and_strength():
    before_rng = torch.get_rng_state().clone()
    with session() as full, session() as split, session() as resumed:
        initial = deepcopy(full.game.network.state_dict())
        assert type(full.game).__module__ == 'particle_sliders.pg09'
        assert type(full.game.critic).__module__ == 'particle_sliders.fibo_hndl'
        assert full.game.response_shape == (1, 2, 8, 6)
        full.train()
        assert any(not torch.equal(initial[k], v) for k, v in full.game.network.state_dict().items())
        split.train(stop_after=1)
        resumed.restore(split.checkpoint())
        resumed.train()
        same(full.checkpoint(), resumed.checkpoint())
        packet = full.checkpoint()
        values = [full.evaluate(strength=v) for v in (0., .5, 1.)]
        assert torch.equal(values[0], full.game.rows[0].neutral)
        assert not torch.equal(values[0], values[-1])
        assert not torch.equal(values[1], values[-1])
        same(packet, full.checkpoint())
        assert full.ema.values is not None
        with pytest.raises(UnsupportedOperation):
            full.render()
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize('damage', ['cohort', 'checksum', 'ema', 'optimizer', 'shape'])
def test_checkpoint_rejection_atomic(damage):
    with session() as source, session() as target:
        source.train(stop_after=1)
        packet, before = source.checkpoint(), target.checkpoint()
        if damage == 'cohort':
            packet['manifest']['config']['cohort'] = 'old-historical-sd3'
        elif damage == 'checksum':
            packet['sha256'] = 'bad'
        elif damage == 'ema':
            next(iter(packet['payload']['ema']['values'].values())).fill_(float('nan'))
        elif damage == 'optimizer':
            packet['payload']['game']['opt_g']['state'] = {}
        else:
            key = next(iter(packet['payload']['game']['network']))
            packet['payload']['game']['network'][key] = torch.zeros(1)
        if damage != 'checksum':
            seal(packet)
        with pytest.raises((CheckpointError, ValueError)):
            target.restore(packet)
        same(before, target.checkpoint())


@pytest.mark.parametrize('damage', ['base', 'pooled', 'text', 'patch', 'vector', 'timestep', 'options'])
def test_admission_before_native_mutation(damage):
    base, data = model(), rows()
    options = config(base).options
    if damage == 'base':
        options['base_sha256'] = '0' * 64
    elif damage in ('pooled', 'text'):
        condition = dict(data[0].neutral)
        condition['pooled' if damage == 'pooled' else 'embeds'] = torch.zeros(1, 2)
        data[0] = replace(data[0], neutral=condition)
    elif damage == 'patch':
        data[0] = replace(data[0], latent=torch.zeros(1, 2, 7, 6))
    elif damage == 'vector':
        data[0] = replace(data[0], latent=torch.zeros(1, 192))
    elif damage == 'timestep':
        data[0] = replace(data[0], timestep=torch.tensor(1001.))
    else:
        options['penalty_input_domain'] = 'continuous'
    base.train()
    next(base.parameters()).requires_grad_(False)
    before = snapshot(base)
    with pytest.raises(ValueError):
        BACKEND.open(SessionConfig(NAME, 'reject', **options), model=base, rows=data)
    restored(before)


@pytest.mark.parametrize('target', ['core', 'graph', 'hndl', 'hypergan', 'native'])
def test_actual_source_tamper_rejected_before_native_mutation(target):
    import particle_sliders, hndl, hypergan.hndl_networks
    import diffusers.models.attention_processor as native_processors
    paths = dict(core=Path(particle_sliders.__file__).with_name('lifecycle.py'),
        native=Path(native_processors.__file__), graph=runtime.ROOT / 'projection.hndl', hndl=Path(hndl.__file__),
        hypergan=Path(hypergan.hndl_networks.__file__).with_name('_build_provenance.json'))
    path = paths[target]
    original = path.read_bytes()
    base, data = model(), rows()
    options = config(base).options
    before = snapshot(base)
    try:
        if target == 'hypergan':
            changed = json.loads(original)
            changed['dirty'] = True
            path.write_text(json.dumps(changed))
        else:
            path.write_bytes(original + b'\n ')
        with pytest.raises((ValueError, json.JSONDecodeError)):
            BACKEND.open(SessionConfig(NAME, 'tamper', **options), model=base, rows=data)
    finally:
        path.write_bytes(original)
    restored(before)


@pytest.mark.parametrize('stage', ['teachers', 'critic', 'game', 'spec', 'session', 'update'])
def test_failure_cleanup_restores_borrowed_model(monkeypatch, stage):
    import particle_sliders_sd3_shared.backend as backend
    import particle_sliders.lifecycle as lifecycle
    base = model()
    base.train()
    base.transformer_blocks[0].eval()
    base.enable_gradient_checkpointing()
    next(base.parameters()).requires_grad_(False)
    before = snapshot(base)
    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + stage)
    if stage == 'teachers':
        monkeypatch.setattr(backend.ImageResponseAdapter, 'prepare', fail)
    elif stage in ('critic', 'game'):
        monkeypatch.setattr(backend, 'HNDLFieldCritic' if stage == 'critic' else 'ConditionalPG094', fail)
    elif stage == 'spec':
        monkeypatch.setattr(type(backend.INFERENCE), 'specification', fail)
    elif stage == 'session':
        monkeypatch.setattr(lifecycle.SliderSession, '__init__', fail)
    with pytest.raises(RuntimeError, match='injected'):
        value = BACKEND.open(config(base), model=base, rows=rows())
        if stage == 'update':
            monkeypatch.setattr(value.game, 'step', fail)
            value.step()
    restored(before)


def test_every_cleanup_runs_and_original_failure_survives():
    reached = []
    def fail():
        reached.append(1)
        raise RuntimeError('cleanup')
    with pytest.raises(RuntimeError, match='cleanup'):
        runtime.restore_all([fail, lambda: reached.append(2)])
    assert reached == [1, 2]
    try:
        raise ValueError('original')
    except ValueError as error:
        runtime.restore_all([fail, lambda: reached.append(3)])
        assert 'cleanup' in error.__notes__[0]
    assert reached[-1] == 3


@pytest.mark.parametrize('selection', ['raw', 'ema'])
def test_fresh_process_native_only_after_training_assets_deleted(tmp_path, selection):
    base, data = model(), rows()
    model_dir = tmp_path / 'model'
    base.save_pretrained(model_dir)
    data_path = tmp_path / 'rows.pt'
    torch.save([dict(neutral=r.neutral, positive=r.positive, negative=r.negative,
        latent=r.latent, timestep=r.timestep) for r in data], data_path)
    options = config(base).options | dict(model_path=str(model_dir), rows_path=str(data_path), rows_sha256=runtime.digest(data_path))
    cfg = SessionConfig(NAME, 'fresh-native-only', **options)
    packet_path = tmp_path / 'training.pt'
    artifact = tmp_path / 'adapter.pt'
    with open_session(cfg) as trained:
        trained.train()
        trained.checkpoint(packet_path)
        packet = trained.export_inference(artifact, selection=selection)
        assert packet['selection'] == selection
    request = dict(latent=data[0].latent, embeds=data[0].neutral['embeds'],
        pooled=data[0].neutral['pooled'], timestep=data[0].timestep)
    with open_inference(artifact, options={'model_path': str(model_dir)}, operation='response') as native:
        expected = {v: native.response(strength=v, **request) for v in (0., .5, 1.)}
        assert not hasattr(native, 'game')
        assert not hasattr(native, 'critic')
    data_path.unlink()
    packet_path.unlink()
    torch.save(request, tmp_path / 'request.pt')
    code = '''
import sys, torch
from particle_sliders.inference import open_inference
import particle_sliders_sd3_shared.backend as backend
backend.prepare = lambda *a, **k: (_ for _ in ()).throw(AssertionError("training factory called"))
torch.set_num_threads(1)
request = torch.load(sys.argv[3], weights_only=True)
with open_inference(sys.argv[1], options={"model_path": sys.argv[2]}, operation="response") as native:
    assert not hasattr(native, "game")
    torch.save({v: native.response(strength=v, **request) for v in (0., .5, 1.)}, sys.argv[4])
'''
    subprocess.run([sys.executable, '-c', code, str(artifact), str(model_dir),
                    str(tmp_path / 'request.pt'), str(tmp_path / 'outputs.pt')], check=True, cwd=tmp_path)
    same(expected, torch.load(tmp_path / 'outputs.pt', weights_only=True))


def test_installed_source_identity():
    import particle_sliders, particle_sliders_sd3_shared
    for package in (particle_sliders, particle_sliders_sd3_shared):
        assert 'site-packages' in package.__file__
    builder, identity = runtime.admitted_builder()
    assert identity['core_commit'] == '7a8a321d04323d49319d0b08e6093a0ddce36b27'
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert builder.__module__ == 'hypergan.hndl_networks'


def test_late_restore_failure_rolls_back_own_game_and_ema():
    with session() as source, session() as target:
        source.train(stop_after=1)
        before = target.checkpoint()
        reached = []
        def late(module, incompatible):
            reached.append(True)
            if len(reached) == 1:
                raise RuntimeError('late native load failure')
        handle = target.game.network.register_load_state_dict_post_hook(late)
        try:
            with pytest.raises(RuntimeError, match='late native'):
                target.restore(source.checkpoint())
        finally:
            handle.remove()
        assert len(reached) == 2  # failed apply followed by valid rollback
        same(before, target.checkpoint())


def test_partial_attach_failure_removes_every_owned_hook(monkeypatch):
    base = model()
    before = snapshot(base)
    second = runtime.targets(base)[1][1]
    def fail(*args, **kwargs):
        raise RuntimeError('second attach failed')
    monkeypatch.setattr(second, 'register_forward_hook', fail)
    with pytest.raises(RuntimeError, match='second attach'):
        BACKEND.open(config(base), model=base, rows=rows())
    restored(before)


def test_cleanup_remove_failure_attempts_other_handles_and_native_restore(monkeypatch):
    base = model()
    base.train()
    before = snapshot(base)
    value = BACKEND.open(config(base), model=base, rows=rows())
    handle = value.game.backend.adapter.handles[-1]
    remove = handle.remove
    def failing_remove():
        remove()
        raise RuntimeError('remove failed after removal')
    monkeypatch.setattr(handle, 'remove', failing_remove)
    with pytest.raises(RuntimeError, match='remove failed'):
        value.close()
    restored(before)


@pytest.mark.parametrize('changed', ['rank', 'alpha'])
def test_inference_bad_configuration_refused_before_native_mutation(monkeypatch, tmp_path, changed):
    import particle_sliders_sd3_shared.backend as backend
    from particle_sliders.inference import InferenceSpec
    with session() as source:
        packet = source.export_inference()
    descriptor = deepcopy(packet['specification'])
    descriptor['configuration'][changed] = 3
    spec = InferenceSpec(descriptor)
    base = model()
    before = snapshot(base)
    monkeypatch.setattr(backend, 'load_model', lambda options: base)
    with pytest.raises(ValueError, match='differs before'):
        backend.prepare_inference({'model_path': str(tmp_path)}, spec)
    restored(before)


def test_callback_failure_restores_native_borrowed_flags(monkeypatch):
    base = model()
    base.train()
    before = snapshot(base)
    value = BACKEND.open(config(base), model=base, rows=rows())
    try:
        def fail(*args, **kwargs):
            base.transformer_blocks[0].train()
            next(base.parameters()).requires_grad_(True)
            raise RuntimeError('native observation failure')
        monkeypatch.setattr(value.game.backend, 'generated', fail)
        with pytest.raises(RuntimeError, match='native observation'):
            value.evaluate()
        assert not any(p.requires_grad for p in base.parameters())
        assert not any(m.training for m in base.modules())
    finally:
        value.close()
    restored(before)


def test_changed_cache_refused_before_loading_model(monkeypatch, tmp_path):
    import particle_sliders_sd3_shared.backend as backend
    path = tmp_path / 'rows.pt'
    path.write_bytes(b'changed')
    options = config().options | dict(rows_path=str(path), rows_sha256='0' * 64, model_path='unused')
    monkeypatch.setattr(backend, 'load_model', lambda options: pytest.fail('native loader executed'))
    with pytest.raises(ValueError, match='cache identity differs before'):
        BACKEND.open(SessionConfig(NAME, 'cache-reject', **options))


def test_installed_cli_shared_training_exports_and_resume(tmp_path):
    base, data = model(), rows()
    model_dir = tmp_path / 'model'
    base.save_pretrained(model_dir)
    cache = tmp_path / 'rows.pt'
    torch.save([dict(neutral=r.neutral, positive=r.positive, negative=r.negative,
        latent=r.latent, timestep=r.timestep) for r in data], cache)
    options = config(base).options | dict(model_path=str(model_dir), rows_path=str(cache), rows_sha256=runtime.digest(cache))
    cfg = tmp_path / 'cohort.json'
    cfg.write_text(json.dumps(SessionConfig(NAME, 'cli-cohort', **options).as_dict()))
    entrypoint = str(Path(sys.executable).with_name('sd3-shared-slider'))
    # Limit threads in this fresh native process without changing the product CLI.
    import os
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    args = [entrypoint, 'train', '--config', str(cfg), '--export-format', 'both']
    subprocess.run([*args, '--out', str(tmp_path / 'first'), '--stop-after', '1'], env=env, cwd=tmp_path, check=True)
    subprocess.run([*args, '--resume', str(tmp_path / 'first/training.pt'), '--out', str(tmp_path / 'resumed')],
                   env=env, cwd=tmp_path, check=True)
    receipt = json.loads((tmp_path / 'resumed/manifest.json').read_text())
    assert receipt['completed_steps'] == 3
    assert set(receipt['files']) == {'training.pt', 'adapter.pt', 'adapter-v2.pt', 'metrics.jsonl'}


def test_legacy_tensor_checkpoint_refused_before_factory(monkeypatch, tmp_path):
    import particle_sliders_sd3_shared.backend as backend
    path = tmp_path / 'historical-style-state-dict.pt'
    # Representative native-only state dictionary, not a claimed trained model artifact.
    torch.save({'lora_unet_joint_attention_lora_down.weight': torch.zeros(2, 8)}, path)
    monkeypatch.setattr(backend, 'load_model', lambda options: pytest.fail('native loader executed'))
    with pytest.raises(CheckpointError):
        open_session(config(), resume=path)
