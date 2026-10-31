from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor

from particle_sliders.lifecycle import SessionConfig, CheckpointError, _hash
from particle_sliders.response import open_response
from particle_sliders.inference import export_inference, open_inference
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.field_critic import FieldCritic
from particle_sliders.fibo_hndl import HNDLFieldCritic
from sdxl_particle_preview import backend
from sdxl_particle_preview.native import HNDLLoRA, NativeSDXL, gate, targets, validate_row


@pytest.fixture(autouse=True)
def threads():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.fixture
def native(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(8,), layers_per_block=1, cross_attention_dim=8,
            attention_head_dim=2, norm_num_groups=4, addition_embed_type='text_time',
            addition_time_embed_dim=2, projection_class_embeddings_input_dim=20)
        model.set_attn_processor(AttnProcessor())
        scheduler = DDIMScheduler(num_train_timesteps=100, prediction_type='epsilon')
        model.save_pretrained(tmp_path / 'base/unet')
        scheduler.save_pretrained(tmp_path / 'base/scheduler')
        model, scheduler = backend.load_model(tmp_path / 'base')
        def condition(offset):
            return dict(embeds=torch.randn(1, 3, 8) + offset, pooled=torch.randn(1, 8) + offset,
                        time_ids=torch.tensor([[64., 64., 0., 0., 64., 64.]]))
        rows = [ResponseRow(condition(0), condition(.3), condition(-.2),
            torch.randn(1, 4, 8, 8), torch.tensor(20.), guidance=1.5) for _ in range(2)]
        rowfile = tmp_path / 'rows.pt'
        torch.save([dict(neutral=r.neutral, positive=r.positive, negative=r.negative,
                         latent=r.latent, timestep=r.timestep, guidance=r.guidance) for r in rows], rowfile)
    options = dict(model_dir=str(tmp_path / 'base'), rows_path=str(rowfile),
                   shape=[1, 4, 8, 8], steps=3, batch=2, seed=7, ema_decay=.9)
    return model, scheduler, rows, options


def test_hndl_ordinary_lora_matched_weights_outputs_and_vjps():
    builder, identity = gate()
    assert builder.__module__ == 'hypergan.hndl_networks'
    base = torch.nn.Linear(7, 5)
    graph = HNDLLoRA(base, 4, 1., builder, identity)
    assert type(graph.graph).__module__ == 'hndl.torch'
    down, up = torch.nn.Linear(7, 4, bias=False), torch.nn.Linear(4, 5, bias=False)
    with torch.no_grad():
        up.weight.normal_()
        graph.graph['down'].weight.copy_(down.weight)
        graph.graph['up'].weight.copy_(up.weight)
    x = torch.randn(2, 3, 7, requires_grad=True)
    expected, actual = up(down(x)), graph(x)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    cotangent = torch.randn_like(actual)
    a = torch.autograd.grad(actual, (x, *graph.parameters()), cotangent, retain_graph=True)
    b = torch.autograd.grad(expected, (x, down.weight, up.weight), cotangent)
    for lhs, rhs in zip(a, b):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


def test_hndl_critic_matches_field_outputs_vjps_and_second_derivatives():
    ref, actual = FieldCritic(4, 6, width=8), HNDLFieldCritic(4, 6, (8, 8), width=8)
    pairs = [('field0', ref.field[0]), ('field2', ref.field[2]), ('condition', ref.condition),
             ('score0', ref.score[0]), ('score2', ref.score[2])]
    for name, module in pairs:
        actual.graph[name].load_state_dict(module.state_dict())
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    c = torch.randn(2, 6, requires_grad=True)
    expected, result = ref(x, c), actual(x, c)
    torch.testing.assert_close(result, expected)
    ga = torch.autograd.grad(result.sum(), (x, c), create_graph=True)
    gb = torch.autograd.grad(expected.sum(), (x, c), create_graph=True)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
    pa, pb = ga[0].square().sum(), gb[0].square().sum()
    sa = torch.autograd.grad(pa, (x, *actual.parameters()), allow_unused=True)
    sb = torch.autograd.grad(pb, (x, *ref.parameters()), allow_unused=True)
    for a, b in zip(sa, sb):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('prediction', ['epsilon', 'v_prediction'])
def test_native_sdxl_call_and_conditioning_contract(native, prediction):
    model, scheduler, rows, options = native
    scheduler.register_to_config(prediction_type=prediction)
    original = deepcopy(model.state_dict())
    r = rows[0]
    expected = model(scheduler.scale_model_input(r.latent, r.timestep), r.timestep.reshape(1),
        encoder_hidden_states=r.neutral['embeds'], added_cond_kwargs=dict(
            text_embeds=r.neutral['pooled'], time_ids=r.neutral['time_ids'])).sample
    owner = NativeSDXL(model, scheduler, (1, 4, 8, 8), 4, 1.)
    try:
        actual = owner.velocity(r.neutral, r.latent, r.timestep, scale=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert owner.metadata()['prediction_type'] == prediction
        for key in ('embeds', 'pooled', 'time_ids'):
            changed = deepcopy(r.neutral)
            changed[key] += 1
            value = owner.velocity(changed, r.latent, r.timestep, scale=0)
            assert not torch.equal(value, actual), key
    finally:
        owner.close()
    equal(original, model.state_dict())
    assert not any(m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_shared_updates_exact_resume_ema_and_native_only_inference(native, tmp_path, preset):
    model, scheduler, rows, options = native
    options['preset'] = preset
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:fixture', **options)
    with open_response(cfg) as full:
        before = deepcopy(full.game.network.state_dict())
        assert type(full.game).__module__ == 'particle_sliders.pg09'
        assert type(full.game.critic.graph).__module__ == 'hndl.torch'
        full.train()
        final = full.checkpoint()
        assert any(not torch.equal(v, full.game.network.state_dict()[k]) for k, v in before.items())
    with open_response(cfg) as split:
        split.train(stop_after=1)
        split.checkpoint(tmp_path / 'train.pt')
    with open_response(cfg, resume=tmp_path / 'train.pt') as resumed:
        resumed.train()
        equal(final, resumed.checkpoint())
        before = resumed.checkpoint()
        rng = torch.get_rng_state().clone()
        for strength in (0., .5, 1.):
            value = resumed.evaluate(strength=strength)
            assert value.shape == (1, 4, 8, 8) and torch.isfinite(value).all()
        torch.testing.assert_close(resumed.evaluate(strength=0), resumed.game.rows[0].neutral, rtol=0, atol=0)
        equal(before, resumed.checkpoint())
        assert torch.equal(rng, torch.get_rng_state())
        exported = export_inference(resumed, tmp_path / 'inference.pt', selection='ema')
        assert exported['selection'] == 'ema'
        saved_raw = deepcopy(resumed.game.network.state_dict())
        resumed.game.network.load_state_dict(resumed.ema.values)
        expected_ema = resumed.evaluate(strength=.5)
        resumed.game.network.load_state_dict(saved_raw)
    Path(options['rows_path']).unlink()
    # The native-only registry must not open training files, build critics, or construct games.
    with open_inference(tmp_path / 'inference.pt', options={'model_dir': options['model_dir']},
                        operation='response') as inference:
        result = inference.response(row=rows[0], strength=.5)
        assert result.shape == (1, 4, 8, 8) and torch.isfinite(result).all()
        torch.testing.assert_close(result, expected_ema, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_before_mutation_conditioning_and_source_refusals(native, monkeypatch):
    model, scheduler, rows, options = native
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:fixture', **options)
    modes = [(m, m.training) for m in model.modules()]
    grads = [(p, p.requires_grad) for p in model.parameters()]
    original = deepcopy(model.state_dict())
    for key, value in [('pooled', torch.ones(1, 7)), ('embeds', torch.ones(1, 3, 7)),
                       ('time_ids', torch.ones(1, 5))]:
        bad = deepcopy(rows[0].neutral)
        bad[key] = value
        with pytest.raises(ValueError, match='dimensions'):
            open_response(cfg, model=model, scheduler=scheduler, rows=[replace(rows[0], neutral=bad), rows[1]])
    with pytest.raises(ValueError, match='discrete'):
        open_response(cfg, model=model, scheduler=scheduler, rows=[replace(rows[0], timestep=torch.tensor(.5)), rows[1]])
    equal(original, model.state_dict())
    assert all(m.training == flag for m, flag in modes)
    assert all(p.requires_grad == flag for p, flag in grads)
    assert not any(m._forward_hooks for m in model.modules())
    import sdxl_particle_preview.native as native_module
    monkeypatch.setattr(native_module, 'digest', lambda path: '0' * 64)
    with pytest.raises(ValueError, match='core source'):
        NativeSDXL(model, scheduler, (1, 4, 8, 8), 4, 1.)
    assert not any(m._forward_hooks for m in model.modules())


def test_cleanup_success_prepare_failure_and_step_failure(native, monkeypatch):
    model, scheduler, rows, options = native
    model.train()
    # Heterogeneous borrowed gradient/mode flags must survive every close/failure path.
    next(model.parameters()).requires_grad_(False)
    next(iter(model.children())).eval()
    modes = [(m, m.training) for m in model.modules()]
    grads = [(p, p.requires_grad) for p in model.parameters()]
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:cleanup', **options)
    def assert_restored():
        assert all(m.training == v for m, v in modes)
        assert all(p.requires_grad == v for p, v in grads)
        assert not any(m._forward_hooks for m in model.modules())
    session = open_response(cfg, model=model, scheduler=scheduler, rows=rows)
    session.close()
    session.close()
    assert_restored()
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise RuntimeError('critic failure')
        patch.setattr(backend, 'HNDLFieldCritic', fail)
        with pytest.raises(RuntimeError, match='critic failure'):
            open_response(cfg, model=model, scheduler=scheduler, rows=rows)
    assert_restored()
    session = open_response(cfg, model=model, scheduler=scheduler, rows=rows)
    def failed_step():
        raise RuntimeError('update failure')
    session.game.step = failed_step
    with pytest.raises(RuntimeError, match='update failure'):
        session.step()
    assert_restored()


def test_checkpoint_mismatch_and_restore_rollback(native, tmp_path):
    model, scheduler, rows, options = native
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:restore', **options)
    with open_response(cfg) as session:
        session.step()
        before = session.checkpoint(tmp_path / 'good.pt')
        bad = deepcopy(before)
        bad['manifest']['config']['cohort'] = 'historical-xl-loss'
        bad['sha256'] = _hash({k: v for k, v in bad.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            session.restore(bad)
        equal(before, session.checkpoint())
    with pytest.raises(CheckpointError, match='kind/config/cohort'):
        open_response(SessionConfig(backend.NAME, 'other', **options), resume=tmp_path / 'good.pt',
                      model=model, scheduler=scheduler, rows=rows)


def test_native_strength_and_nested_scale_restore(native):
    model, scheduler, rows, _ = native
    owner = NativeSDXL(model, scheduler, (1, 4, 8, 8), 4, 1.)
    try:
        for layer in owner.network.layers:
            torch.nn.init.normal_(layer.graph['up'].weight, std=.01)
        r = rows[0]
        a = owner.response(row=r, strength=0)
        b = owner.response(row=r, strength=.5)
        c = owner.response(row=r, strength=1)
        assert not torch.equal(a, b) and not torch.equal(b, c)
        with owner.network.scaled(.5):
            with pytest.raises(RuntimeError):
                with owner.network.scaled(0):
                    raise RuntimeError('nested')
            assert owner.network.scale == .5
        assert owner.network.scale == 0
        for strength in (-1., 1.1, float('nan')):
            with pytest.raises(ValueError):
                owner.response(row=r, strength=strength)
    finally:
        owner.close()


def test_historical_lora_matched_weight_native_outputs_and_vjps(native):
    import importlib.util
    source = Path(__file__).resolve().parents[3] / 'conceptmod/textsliders/lora.py'
    spec = importlib.util.spec_from_file_location('historical_sdxl_lora_fixture', source)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    model, scheduler, rows, _ = native
    reference = deepcopy(model)
    ref_targets = targets(reference)
    owner = NativeSDXL(model, scheduler, (1, 4, 8, 8), 4, 1.)
    refs = []
    try:
        for (name, module), hndl_layer in zip(ref_targets.items(), owner.network.layers):
            old = historical.LoRAModule(name, module, multiplier=.7, lora_dim=4, alpha=1.)
            old.apply_to()
            torch.nn.init.normal_(old.lora_up.weight, std=.01)
            hndl_layer.graph['down'].weight.data.copy_(old.lora_down.weight)
            hndl_layer.graph['up'].weight.data.copy_(old.lora_up.weight)
            refs.append(old)
        row = rows[0]
        x = row.latent.clone().requires_grad_(True)
        kwargs = dict(encoder_hidden_states=row.neutral['embeds'], added_cond_kwargs=dict(
            text_embeds=row.neutral['pooled'], time_ids=row.neutral['time_ids']))
        expected = reference(x, row.timestep, **kwargs).sample
        with owner.network.scaled(.7):
            actual = model(x, row.timestep, **kwargs).sample
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        ga = torch.autograd.grad(actual.square().sum(), (x, *owner.network.parameters()))
        gb = torch.autograd.grad(expected.square().sum(), (x, *(p for l in refs for p in l.parameters())))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    finally:
        owner.close()


def test_every_hook_cleanup_even_when_one_remove_raises(native):
    model, scheduler, _, _ = native
    modes = [(m, m.training) for m in model.modules()]
    grads = [(p, p.requires_grad) for p in model.parameters()]
    owner = NativeSDXL(model, scheduler, (1, 4, 8, 8), 4, 1.)
    handle = owner.network.handles[0]
    original = handle.remove
    def fail_after_remove():
        original()
        raise RuntimeError('injected removal failure')
    handle.remove = fail_after_remove
    with pytest.raises(RuntimeError, match='removal failure'):
        owner.close()
    assert not any(m._forward_hooks for m in model.modules())
    assert all(m.training == v for m, v in modes)
    assert all(p.requires_grad == v for p, v in grads)
    owner.close()


def test_bad_ema_and_late_load_failure_are_atomic(native, monkeypatch):
    model, scheduler, rows, options = native
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:atomic', **options)
    with open_response(cfg, model=model, scheduler=scheduler, rows=rows) as session:
        session.step()
        before = session.checkpoint()
        bad = deepcopy(before)
        bad['payload']['ema']['decay'] = 7
        bad['sha256'] = _hash({k: v for k, v in bad.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            session.restore(bad)
        equal(before, session.checkpoint())
        original = session.game.network.load_state_dict
        count = 0
        def fail_once(state, **kwargs):
            nonlocal count
            count += 1
            result = original(state, **kwargs)
            if count == 1:
                next(session.game.network.parameters()).data.add_(9)
                raise RuntimeError('late tensor load failure')
            return result
        with monkeypatch.context() as patch:
            patch.setattr(session.game.network, 'load_state_dict', fail_once)
            with pytest.raises(RuntimeError, match='late tensor'):
                session.restore(before)
        equal(before, session.checkpoint())


def test_native_inference_rejects_changed_weights_before_hooks(native, tmp_path, monkeypatch):
    model, scheduler, rows, options = native
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:changed-model', **options)
    with open_response(cfg) as session:
        artifact = export_inference(session)
    changed = deepcopy(model)
    next(changed.parameters()).data.add_(1)
    monkeypatch.setattr(backend, 'load_model', lambda path: (changed, scheduler))
    flags = [(p, p.requires_grad) for p in changed.parameters()]
    with pytest.raises(ValueError, match='before hook'):
        open_inference(artifact, options={'model_dir': options['model_dir']}, operation='response')
    assert not any(m._forward_hooks for m in changed.modules())
    assert all(p.requires_grad == value for p, value in flags)


def test_cli_executes_installed_shared_game(native, tmp_path):
    import subprocess
    import os
    _, _, _, options = native
    options['steps'] = 1
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:cli', **options)
    path = tmp_path / 'session.json'
    path.write_text(json.dumps(cfg.as_dict()))
    result = subprocess.run([sys.executable, '-m', 'sdxl_particle_preview.cli', '--config', str(path),
        '--checkpoint', str(tmp_path / 'cli.training.pt'), '--export', str(tmp_path / 'cli.inference.pt'),
        '--selection', 'ema'], capture_output=True, text=True,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'cli.training.pt').is_file() and (tmp_path / 'cli.inference.pt').is_file()


def test_profile_shape_legacy_and_builder_tamper_refuse_before_hooks(native, tmp_path, monkeypatch):
    from particle_sliders.pg09 import validate_pg09_domain
    from particle_sliders.lifecycle import BackendUnavailable
    import particle_sliders.fibo_hndl as hndl_binding
    import particle_sliders.response as response
    model, scheduler, rows, options = native
    cfg = SessionConfig(backend.NAME, 'sdxl-native-hndl-v1:gate', **options)
    with pytest.raises(ValueError, match='width >=170'):
        validate_pg09_domain((1, 170), response_domain='continuous_vector')
    for changes in (dict(shape=[1, 170]), dict(penalty_input_domain='continuous'),
                    dict(pg_source_profile='new-unadmitted-profile'), dict(pg_version='0.9.3')):
        with pytest.raises((ValueError, BackendUnavailable)):
            open_response(SessionConfig(backend.NAME, cfg.cohort, **(options | changes)),
                          model=model, scheduler=scheduler, rows=rows)
        assert not any(m._forward_hooks for m in model.modules())
    torch.save({'lora_unet_historical': torch.zeros(2, 2)}, tmp_path / 'old.pt')
    with monkeypatch.context() as patch:
        patch.setattr(response, 'response_backend', lambda name: pytest.fail('legacy reached native factory'))
        with pytest.raises(CheckpointError):
            open_response(cfg, resume=tmp_path / 'old.pt')
    with monkeypatch.context() as patch:
        patch.setattr(hndl_binding, '_sha', lambda path: '0' * 64)
        with pytest.raises(ValueError, match='HNDL source closure'):
            NativeSDXL(model, scheduler, (1, 4, 8, 8), 4, 1.)
    assert not any(m._forward_hooks for m in model.modules())
