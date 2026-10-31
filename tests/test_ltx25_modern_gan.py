"""Installed-package CPU plumbing, never pretrained/GPU qualification."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest
import torch
from torch import nn

import conceptmod.textsliders.ltx25_gan as product
from particle_sliders.field_critic import FieldCritic
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.pg094 import validate_pg09_domain
from particle_sliders.pg09 import _hash
from particle_sliders.response import open_response


def rows(geometry=(2, 1, 1)):
    g = torch.Generator().manual_seed(123)
    return [product.VideoRow('person still', 'person smiling',
        torch.randn(1, geometry[0]*geometry[1]*geometry[2], 128, generator=g),
        torch.randn(1, 2, 8, generator=g)) for _ in range(2)]


def session(**options):
    return open_response(product.config(steps=3, ema_decay=.9, **options),
                         rows=rows(tuple(options.get('geometry', [2, 1, 1]))))


def test_real_hypergan_graph_matched_values_vjp_and_penalty_second_derivatives():
    torch.manual_seed(9)
    critic = product.VideoFieldCritic(3, 2, (2, 2, 3), width=4).double()
    reference = FieldCritic(3, 2, width=4).double()
    for name, module in [('field0', reference.field[0]), ('field2', reference.field[2]),
                         ('condition', reference.condition), ('score0', reference.score[0]),
                         ('score2', reference.score[2])]:
        module.load_state_dict(critic.graph[name].state_dict())
    assert isinstance(critic.graph['field0'], nn.Conv2d)
    assert isinstance(critic.graph['field2'], nn.Conv2d)
    assert isinstance(critic.graph['score0'], nn.Linear)
    identity = json.loads(critic.extra_repr())
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert identity['hndl_version'] == '0.10.0'
    assert identity['temporal_readout'].startswith('framewise-spatial-score-mean')
    value = torch.randn(2, 3, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    condition = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    actual = critic(value, condition)
    expected = torch.stack([reference(value[:, :, i], condition) for i in range(2)]).mean(0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_params = [critic.graph[n].get_parameter(p) for n in
        ('field0','field2','condition','score0','score2') for p in ('weight','bias')]
    expected_params = [p for module in (reference.field[0], reference.field[2], reference.condition,
                                      reference.score[0], reference.score[2]) for p in module.parameters()]
    a = torch.autograd.grad(actual.sum(), [value, condition, *actual_params], create_graph=True)
    b = torch.autograd.grad(expected.sum(), [value, condition, *expected_params], create_graph=True)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    # Input-gradient penalty differentiates again into every critic parameter.
    aa = torch.autograd.grad(a[0].square().sum(), [value, condition, *actual_params], allow_unused=True)
    bb = torch.autograd.grad(b[0].square().sum(), [value, condition, *expected_params], allow_unused=True)
    for left, right in zip(aa, bb):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('geometry', [(1, 1, 1), (3, 2, 2)])
def test_native_geometry_masking_gradient_ownership_and_zero_strength(geometry):
    with session(geometry=list(geometry)) as s:
        native = s.game.backend.backend
        prepared = s.game.rows[0]
        assert s.game.response_shape == (1, 128, *geometry)
        zero = s.evaluate(strength=0.)
        assert torch.equal(zero, prepared.neutral)
        response = s.game.backend.generated(prepared)
        response.square().mean().backward()
        leaves = list(s.game.network.parameters())
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in leaves)
        assert any(p.grad.abs().sum() > 0 for p in leaves)
        owned = set(map(id, leaves))
        for root in (native.encoder, native.connectors, native.transformer):
            assert all(not p.requires_grad and p.grad is None for p in root.parameters() if id(p) not in owned)
        assert all('audio' not in name for name in native.lora_module_names())
        assert any('te-' in name for name in native.lora_module_names())
        seen = []
        hook = native.connectors.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
        try:
            packed = s.game.backend._pack('person smiling', prepared.row, 0., teacher=True)
        finally:
            hook.remove()
        assert packed.hold_stage == 'pre_connector'
        assert seen[0][1] == 1024  # Native pre-connector padding, before register insertion.
        assert packed.encoder_attention_mask.shape[:2] == packed.encoder_hidden_states.shape[:2]
        assert (packed.encoder_attention_mask == 0).any()


def test_shared_updates_exact_resume_ema_and_rejection_before_mutation():
    with session() as full, session() as first, session() as resumed:
        before = _hash(full.game.network.state_dict())
        base = _hash(full.game.backend.backend.transformer.state_dict())
        full.train()
        assert _hash(full.game.network.state_dict()) != before
        assert _hash(full.game.backend.backend.transformer.state_dict()) == base
        first.train(stop_after=1)
        resumed.restore(first.checkpoint())
        resumed.train()
        assert _hash(full.checkpoint()) == _hash(resumed.checkpoint())
        assert full.game._signature['response_shape'] == [128, 2, 1, 1]
        assert full.game.penalty is not None
        snapshot = _hash(resumed.checkpoint())
        packet = deepcopy(first.checkpoint())
        packet['manifest']['config']['cohort'] = 'historical-uni'
        packet['sha256'] = _hash({k:v for k,v in packet.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            resumed.restore(packet)
        assert _hash(resumed.checkpoint()) == snapshot
        for scale in (0., .5, 1.):
            assert torch.isfinite(full.evaluate(strength=scale)).all()
        assert full.export_inference(selection='ema')['selection'] == 'ema'


def test_native_only_inference_after_training_rows_removed(monkeypatch):
    with session() as s:
        s.train()
        packet = s.export_inference(selection='ema')
        native_row = rows()[0]
    # No training factory, teacher cache, critic or optimizer can be reconstructed.
    def forbidden(*args, **kwargs):
        raise AssertionError('training path invoked by native inference')
    monkeypatch.setattr(product, 'prepare', forbidden)
    monkeypatch.setattr(product, 'VideoFieldCritic', forbidden)
    with open_inference(packet, operation='response') as infer:
        assert not hasattr(infer, 'game')
        values = [infer.response(neutral=native_row.neutral, video_latents=native_row.video_latents,
            audio_latents=native_row.audio_latents, strength=scale) for scale in (0., .5, 1.)]
        assert all(x.shape == (1, 128, 2, 1, 1) and torch.isfinite(x).all() for x in values)
        assert not torch.equal(values[0], values[2])
        # Raw native scale-zero is exactly the frozen response regardless of trained leaves.
        adapter = product.make_native({'dummy': True})
        base = adapter._field(adapter._pack(native_row.neutral, native_row, 0.))
        assert torch.equal(base, values[0])
        adapter.close()
    assert 'rows' not in packet and 'optimizer' not in packet and 'critic' not in packet


@pytest.mark.parametrize('options', [dict(dummy=False), dict(model_id='different'),
    dict(transformer_subfolder='transformer_full'), dict(geometry=[0,1,1]),
    dict(penalty_input_domain='continuous')])
def test_preflight_refuses_before_native_hooks(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError('native hooks attached before refusal')
    monkeypatch.setattr(product, 'LTX25Backend', forbidden)
    with pytest.raises(ValueError):
        product.backend.open(SessionConfig(product.BACKEND, product.COHORT, **({'dummy':True} | options)), rows=rows())


def test_historical_cohort_refused_before_native_hooks(monkeypatch):
    monkeypatch.setattr(product, 'LTX25Backend', lambda **kw: pytest.fail('hooks attached'))
    with pytest.raises(ValueError, match='historical UNI'):
        product.backend.open(SessionConfig(product.BACKEND, 'ltx25_uni_embed', dummy=True), rows=rows())


def test_builder_tamper_refused_before_native_hooks(monkeypatch):
    import particle_sliders.fibo_hndl as gate
    original = gate._sha
    monkeypatch.setattr(gate, '_sha', lambda path: '0' * 64
        if Path(path).name == 'hndl_networks.py' else original(path))
    monkeypatch.setattr(product, 'LTX25Backend', lambda **kw: pytest.fail('hooks attached'))
    with pytest.raises(ValueError, match='builder closure'):
        product.backend.open(product.config(), rows=rows())


def test_native_helper_source_tamper_refused_before_native_hooks(monkeypatch):
    monkeypatch.setitem(product.NATIVE_SOURCES, 'ltx25_uni.py', '0' * 64)
    monkeypatch.setattr(product, 'LTX25Backend', lambda **kw: pytest.fail('hooks attached'))
    with pytest.raises(ValueError, match='native helper source cohort'):
        product.backend.open(product.config(), rows=rows())


def test_native_adapter_matches_existing_backend_output_and_parameter_vjp():
    adapter = product.make_native({'dummy': True})
    reference = product.make_native({'dummy': True})
    row = rows()[0]
    try:
        prepared = adapter.prepare(row)
        actual = adapter.generated(prepared)
        native = reference.backend
        native.network.set_lora_slider(1.)
        with native.network:
            text = native.encode_text(row.neutral, frozen=False)
            packed = native.pack_t2v(text, video_latents=row.video_latents, audio_latents=row.audio_latents,
                                    num_frames=9, height=32, width=32)
        packed = replace(packed, audio_encoder_hidden_states=prepared.audio_condition,
                         audio_encoder_attention_mask=prepared.audio_mask)
        expected = native.forward_velocity(packed, scale=0.).sample.transpose(1,2).reshape(1,128,2,1,1)
        assert torch.equal(actual, expected)
        a = torch.autograd.grad(actual.square().sum(), tuple(adapter.network.parameters()))
        b = torch.autograd.grad(expected.square().sum(), tuple(reference.network.parameters()))
        for left, right in zip(a,b):
            assert torch.equal(left,right)
    finally:
        adapter.close()
        reference.close()


def test_changed_factory_source_artifact_refused_before_native_hooks(monkeypatch):
    with session() as s:
        packet = s.export_inference()
    packet['specification']['factory_sources']['prepare'] = '0' * 64
    packet['sha256'] = _hash({k:v for k,v in packet.items() if k != 'sha256'})
    monkeypatch.setattr(product, 'make_native', lambda *args: pytest.fail('hooks attached'))
    with pytest.raises(CheckpointError, match='factory source'):
        open_inference(packet, operation='response')


def test_core_does_not_admit_connector_tensor_or_wide_vector():
    for shape, domain in [((1,1024,4096), 'continuous_field'), ((1,4096), 'continuous_vector')]:
        with pytest.raises(ValueError):
            validate_pg09_domain(shape, response_domain=domain)


def test_close_restores_host_forward_methods():
    s = session()
    adapter = s.game.backend
    hosts = list(adapter._hosts)
    originals = [leaf.org_forward for leaf in adapter.backend.network.loras]
    s.close()
    s.close()
    assert [host.forward for _, host in hosts] == originals


def test_installed_package_has_exact_opt_in_dependency_and_entry_points():
    from importlib.metadata import distribution
    dist = distribution('particle-sliders-ltx25-modern-preview')
    assert 'site-packages' in str(Path(product.__file__).resolve())
    assert any('4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231' in req and '[hypergan-hndl]' in req for req in dist.requires)
    assert {ep.group for ep in dist.entry_points} == {
        'particle_sliders.response_backends', 'particle_sliders.inference_backends'}
