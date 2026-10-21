"""CPU graph parity and real released-PG lifecycle; no native model weights."""
from copy import deepcopy
from importlib.metadata import version

import pytest
import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLFieldCritic, HNDLParticleProjection, PROFILE
from particle_sliders.field_critic import FieldCritic
from particle_sliders.routed_projection import TinyParticleProjection

pytest.importorskip('hndl')
pytest.importorskip('hypergan.hndl_networks')
if version('hndl') != '0.9.0':
    pytest.skip('Fibo network profile requires the separate HNDL0.9.0 extra', allow_module_level=True)

def copy_affine(destination, source):
    with torch.no_grad():
        destination.weight.copy_(source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)


@pytest.mark.parametrize('shape', [(2, 5), (2, 3, 5), (2, 3, 4, 5)])
def test_all_routed_network_ops_match_original_outputs_and_vjps(shape):
    torch.manual_seed(42)
    host = nn.Linear(5, 6).requires_grad_(False)
    twin = deepcopy(host)
    bank = nn.Parameter(torch.randn(128, 4))
    other_bank = nn.Parameter(bank.detach().clone())
    old = TinyParticleProjection('old', host, bank, 8, 8)
    new = HNDLParticleProjection('new', twin, other_bank, 8, 8)
    copy_affine(new.graph['down'], old.lora_down)
    # Nonzero up maps exercise every route and bank gradient.
    nn.init.normal_(old.lora_up.weight, std=.2)
    copy_affine(new.graph['up'], old.lora_up)
    for family, sequential in [('router', old.bridge.router), ('bridge', old.bridge.net)]:
        for index in (0, 2, 4, 6):
            copy_affine(new.graph[family + str(index)], sequential[index])
    value = torch.randn(shape, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    old.multiplier = new.multiplier = .7
    expected, actual = host(value), twin(other)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected_grad = torch.autograd.grad(expected.square().sum(), (value, bank))
    actual_grad = torch.autograd.grad(actual.square().sum(), (other, other_bank))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    assert actual_grad[1].abs().sum() > 0
    old.multiplier = new.multiplier = 0
    assert torch.equal(twin(value), old.org_forward(value))
    assert not any('particles' in key for key in new.state_dict())


def test_hndl_spatial_critic_preserves_field_outputs_and_penalty_derivatives():
    torch.manual_seed(43)
    old, new = FieldCritic(4, 3), HNDLFieldCritic(4, 3, (3, 4))
    for name, layer in [('field0', old.field[0]), ('field2', old.field[2]),
                        ('condition', old.condition), ('score0', old.score[0]), ('score2', old.score[2])]:
        copy_affine(new.graph[name], layer)
    value = torch.randn(2, 4, 3, 4, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    condition = torch.randn(2, 3)
    expected, actual = old(value, condition), new(other, condition)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    g0 = torch.autograd.grad(expected.sum(), value, create_graph=True)[0]
    g1 = torch.autograd.grad(actual.sum(), other, create_graph=True)[0]
    torch.testing.assert_close(g1, g0, atol=2e-7, rtol=2e-5)
    d0 = torch.autograd.grad(g0.square().sum(), old.field[0].weight)[0]
    d1 = torch.autograd.grad(g1.square().sum(), new.graph['field0'].weight)[0]
    torch.testing.assert_close(d1, d0, atol=2e-7, rtol=3e-5)


def test_low_level_hndl_export_reload_and_wrong_profile_preflight(tmp_path):
    from test_fibo_native import Pipeline, IDENTITY
    from particle_sliders.fibo_native import FiboParticleSlider, target_modules
    model = Pipeline().transformer
    slider = FiboParticleSlider(model, network_profile=PROFILE)
    assert len(list(slider.parameters())) == len({id(p) for p in slider.parameters()})
    assert sum(p is slider.particles for p in slider.parameters()) == 1
    path = tmp_path / 'adapter.safetensors'
    slider.save(path, model_identity=IDENTITY)
    other = Pipeline().transformer
    forwards = [m.forward for m in target_modules(other).values()]
    with pytest.raises(ValueError, match='protocol/model/target'):
        FiboParticleSlider.load(other, path, model_identity=IDENTITY)
    assert forwards == [m.forward for m in target_modules(other).values()]
    loaded = FiboParticleSlider.load(other, path, model_identity=IDENTITY, network_profile=PROFILE)
    assert loaded.metadata(IDENTITY) == slider.metadata(IDENTITY)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in slider.state_dict().items())


def test_reviewed_builder_closure_rejects_tamper_before_attachment(monkeypatch):
    import particle_sliders.fibo_hndl as module
    from test_fibo_native import Pipeline
    from particle_sliders.fibo_native import FiboParticleSlider, target_modules
    model = Pipeline().transformer
    forwards = [m.forward for m in target_modules(model).values()]
    flags = [p.requires_grad for p in model.parameters()]
    original = module._sha
    monkeypatch.setattr(module, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError, match='HyperGAN builder closure'):
        FiboParticleSlider(model, network_profile=PROFILE)
    assert forwards == [m.forward for m in target_modules(model).values()]
    assert flags == [p.requires_grad for p in model.parameters()]


def test_real_pg_shared_factory_resume_ema_inference_and_profile_separation(tmp_path):
    pytest.importorskip('fibo_particle_slider')
    from PIL import Image
    from particle_sliders.lifecycle import open_session, CheckpointError
    from particle_sliders.inference import open_inference
    from fibo_particle_slider.lifecycle import synthetic_config
    from fibo_particle_slider.inference import INFERENCE
    config = synthetic_config(tmp_path / 'cache', steps=4, network_profile=PROFILE)
    assert config.cohort == 'fibo-hypergan-hndl-v1-cpu'
    before = torch.get_rng_state().clone()
    with open_session(config) as full:
        assert isinstance(full.game.critic, HNDLFieldCritic)
        assert all(isinstance(p, HNDLParticleProjection) for p in full.game.backend.slider.adapters.values())
        bank_before = full.game.backend.slider.particles.detach().clone()
        steps = []
        full.train(on_step=lambda metric: steps.append(metric['step']))
        assert steps == [1, 2, 3, 4]
        assert not torch.equal(bank_before, full.game.backend.slider.particles)
        expected = full.checkpoint()['sha256']
    with open_session(config) as split:
        split.train(stop_after=2)
        packet = split.checkpoint(tmp_path / 'training.pt')
    with open_session(config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
        for strength in (0., .5, 1.):
            assert torch.isfinite(resumed.evaluate(strength=strength)).all()
        resumed.export_inference(tmp_path / 'native.pt', selection='ema')
    # Native-only reload cannot need training rows or construct a critic/game.
    import shutil
    shutil.rmtree(tmp_path / 'cache')
    refs = [Image.new('RGB', (8, 8), (64, 20, 30))]  # Protocol fixture, not target data.
    with open_inference(tmp_path / 'native.pt', references=True, output_type='latent', backend=INFERENCE,
                        native_request={'prompt': 'keep jacket', 'images': refs}) as inference:
        for strength in (0., .5, 1.):
            assert torch.isfinite(inference.render(prompt='keep jacket', images=refs,
                strength=strength, seed=7, output_type='latent').images[0]).all()
    with open_session(synthetic_config(tmp_path / 'legacy-cache', steps=4)) as legacy:
        with pytest.raises(CheckpointError):
            legacy.restore(packet)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
