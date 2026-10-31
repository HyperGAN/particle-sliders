"""Installed-package CPU contracts. Never loads pretrained weights or uses CUDA."""
from copy import deepcopy
import json
from importlib.metadata import distribution, entry_points, version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLFieldCritic, HNDLParticleProjection
from particle_sliders.field_critic import FieldCritic
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, open_session
from particle_sliders.pg09 import _hash
from particle_sliders.pg094 import validate_pg09_domain
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.routed_projection import TinyParticleProjection
from sana_particle_gan.backend import (
    BACKEND, COHORT, INFERENCE, SanaParticleSlider, SanaResponseAdapter,
    configuration, staged_model,
)

IDENTITY = {'qualification': 'CPU fixture only', 'fixture': 'native Sana API, not pretrained'}
torch.set_num_threads(1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(6, 6)
        self.to_k = nn.Linear(6, 6)
        self.to_v = nn.Linear(6, 6)
        self.to_out = nn.ModuleList([nn.Linear(6, 6)])

    def forward(self, value, text):
        return self.to_out[0](self.to_q(value) + self.to_k(text) + self.to_v(text))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attention()
        self.attn2 = Attention()

    def forward(self, value, text):
        return self.attn2(self.attn1(value, value), text)


class NativeSana(nn.Module):
    """Exact forward keywords + native field axis, mask, doubled output and time."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_layers=1, in_channels=2, out_channels=4, timestep_scale=2.)
        self.input = nn.Linear(2, 6)
        self.transformer_blocks = nn.ModuleList([Block()])
        self.output = nn.Linear(6, 2)
        self.calls = []

    def forward(self, hidden_states, *, encoder_hidden_states, encoder_attention_mask,
                timestep, return_dict):
        self.calls.append((timestep.detach().clone(), encoder_attention_mask.detach().clone()))
        batch, channels, height, width = hidden_states.shape
        value = self.input(hidden_states.flatten(2).transpose(1, 2))
        mask = encoder_attention_mask[..., None]
        text = (encoder_hidden_states * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        for block in self.transformer_blocks:
            value = block(value + timestep[:, None, None] / 1000, text)
        velocity = self.output(value).transpose(1, 2).reshape(batch, channels, height, width)
        # Native Sana may output a second learned-variance half.
        return (torch.cat((velocity, torch.full_like(velocity, 100.)), dim=1),)


def bundle():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = NativeSana().requires_grad_(False)
        rows = []
        unconditional = {'embeds': torch.randn(1, 3, 6), 'mask': torch.tensor([[1, 1, 0]]), 'prompt': ''}
        for _ in range(3):
            def condition():
                return {'embeds': torch.randn(1, 3, 6), 'mask': torch.tensor([[1, 1, 0]])}
            rows.append(ResponseRow(condition(), condition(), deepcopy(unconditional),
                torch.randn(1, 2, 3, 5), timestep=torch.tensor(250.)))
    config = configuration(model, model_identity=IDENTITY, response_shape=(1, 2, 3, 5), unconditional=rows[0].negative)
    return model, rows, config


def opened(model, rows, config, **kwargs):
    return open_session(config, model=model, model_identity=IDENTITY, rows=rows, **kwargs)


def test_installed_distribution_and_entrypoints():
    import sana_particle_gan, particle_sliders
    assert 'site-packages' in sana_particle_gan.__file__
    assert 'site-packages' in particle_sliders.__file__
    assert version('particle-sliders-core') == '0.12.0'
    assert version('particlegan') == '0.9.4'
    assert version('hndl') == '0.10.0'
    assert next(iter(entry_points(group='particle_sliders.image_backends', name=BACKEND.name))).load() is BACKEND
    assert next(iter(entry_points(group='particle_sliders.inference_backends', name=INFERENCE.name))).load() is INFERENCE
    requirements = distribution('particle-sliders-sana').requires
    assert any('4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231' in r for r in requirements)
    assert not torch.cuda.is_initialized()


def test_native_teacher_student_mask_geometry_and_ownership():
    model, rows, config = bundle()
    base = deepcopy(model.state_dict())
    forwards = [m.forward for m in model.modules()]
    with opened(model, rows, config) as session:
        adapter, slider = session.game.backend, session.game.network
        assert isinstance(adapter, SanaResponseAdapter)
        assert isinstance(session.game.critic, HNDLFieldCritic)
        from particle_sliders.fibo_hndl import verified_builder
        assert verified_builder()[0].__module__ == 'hypergan.hndl_networks'
        assert all(isinstance(p, HNDLParticleProjection) for p in slider.adapters)
        assert all(type(p.graph).__module__ == 'hndl.torch' for p in slider.adapters)
        assert type(session.game.critic.graph).__module__ == 'hndl.torch'
        assert sum(p is slider.particles for p in slider.parameters()) == 1
        assert not any('particles' in key for p in slider.adapters for key in p.state_dict())
        assert not ({id(p) for p in model.parameters()} & {id(p) for p in slider.parameters()})
        assert not any(p.requires_grad for p in model.parameters())
        prepared = session.game.rows[0]
        row = prepared.row
        raw = lambda condition: adapter.velocity(condition, row.latent, row.timestep, scale=0.)
        with torch.no_grad():
            uncond, positive = raw(row.negative), raw(row.positive)
            torch.testing.assert_close(prepared.target, uncond + 4.5 * (positive - uncond), rtol=0, atol=0)
            torch.testing.assert_close(session.evaluate(strength=0.), raw(row.neutral), rtol=0, atol=0)
        assert prepared.target.shape == row.latent.shape == (1, 2, 3, 5)
        assert torch.equal(model.calls[-1][0], torch.tensor([500.]))
        assert torch.equal(model.calls[-1][1], row.neutral['mask'])
        # Nonzero up matrices exercise native input and particle gradients.
        for projection in slider.adapters:
            nn.init.normal_(projection.lora_up.weight, std=.03)
        latent = row.latent.clone().requires_grad_()
        prediction = adapter.velocity(row.neutral, latent, row.timestep, scale=.5)
        prediction.square().sum().backward()
        assert latent.grad is not None and latent.grad.abs().sum() > 0
        assert slider.particles.grad is not None and slider.particles.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.parameters())
        for key, value in base.items():
            assert torch.equal(value, model.state_dict()[key])
        with slider.scaled(.7):
            with pytest.raises(RuntimeError):
                with slider.scaled(.2):
                    raise RuntimeError('restore nested scale')
            assert all(p.multiplier == .7 for p in slider.adapters)
        assert all(p.multiplier == 0 for p in slider.adapters)
    assert forwards == [m.forward for m in model.modules()]
    assert model.training  # Borrowed caller mode restored.


def test_actual_shared_updates_exact_resume_ema_and_native_only_reload(tmp_path, monkeypatch):
    model, rows, config = bundle()
    rng = torch.get_rng_state().clone()
    with opened(model, rows, config) as full:
        bank = full.game.network.particles.detach().clone()
        critic = deepcopy(full.game.critic.state_dict())
        full.train()
        assert not torch.equal(bank, full.game.network.particles)
        assert any(not torch.equal(v, full.game.critic.state_dict()[k]) for k, v in critic.items())
        expected = full.checkpoint()['sha256']
        assert full.game._signature['resolved_penalty_domain'] == 'image'
        assert full.game.response_shape == (1, 2, 3, 5)
    model, rows, _ = bundle()
    with opened(model, rows, config) as split:
        split.train(stop_after=2)
        split.checkpoint(tmp_path / 'training.pt')
    model, rows, _ = bundle()
    with opened(model, rows, config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
        outputs = [resumed.evaluate(strength=s) for s in (0., .5, 1.)]
        assert all(torch.isfinite(v).all() for v in outputs)
        assert not torch.equal(outputs[0], outputs[-1])
        for selection in ('raw', 'ema'):
            resumed.export_inference(tmp_path / f'{selection}.pt', selection=selection)
        ema_values = deepcopy(resumed.ema.values)
        native_row = deepcopy(rows[0])
    # No training rows, checkpoint, game or critic construction may be used.
    del rows
    (tmp_path / 'training.pt').unlink()
    import sana_particle_gan.backend as backend
    monkeypatch.setattr(backend, 'ConditionalPG094', lambda *a, **k: pytest.fail('game in inference'))
    monkeypatch.setattr(backend, 'HNDLFieldCritic', lambda *a, **k: pytest.fail('critic in inference'))
    for selection in ('raw', 'ema'):
        model, _, _ = bundle()
        with staged_model(model, model_identity=IDENTITY):
            with open_inference(tmp_path / f'{selection}.pt', operation='response') as native:
                assert not hasattr(native, 'game')
                if selection == 'ema':
                    for k, v in native.binding.network.state_dict().items():
                        assert torch.equal(v, ema_values[k])
                for strength in (0., .5, 1.):
                    value = native.response(strength=strength, condition=native_row.neutral,
                        latent=native_row.latent, timestep=native_row.timestep)
                    assert value.shape == (1, 2, 3, 5) and torch.isfinite(value).all()
                    if selection == 'raw':
                        torch.testing.assert_close(value, outputs[(0., .5, 1.).index(strength)], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('change', ['source', 'profile', 'shape', 'model', 'weights', 'mask', 'target'])
def test_reject_before_hooks_and_borrowed_state_mutation(change):
    model, rows, config = bundle()
    options = config.options
    if change == 'source':
        options['binding_source_sha256'] = '0' * 64
    elif change == 'profile':
        options['pg_source_profile'] = 'release-0.9.3'
    elif change == 'shape':
        options['response_shape'] = [1, 30]
    elif change == 'model':
        options['model_identity']['declaration']['fixture'] = 'other'
    elif change == 'weights':
        with torch.no_grad():
            model.output.weight.add_(1)
    elif change == 'mask':
        rows[0].neutral['mask'][0, 1] = 2
    elif change == 'target':
        model.transformer_blocks[0].attn2.to_q = nn.Identity()
    from particle_sliders.lifecycle import SessionConfig
    changed = SessionConfig(config.backend, config.cohort, **options)
    forwards = [m.forward for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    state = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        opened(model, rows, changed)
    assert forwards == [m.forward for m in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]
    for k, v in state.items():
        assert torch.equal(v, model.state_dict()[k])
    assert not model.calls


def test_builder_tamper_rejects_before_hooks(monkeypatch):
    model, rows, config = bundle()
    import particle_sliders.fibo_hndl as shared
    old = shared._sha
    monkeypatch.setattr(shared, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else old(p))
    forwards = [m.forward for m in model.modules()]
    with pytest.raises(ValueError, match='builder closure'):
        opened(model, rows, config)
    assert forwards == [m.forward for m in model.modules()] and not model.calls


def test_checkpoint_rejection_preserves_state_and_rejects_unversioned_mapping(tmp_path):
    model, rows, config = bundle()
    with opened(model, rows, config) as session:
        session.step()
        packet = session.checkpoint()
        altered = deepcopy(packet)
        altered['payload']['game']['config']['backend']['teacher_guidance'] = 1.
        altered['sha256'] = _hash({k: v for k, v in altered.items() if k != 'sha256'})
        before = session.checkpoint()['sha256']
        with pytest.raises(CheckpointError):
            session.restore(altered)
        assert session.checkpoint()['sha256'] == before
        with pytest.raises(CheckpointError):
            session.restore({'xattn': torch.ones(2)})
    # Different cohort refused by shared preflight before the factory is opened.
    from particle_sliders.lifecycle import SessionConfig
    path = tmp_path / 'training.pt'
    torch.save(packet, path)
    model, rows, _ = bundle()
    with pytest.raises(CheckpointError):
        opened(model, rows, SessionConfig(config.backend, COHORT + '-other', **config.options), resume=path)
    assert not model.calls


def _copy(destination, source):
    with torch.no_grad():
        destination.weight.copy_(source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)


def test_native_routed_projection_matched_outputs_input_particle_vjps():
    model, _, _ = bundle()
    twin = deepcopy(model)
    shared = SanaParticleSlider(model)
    reference_bank = nn.Parameter(shared.particles.detach().clone())
    host = twin.transformer_blocks[0].attn2.to_q
    former = TinyParticleProjection('matched architecture', host, reference_bank, 8, 8)
    actual = shared.adapters[0]
    _copy(actual.graph['down'], former.lora_down)
    nn.init.normal_(former.lora_up.weight, std=.1)
    _copy(actual.graph['up'], former.lora_up)
    for name, group in [('router', former.bridge.router), ('bridge', former.bridge.net)]:
        for index in (0, 2, 4, 6):
            _copy(actual.graph[f'{name}{index}'], group[index])
    x = torch.randn(1, 15, 6, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    former.multiplier = actual.multiplier = .5
    a, b = former(x), actual(y)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.square().sum(), (x, reference_bank))
    gb = torch.autograd.grad(b.square().sum(), (y, shared.particles))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
    shared.close()


def test_native_field_critic_matched_second_derivatives():
    former, actual = FieldCritic(2, 3), HNDLFieldCritic(2, 3, (3, 5))
    for name, layer in [('field0', former.field[0]), ('field2', former.field[2]),
                        ('condition', former.condition), ('score0', former.score[0]), ('score2', former.score[2])]:
        _copy(actual.graph[name], layer)
    x = torch.randn(2, 2, 3, 5, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    condition = torch.randn(2, 3)
    a, b = former(x, condition), actual(y, condition)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.sum(), x, create_graph=True)[0]
    gb = torch.autograd.grad(b.sum(), y, create_graph=True)[0]
    torch.testing.assert_close(ga, gb)
    da = torch.autograd.grad(ga.square().sum(), former.field[0].weight)[0]
    db = torch.autograd.grad(gb.square().sum(), actual.graph['field0'].weight)[0]
    torch.testing.assert_close(da, db, rtol=3e-5, atol=2e-7)


def test_wide_vector_guard_and_no_render_qualification():
    with pytest.raises(ValueError, match='width >=170'):
        validate_pg09_domain((1, 4096), response_domain='continuous_vector')
    model, rows, config = bundle()
    from particle_sliders.lifecycle import UnsupportedOperation
    with pytest.raises(UnsupportedOperation):
        BACKEND.open(config, operation='render', model=model, model_identity=IDENTITY, rows=rows)
    assert not model.calls


def test_shared_source_tamper_refused_before_attachment(monkeypatch):
    model, rows, config = bundle()
    original = Path.read_bytes
    def altered(path):
        result = original(path)
        return result + b'\n# tampered\n' if path.name == 'pg09_images.py' else result
    monkeypatch.setattr(Path, 'read_bytes', altered)
    forwards = [m.forward for m in model.modules()]
    with pytest.raises(ValueError, match='shared core source/data differs'):
        opened(model, rows, config)
    assert forwards == [m.forward for m in model.modules()] and not model.calls


def test_partial_graph_construction_rolls_back_without_attachment(monkeypatch):
    model, _, _ = bundle()
    forwards = [m.forward for m in model.modules()]
    import sana_particle_gan.backend as backend
    from particle_sliders.fibo_hndl import verified_builder
    real, identity = verified_builder()
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('interrupted third graph')
        return real(*args, **kwargs)
    monkeypatch.setattr(backend, 'verified_builder', lambda: (interrupted, identity))
    with pytest.raises(RuntimeError, match='third graph'):
        SanaParticleSlider(model)
    assert len(calls) == 3 and not model.calls
    assert forwards == [m.forward for m in model.modules()]
    assert model.training and not any(p.requires_grad for p in model.parameters())


def test_native_reload_changed_model_is_rejected_before_attachment(tmp_path):
    model, rows, config = bundle()
    with opened(model, rows, config) as session:
        session.export_inference(tmp_path / 'native.pt')
    model, _, _ = bundle()
    with torch.no_grad():
        model.output.bias.add_(1)
    forwards = [m.forward for m in model.modules()]
    with staged_model(model, model_identity=IDENTITY):
        with pytest.raises(ValueError, match='before attachment'):
            open_inference(tmp_path / 'native.pt', operation='response')
    assert forwards == [m.forward for m in model.modules()] and not model.calls


@pytest.mark.parametrize('change', ['missing_prompt', 'negative_prompt', 'different_encoding'])
def test_negative_slot_requires_declared_empty_prompt_and_matching_encoding(change):
    model, rows, config = bundle()
    if change == 'missing_prompt':
        rows[0].negative.pop('prompt')
    elif change == 'negative_prompt':
        rows[0].negative['prompt'] = 'a sad person'
    else:
        rows[0].negative['embeds'].add_(1)
    with pytest.raises(ValueError, match='negative slot'):
        opened(model, rows, config)
    assert not model.calls
    assert all('forward' not in m.__dict__ for m in model.modules())


@pytest.mark.parametrize('guidance', [True, False, None, '4.5', float('nan'), float('inf'), -1.])
def test_guidance_rejects_bools_and_invalid_numbers(guidance):
    model, rows, config = bundle()
    with pytest.raises(ValueError, match='teacher CFG'):
        configuration(model, model_identity=IDENTITY, response_shape=(1, 2, 3, 5),
                      unconditional=rows[0].negative, guidance=guidance)
    from particle_sliders.lifecycle import SessionConfig
    # Nonfinite values are already refused by SessionConfig's JSON boundary.
    if type(guidance) in (int, float) and not torch.isfinite(torch.tensor(guidance)):
        return
    options = config.options
    options['guidance'] = guidance
    with pytest.raises(ValueError, match='teacher CFG'):
        opened(model, rows, SessionConfig(config.backend, config.cohort, **options))
    assert not model.calls


@pytest.mark.parametrize('filename', ['pg094_release.json', 'fibo_projection.hndl', 'LICENSE'])
def test_complete_core_data_tamper_refused_before_hooks(monkeypatch, filename):
    model, rows, config = bundle()
    original = Path.read_bytes
    def altered(path):
        data = original(path)
        return data + b'\nchanged\n' if path.name == filename and path.parent.name == 'particle_sliders' else data
    monkeypatch.setattr(Path, 'read_bytes', altered)
    with pytest.raises(ValueError, match='shared core source/data differs'):
        opened(model, rows, config)
    assert not model.calls


@pytest.mark.parametrize('provenance', [None, {},
    {'url': 'https://github.com/HyperGAN/particle-sliders.git',
     'subdirectory': 'packages/particle-sliders-core',
     'vcs_info': {'vcs': 'git', 'commit_id': '0' * 40, 'requested_revision': '0' * 40}},
    {'url': 'file:///tmp/particle_sliders_core-0.12.0-py3-none-any.whl',
     'archive_info': {'hashes': {'sha256': '0' * 64}}}])
def test_unrecognized_core_git_or_wheel_identity_refused(monkeypatch, provenance):
    model, rows, config = bundle()
    import sana_particle_gan.backend as backend
    original = backend.distribution
    class WrongDistribution:
        def read_text(self, filename):
            return None if provenance is None else json.dumps(provenance)
    monkeypatch.setattr(backend, 'distribution', lambda name: WrongDistribution()
                        if name == 'particle-sliders-core' else original(name))
    with pytest.raises(ValueError, match='exact core Git commit or candidate wheel provenance'):
        opened(model, rows, config)
    assert not model.calls


def test_close_restores_exact_instance_attributes_and_preserves_foreign_replacement():
    model, _, _ = bundle()
    host = model.transformer_blocks[0].attn2.to_q
    host.forward = host.forward  # Caller-owned instance override, distinct from class descriptor.
    original = host.__dict__['forward']
    other = model.transformer_blocks[0].attn2.to_k
    slider = SanaParticleSlider(model)
    replacement = lambda value: value
    other.forward = replacement
    slider.close()
    assert slider._closed
    assert host.__dict__['forward'] is original
    assert other.__dict__['forward'] is replacement
    assert all('forward' not in h.__dict__ for h in
               (model.transformer_blocks[0].attn2.to_v, model.transformer_blocks[0].attn2.to_out[0]))
    slider.close()
    assert other.forward is replacement


def test_close_retries_partial_forward_restoration(monkeypatch):
    model, _, _ = bundle()
    host = model.transformer_blocks[0].attn2.to_q
    slider = SanaParticleSlider(model)
    original = nn.Module.__delattr__
    calls = []
    def fail_once(module, name):
        if module is host and name == 'forward' and not calls:
            calls.append(name)
            raise RuntimeError('transient owned-forward restoration failure')
        return original(module, name)
    monkeypatch.setattr(nn.Module, '__delattr__', fail_once)
    with pytest.raises(ExceptionGroup, match='cleanup incomplete'):
        slider.close()
    assert not slider._closed and 'forward' in host.__dict__
    assert all('forward' not in record['host'].__dict__ for record in slider._attachments[1:])
    slider.close()
    assert slider._closed and 'forward' not in host.__dict__
    assert model.training


def test_close_retries_native_restore_failure():
    model, _, _ = bundle()
    slider = SanaParticleSlider(model)
    restore = slider._restore
    calls = []
    def fail_once():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('transient native restore failure')
        restore()
    slider._restore = fail_once
    with pytest.raises(ExceptionGroup, match='cleanup incomplete'):
        slider.close()
    assert not slider._closed
    slider.close()
    assert slider._closed and model.training and len(calls) == 2
