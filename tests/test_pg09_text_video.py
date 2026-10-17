from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from conceptmod.textsliders.pg09_backends import ResponseRow
from conceptmod.textsliders.pg09_text_video import (
    TinyQwenResponseAdapter, TextResponseRow, LTX25VideoResponseAdapter,
    ResponseCompatibilityError, require_continuous_vector,
)
from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
from conceptmod.textsliders.tiny_llm_particle import TinyParticleSlider
from conceptmod.textsliders.ltx25_backend import LTX25Backend


class Critic(nn.Module):
    def __init__(self, width, conditions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + conditions, 16), nn.SiLU(), nn.Linear(16, 1))
        self.seen = []

    def forward(self, value, condition):
        self.seen.append(tuple(value.shape))
        return self.net(torch.cat((value.flatten(1), condition), dim=1)).squeeze(-1)


def make(family, preset='bispf_optimized'):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    if family == 'tiny':
        native = TinyLLMBackend(dummy=True)
        slider = TinyParticleSlider(native.model).float()
        adapter = TinyQwenResponseAdapter(native, slider)
        rows = [TextResponseRow(tuple(native.encode(n).ids), tuple(native.encode(p).ids))
                for n, p in [('a person', 'a smiling person'), ('a child', 'a happy child')]]
        domain, particles = 'continuous_vector', adapter.particles
    else:
        native = LTX25Backend(dummy=True, recipe='velocity', lora_rank=2, lora_alpha=2)
        adapter = LTX25VideoResponseAdapter(native)
        video = torch.randn(1, 128, 2, 2, 2)
        audio = torch.randn(1, 2, 8)
        rows = [ResponseRow((n, audio), (p, audio), ('', audio), video * k,
                            timestep=torch.tensor(t))
                for n, p, k, t in [('a person', 'a smiling person', 1, 725.),
                                    ('a child', 'a happy child', .9, 909.375)]]
        domain, particles = 'continuous_field', None
    fixed = [adapter.prepare(row) for row in rows]
    # Context is frozen native neutral response, plus declared sign/time.
    # No row-id proxy or invented context embedding enters the critic.
    conditions = torch.cat([torch.cat((r.neutral.flatten(), torch.tensor([
        r.row.scale, float(r.row.timestep) / 1000 if family == 'ltx' else 0.])))
                            [None] for r in fixed])
    critic = Critic(fixed[0].target.numel(), conditions.shape[1])
    game = ConditionalPG09(adapter, fixed, critic, conditions=conditions, particles=particles,
                           response_domain=domain, preset=preset, batch_size=2,
                           total_steps=4, seed=15)
    return adapter, fixed, game


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('family', ['tiny', 'ltx'])
@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_shared_game_gradient_ownership_native_domain_and_objective(family, preset, monkeypatch):
    adapter, fixed, game = make(family, preset)
    base_module = adapter.backend.model if family == 'tiny' else adapter.backend.transformer
    base = deepcopy(base_module.state_dict())
    before = deepcopy(adapter.network.state_dict())
    def forbidden(*a, **kw):
        raise AssertionError('MSE forbidden')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    for _ in range(3):
        result = game.step()
        assert result['g_loss'] == pytest.approx(result['g_adv'] + result['prior_regularization'])
        assert result['d_loss'] == pytest.approx(result['d_adv'] + result['d_pen'])
    equal(base, base_module.state_dict())
    assert any(not torch.equal(before[n], t) for n, t in adapter.network.state_dict().items())
    assert all(p.grad is not None for p in adapter.network.parameters())
    owned = [p for g in game.opt_g.param_groups for p in g['params']]
    assert len(owned) == len(set(map(id, owned))) == len(list(adapter.network.parameters()))
    assert set(map(id, owned)) == set(map(id, adapter.network.parameters()))
    if family == 'tiny':
        assert game.prior.z is adapter.slider.particles
        assert not torch.equal(before['weights.0'], adapter.network.state_dict()['weights.0'])
        assert game.state_dict()['config']['generator_arm'] == 'routed_particle_gan'
    else:
        assert game.prior is None
        assert all(len(shape) == 5 for shape in game.critic.seen)
        assert fixed[0].target.shape == (1, 128, 2, 2, 2)
        assert game.state_dict()['config']['generator_arm'] == 'adapter_gan'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['tiny', 'ltx'])
@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_exact_resume_evaluation_and_export(family, preset):
    adapter, fixed, game = make(family, preset)
    game.step()
    state = game.state_dict()
    sample = game.evaluate(0, seed=3)
    equal(state, game.state_dict())
    metrics = game.step()
    expected = game.state_dict()
    other, _, resumed = make(family, preset)
    resumed.load_state_dict(state)
    assert torch.equal(sample, resumed.evaluate(0, seed=3))
    assert metrics == resumed.step()
    equal(expected, resumed.state_dict())
    resumed.load_export_state_dict(game.export_state_dict())
    assert torch.equal(game.evaluate(0), resumed.evaluate(0))
    with pytest.raises(RuntimeError, match='inference export'):
        resumed.step()
    for scale in (0, .5, 1):
        if family == 'tiny':
            assert torch.equal(adapter.render(fixed[0].row, scale=scale),
                               other.render(fixed[0].row, scale=scale))
        else:
            assert torch.equal(adapter._response(fixed[0].row, scale=scale),
                               other._response(fixed[0].row, scale=scale))


@pytest.mark.parametrize('family', ['tiny', 'ltx'])
def test_neutral_student_scale_zero_and_row_snapshot(family):
    adapter, fixed, game = make(family)
    row = fixed[0].row
    calls = []
    if family == 'tiny':
        original = adapter.backend.hidden
        def hidden(ids):
            calls.append(tuple(ids))
            return original(ids)
        adapter.backend.hidden = hidden
        adapter.generated(fixed[0])
        assert calls == [row.neutral_ids]
        assert torch.equal(adapter.render(row, scale=0), fixed[0].neutral)
    else:
        original = adapter.backend.encode_text
        def encode(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)
        adapter.backend.encode_text = encode
        adapter.generated(fixed[0])
        assert calls == [row.neutral[0]]
        assert torch.equal(adapter._response(row, scale=0), fixed[0].neutral)
        with pytest.raises(ResponseCompatibilityError, match='raw one-step'):
            adapter.prepare(replace(row, guidance=3))
        with pytest.raises(ResponseCompatibilityError, match='identical audio'):
            adapter.prepare(replace(row, positive=(row.positive[0], row.positive[1] + 1)))
        original_latent, original_audio = row.latent.clone(), row.neutral[1].clone()
        prepared = adapter.prepare(row)
        row.latent.add_(1)
        row.neutral[1].add_(1)
        assert torch.equal(prepared.row.latent, original_latent)
        assert torch.equal(prepared.row.neutral[1], original_audio)


@pytest.mark.parametrize('family,width', [('TinyQwen', 1024), ('Bonsai', 5120), ('boundary', 170)])
def test_wide_vectors_fail_closed(family, width):
    with pytest.raises(ResponseCompatibilityError, match=f'width {width}'):
        require_continuous_vector(width, family)


def test_default_ltx_embeddings_not_silently_switched():
    with pytest.raises(ResponseCompatibilityError, match='connector-embedding'):
        LTX25VideoResponseAdapter(LTX25Backend(dummy=True))


def test_tiny_wrong_attached_backend_rejected():
    first, second = TinyLLMBackend(dummy=True), TinyLLMBackend(dummy=True)
    slider = TinyParticleSlider(first.model).float()
    with pytest.raises(ResponseCompatibilityError, match='exact backend'):
        TinyQwenResponseAdapter(second, slider)


def test_tiny_actual_wide_config_fails_before_response_or_optimizer():
    backend = TinyLLMBackend(dummy=True)
    slider = TinyParticleSlider(backend.model).float()
    backend.model.config['hidden_size'] = 1024
    with pytest.raises(ResponseCompatibilityError, match='width 1024'):
        TinyQwenResponseAdapter(backend, slider)


def test_ltx_reversible_scale_and_coordinate_order():
    adapter, fixed, _ = make('ltx')
    native = adapter.backend.network
    previous = [m.multiplier for m in native.loras]
    old = native.lora_scale
    with adapter.network.scaled(-.4):
        adapter.generated(fixed[0])
    assert old == native.lora_scale
    assert previous == [m.multiplier for m in native.loras]
    # Backend receives exactly native time-major/spatial-major video tokens.
    observed = []
    original = adapter.backend.forward_velocity
    def forward(packed, **kw):
        observed.append(packed)
        return original(packed, **kw)
    adapter.backend.forward_velocity = forward
    adapter.generated(fixed[0])
    packed = observed[0]
    volume = fixed[0].row.latent
    for t in range(2):
        for h in range(2):
            for w in range(2):
                assert torch.equal(packed.hidden_states[0, 4*t+2*h+w], volume[0, :, t, h, w])
    assert torch.all(packed.timestep == 725)
    assert torch.all(packed.audio_timestep == 725)


@pytest.mark.parametrize('family', ['tiny', 'ltx'])
def test_source_and_coordinate_signature_bound(family):
    adapter, fixed, game = make(family)
    sources = game.state_dict()['config']['implementation_sources']
    assert 'conceptmod.textsliders.pg09_text_video' in sources
    assert ('conceptmod.textsliders.tiny_llm_particle' if family == 'tiny'
            else 'conceptmod.textsliders.ltx25_backend') in sources
    state = game.state_dict()
    state['config']['conditions_sha256'] = 'wrong'
    before = game.state_dict()
    with pytest.raises(ValueError, match='conditioning mismatch'):
        game.load_state_dict(state)
    equal(before, game.state_dict())


def test_ltx_checkpoint_recomputation_captures_scale():
    from conceptmod.textsliders.pg09_backends import checkpoint_with_adapter_scale
    adapter, fixed, _ = make('ltx')
    first = adapter.generated(fixed[0])
    first.sum().backward()
    expected = [p.grad.clone() for p in adapter.network.parameters()]
    adapter.network.zero_grad(set_to_none=True)
    for block in adapter.backend.transformer.transformer_blocks:
        checkpoint_with_adapter_scale(block, adapter.network)
    second = adapter.generated(fixed[0])
    second.sum().backward()
    assert torch.equal(first, second)
    for p, e in zip(adapter.network.parameters(), expected):
        assert torch.equal(p.grad, e)


def test_tiny_fp32_adapter_on_bf16_frozen_host():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    native = TinyLLMBackend(dummy=True)
    native.model.to(dtype=torch.bfloat16)
    slider = TinyParticleSlider(native.model).float()
    adapter = TinyQwenResponseAdapter(native, slider)
    rows = [adapter.prepare(TextResponseRow(tuple(native.encode(n).ids), tuple(native.encode(p).ids)))
            for n, p in [('a person', 'a smiling person'), ('a child', 'a happy child')]]
    conditions = torch.cat([r.neutral for r in rows])
    game = ConditionalPG09(adapter, rows, Critic(32, 32), conditions=conditions,
                           particles=adapter.particles, response_domain='continuous_vector',
                           total_steps=2, batch_size=2)
    base = deepcopy(native.model.state_dict())
    game.step()
    equal(base, native.model.state_dict())
    assert all(p.dtype == torch.float32 and p.grad is not None for p in adapter.network.parameters())
    assert torch.equal(adapter.render(rows[0].row, scale=0), rows[0].neutral)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_ltx_shared_checkpoint_helper_through_complete_game_step(preset):
    from conceptmod.textsliders.pg09_backends import checkpoint_with_adapter_scale
    plain, _, first = make('ltx', preset)
    checkpointed, _, second = make('ltx', preset)
    for block in checkpointed.backend.transformer.transformer_blocks:
        checkpoint_with_adapter_scale(block, checkpointed.network)
    # Includes protected-geometry's repeated gradient queries and the penalty,
    # rather than testing only a single ordinary backward.
    assert first.step() == second.step()
    equal(first.state_dict(), second.state_dict())
