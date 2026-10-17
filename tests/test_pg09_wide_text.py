from copy import deepcopy

import pytest
import torch

from particle_sliders.pg09 import ConditionalPG09
from conceptmod.textsliders.pg09_text_video import (
    TinyQwenResponseAdapter, TextResponseRow, BonsaiReadoutResponseAdapter, BonsaiResponseRow,
    ResponseCompatibilityError,
)
from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend, DummyTinyCausalLM
from conceptmod.textsliders.tiny_llm_particle import TinyParticleSlider
from conceptmod.textsliders.bonsai_gguf_backend import BonsaiGGUFBackend
from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
from test_pg09_text_video import Critic, equal


def make_wide(family, preset='bispf_optimized', *, profile='alz-continuous-v1', width=1024):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    options = dict(pg_source_profile=profile,
                   penalty_input_domain='continuous' if profile == 'alz-continuous-v1' else 'auto')
    if family == 'tiny':
        native = TinyLLMBackend(dummy=True)
        native.model = DummyTinyCausalLM(hidden_size=width, num_hidden_layers=1,
                                        intermediate_size=64, vocab_size=64)
        native.model.eval().requires_grad_(False)
        slider = TinyParticleSlider(native.model).float()
        adapter = TinyQwenResponseAdapter(native, slider, **options)
        rows = [TextResponseRow(tuple(native.encode(n).ids), tuple(native.encode(p).ids))
                for n, p in [('a person', 'a smiling person'), ('a child', 'a happy child')]]
    else:
        native = BonsaiGGUFBackend(dummy=True)
        head = BonsaiParticleHead().float()
        adapter = BonsaiReadoutResponseAdapter(native, head,
                    model_identity={'qualification': 'CPU synthetic native-width fixture'}, **options)
        rows = [BonsaiResponseRow(n, p) for n, p in
                [('a garden', 'a rose garden'), ('a sea', 'a turquoise sea')]]
    fixed = [adapter.prepare(row) for row in rows]
    conditions = torch.cat([torch.cat((r.neutral.flatten(), torch.tensor([1., 0.])))
                            [None] for r in fixed])
    critic = Critic(fixed[0].target.numel(), conditions.shape[1])
    game = ConditionalPG09(adapter, fixed, critic, conditions=conditions, particles=adapter.particles,
                           **adapter.domain_options, preset=preset, total_steps=4, batch_size=2, seed=15)
    return adapter, fixed, game


@pytest.mark.parametrize('family', ['tiny', 'bonsai'])
@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_explicit_native_width_game_has_actual_routed_bank_and_gan_only(family, preset, monkeypatch):
    adapter, fixed, game = make_wide(family, preset)
    before = deepcopy(adapter.network.state_dict())
    bank = adapter.particles.detach().clone()
    critic = deepcopy(game.critic.state_dict())
    base = deepcopy(adapter.backend.model.state_dict()) if family == 'tiny' else None
    def forbidden(*a, **kw):
        raise AssertionError('optimized MSE forbidden')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    for _ in range(3):
        result = game.step()
        assert result['g_loss'] == pytest.approx(result['g_adv'] + result['prior_regularization'])
        assert result['d_loss'] == pytest.approx(result['d_adv'] + result['d_pen'])
        assert result['d_pen'] > 0
        assert game.penalty.last_stats['input_domain'] == 'continuous'
        assert game.penalty.last_stats['resolved_domain'] == 'continuous'
    assert not torch.equal(bank, adapter.particles)
    assert any(not torch.equal(before[k], v) for k, v in adapter.network.state_dict().items())
    assert any(not torch.equal(critic[k], v) for k, v in game.critic.state_dict().items())
    assert all(p.grad is not None for p in adapter.network.parameters())
    assert game.prior.z is adapter.particles
    owned = [p for group in game.opt_g.param_groups for p in group['params']]
    assert len(owned) == len(set(map(id, owned)))
    assert set(map(id, owned)) == set(map(id, adapter.network.parameters()))
    if family == 'tiny':
        equal(base, adapter.backend.model.state_dict())
    else:
        assert torch.equal(adapter.render(fixed[0].row, scale=0), fixed[0].neutral)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['tiny', 'bonsai'])
@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_native_width_resume_export_and_source_profile(family, preset, tmp_path):
    adapter, rows, game = make_wide(family, preset)
    game.step()
    before = game.state_dict()
    sample = game.evaluate(0, seed=18)
    equal(before, game.state_dict())
    torch.save(before, tmp_path / 'state.pt')
    safe = torch.load(tmp_path / 'state.pt', weights_only=True)
    metrics = game.step()
    expected = game.state_dict()
    other, _, resumed = make_wide(family, preset)
    resumed.load_state_dict(safe)
    assert metrics == resumed.step()
    equal(expected, resumed.state_dict())
    resumed.load_export_state_dict(game.export_state_dict())
    for scale in (0, .5, 1):
        assert torch.equal(adapter.render(rows[0].row, scale=scale),
                           other.render(rows[0].row, scale=scale))
    state = game.state_dict()
    assert state['config']['pg_source_profile'] == 'alz-continuous-v1'
    assert state['config']['penalty_input_domain'] == 'continuous'
    assert state['config']['resolved_penalty_domain'] == 'continuous'
    state['config']['pg_source_profile'] = 'release'
    before = game.state_dict()
    with pytest.raises(ValueError, match='mismatch'):
        game.load_state_dict(state)
    equal(before, game.state_dict())


def test_bonsai_release_guard_runs_before_backend_or_head_access():
    class NeverRead:
        def __getattr__(self, key):
            raise AssertionError('must reject before model/readout access')
    with pytest.raises(ResponseCompatibilityError, match='width 5120'):
        BonsaiReadoutResponseAdapter(NeverRead(), NeverRead(), model_identity={'test': True})


def test_tiny_release_guard_keeps_native_1024_blocked():
    native = TinyLLMBackend(dummy=True)
    native.model.config['hidden_size'] = 1024
    with pytest.raises(ResponseCompatibilityError, match='width 1024'):
        TinyQwenResponseAdapter(native, object())
