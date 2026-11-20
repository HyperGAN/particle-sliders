"""Explicit release/profile admission and same-version identity boundaries."""
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata as metadata

import pytest
import torch

from particle_sliders.lifecycle import Backend, BackendUnavailable, Capabilities, SessionConfig
from particle_sliders import engine
from particle_sliders.engine import (ConditionalParticleGAN, PG_SOURCE_PROFILE, PG_VERSION,
                                     engine_options)
from test_conditional_engine import fixture, equal


def test_explicit_release_manifest_and_public_export():
    import particle_sliders
    assert PG_VERSION == '0.10.1'
    assert PG_SOURCE_PROFILE == 'release-0.10.1'
    assert particle_sliders.ConditionalParticleGAN is ConditionalParticleGAN
    assert engine._released_sources() == 'cd1a8b2e1b2a05721fb994fd267a78637c883e6576bc2ffa0cf3d923295948c6'
    assert engine_options({'pg_version': '0.10.1'}) == dict(pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain='auto')


def test_default_game_resolves_upstream_recipe_without_hidden_decay_or_ema(monkeypatch):
    from particle_sliders.lifecycle import EMA
    from particle_sliders.optimizer_policy import configure_dense_profile

    calls = []
    original = engine.get_recipe

    def resolve(name, **options):
        calls.append((name, options))
        return original(name, **options)

    monkeypatch.setattr(engine, 'get_recipe', resolve)
    game = fixture()
    assert len(calls) == 1 and calls[0][0] == 'bispf_optimized'
    name, options = calls[0]
    resolved, _ = configure_dense_profile(original(name, **options))
    assert asdict(game.recipe) == asdict(resolved)
    assert game.recipe.ema_decay == 0
    assert EMA(game.network).values is None
    rates = [[group['lr'] for group in optimizer.param_groups]
             for optimizer in (game.opt_g, game.opt_d)]
    # Inspect the scheduler at all historical phase boundaries. These are
    # schedule queries, not optimizer updates or fabricated training steps.
    for step in (0, 199, 200, 799, 800, 959, 960, 1024, 1600):
        engine.scale_learning_rates(step, game.recipe, (game.opt_g, game.opt_d),
                                 game.base_lrs, game.prior)
        assert [[group['lr'] for group in optimizer.param_groups]
                for optimizer in (game.opt_g, game.opt_d)] == rates
    assert game.completed_steps == 0 and not torch.cuda.is_initialized()


@pytest.mark.parametrize('options', [dict(pg_version='0.9.3'), dict(pg_source_profile='release-0.9.3'), dict(pg_version='0.9.1'), dict(pg_source_profile='release'),
                                    dict(pg_source_profile='alz-continuous-v1'), dict(pg_version='0.9.5')])
def test_no_version_or_overlay_fallback_before_factory(options):
    calls = []
    contract = Backend('cpu', Capabilities('adapter_gan', False),
                       lambda options: calls.append('allocate'), lambda options: None)
    with pytest.raises(BackendUnavailable):
        contract.open(SessionConfig('cpu', 'new093', **options))
    assert calls == []
    with pytest.raises(ValueError):
        engine_options(options)


def test_installed_wrong_version_refused_before_factory(monkeypatch):
    original = metadata.version
    monkeypatch.setattr(metadata, 'version', lambda name: '0.9.1' if name == 'particlegan' else original(name))
    contract = Backend('cpu', Capabilities('adapter_gan', False), lambda options: pytest.fail('allocated'), lambda options: None)
    with pytest.raises(BackendUnavailable, match='0.10.1'):
        contract.open(SessionConfig('cpu', 'new093'))


def test_old_version_and_source_receipts_cannot_rebind():
    game = fixture()
    game.step()
    before = game.state_dict()
    for key, wrong in [('pg_version', '0.9.1'), ('pg_source_profile', 'release'), ('pg_sources_sha256', 'other')]:
        packet = deepcopy(before)
        packet['config'][key] = wrong
        with pytest.raises(ValueError, match='mismatch'):
            game.load_state_dict(packet)
        equal(before, game.state_dict())
    assert not torch.cuda.is_initialized()
