"""Explicit release/profile admission and same-version identity boundaries."""
from copy import deepcopy
import importlib.metadata as metadata

import pytest
import torch

from particle_sliders.lifecycle import Backend, BackendUnavailable, Capabilities, SessionConfig
from particle_sliders.pg093 import ConditionalPG093, PG_VERSION, PG_SOURCE_PROFILE, engine_options
from particle_sliders import pg09
from test_conditional_pg09 import fixture, equal


def test_explicit_release_manifest_and_public_alias():
    assert PG_VERSION == '0.9.3'
    assert PG_SOURCE_PROFILE == 'release-0.9.3'
    assert ConditionalPG093 is pg09.ConditionalPG09
    assert pg09._released_sources() == '966027b710f3abb48ece4cfe0261f12fd119fc8e29eb43aa2f8957e406ecdae1'
    assert engine_options({'pg_version': '0.9.3'}) == dict(pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain='auto')


@pytest.mark.parametrize('options', [dict(pg_version='0.9.1'), dict(pg_source_profile='release'),
                                    dict(pg_source_profile='alz-continuous-v1'), dict(penalty_input_domain='continuous')])
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
    with pytest.raises(BackendUnavailable, match='0.9.3'):
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
