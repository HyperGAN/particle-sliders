"""Exact release/build identities and unchanged continuous-domain boundaries."""
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path

import pytest
import torch
from particle_sliders import pg09
from particle_sliders.fibo_hndl import PROFILE, verified_builder
from test_conditional_pg09 import fixture, equal


@pytest.mark.parametrize('width', [170, 2048, 4096, 1398528])
def test_native_wide_vectors_remain_refused(width):
    with pytest.raises(ValueError, match='width >=170'):
        pg09.validate_pg09_domain((1, width), response_domain='continuous_vector')
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile,domain', [('release-0.9.3', 'auto'),
    ('release', 'auto'), ('alz-continuous-v1', 'continuous'), ('release-0.9.4', 'continuous')])
def test_old_or_unqualified_profiles_never_become_a_version_override(profile, domain):
    with pytest.raises(ValueError, match='qualified pair'):
        pg09.validate_pg09_domain((1, 2048), response_domain='continuous_vector',
            pg_source_profile=profile, penalty_input_domain=domain)


@pytest.mark.parametrize('key,value', [('pg_version', '0.9.3'),
    ('pg_source_profile', 'release-0.9.3'), ('pg_sources_sha256', 'old-source-map')])
def test_previous_release_checkpoint_identity_refuses_transactionally(key, value):
    game = fixture()
    game.step()
    before = game.state_dict()
    wrong = deepcopy(before)
    wrong['config'][key] = value
    with pytest.raises(ValueError, match='mismatch') as refused:
        game.load_state_dict(wrong)
    assert 'original pinned particle-sliders-core/ParticleGAN source environment' in str(refused.value)
    assert 'fresh cohort' in str(refused.value)
    equal(before, game.state_dict())


def test_exact_actual_framework_develop_identity():
    pytest.importorskip("hndl")
    pytest.importorskip("hypergan.hndl_networks")
    _, identity = verified_builder()
    assert identity['profile'] == PROFILE == 'hypergan-hndl-v2'
    assert identity['hndl_version'] == version('hndl') == '0.10.0'
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert len(identity['hypergan_build_provenance_sha256']) == 64


@pytest.mark.parametrize('change', [{'dirty': True}, {'commit': '0' * 40}])
def test_same_builder_bytes_cannot_claim_wrong_or_dirty_framework_commit(monkeypatch, change):
    pytest.importorskip("hndl")
    pytest.importorskip("hypergan.hndl_networks")
    read = Path.read_text
    def altered(path, *args, **kwargs):
        text = read(path, *args, **kwargs)
        if path.name == '_build_provenance.json':
            document = json.loads(text)
            document.update(change)
            return json.dumps(document)
        return text
    monkeypatch.setattr(Path, 'read_text', altered)
    with pytest.raises(ValueError, match='clean HyperGAN develop build provenance'):
        verified_builder()


def test_old_network_profile_refuses_before_host_mutation():
    from particle_sliders.fibo_native import FiboParticleSlider, target_modules
    from test_fibo_native import Pipeline
    model = Pipeline().transformer
    before = [module.forward for module in target_modules(model).values()]
    with pytest.raises(ValueError, match='unknown Fibo network profile'):
        FiboParticleSlider(model, network_profile='hypergan-hndl-v1')
    assert before == [module.forward for module in target_modules(model).values()]
