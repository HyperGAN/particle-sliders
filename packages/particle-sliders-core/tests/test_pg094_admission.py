"""Exact release/build identities and preserved auto-dispatch domain boundaries."""
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path

import pytest
import torch
from particle_sliders import pg09
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








def test_previous_host_source_refusal_explains_original_cohort_without_mutation():
    game = fixture()
    game.step()
    before = game.state_dict()
    wrong = deepcopy(before)
    wrong['config']['implementation_sources']['particle_sliders.pg09'] = 'previous-core-source'
    with pytest.raises(ValueError, match='original pinned source/model/data cohort'):
        game.load_state_dict(wrong)
    equal(before, game.state_dict())
