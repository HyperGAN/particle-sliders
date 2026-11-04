"""Source-bound continuous-vector admission; CPU conformance, not model quality."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import particlegan
from particlegan import init

from particle_sliders import pg09
from test_conditional_pg09 import equal


def installed_profile():
    root = Path(particlegan.__file__).resolve().parent
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    matches = []
    for filename in ('pg094_release.json', 'pg094_continuous.json'):
        document = json.loads(Path(pg09.__file__).with_name(filename).read_text())
        if actual == document['python_sources']:
            matches.append(document['source_profile'])
    assert len(matches) == 1, 'tests require one exact admitted source environment'
    return matches[0]


@dataclass(frozen=True)
class Row:
    row: dict
    target: torch.Tensor
    neutral: torch.Tensor


class Network(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.particles = nn.Parameter(torch.linspace(-1, 1, 16).reshape(8, 2))
        self.projection = nn.Linear(2, width)
        init.deterministic_orthogonal_(self.projection, seed=0)

    def forward(self, value):
        return self.projection(value + self.particles.mean(0))


class Backend:
    def __init__(self, width):
        self.network = Network(width)

    def metadata(self):
        return {'model': 'bounded CPU vector conformance', 'width': self.network.projection.out_features}

    def generated(self, prepared):
        return prepared.neutral + prepared.row['sign'] * self.network(prepared.row['input'])


class Critic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width + 1, 8), nn.SiLU(), nn.Linear(8, 1))
        init.deterministic_orthogonal_(self.layers, seed=1)

    def forward(self, value, condition):
        return self.layers(torch.cat((value, condition), 1)).squeeze(-1)


def game(width):
    backend = Backend(width)
    neutral = torch.linspace(-1, 1, width).reshape(1, width)
    rows = [Row({'input': torch.tensor([[0.1 * index, -0.2]]), 'sign': sign},
                neutral + sign * 0.2, neutral) for index, sign in enumerate((-1., 1., -1., 1.))]
    return pg09.ConditionalPG09(backend, rows, Critic(width),
        particles=backend.network.particles, conditions=torch.arange(4.).reshape(4, 1),
        response_domain='continuous_vector', response_scale=1., total_steps=4, seed=0,
        pg_source_profile=pg09.PG_CONTINUOUS_SOURCE_PROFILE, penalty_input_domain='continuous',
        recipe_overrides={'lr': 1e-5, 'd_lr_mult': 3., 'prior_lr_mult': 4.})


@pytest.mark.parametrize('width', [16, 169, 170, 1024, 2048, 4096])
def test_source_and_domain_are_both_required_before_native_admission(width):
    current = installed_profile()
    for profile, domain in pg09._DOMAIN_PAIRS:
        if current == profile:
            if profile == pg09.PG_SOURCE_PROFILE and width >= 170:
                with pytest.raises(ValueError, match='width >=170'):
                    pg09.validate_pg09_domain((1, width), response_domain='continuous_vector',
                        pg_source_profile=profile, penalty_input_domain=domain)
            else:
                result = pg09.validate_pg09_domain((1, width), response_domain='continuous_vector',
                    pg_source_profile=profile, penalty_input_domain=domain)
                assert result['resolved_penalty_domain'] == 'continuous'
                assert result['pg_source_profile'] == current
        else:
            with pytest.raises(RuntimeError, match='sources differ'):
                pg09.validate_pg09_domain((1, width), response_domain='continuous_vector',
                    pg_source_profile=profile, penalty_input_domain=domain)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('width', [170, 1024, 2048, 4096])
def test_wide_native_geometry_full_bank_rates_resume_and_inference(width):
    if installed_profile() != pg09.PG_CONTINUOUS_SOURCE_PROFILE:
        with pytest.raises(RuntimeError, match='sources differ'):
            game(width)
        return
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original = game(width)
        assert original.response_shape == (1, width)
        assert original.penalty.regularizer.input_domain == 'continuous'
        assert all(group['lr'] == pytest.approx(3e-5) for group in original.opt_d.param_groups)
        rates = {group['role']: group['lr'] for group in original.opt_g.param_groups}
        assert rates == pytest.approx({'generator': 1e-5, 'prior': 4e-5})
        original.step()
        checkpoint = original.state_dict()
        assert checkpoint['config']['pg_source_profile'] == pg09.PG_CONTINUOUS_SOURCE_PROFILE
        expected = original.step()
        expected_state = original.state_dict()
        restored = game(width)
        restored.load_state_dict(checkpoint)
        assert restored.step() == expected
        equal(expected_state, restored.state_dict())
        exported = original.export_state_dict()
        inference = game(width)
        inference.load_export_state_dict(exported)
        assert torch.equal(original.evaluate(0, seed=0), inference.evaluate(0, seed=0))
        for key, wrong in (('pg_source_profile', pg09.PG_SOURCE_PROFILE),
                           ('penalty_input_domain', 'auto'), ('pg_sources_sha256', '0' * 64)):
            corrupt = deepcopy(checkpoint)
            corrupt['config'][key] = wrong
            before = restored.state_dict()
            with pytest.raises(ValueError, match='mismatch'):
                restored.load_state_dict(corrupt)
            equal(before, restored.state_dict())
    finally:
        torch.set_num_threads(before_threads)


def test_continuous_profile_never_aliases_release_or_admits_image_domain():
    from particle_sliders.pg094 import engine_options
    installed_profile()
    assert engine_options({}) == {'pg_source_profile': pg09.PG_SOURCE_PROFILE, 'penalty_input_domain': 'auto'}
    candidate = {'pg_source_profile': pg09.PG_CONTINUOUS_SOURCE_PROFILE, 'penalty_input_domain': 'continuous'}
    assert engine_options(candidate) == candidate
    for profile, domain in ((pg09.PG_SOURCE_PROFILE, 'continuous'),
                            (pg09.PG_CONTINUOUS_SOURCE_PROFILE, 'auto')):
        with pytest.raises(ValueError, match='qualified source/domain pair'):
            engine_options({'pg_source_profile': profile, 'penalty_input_domain': domain})
        with pytest.raises(ValueError, match='qualified pair'):
            pg09.validate_pg09_domain((1, 1024), response_domain='continuous_vector',
                pg_source_profile=profile, penalty_input_domain=domain)
    if installed_profile() == pg09.PG_CONTINUOUS_SOURCE_PROFILE:
        with pytest.raises(ValueError, match='genuine native'):
            pg09.validate_pg09_domain((1, 4, 4, 4), response_domain='continuous_field',
                pg_source_profile=pg09.PG_CONTINUOUS_SOURCE_PROFILE, penalty_input_domain='continuous')


def test_common_lifecycle_checks_selected_source_before_factory():
    from particle_sliders.lifecycle import Backend, BackendUnavailable, Capabilities, SessionConfig
    calls = []
    def factory(options):
        calls.append('factory')
        raise RuntimeError('source gate passed; bounded factory sentinel')
    backend = Backend('continuous-source-gate', Capabilities('adapter_gan', False), factory, lambda options: None)
    options = {'pg_source_profile': pg09.PG_CONTINUOUS_SOURCE_PROFILE, 'penalty_input_domain': 'continuous'}
    if installed_profile() == pg09.PG_CONTINUOUS_SOURCE_PROFILE:
        with pytest.raises(RuntimeError, match='bounded factory sentinel'):
            backend.open(SessionConfig(backend.name, 'new-explicit-vector-cohort', **options))
        assert calls == ['factory']
    else:
        with pytest.raises((BackendUnavailable, RuntimeError), match='sources differ'):
            backend.open(SessionConfig(backend.name, 'new-explicit-vector-cohort', **options))
        assert not calls
