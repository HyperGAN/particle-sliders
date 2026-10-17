"""CPU gates for objective, domains, ownership, conditions and transactional resume."""
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.reference import RoutedMLP


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@dataclass(frozen=True)
class Prepared:
    row: dict
    target: torch.Tensor
    neutral: torch.Tensor


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(8, 2))
        self.routed = RoutedMLP(2, 2, z_dim=2, width=6, router_width=4)

    def forward(self, x):
        return self.routed(torch.nn.functional.dropout(x, .1, training=True), self.particles)


class Backend:
    def __init__(self):
        self.network = Network()
        self.detached = False

    def metadata(self):
        return dict(model='bounded-cpu-fixture', protocol='neutral-signed-native-v1',
                    cfg_adapter_branch='conditional_only')

    def generated(self, prepared):
        x = self.network(prepared.row['input'])
        shape = prepared.target.shape
        response = prepared.neutral + x.reshape(1, 2, 1, 1).expand(shape) * prepared.row['sign']
        return response.detach() if self.detached else response


class Critic(nn.Module):
    def __init__(self, width=200):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width + 3, 8), nn.SiLU(), nn.Linear(8, 1))

    def forward(self, response, condition):
        return self.layers(torch.cat((response.flatten(1), condition), 1)).squeeze(-1)


def fixture(*, particles=True, preset='bispf_optimized', overrides=None, width=200, domain='continuous_field', activation=None):
    with torch.random.fork_rng():
        torch.manual_seed(123)
        backend = Backend()
        critic = Critic(width)
        if activation is not None:
            critic.layers[1] = activation
        generator = torch.Generator().manual_seed(77)
        shape = (1, 2, 10, 10) if domain == 'continuous_field' else (1, width)
        rows = []
        for i in range(6):
            neutral = torch.randn(shape, generator=generator)
            row = dict(input=torch.randn(1, 2, generator=generator), sign=(-1. if i % 2 else 1.))
            rows.append(Prepared(row, neutral + row['sign'] * .2, neutral))
        conditions = torch.tensor([[i, r.row['sign'], 500.] for i, r in enumerate(rows)])
        game = ConditionalPG09(backend, rows, critic, conditions=conditions,
            particles=backend.network.particles if particles else None, preset=preset,
            batch_size=4, total_steps=5, recipe_overrides=overrides, response_domain=domain)
    return game


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_public_objective_ownership_particle_updates_and_condition_binding(monkeypatch):
    game = fixture()
    assert game.prior.z is game.network.particles
    params = [p for group in game.opt_g.param_groups for p in group['params']]
    assert sum(p is game.prior.z for p in params) == 1
    assert len(params) == len(set(map(id, params)))
    before = deepcopy(game.network.state_dict())
    d_before = deepcopy(game.critic.state_dict())
    def forbidden(*a, **kw):
        raise AssertionError('MSE forbidden')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    bindings = []
    original = game.opt_g.set_sampled_rows
    def bind(bank, ids):
        assert bank is game.prior.z
        bindings.append(ids.clone())
        return original(bank, ids)
    monkeypatch.setattr(game.opt_g, 'set_sampled_rows', bind)
    observed = []
    game.critic.register_forward_pre_hook(lambda m, args: observed.append((args[0].shape, args[1].detach().clone())))
    global_rng = torch.get_rng_state().clone()
    metrics = game.step()
    assert torch.equal(global_rng, torch.get_rng_state())
    assert metrics['d_pen'] > 0
    assert metrics['g_loss'] == pytest.approx(metrics['g_adv'] + metrics['prior_regularization'], abs=1e-6)
    assert metrics['d_loss'] == pytest.approx(metrics['d_adv'] + metrics['d_pen'], abs=1e-6)
    assert all(shape == (4, 2, 10, 10) for shape, _ in observed)
    d_condition = game.conditions[metrics['d_rows']]
    g_condition = game.conditions[metrics['g_rows']]
    assert torch.equal(observed[0][1], d_condition)
    assert torch.equal(observed[1][1], d_condition)
    assert torch.equal(observed[-1][1], g_condition)
    assert torch.equal(observed[-2][1], g_condition)
    assert len(bindings) == 1 and torch.equal(bindings[0], torch.arange(8))
    assert not torch.equal(before['particles'], game.network.particles)
    assert any(not torch.equal(before[n], p) for n, p in game.network.state_dict().items() if n != 'particles')
    assert any(not torch.equal(d_before[n], p) for n, p in game.critic.state_dict().items())


def test_exact_stochastic_resume_sampling_and_export_reload():
    game = fixture()
    game.step()
    game.step()
    state = game.state_dict()
    expected = game.step()
    expected_state = game.state_dict()
    other = fixture()
    other.load_state_dict(state)
    before_sample = other.state_dict()
    sample = other.evaluate(0, seed=55)
    equal(before_sample, other.state_dict())
    assert torch.equal(sample, other.evaluate(0, seed=55))
    torch.randn(19)
    assert other.step() == expected
    equal(expected_state, other.state_dict())
    exported = game.export_state_dict()
    loader = fixture()
    loader.load_export_state_dict(exported)
    assert torch.equal(game.evaluate(0, seed=7), loader.evaluate(0, seed=7))
    with pytest.raises(RuntimeError, match='inference export'):
        loader.step()


def test_training_packet_loads_with_weights_only_safe_loader():
    import io
    game = fixture()
    game.step()
    buffer = io.BytesIO()
    torch.save(game.state_dict(), buffer)
    buffer.seek(0)
    packet = torch.load(buffer, weights_only=True)
    other = fixture()
    other.load_state_dict(packet)
    equal(game.state_dict(), other.state_dict())


@pytest.mark.parametrize('preset', ['bispf_optimized', 'bispf'])
def test_detached_or_disconnected_bank_rejected_before_critic_update(preset):
    game = fixture(preset=preset)
    game.backend.detached = True
    before = deepcopy(game.critic.state_dict())
    with pytest.raises(ValueError, match='detached generator'):
        game.step()
    equal(before, game.critic.state_dict())
    with pytest.raises(RuntimeError, match='failed update'):
        game.step()


@pytest.mark.parametrize('width', [170, 1024, 18080])
def test_flat_continuous_domain_never_uses_categorical_fallback(width):
    with pytest.raises(ValueError, match='width >=170'):
        fixture(width=width, domain='continuous_vector')


@pytest.mark.parametrize('options', [{'num_particles': 19}, {'z_dim': 5},
                                   {'batch_size': 8}, {'total_steps': 17}, {'loss_type': 'wasserstein'}])
def test_recipe_cannot_misrepresent_actual_bank_or_objective(options):
    with pytest.raises(ValueError):
        fixture(overrides=options)


@pytest.mark.parametrize('corruption', ['domain', 'core_source', 'module', 'optimizer_groups', 'optimizer_moment', 'rng', 'alias'])
def test_invalid_checkpoint_rejected_transactionally(corruption):
    game = fixture()
    game.step()
    packet = game.state_dict()
    before = game.state_dict()
    if corruption == 'domain':
        packet['config']['response_domain'] = 'categorical'
    elif corruption == 'core_source':
        packet['config']['implementation_sources']['particle_sliders.pg09'] = 'wrong'
    elif corruption == 'module':
        packet['network']['particles'] = torch.zeros(99, 1)
    elif corruption == 'optimizer_groups':
        packet['opt_g']['param_groups'] = []
    elif corruption == 'optimizer_moment':
        # Any unsupported moment shape must fail before parameter mutation.
        key = packet['opt_g']['param_groups'][0]['params'][0]
        packet['opt_g']['state'].setdefault(key, {})['exp_avg'] = torch.zeros(999)
    elif corruption == 'rng':
        packet['streams']['g_rows'] = torch.zeros(2, dtype=torch.uint8)
    elif corruption == 'alias':
        packet['prior']['z'] = packet['prior']['z'] + 1
    with pytest.raises((ValueError, RuntimeError)):
        game.load_state_dict(packet)
    equal(before, game.state_dict())


def test_installed_pg_source_override_fails_closed(monkeypatch):
    from particle_sliders import pg09
    original = pg09.Path.read_text
    def altered(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return text.replace('"__init__.py": "', '"__init__.py": "invalid') if path.name == 'pg09_release_sources.json' else text
    monkeypatch.setattr(pg09.Path, 'read_text', altered)
    with pytest.raises(RuntimeError, match='sources differ'):
        fixture()


def test_native_integer_config_keys_keep_types_and_primitive_backend_flags_are_not_objects():
    from particle_sliders.pg09 import _hash, _implementation_sources, _architectures
    game = fixture()
    game.critic.backend = 'sdpa'
    sources = _implementation_sources(game.backend, game.critic)
    assert 'builtins' not in sources
    assert _hash({0: 'label'}) != _hash({'0': 'label'})
    assert _hash({'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}})
    assert _architectures(game.backend, game.critic)['critic']['tree'][0][-1]['backend'] == 'sdpa'


def test_same_parameter_schema_with_different_activation_cannot_resume():
    game = fixture()
    game.step()
    changed = fixture(activation=nn.ReLU())
    before = changed.state_dict()
    with pytest.raises(ValueError, match='mismatch'):
        changed.load_state_dict(game.state_dict())
    equal(before, changed.state_dict())


class VirtualRows:
    response_shape = (1, 2, 10, 10)

    def __init__(self, count=10, identity='frozen-v1'):
        self.count, self.token, self.reads = count, identity, []

    def __len__(self):
        return self.count

    def __iter__(self):
        raise AssertionError('provider must not be eagerly traversed')

    def __deepcopy__(self, memo):
        raise AssertionError('provider must not be eagerly copied')

    def identity(self):
        return dict(count=self.count, frozen_data_digest=self.token,
                    conditions='context+signed-native-time-v1')

    def __getitem__(self, index):
        self.reads.append(index)
        rng = torch.Generator().manual_seed(index)
        neutral = torch.randn(self.response_shape, generator=rng)
        row = dict(input=torch.randn(1, 2, generator=rng), sign=(-1. if index % 2 else 1.))
        return Prepared(row, neutral + .2 * row['sign'], neutral)

    def conditions(self, indices):
        return torch.tensor([[i % 3, (-1. if i % 2 else 1.), 500.] for i in indices.tolist()])


def lazy_fixture(count=10, identity='frozen-v1'):
    base = fixture()
    provider = VirtualRows(count, identity)
    game = ConditionalPG09(base.backend, provider, base.critic, particles=base.network.particles,
                          batch_size=4, total_steps=5)
    return game, provider


def test_lazy_native_fields_never_materialized_at_init_and_reads_bounded():
    game, provider = lazy_fixture(50000)
    assert game.rows is provider and game.targets is None and game.conditions is None
    assert provider.reads == []
    game.step()
    assert len(provider.reads) <= 8
    assert game._signature['row_count'] == 50000


def test_balanced_independent_cycles_and_mid_cycle_exact_resume():
    game, _ = lazy_fixture()
    phases = {name: [] for name in ('d', 'g')}
    first = game.step()
    packet = game.state_dict()
    other, _ = lazy_fixture()
    other.load_state_dict(packet)
    phases['d'] += first['d_rows']
    phases['g'] += first['g_rows']
    for _ in range(4):
        actual = game.step()
        assert actual == other.step()
        equal(game.state_dict(), other.state_dict())
        for phase in phases:
            phases[phase] += actual[phase + '_rows']
    for phase, indices in phases.items():
        assert sorted(indices[:10]) == sorted(indices[10:]) == list(range(10))
        assert torch.equal(game.exposures[phase], torch.full((10,), 2))
    assert phases['d'] != phases['g']


@pytest.mark.parametrize('change', ['provider', 'cursor', 'order', 'exposures'])
def test_lazy_provider_and_balanced_sampler_corruption_rejected_atomically(change):
    game, _ = lazy_fixture()
    game.step()
    packet = game.state_dict()
    if change == 'provider':
        receiver, _ = lazy_fixture(identity='changed-cache')
    else:
        receiver, _ = lazy_fixture()
        if change == 'cursor':
            packet['row_cursors']['g'] = 1
        elif change == 'order':
            packet['row_orders']['g'][0] = packet['row_orders']['g'][1]
        else:
            packet['exposures']['g'].zero_()
    before = receiver.state_dict()
    with pytest.raises(ValueError):
        receiver.load_state_dict(packet)
    equal(before, receiver.state_dict())
