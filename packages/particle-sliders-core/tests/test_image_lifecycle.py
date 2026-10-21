"""Bounded CPU conformance of the public lifecycle with real released PG updates."""
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from particle_sliders.lifecycle import (
    Backend, Binding, Capabilities, CheckpointError, SessionConfig, UnsupportedOperation,
    SliderSession, read_packet,
)
from particle_sliders.pg09 import ConditionalPG09, _hash
from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter, ResponseRow
from particle_sliders.field_critic import FieldCritic


class Response(FrozenResponseAdapter):
    family = 'cpu-contract'
    time_coordinates = 'native scalar'

    def _velocity(self, condition, latent, timestep):
        # Deliberate dropout exercises private model RNG across split resume.
        x = nn.functional.dropout(latent, .1, training=self.network.training)
        weights = sum(parameter.mean() for parameter in self.network.parameters())
        return condition + x * weights * self.network.scale


@pytest.fixture(autouse=True)
def cpu_only():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def session(particles=True, render=True):
    with torch.random.fork_rng():
        torch.manual_seed(9)
        @contextmanager
        def scaled(scale):
            yield
        named = [('weight', nn.Parameter(torch.randn(2)))]
        if particles:
            named.append(('particles', nn.Parameter(torch.randn(8, 2))))
        network = AdapterParameters(named, scaled)
        response = Response(network)
        rows = [response.prepare(ResponseRow(torch.zeros(1, 2, 10, 10),
            torch.full((1, 2, 10, 10), .2), torch.zeros(1, 2, 10, 10),
            torch.ones(1, 2, 10, 10), timestep=torch.tensor(500.))) for _ in range(4)]
        critic = FieldCritic(2, 3, width=4)
        game = ConditionalPG09(response, rows, critic, conditions=torch.ones(4, 3),
            particles=network.particles if particles else None, total_steps=4, batch_size=2)
        closed = []
        def draw(**kwargs):
            return game.backend.generated(game.rows[0]).clone()
        result = SliderSession(SessionConfig('cpu', 'new-v1', ema_decay=.9),
            Capabilities('routed_particle_gan' if particles else 'adapter_gan', render,
                         strength_min=None, strength_max=None),
            Binding(game, lambda: closed.append(True), draw if render else None))
        result.cleanup_calls = closed
        return result


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def seal(packet):
    packet['sha256'] = _hash({key: value for key, value in packet.items() if key != 'sha256'})
    return packet


@pytest.mark.parametrize('particles', [True, False])
def test_exact_resume_ema_and_observation(tmp_path, particles):
    full, split, resumed = session(particles), session(particles), session(particles)
    before_rng = torch.get_rng_state().clone()
    full.train()
    split.train(stop_after=2)
    split.checkpoint(tmp_path / 'training.pt')
    resumed.restore(read_packet(tmp_path / 'training.pt'))
    state = resumed.checkpoint()
    for strength in (0., -.5, .5, 1.):
        first = resumed.evaluate(strength=strength, seed=6)
        assert torch.equal(first, resumed.evaluate(strength=strength, seed=6))
        equal(state, resumed.checkpoint())
    assert torch.equal(resumed.evaluate(strength=0), resumed.game.rows[0].neutral)
    resumed.render()
    equal(state, resumed.checkpoint())
    resumed.train()
    equal(full.checkpoint(), resumed.checkpoint())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['checksum', 'ema', 'sampler', 'optimizer', 'cohort', 'metrics'])
def test_invalid_restore_is_atomic(damage):
    source, target = session(), session()
    source.train(stop_after=2)
    packet = source.checkpoint()
    before = target.checkpoint()
    if damage == 'checksum':
        packet['sha256'] = 'invalid'
    elif damage == 'ema':
        next(iter(packet['payload']['ema']['values'].values())).fill_(float('nan'))
    elif damage == 'sampler':
        packet['payload']['game']['row_cursors']['g'] = 999
    elif damage == 'optimizer':
        packet['payload']['game']['opt_g']['state'] = {}
    elif damage == 'cohort':
        packet['manifest']['config']['cohort'] = 'wrong'
    elif damage == 'metrics':
        packet['payload']['last_metrics'] = {'step': 999}
    if damage != 'checksum':
        seal(packet)
    with pytest.raises(CheckpointError):
        target.restore(packet)
    equal(before, target.checkpoint())


def test_export_selection_is_inference_only_and_does_not_change_training(tmp_path):
    trained = session()
    trained.train(stop_after=2)
    before = trained.checkpoint()
    raw = trained.export(selection='raw')
    ema = trained.export(tmp_path / 'adapter.pt', selection='ema')
    equal(before, trained.checkpoint())
    assert any(not torch.equal(raw['payload']['adapter']['network'][key], value)
               for key, value in ema['payload']['adapter']['network'].items())
    loaded = session()
    loaded.load_inference(read_packet(tmp_path / 'adapter.pt'))
    equal(loaded.game.network.state_dict(), trained.ema.values)
    for action in (lambda: loaded.restore(ema), loaded.step, loaded.train, loaded.checkpoint):
        with pytest.raises(UnsupportedOperation, match='inference-loaded'):
            action()
    assert not loaded.closed and loaded.cleanup_calls == []
    loaded.close()
    assert loaded.cleanup_calls == [True]


def test_failures_cleanup_and_preflight_before_allocation():
    calls = []
    contract = Backend('iris', Capabilities('routed_particle_gan', render=False),
        lambda *a, **kw: calls.append('allocate'), lambda *a: calls.append('validate'))
    for kwargs in ({'operation': 'render'}, {'strength': -1.}, {'references': True}):
        with pytest.raises(UnsupportedOperation):
            contract.open(SessionConfig('iris', 'new'), **kwargs)
    assert calls == []
    active = session()
    active.game.backend.generated = lambda prepared: (_ for _ in ()).throw(RuntimeError('failure'))
    with pytest.raises(RuntimeError, match='failure'):
        active.step()
    active.close()
    assert active.cleanup_calls == [True]


def test_atomic_save_failure_keeps_previous_file(tmp_path, monkeypatch):
    import particle_sliders.lifecycle as lifecycle
    path = tmp_path / 'training.pt'
    active = session()
    active.checkpoint(path)
    before = path.read_bytes()
    def fail(*args):
        raise OSError('interrupted rename')
    monkeypatch.setattr(lifecycle.os, 'replace', fail)
    with pytest.raises(OSError):
        active.checkpoint(path)
    assert path.read_bytes() == before
    assert {file.name for file in tmp_path.iterdir()} == {'training.pt'}


def test_config_snapshot_and_versions():
    source = {'rows': [{'neutral': 'a', 'positive': 'b'}]}
    config = SessionConfig('cpu', 'new', **source)
    source['rows'][0]['neutral'] = 'changed'
    assert config.options['rows'][0]['neutral'] == 'a'
    packet = session().checkpoint()
    packet['schema'] = 'legacy'
    with pytest.raises(CheckpointError, match='version'):
        session().restore(packet)


def test_native_base_precision_and_render_output_preflight():
    contract = Capabilities('routed_particle_gan', True, base_precisions=('bfloat16',),
                            render_outputs=('latent',))
    contract.check('response')
    with pytest.raises(UnsupportedOperation, match='precision'):
        contract.check('response', precision='float32')
    with pytest.raises(UnsupportedOperation, match='output'):
        contract.check('render', output_type='pil')


def test_run_observer_seam_receives_detached_metrics_and_manifest_snapshots():
    active = session()
    observed = []
    def observer(metrics):
        observed.append(metrics['step'])
        metrics['step'] = 999
        snapshot = active.manifest
        snapshot['config']['cohort'] = 'cannot-rebind'
    active.train(stop_after=2, on_step=observer)
    assert observed == [1, 2]
    assert active.last_metrics['step'] == 2
    assert active.manifest['config']['cohort'] == 'new-v1'


def test_unimplemented_operations_have_specific_reasons_without_precision():
    caps = Capabilities('unimplemented', False, base_precisions=(), operations=(),
                        unavailable_reasons=(('train', 'native PG binding is not implemented'),))
    with pytest.raises(UnsupportedOperation, match='native PG binding'):
        caps.check('train')
    with pytest.raises(UnsupportedOperation, match='render is unavailable'):
        caps.check('render')


def test_nested_device_discovery_without_allocating_cuda():
    from particle_sliders.lifecycle import _rng_devices
    assert _rng_devices({'native_args': {'device': 'cuda:3'}, 'device': 'cpu'},
                        {'device': torch.device('cuda:2')}) == [2, 3]
    assert _rng_devices({'model_path': 'cuda:7'}) == []


def test_native_admission_rejection_preserves_training_state():
    active = session()
    before = active.checkpoint()
    rng = torch.get_rng_state().clone()
    def reject(request):
        assert request['completed_steps'] == 0
        request['manifest']['config']['cohort'] = 'cannot-rebind'
        torch.rand(2)
        next(active.game.network.parameters()).data.add_(2)
        return False
    active.binding.before_update = reject
    with pytest.raises(UnsupportedOperation, match='admission'):
        active.step()
    equal(before, active.checkpoint())
    assert torch.equal(rng, torch.get_rng_state()) and not active.closed
    active.binding.before_update = lambda request: True
    active.step()
    assert active.game.completed_steps == 1


def test_native_request_preflight_precedes_prepare():
    calls = []
    def refuse(options, request, staged):
        calls.append(request['native']['mask'])
        raise UnsupportedOperation('native mask geometry differs')
    contract = Backend('cpu', Capabilities('adapter_gan', True),
                       lambda *a, **kw: calls.append('allocate'), lambda options: None,
                       preflight=refuse)
    with pytest.raises(UnsupportedOperation, match='mask geometry'):
        contract.open(SessionConfig('cpu', 'new'), native_request={'mask': 'invalid'})
    assert calls == ['invalid']


def test_owned_directory_and_borrowed_model_restoration(tmp_path):
    from particle_sliders.lifecycle import new_directory, native_cleanup
    root = tmp_path / 'cache'
    with pytest.raises(KeyboardInterrupt):
        with new_directory(root):
            (root / 'partial').write_text('incomplete')
            raise KeyboardInterrupt()
    assert not root.exists()
    model = nn.Linear(2, 2)
    restore = native_cleanup(model)
    model.eval().requires_grad_(False)
    model.gradient_checkpointing = True
    restore()
    restore()
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert not hasattr(model, 'gradient_checkpointing')


def test_observation_context_restores_online_provider_on_failure():
    active = session()
    calls = []
    @contextmanager
    def observation():
        calls.append('enter')
        try:
            yield
        finally:
            calls.append('exit')
    active.binding.observation = observation
    active.binding.render = lambda **kw: (_ for _ in ()).throw(RuntimeError('native sampling'))
    state = active.checkpoint()
    with pytest.raises(RuntimeError, match='native sampling'):
        active.render()
    assert calls == ['enter', 'exit']
    equal(state, active.checkpoint())
