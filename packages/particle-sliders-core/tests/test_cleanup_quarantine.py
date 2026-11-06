from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.inference import InferenceSession
from particle_sliders.lifecycle import (Binding, Capabilities, LifecycleError, SessionConfig, SliderSession)
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter, ResponseRow
from particle_sliders.field_critic import FieldCritic


class Response(FrozenResponseAdapter):
    family = 'cleanup-cpu'
    time_coordinates = 'native scalar'

    def _velocity(self, condition, latent, timestep):
        return condition + latent * next(self.network.parameters()).mean() * self.network.scale


def make_session(kind):
    network = nn.Linear(2, 2)
    calls = []
    capabilities = Capabilities('adapter_gan', True)
    if kind == 'inference':
        binding = SimpleNamespace(network=network, close=lambda: calls.append('close'),
            observation=None, render=lambda **kw: torch.ones(1), response=lambda **kw: torch.ones(1))
        return InferenceSession(binding, capabilities, {'selection': 'raw'}, 'pil', lambda *args: calls.append('preflight')), calls
    @contextmanager
    def scaled(scale):
        yield
    network = AdapterParameters([('weight', nn.Parameter(torch.ones(2)))], scaled)
    backend = Response(network)
    rows = [backend.prepare(ResponseRow(torch.zeros(1, 2, 10, 10), torch.ones(1, 2, 10, 10),
        torch.zeros(1, 2, 10, 10), torch.ones(1, 2, 10, 10), timestep=torch.tensor(1.))) for _ in range(2)]
    game = ConditionalPG09(backend, rows, FieldCritic(2, 1, width=4), conditions=torch.ones(2, 1), total_steps=2, batch_size=2)
    binding = Binding(game, lambda: calls.append('close'), lambda **kw: torch.ones(1))
    return SliderSession(SessionConfig('cpu', 'cleanup-test', ema_decay=None), capabilities, binding), calls


@pytest.mark.parametrize('kind', ['training', 'inference'])
def test_failed_close_only_allows_retrying_cleanup(kind):
    session, calls = make_session(kind)
    original_close = session.binding.close
    def fail():
        calls.append('failed close')
        raise RuntimeError('close failed')
    session.binding.close = fail
    with pytest.raises(RuntimeError, match='close failed'):
        session.close()
    assert not session.closed
    for operation in (lambda: session.render(), lambda: session.__enter__()):
        with pytest.raises(LifecycleError, match='requires cleanup'):
            operation()
    if kind == 'training':
        for operation in (session.step, session.checkpoint, session.export):
            with pytest.raises(LifecycleError, match='requires cleanup'):
                operation()
    assert calls == ['failed close']
    session.binding.close = original_close
    session.close()
    session.close()
    assert session.closed and calls == ['failed close', 'close']


@pytest.mark.parametrize('kind', ['training', 'inference'])
@pytest.mark.parametrize('body_failure', [False, True])
def test_observation_rollback_failure_quarantines_session(kind, body_failure):
    session, _ = make_session(kind)
    network = session.game.network if kind == 'training' else session.binding.network
    handle = network.register_load_state_dict_post_hook(lambda *args: (_ for _ in ()).throw(RuntimeError('rollback failed')))
    try:
        with pytest.raises(RuntimeError, match='body failed' if body_failure else 'rollback failed') as caught:
            with session._observation(0):
                if body_failure:
                    raise RuntimeError('body failed')
        if body_failure:
            assert any('rollback failed' in note for note in caught.value.__notes__)
    finally:
        handle.remove()
    with pytest.raises(LifecycleError, match='requires cleanup'):
        session.render()
    session.close()
    assert session.closed


@pytest.mark.parametrize('kind', ['training', 'inference'])
def test_native_context_exit_failure_quarantines_but_body_error_alone_does_not(kind):
    session, _ = make_session(kind)
    with pytest.raises(RuntimeError, match='body only'):
        with session._observation(0):
            raise RuntimeError('body only')
    assert not session._cleanup_failed
    @contextmanager
    def observation():
        try:
            yield
        finally:
            raise RuntimeError('native exit failed')
    session.binding.observation = observation
    with pytest.raises(RuntimeError, match='native exit failed'):
        session.render()
    with pytest.raises(LifecycleError, match='requires cleanup'):
        session.render()
    session.close()
