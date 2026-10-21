"""Evaluation's native observation transaction includes lazy-provider access."""
from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch

from test_image_lifecycle import session, equal


@pytest.mark.parametrize('failure', [None, 'fetch', 'forward'])
def test_provider_fetch_forward_and_failures_share_observation_scope(failure):
    active = session(particles=False, render=False)
    original_rows, generated = active.game.rows, active.game.backend.generated
    native_buffer = torch.tensor([9.])
    state = dict(observing=False, cache={'old': 1}, latest=0.)
    events = []

    class LazyRows:
        def __len__(self):
            assert state['observing'], 'provider bounds access escaped observation'
            events.append('len')
            native_buffer.add_(1)
            state['cache']['bounds'] = torch.rand(()).item()
            return len(original_rows)

        def __getitem__(self, index):
            assert state['observing'], 'provider fetch escaped observation'
            events.append('fetch')
            native_buffer.add_(2)
            state['latest'] = torch.rand(()).item()
            state['cache']['row'] = index
            if failure == 'fetch':
                raise RuntimeError('native provider fetch failure')
            return original_rows[index]

    @contextmanager
    def observe_native():
        assert not state['observing']
        before, buffer = deepcopy(state), native_buffer.clone()
        events.append('enter')
        state['observing'] = True
        try:
            yield
        finally:
            state.clear()
            state.update(before)
            native_buffer.copy_(buffer)
            events.append('exit')

    def forward(prepared):
        assert state['observing'], 'native forward escaped observation'
        events.append('forward')
        native_buffer.add_(3)
        torch.rand(())
        if failure == 'forward':
            raise RuntimeError('native forward failure')
        return generated(prepared) + state['latest']

    active.game.rows = LazyRows()
    active.game.backend.generated = forward
    active.binding.observation = observe_native
    checkpoint, rng = active.checkpoint(), torch.get_rng_state().clone()
    try:
        if failure:
            with pytest.raises(RuntimeError, match='failure'):
                active.evaluate(seed=21)
        else:
            first = active.evaluate(seed=21)
            assert torch.equal(first, active.evaluate(seed=21))
        equal(checkpoint, active.checkpoint())
        assert torch.equal(rng, torch.get_rng_state())
        assert state == dict(observing=False, cache={'old': 1}, latest=0.)
        assert native_buffer.item() == 9.
        assert events[0] == 'enter' and events[-1] == 'exit'
        assert events.index('enter') < events.index('len') < events.index('fetch')
        if failure != 'fetch':
            assert events.index('fetch') < events.index('forward')
        assert not active.closed  # Observation failures do not corrupt a training session.
    finally:
        active.close()


def test_out_of_bounds_provider_check_is_also_transactional():
    active = session(particles=False, render=False)
    events = []
    state = dict(observing=False)
    rng = torch.get_rng_state().clone()

    class Rows:
        def __len__(self):
            assert state['observing']
            torch.rand(())
            events.append('len')
            return 2

        def __getitem__(self, index):
            pytest.fail('out-of-bounds row fetched')

    @contextmanager
    def observation():
        state['observing'] = True
        try:
            yield
        finally:
            state['observing'] = False
            events.append('restored')

    active.game.rows = Rows()
    active.binding.observation = observation
    try:
        with pytest.raises(ValueError, match='out of bounds'):
            active.evaluate(index=2)
        assert events == ['len', 'restored']
        assert not state['observing'] and torch.equal(rng, torch.get_rng_state())
    finally:
        active.close()
