"""Named native/dev rows share observation, source identity and lifetime ownership."""
from contextlib import contextmanager
from copy import deepcopy
import pytest
import torch
from particle_sliders.lifecycle import EvaluationRows, UnsupportedOperation, SliderSession
from test_image_lifecycle import session, equal


@pytest.mark.parametrize('failure', [None, 'len', 'fetch', 'forward'])
def test_dev_split_fetch_forward_failures_preserve_native_game_and_rng(failure):
    active = session(particles=False, render=False)
    rows = active.game.rows
    events, native = [], {'cache': 1}
    closed = []
    @contextmanager
    def observe():
        before = deepcopy(native)
        events.append('enter')
        try:yield
        finally:
            native.clear();native.update(before);events.append('exit')
    class DevRows:
        def __len__(self):
            assert events[-1] == 'enter'
            native['cache'] += 1;torch.rand(())
            if failure == 'len':raise RuntimeError('dev len failure')
            return len(rows)
        def __getitem__(self, index):
            events.append('fetch');native['cache'] += 2;torch.rand(())
            if failure == 'fetch':raise RuntimeError('dev fetch failure')
            return rows[index]
    original = active.game.backend.generated
    def generated(prepared):
        assert 'fetch' in events
        native['cache'] += 3
        if failure == 'forward':raise RuntimeError('dev forward failure')
        return original(prepared)
    binding = active.binding
    binding.evaluation_rows = {'dev': EvaluationRows(DevRows(), {'dataset':'frozen-dev-v1'})}
    binding.observation = observe
    binding.close = lambda:closed.append('both train and dev resources closed')
    active = SliderSession(active.config, active.capabilities, binding)
    active.game.backend.generated = generated
    checkpoint, rng = active.checkpoint(), torch.get_rng_state().clone()
    if failure:
        with pytest.raises(RuntimeError, match='dev'):active.evaluate(split='dev', seed=7)
    else:
        assert torch.equal(active.evaluate(split='dev',seed=7),active.evaluate(split='dev',seed=7))
    equal(checkpoint, active.checkpoint())
    assert torch.equal(rng, torch.get_rng_state()) and native=={'cache':1}
    assert events[0]=='enter' and events[-1]=='exit'
    active.close();active.close();assert len(closed)==1


def test_invalid_split_and_index_reject_before_native_hook_or_fetch():
    active = session(particles=False, render=False)
    active.binding.observation = lambda:pytest.fail('invalid request entered native scope')
    for split in ('missing',None):
        with pytest.raises(UnsupportedOperation):active.evaluate(split=split)
    for index in (-1,True,1.):
        with pytest.raises(ValueError):active.evaluate(index)
    active.close()


def test_evaluation_identity_is_bound_to_resume_manifest():
    active = session(particles=False,render=False)
    binding = active.binding
    binding.evaluation_rows={'dev':EvaluationRows(active.game.rows,{'dataset':'A'})}
    first=SliderSession(active.config,active.capabilities,binding)
    binding.evaluation_rows={'dev':EvaluationRows(active.game.rows,{'dataset':'B'})}
    second=SliderSession(active.config,active.capabilities,binding)
    assert first.manifest['evaluation_splits']!=second.manifest['evaluation_splits']
    from particle_sliders.lifecycle import CheckpointError
    before=second.checkpoint()
    with pytest.raises(CheckpointError):second.restore(first.checkpoint())
    equal(before,second.checkpoint())
    first.close();second.close()


@pytest.mark.parametrize('failure', [None, 'fetch', 'forward'])
def test_native_response_callback_runs_inside_observation_and_preserves_guards(failure):
    active = session(particles=False, render=False)
    original_rows = active.game.rows
    events, native = [], {'cache': 1}
    @contextmanager
    def observe():
        before = deepcopy(native)
        events.append('enter')
        try:
            yield
        finally:
            native.clear(); native.update(before); events.append('exit')
    def response(index=0, *, split='train', strength=1., seed=0):
        assert events[-1] == 'enter'
        if split not in ('train', 'dev'):
            raise UnsupportedOperation('unsupported native split')
        if index >= len(original_rows):
            raise ValueError('native bounds')
        events.append('guard-fetch')
        native['cache'] += 1
        torch.rand(())
        if failure == 'fetch':
            raise RuntimeError('native fetch failure')
        from dataclasses import replace
        prepared = original_rows[index]
        prepared = replace(prepared, row=replace(prepared.row, scale=strength))
        result = active.game.backend.generated(prepared).detach().clone()
        if failure == 'forward':
            raise RuntimeError('native forward failure')
        events.append('guard-return')
        return result
    active.binding.response = response
    active.binding.observation = observe
    active = SliderSession(active.config, active.capabilities, active.binding)
    before, rng = active.checkpoint(), torch.get_rng_state().clone()
    if failure:
        with pytest.raises(RuntimeError, match='native'):
            active.evaluate(split='dev', strength=.5, seed=17)
    else:
        assert torch.equal(active.evaluate(split='dev', strength=.5, seed=17),
                           active.evaluate(split='dev', strength=.5, seed=17))
        assert 'guard-return' in events
    equal(before, active.checkpoint())
    assert torch.equal(rng, torch.get_rng_state()) and native == {'cache': 1}
    assert active.game.rows is original_rows and events[-1] == 'exit'
    assert active.manifest['native_response']['source_sha256']
    events.clear()
    with pytest.raises(UnsupportedOperation):
        active.evaluate(split='missing')
    assert events == ['enter', 'exit']
    events.clear()
    with pytest.raises(ValueError):
        active.evaluate(-1, split='dev')
    assert not events
    active.close()


def test_native_response_cannot_ambiguously_override_named_providers():
    active = session(particles=False, render=False)
    active.binding.evaluation_rows = {'dev': EvaluationRows(active.game.rows, {'data': 'dev'})}
    active.binding.response = lambda **request: None
    with pytest.raises(ValueError, match='cannot combine'):
        SliderSession(active.config, active.capabilities, active.binding)
    active.close()
