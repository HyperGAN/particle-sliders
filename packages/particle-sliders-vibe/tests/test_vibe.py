import json
from pathlib import Path
import sys

import pytest

from particle_sliders_vibe import LEGACY_ENGINE_SHA256, SCHEMA, collect, digest, read_json, score_value, write_json


class Product:
    identity = {'fixture': 'finished-output-curation-v1'}

    def __init__(self):
        self.calls = []

    def propose(self, vibe, count, history, *, request_id):
        self.calls.append(('propose', count))
        offset = len(history) * 10
        return [dict(neutral=f'subject-{offset+i}', positive=f'boss-{offset+i}') for i in range(count)]

    def render(self, prompt, seed, directory, *, request_id):
        self.calls.append(('render', prompt, seed))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'output.txt'
        path.write_text(prompt)
        return dict(path=str(path), prompt=prompt, seed=seed)

    def score(self, vibe, output, *, request_id):
        self.calls.append(('score', output['prompt'], output['seed']))
        index = int(output['prompt'].split('-')[-1])
        score = 0.1 if output['prompt'].startswith('subject') else 0.6 + (index % 3) * 0.1
        return dict(score=score, raw={'question': vibe, 'model': 'fixture'})


def test_replace_low_prompts_keep_history_and_resume_without_replay(tmp_path):
    p = Product()
    first = collect(p, vibe='final boss', directory=tmp_path, pool_size=3, replace=2, best_of=2)
    assert [r['prompt']['positive'] for r in first['rounds'][0]['ranking']] == ['boss-2', 'boss-1', 'boss-0']
    second = collect(p, vibe='final boss', directory=tmp_path, pool_size=3, replace=2, best_of=2, rounds=2)
    assert p.calls.count(('propose', 3)) == 1 and p.calls.count(('propose', 2)) == 1
    assert len(second['selected']) == 6 and second['rounds'][0] == first['rounds'][0]
    assert {r['prompt']['positive'] for r in second['rounds'][1]['ranking']} == {'boss-2', 'boss-10', 'boss-11'}
    calls = p.calls.copy()
    assert collect(p, vibe='final boss', directory=tmp_path, pool_size=3, replace=2, best_of=2, rounds=2) == second
    assert p.calls == calls
    for round_ in second['rounds']:
        assert len({tuple(c['seed'] for c in r['cases']) for r in round_['ranking']}) == 1
    with pytest.raises(ValueError, match='identity'):
        collect(p, vibe='different', directory=tmp_path)


def test_interrupted_round_reuses_committed_calls_and_refuses_unknown(tmp_path):
    p = Product()
    original = p.score
    def fail(*args, **kwargs):
        raise RuntimeError('provider response lost')
    p.score = fail
    with pytest.raises(RuntimeError, match='response lost'):
        collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1)
    p.score = original
    with pytest.raises(RuntimeError, match='unresolved call'):
        collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1)
    state = collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1, retry_pending=True)
    assert len(state['rounds']) == 1 and p.calls.count(('propose', 1)) == 1
    assert p.calls.count(('render', 'subject-0', 7)) == 1


@pytest.mark.parametrize('score', [None, True, '0.2', float('nan'), float('inf'), -1, 1.01])
def test_invalid_scores_fail(score):
    with pytest.raises(ValueError, match='finite score'):
        score_value({'score': score})


def test_pending_resume_checks_media_hash(tmp_path):
    p = Product()
    p.score = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('timeout'))
    with pytest.raises(RuntimeError):
        collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1)
    next(tmp_path.glob('outputs/*/output.txt')).write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1)


def test_collector_cli_factory(monkeypatch, tmp_path):
    from particle_sliders_vibe import main
    import types
    p = Product()
    monkeypatch.setitem(sys.modules, 'vibe_fixture', types.SimpleNamespace(factory=lambda: p))
    main(['--vibe', 'boss', '--workflow', 'vibe_fixture:factory', '--out', str(tmp_path),
          '--pool-size', '1', '--replace', '1'])
    assert json.loads((tmp_path / 'collection.json').read_text())['rounds']


def test_definitive_rejection_is_retryable_without_replaying_generation(tmp_path):
    from particle_sliders_vibe import ProviderRejected
    p = Product()
    original = p.score
    p.score = lambda *a, **k: (_ for _ in ()).throw(ProviderRejected('TypeSafe', 402))
    with pytest.raises(ProviderRejected):
        collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1)
    assert any(json.loads(path.read_text())['status'] == 'failed' for path in (tmp_path / 'calls').glob('*.json'))
    p.score = original
    collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1)
    assert p.calls.count(('render', 'subject-0', 7)) == 1


def test_metadata_does_not_disguise_duplicate_prompts(tmp_path):
    p = Product()
    p.propose = lambda *a, **kw: [dict(neutral='a', positive='b', metadata=i) for i in range(2)]
    with pytest.raises(ValueError, match='duplicate'):
        collect(p, vibe='boss', directory=tmp_path, pool_size=2, replace=1)


def test_independent_collections_scope_all_provider_calls(tmp_path):
    cache, requests = {}, []

    class IdempotentProduct:
        def propose(self, vibe, count, history, *, request_id):
            requests.append(request_id)
            return cache.setdefault(request_id, [dict(neutral='portrait', positive=vibe + ' portrait')])

        def render(self, prompt, seed, directory, *, request_id):
            requests.append(request_id)
            return cache.setdefault(request_id, dict(prompt=prompt, model=self.identity))

        def score(self, vibe, output, *, request_id):
            requests.append(request_id)
            return cache.setdefault(request_id, dict(score=float(vibe in output['prompt'])))

    p = IdempotentProduct()
    for i, (vibe, model) in enumerate([('moonlight', 'a'), ('sunlight', 'a'), ('sunlight', 'b'), ('sunlight', 'b')]):
        p.identity = model
        options = dict(directory=tmp_path / str(i), vibe=vibe, pool_size=1, replace=1, best_of=1)
        state = collect(p, **options)
        row, = state['selected']
        assert row['prompt']['positive'] == vibe + ' portrait'
        assert row['neutral']['output']['model'] == model and row['gain'] == 1
        assert collect(p, **options) == state
    assert len(set(requests)) == len(requests) == 20


@pytest.mark.parametrize('invalid', [
    [dict(neutral='a', positive='b'), dict(neutral='a', positive='b')],
    [dict(neutral='same', positive='same'), dict(neutral='c', positive='d')],
    [],
    {'path': 'this is a malformed proposal, not a media file'},
])
def test_rejected_proposal_gets_one_fresh_attempt_then_resumes_that_attempt(tmp_path, invalid):
    p = Product()
    propose = p.propose
    keys = []
    def respond(vibe, count, history, *, request_id):
        keys.append(request_id)
        if len(keys) == 1:
            return invalid
        if len(keys) == 2:
            raise TimeoutError('response lost')
        return propose(vibe, count, history, request_id=request_id)
    p.propose = respond
    options = dict(vibe='boss', directory=tmp_path, pool_size=2, replace=1, best_of=1)
    with pytest.raises(ValueError):
        collect(p, **options)
    first, = tmp_path.glob('calls/*.json')
    receipt = read_json(first)
    assert receipt['status'] == 'failed' and receipt['result'] == invalid
    assert not p.calls
    with pytest.raises(TimeoutError):
        collect(p, **options)
    with pytest.raises(RuntimeError, match=keys[1]):
        collect(p, **options)
    assert len(keys) == 2 and keys[0] != keys[1]
    assert len(collect(p, **options, retry_pending=True)['selected']) == 2
    assert keys[2] == keys[1] and read_json(first) == receipt


def legacy_collection(directory, product):
    identity = dict(schema=SCHEMA, vibe='boss', product=product.identity, engine_sha256=LEGACY_ENGINE_SHA256,
                    pool_size=1, replace=1, best_of=1, seed=7,
                    metric='equal-case-mean-positive-minus-neutral', selection='raw')
    write_json(directory / 'collection.json', dict(identity=identity, rounds=[], selected=[]))
    return identity


def legacy_call(directory, kind, inputs, status, **fields):
    key = digest(dict(kind=kind, inputs=inputs))
    write_json(directory / 'calls' / (key + '.json'),
               dict(kind=kind, inputs=inputs, request_id=key, status=status, **fields))
    return key


def test_legacy_pending_uses_original_id_and_reuses_paid_work(tmp_path):
    p = Product()
    identity = legacy_collection(tmp_path, p)
    proposal_input = dict(round=0, count=1, history=[])
    legacy_call(tmp_path, 'propose', proposal_input, 'done',
                result=[dict(neutral='subject-0', positive='boss-0')])
    output = dict(prompt='subject-0', seed=7)
    legacy_call(tmp_path, 'render', output, 'done', result=output)
    pending = legacy_call(tmp_path, 'score', dict(vibe='boss', output=output), 'pending')
    options = dict(vibe='boss', directory=tmp_path, pool_size=1, replace=1, best_of=1)
    with pytest.raises(RuntimeError, match=pending):
        collect(p, **options)
    assert not p.calls
    score = p.score
    keys = []
    def record_score(*args, request_id):
        keys.append(request_id)
        return score(*args, request_id=request_id)
    p.score = record_score
    state = collect(p, **options, retry_pending=True)
    assert keys[0] == pending and state['identity'] == identity
    assert [c for c in p.calls if c[0] != 'score'] == [('render', 'boss-0', 7)]
    assert state['selected'][0]['neutral']['output'] == output


def test_legacy_done_duplicate_is_replaced_without_replay(tmp_path):
    p = Product()
    legacy_collection(tmp_path, p)
    invalid = [dict(neutral='a', positive='a')]
    key = legacy_call(tmp_path, 'propose', dict(round=0, count=1, history=[]), 'done', result=invalid)
    state = collect(p, vibe='boss', directory=tmp_path, pool_size=1, replace=1, best_of=1)
    first = read_json(tmp_path / 'calls' / (key + '.json'))
    assert first['status'] == 'failed' and first['result'] == invalid
    assert state['selected'][0]['prompt']['positive'] == 'boss-0'
    assert p.calls.count(('propose', 1)) == 1
