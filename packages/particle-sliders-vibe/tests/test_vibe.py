import json
from pathlib import Path
import sys

import pytest

from particle_sliders_vibe import collect, score_value


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
