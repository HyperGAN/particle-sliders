"""Persistent, model-independent finished-output curation. No model imports.

Products implement propose(vibe, count, history), render(prompt, seed, directory),
and score(vibe, output). Each receives request_id for external idempotency.
Outputs are JSON objects; local media uses an absolute `path`. Scores are JSON
objects with a finite `score` in [0, 1] and optional raw judge evidence.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = 'particle-slider-vibe-v1'


class ProviderRejected(RuntimeError):
    """A definitive provider rejection; unlike timeout its outcome is known."""
    def __init__(self, provider, status):
        super().__init__(f'{provider} rejected the request with HTTP {status}')
        self.provider, self.status = provider, status


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.vibe-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prompt_pair(value):
    if not isinstance(value, dict) or set(value) - {'neutral', 'positive', 'metadata'}:
        raise ValueError('prompt must contain neutral, positive and optional metadata')
    for key in ('neutral', 'positive'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError('prompt requires nonempty ' + key)
    if value['neutral'] == value['positive']:
        raise ValueError('positive and neutral prompts must differ')
    canonical(value)
    return value


def score_value(value):
    score = value.get('score') if isinstance(value, dict) else None
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('scorer must return a finite score in [0,1]')
    canonical(value)
    return value


@contextmanager
def collection_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('this collection already has an active writer') from exc
        yield


def _call(root, kind, inputs, callback, *, retry_pending=False):
    """Commit each call independently; ambiguous paid work never silently replays."""
    key = digest(dict(kind=kind, inputs=inputs))
    path = root / 'calls' / (key + '.json')
    if path.exists():
        record = read_json(path)
        if record['status'] == 'done':
            result = record['result']
            if isinstance(result, dict) and 'path' in result:
                if file_digest(result['path']) != result['sha256']:
                    raise ValueError('saved output changed: ' + result['path'])
            return result
        if record['status'] != 'failed' and not retry_pending:
            raise RuntimeError('unresolved call ' + key + '; reconcile the provider receipt before --retry-pending')
    write_json(path, dict(status='pending', kind=kind, inputs=inputs, request_id=key))
    try:
        result = callback(key)
    except ProviderRejected as exc:
        write_json(path, dict(status='failed', kind=kind, inputs=inputs, request_id=key,
                              provider=exc.provider, http_status=exc.status))
        raise
    canonical(result)
    if isinstance(result, dict) and 'path' in result:
        result = dict(result, path=str(Path(result['path']).resolve()))
        result['sha256'] = file_digest(result['path'])
    write_json(path, dict(status='done', kind=kind, inputs=inputs, request_id=key, result=result))
    return result


def collect(product, *, vibe, directory, rounds=1, pool_size=8, replace=4,
            best_of=3, seed=7, retry_pending=False, on_round=None):
    """Extend a collection to `rounds` total, retaining all measured history.

    Prompt fitness is the equal-case mean positive-minus-neutral score. Both
    sides and every candidate use matched seeds. Best-of-N selects one successful
    context per prompt for target preparation; it does not rank checkpoints/LRs.
    """
    if not isinstance(vibe, str) or not vibe.strip():
        raise ValueError('vibe must be nonempty')
    for name, value in dict(rounds=rounds, pool_size=pool_size, replace=replace, best_of=best_of).items():
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if replace > pool_size or type(seed) is not int or not 0 <= seed < 2**63 - rounds * best_of:
        raise ValueError('invalid replacement count or seed')
    identity = dict(schema=SCHEMA, vibe=vibe, product=product.identity,
                    engine_sha256=file_digest(__file__),
                    pool_size=pool_size, replace=replace, best_of=best_of, seed=seed,
                    metric='equal-case-mean-positive-minus-neutral', selection='raw')
    canonical(identity)
    root = Path(directory).resolve()
    with collection_lock(root):
        path = root / 'collection.json'
        if path.exists():
            state = read_json(path)
            if state['identity'] != identity:
                raise ValueError('collection identity changed; use a new output directory')
        else:
            if any(p.name != '.lock' for p in root.iterdir()):
                raise ValueError('new collection requires an empty directory')
            state = dict(identity=identity, rounds=[], selected=[])
            write_json(path, state)
        call = lambda kind, inputs, callback: _call(root, kind, inputs, callback, retry_pending=retry_pending)
        for index in range(len(state['rounds']), rounds):
            retained = state['rounds'][-1]['ranking'][:pool_size-replace] if index else []
            pool = [r['prompt'] for r in retained]
            count = pool_size - len(pool)
            proposals = call('propose', dict(round=index, count=count, history=state['rounds']),
                lambda key: product.propose(vibe, count, state['rounds'], request_id=key))
            if not isinstance(proposals, list) or len(proposals) != count:
                raise ValueError('proposer must return exactly the requested number of prompt pairs')
            pair_key = lambda p: digest({k: p[k] for k in ('neutral', 'positive')})
            seen = {pair_key(r['prompt']) for history in state['rounds'] for r in history['ranking']}
            for value in proposals:
                pair = prompt_pair(value)
                key = pair_key(pair)
                if key in seen or any(pair_key(p) == key for p in pool):
                    raise ValueError('proposer returned a duplicate/previously tried prompt pair')
                pool.append(pair)
            ranking = []
            for pair in pool:
                cases = []
                for take in range(best_of):
                    case_seed = seed + index * best_of + take
                    sides = {}
                    for side in ('neutral', 'positive'):
                        prompt = pair[side]
                        output = call('render', dict(prompt=prompt, seed=case_seed),
                            lambda key: product.render(prompt, case_seed, root / 'outputs' / key, request_id=key))
                        judgment = score_value(call('score', dict(vibe=vibe, output=output),
                            lambda key: score_value(product.score(vibe, output, request_id=key))))
                        sides[side] = dict(output=output, judgment=judgment)
                    cases.append(dict(seed=case_seed, **sides,
                        gain=sides['positive']['judgment']['score'] - sides['neutral']['judgment']['score']))
                # Stable order retains all ties in history without a second ranking metric.
                winner = max(range(best_of), key=lambda i: cases[i]['gain'])
                ranking.append(dict(prompt=pair, cases=cases, best=winner,
                    fitness=sum(c['gain'] for c in cases) / len(cases)))
            ranking.sort(key=lambda row: -row['fitness'])
            state['rounds'].append(dict(index=index, ranking=ranking))
            # Keep every best-of-N training context, not just the surviving prompt pool.
            for row in ranking:
                case = row['cases'][row['best']]
                state['selected'].append(dict(id=digest(dict(round=index, prompt=row['prompt'])),
                    round=index, prompt=row['prompt'], seed=case['seed'], fitness=row['fitness'],
                    gain=case['gain'], neutral=case['neutral'], positive=case['positive']))
            write_json(path, state)
            if on_round:
                on_round(state['rounds'][-1])
        return state


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vibe', required=True, help='fixed intended-output criterion')
    parser.add_argument('--workflow', required=True, help='installed product module:factory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=1, help='total rounds; raise to extend saved work')
    parser.add_argument('--pool-size', type=int, default=8)
    parser.add_argument('--replace', type=int, default=4)
    parser.add_argument('--best-of', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--retry-pending', action='store_true', help='explicitly retry reconciled ambiguous provider calls')
    args = parser.parse_args(argv)
    module, name = args.workflow.split(':', 1)
    product = getattr(importlib.import_module(module), name)()
    try:
        state = collect(product, vibe=args.vibe, directory=args.out, rounds=args.rounds,
            pool_size=args.pool_size, replace=args.replace, best_of=args.best_of, seed=args.seed,
            retry_pending=args.retry_pending)
        print(json.dumps(dict(collection=str(args.out.resolve() / 'collection.json'),
                              rounds=len(state['rounds']), selected=len(state['selected']))))
    finally:
        close = getattr(product, 'close', None)
        if close:
            close()
    return 0
