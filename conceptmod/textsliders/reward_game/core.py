"""Content identities, immutable observations, atomic claims and scorecards.

This module has no model imports. A renderer must implement the job contract in
worker.py; the controller consumes decisions, never training loss or exit codes.
"""
from contextlib import contextmanager
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = ROOT / 'analysis/reward_slider_game_20260908/benchmark.json'
DEFAULT_HOME = ROOT / 'analysis/reward_game_v1_20260908'
STRUCTURE = dict(kind='language_model', rank=8, alpha=8,
                 target_replace=['Qwen3Attention'], prefix='lora_te', delimiter='-',
                 train_method='full', unit_scale=1.0)


class IntegrityError(ValueError):
    pass


class BusyError(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def locked(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise BusyError(f'Already owned: {path}') from exc
        yield


def immutable(path, value):
    path = Path(path)
    with locked(path.with_suffix(path.suffix + '.lock'), blocking=True):
        if path.exists():
            if read(path) != value:
                raise IntegrityError(f'Immutable artifact changed: {path}')
        else:
            write(path, value)


class Store:
    def __init__(self, home):
        self.home = Path(home).resolve(); self.home.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **data):
        with locked(self.home / 'ledger.lock', blocking=True):
            row = dict(kind=kind, time_unix=time.time(), pid=os.getpid(), **data)
            with (self.home / 'ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
        return row

    def status(self, state, **data):
        write(self.home / 'status.json', dict(state=state, updated_unix=time.time(), **data))

    def observation(self, key):
        path = self.home / 'renders' / key / 'observation.json'
        if not path.exists():
            return None
        row = read(path)
        if row['key'] != key or digest(row['identity']) != key:
            raise IntegrityError('Render identity changed')
        if row.get('audio_sha256') and (not Path(row['audio']).exists() or sha(row['audio']) != row['audio_sha256']):
            raise IntegrityError(f'Corrupted cached audio: {row["audio"]}')
        return row

    @contextmanager
    def claim(self, key):
        folder = self.home / 'renders' / key
        with locked(folder / 'claim.lock'):
            existing = self.observation(key)
            if existing is not None:
                raise BusyError(f'Attempt already recorded for {key}; retained without reroll')
            immutable(folder / 'attempt.json', dict(key=key, pid=os.getpid(), started_unix=time.time()))
            self.event('case_claimed', key=key)
            yield folder

    def score(self, row, reward_hash):
        if not row.get('audio_sha256'):
            return None
        key = digest(dict(audio_sha256=row['audio_sha256'], reward_sha256=reward_hash))
        path = self.home / 'scores' / (key + '.json')
        if not path.exists():
            return None
        score = read(path)
        if score['audio_sha256'] != row['audio_sha256'] or score['reward_identity'] != reward_hash:
            raise IntegrityError('Scoring cache identity changed')
        if score.get('valid') and (not math.isfinite(score['scalar']) or not 1 <= score['scalar'] <= 10):
            raise IntegrityError('Nonfinite or out-of-range cached CE')
        return score

    def save_score(self, score, reward_hash):
        value = dict(score, reward_identity=reward_hash)
        key = digest(dict(audio_sha256=score['audio_sha256'], reward_sha256=reward_hash))
        immutable(self.home / 'scores' / (key + '.json'), value)
        return value


def checkpoint(path, multiplier):
    """Audit actual tensors, structural metadata and sound-only prompt provenance."""
    if not math.isfinite(multiplier) or multiplier < 0:
        raise IntegrityError('Multiplier must be finite and nonnegative')
    if path is None:
        return dict(weights_sha256=None, structure=None, multiplier=0.0, path=None)
    import sys
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    path = Path(path).resolve(); meta = read(path.with_suffix('.json'))
    h = sha(path)
    if h != meta.get('weights_sha256') or any(meta.get(k) != v for k, v in STRUCTURE.items()):
        raise IntegrityError('Checkpoint hash or supported rank-8 LM structure mismatch')
    if _artist_name_hit('', json.dumps(meta)):
        raise IntegrityError('Prohibited names in adapter metadata')
    prompt = Path(meta['prompts_file'])
    if not prompt.is_absolute(): prompt = ROOT / prompt
    if _artist_name_hit('', prompt.read_text()):
        raise IntegrityError('Prohibited training prompt provenance; strip and retrain')
    weights = load_file(str(path))
    expected = set()
    for layer in range(36):
        for projection in ('q', 'k', 'v', 'o'):
            prefix = f'lora_te-model-layers-{layer}-self_attn-{projection}_proj'
            expected.update(prefix + suffix for suffix in ('.lora_up.weight', '.lora_down.weight', '.alpha'))
            if not expected.issubset(weights.keys()):
                raise IntegrityError('Missing LM attention projection tensors')
            up, down, alpha = (weights[prefix + suffix] for suffix in ('.lora_up.weight', '.lora_down.weight', '.alpha'))
            out_dim = 1024 if projection in ('k', 'v') else 4096
            if tuple(up.shape) != (out_dim, 8) or tuple(down.shape) != (8, 4096) or alpha.numel() != 1 or float(alpha) != 8:
                raise IntegrityError(f'Wrong projection topology: {prefix}')
    if set(weights) != expected or not all(torch.isfinite(t).all() for t in weights.values()):
        raise IntegrityError('Unexpected or nonfinite adapter tensors')
    return dict(path=str(path), weights_sha256=h, structure={k: meta[k] for k in STRUCTURE},
                multiplier=float(multiplier), prompt_sha256=sha(prompt), sidecar_sha256=sha(path.with_suffix('.json')),
                tensor_count=len(weights))


def candidate_identity(candidate):
    return {k: candidate[k] for k in ('weights_sha256', 'structure', 'multiplier')}


def generation_identity(game, manifest, case, candidate):
    styles = []
    for comp in case['style_components']:
        styles.append(dict(structure={k: comp[k] for k in STRUCTURE if k != 'unit_scale'},
                           weights_sha256=manifest['style_hashes'][comp['weights']], multiplier=comp['multiplier']))
    energy=sum(abs(c['multiplier']*c['alpha']/c['rank']) for c in case['style_components'])
    if candidate['structure']:
        structure=candidate['structure'];rank=structure['rank']
        energy+=abs(candidate['multiplier']*structure.get('alpha',rank)/rank)
    if energy>game['host_energy_max']+1e-8:
        raise IntegrityError('Candidate plus fixed styles exceeds the declared host energy contract')
    return dict(schema='ordinary-music3-render-v1', candidate=candidate_identity(candidate),
                case={k: case[k] for k in ('id', 'family', 'seed', 'physical_gpu')},
                styles=styles, base_hashes=manifest['model_hashes'],
                renderer_sources=manifest['source_hashes'], bridge_sha256=sha(Path(__file__).with_name('worker.py')), sampler=game['sampler'],
                duration_seconds=game['duration_seconds'], host_energy_max=game['host_energy_max'])


def reference_summary(cases, label):
    scores, deltas, comparisons = [], [], []
    families = defaultdict(list)
    for case in cases:
        value = case['controls'][label]['ce']; delta = value - case['controls']['off']['ce']
        scores.append(value); deltas.append(delta); families[case['family']['family']].append(delta)
        comparisons.append(value - case['controls']['v1-original']['ce'])
    means = {k: sum(v) / len(v) for k, v in families.items()}
    return dict(cases=len(cases), wins_vs_off=sum(x > 0 for x in deltas),
                meaningful_wins_vs_off=sum(x >= .02 for x in deltas), mean_ce=sum(scores)/len(scores),
                equal_family_ce_gain=sum(means.values())/len(means), worst_ce_delta=min(deltas),
                wins_vs_original=sum(x > 0 for x in comparisons), mean_gain_vs_original=sum(comparisons)/len(comparisons),
                family_deltas=means)


def scorecard(game, observations, stage):
    cases = {c['id']: c for c in game['cases']}
    ids = game['stages'][str(stage)]; rows = []; invalid = []
    for ident in ids:
        case = cases[ident]; row = observations.get(ident); reward = (row or {}).get('reward') or {}
        if not row or row['status'] != 'complete' or not reward.get('valid') or not math.isfinite(reward.get('scalar', float('nan'))):
            invalid.append(dict(case=ident, status=(row or {}).get('status', 'unattempted'), error=(row or {}).get('error')))
            continue
        ce = reward['scalar']
        rows.append(dict(case=ident, family=case['family']['family'], voice=case['family']['voice'], ce=ce,
                         deltas={label: ce - c['ce'] for label, c in case['controls'].items()}, audio=row.get('audio')))
    def aggregate(selected, label):
        groups = defaultdict(list)
        for row in selected: groups[row['family']].append(row['deltas'][label])
        means = {k: sum(v)/len(v) for k, v in groups.items()}
        values = [r['deltas'][label] for r in selected]
        return dict(wins=sum(d > 0 for d in values), meaningful_wins=sum(d >= .02 for d in values),
                    ties=sum(d == 0 for d in values), equal_family_gain=sum(means.values())/len(means) if means else None,
                    worst_delta=min(values) if values else None, family_deltas=means)
    comparisons = {label: aggregate(rows, label) for label in game['controls']}
    off = comparisons['off']; original = comparisons['v1-original']; rule = game['rules'][f'stage{stage}']
    passed = not invalid and off['wins'] >= rule['min_wins'] and off['equal_family_gain'] >= rule['min_mean_ce_gain'] and off['worst_delta'] >= rule['min_worst_ce_delta']
    if stage == 16:
        passed = passed and original['wins'] >= rule['min_wins_vs_original'] and original['equal_family_gain'] >= rule['min_mean_gain_vs_original']
    return dict(stage=stage, scheduled_cases=len(ids), valid_cases=len(rows), advance=bool(passed),
                decision='development_pass_requires_confirmation' if passed and stage == 16 else 'advance' if passed else 'rejected',
                comparisons=comparisons, invalid=invalid, rows=rows,
                losses=[r for r in rows if r['deltas']['off'] < 0],
                voice_groups={v: {label: aggregate([r for r in rows if r['voice'] == v], label) for label in game['controls']}
                              for v in sorted({r['voice'] for r in rows})})


def markdown(card):
    lines = [f"Run {card.get('run_id', 'reference')}: {card['decision']}; {card['valid_cases']}/{card['stage']} valid development cases.", '',
             '| Control | Wins | Meaningful wins | Family mean CE gain | Worst delta |', '|---|---:|---:|---:|---:|']
    for label, row in card['comparisons'].items():
        gain = '—' if row['equal_family_gain'] is None else f"{row['equal_family_gain']:+.4f}"
        worst = '—' if row['worst_delta'] is None else f"{row['worst_delta']:+.4f}"
        lines.append(f"| {label} | {row['wins']}/{card['stage']} | {row['meaningful_wins']} | {gain} | {worst} |")
    if card.get('early_rejected'): lines += ['', 'Early rejected at the declared stage. No extrapolated full score.']
    if card.get('forced_stage'): lines += ['', 'Explicit development diagnostic; early stopping bypassed.']
    lines += ['', f"New clips: {card.get('new_clips', 0)}; render cache hits: {card.get('cache_hits', 0)}; elapsed seconds: {card.get('elapsed_seconds', 0):.1f}."]
    if card['losses']:
        lines += ['', 'Losses versus Off:'] + [f"- {r['case']}: {r['deltas']['off']:+.4f} CE" for r in card['losses']]
    if card['invalid']: lines += ['', 'Invalid/unattempted: ' + json.dumps(card['invalid'])]
    return '\n'.join(lines) + '\n'
