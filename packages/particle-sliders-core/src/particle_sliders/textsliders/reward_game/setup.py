"""Verify frozen evidence and import controls without producing audio."""
import importlib.metadata
import sys
from pathlib import Path

from .core import (Store, IntegrityError, read, write, immutable, sha, digest,
                   checkpoint, generation_identity, reference_summary, ROOT)


def verify(home, full=False):
    config = read(Path(home) / 'game.json')
    game = read(config['spec'])
    if digest(game) != config['benchmark_sha256'] or sha(game['source_manifest']) != game['source_manifest_sha256']:
        raise IntegrityError('Frozen benchmark or source manifest changed; create a new game version')
    manifest = read(game['source_manifest'])
    bridge = Path(home) / 'renderer.json'
    if bridge.exists():
        for path, expected in read(bridge)['sources'].items():
            if sha(path) != expected: raise IntegrityError('Game renderer bridge changed; explicit version amendment required: ' + path)
    for path, expected in manifest['source_hashes'].items():
        if sha(path) != expected: raise IntegrityError('Renderer source changed: ' + path)
    for package, version in manifest['packages'].items():
        if importlib.metadata.version(package) != version: raise IntegrityError('Renderer package changed: ' + package)
    inventory = {**manifest['model_hashes'], **manifest['style_hashes'], **game['reward_spec']['hashes']}
    for path, expected in inventory.items():
        stat = Path(path).stat()
        if full or dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != config['verified_stats'][path]:
            if sha(path) != expected: raise IntegrityError('Frozen model/style/reward file changed: ' + path)
    return game, manifest


def init(home, spec):
    from ..reward_sliders.specs import digest as legacy_digest, validate_families
    store = Store(home); spec = Path(spec).resolve(); game = read(spec)
    if sha(game['source_manifest']) != game['source_manifest_sha256']:
        raise IntegrityError('Source manifest changed')
    manifest = read(game['source_manifest'])
    references = read(spec.parent / 'reference-scorecards.json'); audit = read(spec.parent / 'audit.json')
    if references['benchmark_sha256'] != digest(game) or audit['benchmark_sha256'] != sha(spec) or audit['reference_scorecards_sha256'] != sha(spec.parent / 'reference-scorecards.json'):
        raise IntegrityError('Prepared game integrity failure')
    if game['reward_spec'] != manifest['reward_spec'] or game['sampler'] != manifest['sampler']:
        raise IntegrityError('Frozen reward/sampler mismatch')
    validate_families(manifest['families'])
    original = checkpoint(game['original_reference']['checkpoint'], 1.)
    if original['weights_sha256'] != game['original_reference']['checkpoint_sha256']:
        raise IntegrityError('Original comparison weights changed')
    inventory = {**manifest['model_hashes'], **manifest['style_hashes'], **game['reward_spec']['hashes']}
    stats = {}
    previous = None
    if (store.home / 'game.json').exists():
        verify(home)
        previous = read(store.home / 'game.json')['verified_stats']
    for path, expected in inventory.items():
        s = Path(path).stat(); stats[path] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns)
        if previous is None or previous.get(path) != stats[path]:
            if sha(path) != expected: raise IntegrityError('Frozen asset changed: ' + path)
    config = dict(spec=str(spec), benchmark_sha256=digest(game), game_id=game['game_id'], verified_stats=stats)
    immutable(store.home / 'game.json', config)
    immutable(store.home / 'renderer.json', dict(version=1, sources={str(Path(__file__).with_name('worker.py')): sha(Path(__file__).with_name('worker.py'))},
              contract='SearchRenderer.generate unchanged; imported controls require the recorded Off PCM audit'))
    verify(home)
    sys.path.insert(0, str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    for case in game['cases']:
        if case['style_components'] != manifest['style_components'][case['family']['family']]:
            raise IntegrityError('Frozen styles differ from source manifest')
        for comp in case['style_components']:
            metadata = read(Path(comp['weights']).with_suffix('.json'))
            prompt = Path(metadata['prompts_file'])
            if not prompt.is_absolute(): prompt = ROOT / prompt
            if _artist_name_hit('', prompt.read_text()): raise IntegrityError('Prohibited style provenance')
        for label, control in case['controls'].items():
            if sha(control['observation']) != control['observation_sha256'] or sha(control['audio']) != control['audio_sha256']:
                raise IntegrityError('Frozen observation or audio changed')
            row = read(control['observation']); reward = row['reward']; provenance = row['provenance']
            if provenance['manifest_sha256'] not in [legacy_digest(manifest), *manifest.get('compatible_manifest_sha256', [])]:
                raise IntegrityError('Reference came from an incompatible source manifest')
            expected_components = list(case['style_components'])
            if label != 'off':
                expected_components += [dict(original['structure'],weights=original['path'],
                                              multiplier=.5 if label == 'v1-half' else 1.)]
            structural = lambda comps: [{k:v for k,v in comp.items() if k != 'mtime'} for comp in comps]
            if structural(provenance['components']) != structural(expected_components):
                raise IntegrityError('Reference applied different style or reward components')
            windows = reward['windows']
            if (row['status'] != 'complete' or not reward['valid'] or reward['scalar'] != control['ce'] or
                reward['audio_sha256'] != control['audio_sha256'] or reward['reward_spec_sha256'] != legacy_digest(game['reward_spec']) or
                [(w['start_s'], w['end_s']) for w in windows] != [(0., 10.), (10., 20.)] or
                sum(w['axes']['CE']*10 for w in windows)/20 != reward['scalar'] or
                row['family'] != case['family']['family'] or row['seed'] != case['seed'] or
                provenance['physical_gpu'] != case['physical_gpu'] or provenance['family_sha256'] != legacy_digest(case['family']) or
                provenance['base_sha256'] != legacy_digest(manifest['model_hashes']) or provenance['sampler'] != game['sampler']):
                raise IntegrityError('Frozen observation generation/reward identity mismatch')
            candidate = checkpoint(None, 0) if label == 'off' else dict(original, multiplier=.5 if label == 'v1-half' else 1.)
            if label != 'off' and (row['arm']['checkpoint_sha256'] != original['weights_sha256'] or row['arm']['coefficient'] != candidate['multiplier']):
                raise IntegrityError('Reference adapter mismatch')
            identity = generation_identity(game, manifest, case, candidate); key = digest(identity)
            imported = dict(row, key=key, identity=identity, source_observation=control['observation'], imported_reference=True)
            immutable(store.home / 'renders' / key / 'observation.json', imported)
            store.save_score(reward, digest(game['reward_spec']))
    cases = {c['id']: c for c in game['cases']}
    rebuilt = {n: {label: reference_summary([cases[k] for k in keys], label) for label in game['controls']}
               for n, keys in game['stages'].items()}
    if rebuilt != references['stages']: raise IntegrityError('Reference scorecards do not reproduce exactly')
    result = dict(passed=True, reference_audio_hashes_verified=48, exact_reference_scorecards=True,
                  new_clips=0, benchmark_sha256=digest(game), checkpoint=original)
    immutable(store.home / 'audit' / 'initialization.json', result)
    store.event('initialized', benchmark_sha256=digest(game), new_clips=0)
    store.status('ready', research_complete=False)
    return result
