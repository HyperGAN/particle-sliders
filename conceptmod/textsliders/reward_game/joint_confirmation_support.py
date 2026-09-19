"""Resolve native pair/control arms and reject exposure across sibling campaigns."""
from pathlib import Path
from .core import read, sha, IntegrityError


def component(path, multiplier):
    if Path(path).suffix == '.json':
        from .joint_artifact import component as resolve
        return resolve(path, multiplier)
    from ..reward_sliders.evaluate import component as resolve
    from .core import checkpoint
    checkpoint(path, multiplier)
    return [resolve(path, multiplier)]


def check_fresh(home, base, game, families, seeds):
    parent = Path(read(Path(home)/'game.json')['parent_home']).resolve()
    previous = list(base['families'])
    used_seeds = {c['seed'] for c in game['cases']}
    protocols = {}
    # The root and every child campaign share exposure, including failed or
    # interrupted frozen batches. Draft files without a protocol are unused.
    for old in sorted(parent.rglob('protocol.json')):
        if old.parent.parent.name not in ('confirmation', 'composition'):
            continue
        p = read(old)
        previous += read(old.parent/'manifest.json')['families']
        used_seeds.update(p.get('seeds', []))
        used_seeds.update(c['seed'] for c in p.get('cases', []))
        protocols[str(old)] = sha(old)
    for family in families:
        if any(family[key] == old.get(key) for old in previous for key in ('family', 'caption', 'lyrics')):
            raise IntegrityError('Confirmation fixture already exposed, including sibling campaigns')
    if set(seeds) & used_seeds:
        raise IntegrityError('Confirmation seed already exposed, including sibling campaigns')
    return dict(previous_protocols=protocols, compared_families=len(previous),
                used_seeds=sorted(used_seeds), sibling_campaigns_included=True)
