"""Resolve the two explicitly declared ordinary hosts in fresh comparisons."""
from pathlib import Path
from .core import read, IntegrityError


def component(path, multiplier):
    kind = read(Path(path).with_suffix('.json')).get('kind')
    if kind == 'transformer':
        from .acoustic_artifact import component as resolve
    elif kind == 'language_model':
        from ..reward_sliders.evaluate import component as resolve
    else:
        raise IntegrityError('Unknown fresh-comparison adapter host')
    return resolve(path, multiplier)
