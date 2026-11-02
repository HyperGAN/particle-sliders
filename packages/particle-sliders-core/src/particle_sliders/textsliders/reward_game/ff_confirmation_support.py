"""Single acoustic feed-forward candidate and original LM control resolution."""
from pathlib import Path
from .core import read, IntegrityError
from .joint_confirmation_support import check_fresh


def component(path,multiplier):
    kind=read(Path(path).with_suffix('.json')).get('kind')
    if kind=='transformer':
        from .ff_artifact import component as resolve
    elif kind=='language_model':
        from ..reward_sliders.evaluate import component as resolve
    else:
        raise IntegrityError('Unknown feed-forward fresh comparison host')
    return resolve(path,multiplier)
