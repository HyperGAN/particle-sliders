"""Archive trained block factors, complete batches and the fixed parent."""
from pathlib import Path
from .core import read, immutable, sha, IntegrityError
from .acoustic_artifacts import archive as archive_training


def archive(home, attempt):
    archive_training(home, attempt)
    destination = Path(home)/'attempts'/attempt
    recipe = read(destination/'recipe.json'); source = Path(recipe['training_status']).parent
    parent = read(source/'recipe.json')['parent']
    if sha(parent['path']) != parent['weights_sha256']:
        raise IntegrityError('Joint block parent changed')
    immutable(destination/'parent-checkpoint.json', parent)
    for name in ('native-step1-audit.json', 'native-step2-audit.json'):
        immutable(destination/'acoustic-training'/name, read(source/name))
    for name in ('batches',):
        link = destination/'acoustic-training'/name
        if not link.exists():
            link.symlink_to((source/name).resolve(), target_is_directory=True)


def archive_selection(home, attempt):
    raise IntegrityError('Robust block training has no direct-search Off selection')
