"""Preserve both ordinary LoRAs and their inherited training provenance."""
from pathlib import Path
from .core import Store,read,immutable,sha,IntegrityError
from .artifacts import archive as archive_development
from .joint_artifact import checkpoint


def archive(home,attempt):
    archive_development(home,attempt)
    folder=Store(home).home/'attempts'/attempt;recipe=read(folder/'recipe.json');pair=checkpoint(recipe['checkpoint'],1.)
    immutable(folder/'pair-provenance.json',pair)
    for part in pair['components']:
        source=Path(part['path']);target=folder/'components'/part['structure']['kind'];target.mkdir(parents=True,exist_ok=True)
        for file in (source,source.with_suffix('.json'),source.parent/'recipe.json'):
            link=target/file.name
            if not link.exists():link.symlink_to(file.resolve())
        data=target/'inherited-training'
        if not data.exists():data.symlink_to(source.parent.resolve(),target_is_directory=True)
        if sha(source)!=part['weights_sha256']:raise IntegrityError('Joint inherited weights changed')


def archive_selection(home,attempt):
    raise IntegrityError('Joint v1 has no adaptive selection protocol')
