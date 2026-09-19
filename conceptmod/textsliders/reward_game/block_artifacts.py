"""Retain fixed-composition ancestry separately from new optimizer work."""
from pathlib import Path
from .core import Store,read,immutable,sha,IntegrityError
from .artifacts import archive as archive_development


def archive(home,attempt):
    archive_development(home,attempt)
    folder=Store(home).home/'attempts'/attempt;recipe=read(folder/'recipe.json')
    source=Path(recipe['checkpoint']).parent;target=folder/'composition'
    for name in ('recipe.json','composition.json','training-prompts.json'):
        immutable(target/name,read(source/name))
    index=[]
    for name,evidence in recipe['parent_evidence'].items():
        path=Path(evidence['path'])
        if sha(path)!=evidence['sha256']:raise IntegrityError('Composition parent evidence changed')
        link=target/'parent-evidence'/(name+path.suffix);link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(path.resolve())
        index.append(dict(name=name,source=str(path),link=str(link),sha256=evidence['sha256']))
    for name,parent in recipe['parents'].items():
        for suffix in ('.safetensors','.json'):
            path=Path(parent['path']).with_suffix(suffix)
            expected=parent['weights_sha256'] if suffix=='.safetensors' else parent['sidecar_sha256']
            if sha(path)!=expected:raise IntegrityError('Composition parent checkpoint changed')
            link=target/'parents'/(name+suffix);link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(path.resolve())
            index.append(dict(name=name,source=str(path),link=str(link),sha256=expected))
    immutable(target/'evidence-index.json',dict(files=index,new_optimizer_updates=0,new_training_clips=0,
        shared_capture_count=8,parents_are_not_independent_training_datasets=True))


def archive_selection(home,attempt):
    raise IntegrityError('Block composition has no direct-search Off-selection protocol')
