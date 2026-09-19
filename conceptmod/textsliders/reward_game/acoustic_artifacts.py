"""Auditable acoustic training captures alongside ordinary development artifacts."""
from pathlib import Path
from .core import Store,read,immutable,sha,IntegrityError
from .artifacts import archive as archive_development


def archive(home,attempt):
    archive_development(home,attempt)
    folder=Store(home).home/'attempts'/attempt;recipe=read(folder/'recipe.json');source=Path(recipe['training_status']).parent
    target=folder/'acoustic-training';target.mkdir(parents=True,exist_ok=True)
    for name in ('recipe.json','collection-recipe.json','captures.json','collection-status.json','collection-off-restoration.json','training-prompts.json','status.json'):
        immutable(target/name,read(source/name))
    for name in ('updates.jsonl','state.pt',f"state-step{recipe['expected_updates']}.pt",f"off-restoration-{recipe['expected_updates']}.json"):
        file=source/name
        if not file.exists():raise IntegrityError('Missing acoustic training artifact: '+name)
        link=target/name
        if not link.exists():link.symlink_to(file.resolve())
    index=[]
    for row in read(source/'captures.json'):
        for kind in ('audio','capture'):
            file=Path(row[kind]);expected=row[kind+'_sha256']
            if sha(file)!=expected:raise IntegrityError('Acoustic capture artifact changed')
            link=target/kind/(row['id']+file.suffix);link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(file.resolve())
            index.append(dict(id=row['id'],kind=kind,source=str(file),link=str(link),sha256=expected,reused_exact_capture=row['reused_exact_capture']))
    immutable(target/'evidence-index.json',index)


def archive_selection(home,attempt):
    raise IntegrityError('This acoustic version has no direct-search Off-selection protocol')
