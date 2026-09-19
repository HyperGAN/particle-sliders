"""Make a completed attempt independently inspectable without copying WAVs."""
from pathlib import Path

from .core import Store,read,immutable,sha


def archive_selection(home,attempt):
    """Retain training evidence even when Off wins and no development is run."""
    store=Store(home);folder=store.home/'attempts'/attempt;recipe=read(folder/'recipe.json')
    source=Path(recipe['selection_result']).parent
    for name in ('selection.json','audio-observations.json','generation-jobs.json'):
        immutable(folder/('training-'+name),read(source/name))
    immutable(folder/'training-state.json',dict(local_optimizer_updates=0,method='direct adapter parameter search',
              selection_result=recipe['selection_result'],sha256=sha(recipe['selection_result'])))
    index=[]
    for row in read(source/'audio-observations.json'):
        if not row or not row.get('audio_sha256'):continue
        path=Path(row['audio']);target=folder/'training-audio'/(row['key']+'.wav');target.parent.mkdir(exist_ok=True)
        if not target.exists():target.symlink_to(path.resolve())
        index.append(dict(link=str(target),source=str(path),sha256=row['audio_sha256']))
    immutable(folder/'training-audio-index.json',index)


def archive_matched(folder,recipe):
    source=Path(recipe['training_status']).parent;destination=folder/'matched-training'
    for name in ('collection-recipe.json','manifest.json','cases.json','preferences.json',
                 'collection-status.json','capture-parity.json','collection-off-restoration.json',
                 'recovery-recipe.json','prepare-status.json','prepare-off-restoration.json','live-loss-audit.json','recipe.json'):
        immutable(destination/name,read(source/name))
    index=[]
    for pair in read(source/'preferences.json'):
        for role in ('chosen','rejected'):
            row=pair[role+'_observation']
            for label in ('audio','trajectory'):
                path=Path(row[label]);expected=row[label+'_sha256']
                if sha(path)!=expected:raise ValueError('Matched training evidence changed')
                target=destination/label/(row['id']+path.suffix);target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.symlink_to(path.resolve())
                index.append(dict(role=role,id=row['id'],kind=label,link=str(target),source=str(path),sha256=expected))
        audit=source/'tokens'/f"{pair['chosen']}.json"
        immutable(destination/'target-audits'/audit.name,read(audit))
    immutable(destination/'evidence-index.json',index)


def archive_parent_policy(folder,recipe):
    source=Path(recipe['training_status']).parent;destination=folder/'parent-policy-training'
    for name in ('episodes.json','policy-parent.json','manifest.json','recovery-recipe.json','prepare-status.json','recipe.json'):
        immutable(destination/name,read(source/name))
    index=[]
    for episode in read(source/'episodes.json'):
        row=episode['observation']
        for label in ('audio','trajectory'):
            path=Path(row[label]);expected=row[label+'_sha256']
            if sha(path)!=expected:raise ValueError('Parent-policy evidence changed')
            target=destination/label/(row['id']+path.suffix);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.symlink_to(path.resolve())
            index.append(dict(id=row['id'],kind=label,link=str(target),sha256=expected,advantage=episode['advantage']))
        audit=source/'tokens'/f"{row['id']}.json";immutable(destination/'target-audits'/audit.name,read(audit))
    immutable(destination/'evidence-index.json',index)


def archive(home,attempt):
    store=Store(home);folder=store.home/'attempts'/attempt
    recipe=read(folder/'recipe.json');card=read(folder/'scorecard.json')
    if recipe.get('selection_result'):archive_selection(home,attempt)
    if recipe.get('preprocessing',{}).get('collection_recipe'):archive_matched(folder,recipe)
    if recipe.get('preprocessing',{}).get('parent_policy_source'):archive_parent_policy(folder,recipe)
    observations=[store.observation(key) for key in card['render_keys']]
    immutable(folder/'audio-observations.json',observations)
    evaluation=store.home/'evaluations'/card['run_id']
    jobs=[dict(path=str(path),sha256=sha(path),jobs=read(path)) for path in sorted(evaluation.glob('jobs-stage*.json'))]
    immutable(folder/'generation-jobs.json',jobs)
    index=[]
    for row in observations:
        if not row or not row.get('audio_sha256'):continue
        path=Path(row['audio']);target=folder/'audio'/(row['key']+'.wav');target.parent.mkdir(exist_ok=True)
        if not target.exists():target.symlink_to(path.resolve())
        index.append(dict(link=str(target),source=str(path),sha256=row['audio_sha256']))
    immutable(folder/'audio-index.json',index)
    if recipe.get('training_state'):
        state=Path(recipe['training_state']);link=folder/'training-state.pt'
        if not link.exists():link.symlink_to(state.resolve())
        immutable(folder/'training-state.json',dict(path=str(state),sha256=sha(state),actual_updates=read(folder/'training-result.json')['actual_updates']))
    elif not recipe.get('selection_result'):
        immutable(folder/'training-state.json',dict(local_optimizer_updates=0,method='manually supplied checkpoint',
                  parent_checkpoint=recipe['parent_checkpoint']))
    return dict(attempt=attempt,observations=len(observations),new_clips=card['new_clips'])
