"""A fixed pair of ordinary LoRAs, one per native host; no adaptive routing."""
from pathlib import Path
from .core import read,sha,digest,immutable,checkpoint as lm_checkpoint,candidate_identity,IntegrityError
from .acoustic_artifact import checkpoint as acoustic_checkpoint

FORMAT='ordinary-lora-pair-v1'


def construct(language_model,acoustic,folder):
    folder=Path(folder).resolve()
    parts=[lm_checkpoint(language_model,1.),acoustic_checkpoint(acoustic,1.)]
    recipe=dict(format=FORMAT,method='fixed-cross-host-composition-v1',components=parts,
        local_optimizer_updates=0,new_training_audio=0,
        contract='Two ordinary rank-8 alpha-8 LoRAs at unit multiplier on separate hosts. Fixed weights and multipliers for every case; ordinary deployment merging.',
        provenance=[dict(recipe=str(Path(p['path']).parent/'recipe.json'),recipe_sha256=sha(Path(p['path']).parent/'recipe.json')) for p in parts])
    immutable(folder/'recipe.json',recipe);immutable(folder/'candidate.json',recipe)
    return checkpoint(folder/'candidate.json',1.)


def checkpoint(path,multiplier=1.):
    if path is None:return lm_checkpoint(None,multiplier)
    if multiplier!=1.:raise IntegrityError('Joint v1 declares fixed unit multipliers on both hosts')
    path=Path(path).resolve();bundle=read(path)
    if bundle.get('format')!=FORMAT or len(bundle.get('components',[]))!=2:
        raise IntegrityError('Need the declared pair of ordinary LoRAs')
    parts=[]
    for expected,validate in zip(bundle['components'],(lm_checkpoint,acoustic_checkpoint)):
        if expected['multiplier']!=1.:raise IntegrityError('Joint components require unit multipliers')
        actual=validate(expected['path'],expected['multiplier'])
        if actual!=expected:raise IntegrityError('Joint component bytes or provenance changed')
        parts.append(actual)
    if parts[0]['path']==parts[1]['path']:raise IntegrityError('Ordinary native hosts require distinct typed weight files')
    return dict(path=str(path),weights_sha256=sha(path),artifact_format=FORMAT,
        structure=dict(kind='ordinary_lora_pair',components=[candidate_identity(p) for p in parts]),
        multiplier=1.,components=parts,tensor_count=sum(p['tensor_count'] for p in parts),
        prompt_sha256=digest([p['prompt_sha256'] for p in parts]),sidecar_sha256=sha(path),
        interpretation='weights_sha256 identifies the immutable pair manifest; constituent ordinary tensor hashes are recorded separately')


def component(path,multiplier=1.):
    pair=checkpoint(path,multiplier)
    return [dict(p['structure'],weights=p['path'],mtime=Path(p['path']).stat().st_mtime,multiplier=p['multiplier']) for p in pair['components']]
