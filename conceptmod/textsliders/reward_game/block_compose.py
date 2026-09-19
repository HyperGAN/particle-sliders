"""Fixed attention/FF composition on disjoint projections; no new optimization."""
import argparse
from pathlib import Path
import shutil

from .core import read,write,immutable,sha,digest,IntegrityError
from .acoustic_artifact import checkpoint as attention_checkpoint
from .ff_artifact import checkpoint as ff_checkpoint
from .block_artifact import checkpoint,STRUCTURE


def compose(home,folder):
    import torch
    from safetensors.torch import load_file,save_file
    home=Path(home).resolve();folder=Path(folder).resolve()
    attention=attention_checkpoint(home/'training/acoustic-gain150-v1/reward-ce-acoustic-gain150.safetensors',1.)
    feed_forward=ff_checkpoint(home/'training/acoustic-ff-ce-v1/reward-ce-acoustic-ff_step8.safetensors',1.)
    if attention['prompt_sha256']!=feed_forward['prompt_sha256']:
        raise IntegrityError('This composition declares the same eight training captures for both parents')
    source_paths=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('block_artifact.py','acoustic_artifact.py','ff_artifact.py')]
    recipe=dict(method='disjoint-acoustic-projection-composition-v1',parents=dict(attention=attention,feed_forward=feed_forward),
        hypothesis='The independently fitted attention and feed-forward updates each improved all sixteen development cases; their joint effect may retain consistency and increase the gain',
        failure_mechanism='The FF-only mean was .05345 below .10; the earlier calibrated attention adapter missed fresh confirmation. Test complementary layer support with the language model fixed.',
        operation='Copy attention up/down/alpha tensors from the fixed gain150 parent, and feed-forward up/down/alpha tensors from the fixed FF-step8 parent; no arithmetic, gain choice, or per-case selection',
        structure=STRUCTURE,multiplier=1.,local_optimizer_updates=0,new_training_clips=0,
        shared_training_evidence='Both parents use the same eight Off captures from four training families; these are not independent training datasets',
        sources={str(p.resolve()):sha(p) for p in source_paths})
    immutable(folder/'recipe.json',recipe)
    for path,h in recipe['sources'].items():
        target=folder/'sources'/h/Path(path).name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        if sha(target)!=h:raise IntegrityError('Archived composition source differs')
    a=load_file(attention['path']);f=load_file(feed_forward['path'])
    attention_keys={key for key in f if '-attn-' in key}
    if set(a)!=attention_keys or len(a)!=432 or len(f)!=648:
        raise IntegrityError('Parent projection supports do not match the declared composition')
    tensors={key:(a[key] if key in attention_keys else value).clone().contiguous() for key,value in f.items()}
    path=folder/'reward-ce-acoustic-block-compose.safetensors'
    if path.exists():
        saved=load_file(str(path))
        if saved.keys()!=tensors.keys() or not all(torch.equal(saved[k],v) for k,v in tensors.items()):
            raise IntegrityError('Existing composition differs from the fixed parent factors')
    else:save_file(tensors,str(path))
    metadata=read(Path(attention['path']).with_suffix('.json'));prompt=Path(metadata['prompts_file'])
    if not prompt.is_absolute():
        from .core import ROOT
        prompt=ROOT/prompt
    target_prompt=folder/'training-prompts.json'
    if target_prompt.exists():
        if sha(target_prompt)!=attention['prompt_sha256']:raise IntegrityError('Copied training prompts changed')
    else:shutil.copy2(prompt,target_prompt)
    immutable(path.with_suffix('.json'),dict(STRUCTURE,weights_sha256=sha(path),prompts_file=str(target_prompt),
        local_optimizer_updates=0,recipe_sha256=digest(recipe),parent_checkpoints=recipe['parents'],
        reward=dict(name='reward-ce-acoustic-block-compose',method=recipe['method'],interpretation=recipe['operation'])))
    result=dict(candidate=checkpoint(path,1.),attention_source_tensors=len(a),feed_forward_source_tensors=len(f)-len(a),
        all_factors_exact_parent_copies=True,recipe_sha256=digest(recipe),new_optimizer_updates=0,new_audio_generated=0,
        actual_native_merge_audit_required=True)
    immutable(folder/'composition.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);a=p.parse_args()
    print(__import__('json').dumps(compose(a.home,a.folder),indent=2))
