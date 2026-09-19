"""Check the actual eight-update complete-flow attention state and ordinary exported tensors on CPU."""
import argparse
from pathlib import Path

from .core import read,immutable,sha,digest,IntegrityError
from .setup import verify
from .acoustic_artifact import checkpoint
def make(host, device):
    from app.lora_runtime import LoRANetwork
    return LoRANetwork(host, rank=8, alpha=8., multiplier=1.,
        target_replace=['MiniMaxMusic3Attention'], prefix='lora_unet', delimiter='-',
        train_method='full', attach=False).to(device).requires_grad_(True)


def audit(home,folder):
    import json
    import torch
    from safetensors.torch import load_file
    from diffusers.models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
    from ..reward_sliders.specs import MODEL
    home=Path(home).resolve();folder=Path(folder).resolve();verify(home)
    status=read(folder/'status.json');recipe=read(folder/'recipe.json')
    if status['state']!='checkpoint_ready' or status['actual_updates']!=8:
        raise IntegrityError('Expected the completed eight-update complete-flow attention export')
    if not read(folder/'off-restoration-8.json')['exact']:raise IntegrityError('Training Off restoration failed')
    candidate=checkpoint(status['checkpoint'],1.);exported=load_file(candidate['path'])
    state_path=folder/'state-step8.pt';state=torch.load(state_path,map_location='cpu',weights_only=True)
    updates=[json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines() if line]
    if state['completed']!=8 or state['recipe_sha256']!=digest(recipe) or state['history']!=updates or [r['step'] for r in updates]!=list(range(1,9)):
        raise IntegrityError('Saved optimizer history differs from the declared training')
    if exported.keys()!=state['network'].keys() or not all(torch.equal(value,state['network'][key]) for key,value in exported.items()):
        raise IntegrityError('Export does not exactly match the trained state')
    torch.set_num_threads(2)
    config=read(MODEL/'transformer/config.json')
    with torch.device('meta'):
        host=MiniMaxMusic3Transformer1DModel.from_config(config).to(torch.bfloat16).requires_grad_(False)
    torch.manual_seed(recipe['seed']);network=make(host,'cpu');initial=network.state_dict()
    attention=[k for k in initial if k.endswith('.alpha')]
    if not all(torch.equal(initial[k],exported[k]) for k in attention):
        raise IntegrityError('Alpha buffers differ from the actual initialization')
    parameters=list(network.named_parameters());groups=state['optimizer']['param_groups']
    if len(groups)!=1 or len(groups[0]['params'])!=len(parameters):raise IntegrityError('Unexpected optimizer groups')
    group=groups[0]
    if group['lr']!=recipe['learning_rate'] or tuple(group['betas'])!=tuple(recipe['betas']) or group['weight_decay']!=recipe['weight_decay']:
        raise IntegrityError('Saved optimizer settings differ from the recipe')
    selected={ident:name for ident,(name,p) in zip(groups[0]['params'],parameters) if p.requires_grad}
    optimizer=state['optimizer']['state']
    if len(selected)!=288 or optimizer.keys()!=selected.keys():raise IntegrityError('Optimizer state omits or adds attention factors')
    for ident,values in optimizer.items():
        name=selected[ident]
        if float(values['step'])!=8:raise IntegrityError('Optimizer factor has the wrong step count')
        for moment in ('exp_avg','exp_avg_sq'):
            if values[moment].shape!=exported[name].shape or not torch.isfinite(values[moment]).all():
                raise IntegrityError('Malformed optimizer moment')
        if bool((values['exp_avg_sq']<0).any()):raise IntegrityError('Negative second optimizer moment')
    changed=[name for name,p in parameters if p.requires_grad and not torch.equal(initial[name],exported[name])]
    if not changed:raise IntegrityError('No attention factor changed')
    result=dict(passed=True,checkpoint=candidate,training_state_sha256=sha(state_path),actual_updates=8,
        tensor_count=len(exported),export_matches_state_exactly=True,alpha_tensors_unchanged=len(attention),
        attention_optimizer_states=len(optimizer),all_optimizer_steps=8,changed_attention_factors=len(changed),
        optimizer_settings={key:group[key] for key in ('lr','betas','eps','weight_decay')},
        off_restoration_exact=True,model_config_sha256=sha(MODEL/'transformer/config.json'),
        source_sha256=sha(__file__),new_audio_generated=0,new_optimizer_updates=0,
        interpretation='Actual training/export integrity only; ordinary development audio determines improvement')
    immutable(home/'audit/acoustic-full-flow-legacy-export-v1.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);a=p.parse_args()
    print(__import__('json').dumps(audit(a.home,a.folder),indent=2))
