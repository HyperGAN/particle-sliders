"""Continue the existing rank-8 weights with scored semantic preferences."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
from types import SimpleNamespace
import time

import torch
from safetensors.torch import load_file,save_file

from ..reward_sliders.specs import MODEL,read_json,write_json,sha,digest,save_tensor
from ..reward_sliders.experiment import verify
from ..reward_sliders.data import generation_hidden,set_scale
from ..reward_sliders.train import audit_adapter
from ..gan_v2.state import cpu
from .objective import guided_log_probs,selected_mean,detached_pair_coefficients,preference_loss,reference_kl


def policy(lm,row,device):
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET,_SEMANTIC_VOCAB_SIZE,_AUDIO_END_TOKEN_ID)
    embeds=torch.cat((row['prompt_embeds'],row['frame_embeds']),1).to(device)
    boundary=row['prompt_embeds'].shape[1]
    hidden=generation_hidden(lm,embeds)
    distribution=guided_log_probs(hidden[:,boundary:],lm.lm_head.weight,
        offset=_AUDIO_CODE_OFFSET,vocabulary=_SEMANTIC_VOCAB_SIZE,eos=_AUDIO_END_TOKEN_ID)
    return distribution,hidden[:,:boundary]


def train(run,arm,gpu,total):
    from app import generator
    from app.lora_runtime import LoRANetwork
    from transformers import AutoModelForCausalLM
    run=Path(run);m=read_json(run/'manifest.json');verify(m);settings=m['preference']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong GPU assignment')
    if arm not in settings['learning_rates'] or total not in settings['checkpoints']:raise ValueError('Undeclared training point')
    folder=run/'students'/arm;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
    seed=settings['seed'];torch.manual_seed(seed);random.seed(seed)
    initial=settings['initial'];assert sha(initial['checkpoint'])==initial['checkpoint_sha256']
    pairs=read_json(run/'preferences.json')
    signature=dict(manifest_sha256=digest(m),preference_sha256=sha(run/'preferences.json'),arm=arm,
                   token_sha256={ident:sha(run/'tokens'/f'{ident}.pt') for p in pairs for ident in (p['chosen'],p['rejected'])})
    frozen=folder/'manifest.json'
    if frozen.exists():assert read_json(frozen)==signature
    write_json(frozen,signature)
    lm=AutoModelForCausalLM.from_pretrained(str(MODEL/'language_model'),torch_dtype=torch.bfloat16,
                                           local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache=False;pipe=SimpleNamespace(language_model=lm)
    def styles(family):
        with torch.random.fork_rng(devices=[0]):generator._merge_sliders(pipe,'cuda:0',m['style_components'][family])
    network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],
                        prefix='lora_te',delimiter='-',train_method='full').to(device).requires_grad_(True)
    weights=load_file(initial['checkpoint'])
    for key in weights:
        if key.endswith('.lora_up.weight'):weights[key]=weights[key]*initial['coefficient']
    network.load_state_dict(weights,strict=True);audit_adapter(network,lm)
    # New preference optimizer is intentional: the old optimizer belonged to a GAN game.
    optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rates'][arm],betas=(.9,.999),weight_decay=0.)
    rows={}
    for pair in pairs:
        for ident in (pair['chosen'],pair['rejected']):
            value=torch.load(run/'tokens'/f'{ident}.pt',map_location='cpu',weights_only=True)
            if not value['exact_feedback_reconstruction']:raise ValueError('Unverified semantic targets')
            rows[ident]=dict(value,family=pair['family'])
    cache=run/'reference.pt';cache_signature=digest(dict(initial=initial,tokens=signature['token_sha256'],manifest=digest(m)))
    with (run/'reference.lock').open('a') as cache_lock:
        fcntl.flock(cache_lock,fcntl.LOCK_EX)
        if cache.exists():
            value=torch.load(cache,map_location='cpu',weights_only=True)
            if value['signature']!=cache_signature:raise ValueError('Reference cache changed')
            reference=value['rows'];del value
        else:
            reference={}
            with torch.no_grad():
                for ident,row in rows.items():
                    styles(row['family']);logp,prompt=policy(lm,row,device)
                    reference[ident]=dict(log_probs=logp.cpu(),mean=selected_mean(logp,row['tokens']).cpu(),prompt=prompt.cpu())
                    print('REFERENCE '+ident,flush=True)
            save_tensor(cache,dict(signature=cache_signature,rows=reference))
    torch.manual_seed(seed);random.seed(seed)
    sampler=torch.Generator().manual_seed(seed);order=[];cursor=0;completed=0;history=[]
    state_path=folder/'state.pt'
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=True)
        if saved['signature']!=signature:raise ValueError('Continuation state identity changed')
        network.load_state_dict(saved['network'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        completed=saved['completed'];history=saved['history'];order=saved['order'];cursor=saved['cursor']
        sampler.set_state(saved['sampler']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    stop=False
    def cancel(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def save():
        save_tensor(state_path,dict(signature=signature,network=cpu(network.state_dict()),optimizer=cpu(optimizer.state_dict()),
            completed=completed,history=history,order=order,cursor=cursor,sampler=sampler.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    try:
        with (folder/'updates.jsonl').open('a') as log:
            while completed<total and not stop:
                began=time.monotonic()
                if cursor>=len(order):order=torch.randperm(len(pairs),generator=sampler).tolist();cursor=0
                pair=pairs[order[cursor]];cursor+=1;styles(pair['family']);ids=[pair['chosen'],pair['rejected']]
                with torch.no_grad():
                    means=[selected_mean(policy(lm,rows[ident],device)[0],rows[ident]['tokens']) for ident in ids]
                refs=[reference[ident]['mean'].to(device) for ident in ids]
                coefficients=detached_pair_coefficients(*means,*refs,beta=settings['beta'],weight=pair['weight'])
                optimizer.zero_grad(set_to_none=True);metrics=[]
                for ident,coefficient in zip(ids,coefficients):
                    row=rows[ident];logp,prompt=policy(lm,row,device)
                    mean=selected_mean(logp,row['tokens']);kl=reference_kl(logp,reference[ident]['log_probs'])
                    anchor=reference[ident]['prompt'].to(prompt).float()
                    prompt_loss=(prompt.float()-anchor).square().mean()/anchor.square().mean().clamp_min(1e-8)
                    loss=coefficient*mean + .5*settings['kl_weight']*kl + .5*settings['prompt_weight']*prompt_loss
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preference objective')
                    loss.backward();metrics.append(dict(kl=float(kl.detach()),prompt=float(prompt_loss.detach())))
                grad=torch.nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_norm'],error_if_nonfinite=True)
                optimizer.step();completed+=1
                update=dict(step=completed,family=pair['family'],preference_loss=float(preference_loss(*means,*refs,beta=settings['beta'])),
                            gradient_norm=float(grad),metrics=metrics,seconds=time.monotonic()-began)
                history.append(update);log.write(json.dumps(update,allow_nan=False)+'\n');log.flush()
                write_json(folder/'status.json',dict(stage='training',step=completed,total=total,updated_unix=time.time()))
                print(f'UPDATE {arm} {completed}/{total} preference={update["preference_loss"]:.4f} seconds={update["seconds"]:.2f}',flush=True)
                if completed%10==0:save()
                if max(x['kl'] for x in metrics)>settings['gross_kl_stop']:raise FloatingPointError('Reference-policy drift exceeds declared bound')
            save()
            if stop:raise SystemExit(130)
            audit_adapter(network,lm)
            path=folder/f'reward-ce-preference-{arm}_step{completed}.safetensors'
            tensors={k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()};save_file(tensors,str(path))
            loaded=load_file(str(path));assert set(loaded)==set(tensors) and all(torch.equal(loaded[k],v) for k,v in tensors.items())
            metadata=read_json(Path(initial['checkpoint']).with_suffix('.json'))
            metadata.update(steps=completed,prompts_file=str(run/'manifest.json'),weights_sha256=sha(path),
                reward=dict(name='reward-ce-preference-'+arm,method='semantic preference continuation',
                    initial=initial,manifest_sha256=digest(m),signature_sha256=digest(signature),
                    spec=m['reward_spec'],interpretation='experimental continuation; small audio screen pending',
                    reference_optimizer='new AdamW for changed objective',full_audio_likelihood=False))
            write_json(path.with_suffix('.json'),metadata)
            write_json(folder/'status.json',dict(stage='checkpoint_ready',step=completed,weights=str(path),updated_unix=time.time()))
    except BaseException as error:
        save();write_json(folder/'status.json',dict(stage='error',step=completed,error=repr(error)));raise
    finally:
        set_scale(network,0.);network.detach();generator._merge_sliders(pipe,'cuda:0',[])
        exact=all(torch.equal(module.weight.detach().cpu(),base) for module,base in generator._merge_state('cuda:0').pristine.items())
        write_json(folder/f'off-restoration-{total}.json',dict(exact=exact,physical_gpu=gpu))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--total',type=int,required=True)
    args=p.parse_args();train(args.run_dir,args.arm,args.gpu,args.total)
