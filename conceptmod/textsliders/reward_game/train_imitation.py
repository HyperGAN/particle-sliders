"""Positive-target semantic imitation with deployment-matched numerical output.

Only previously captured training takes are used. This is a semantic objective,
not waveform likelihood or residual-code likelihood. Rejected takes are not
pushed downward. Every selected generation position retains its causal history.
"""
import argparse
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

from .core import ROOT, read, write, immutable, digest, sha, locked
from .merged_forward import MergedForward


def train(folder,gpu,total,lr=1e-5):
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file,save_file
    from ..reward_sliders.specs import MODEL,save_tensor
    from ..reward_sliders.data import generation_hidden
    from ..reward_preference.objective import guided_log_probs,reference_kl
    from ..reward_sliders.evaluate import component
    from ..reward_sliders.train import audit_adapter
    from ..gan_v2.state import cpu
    from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CODE_OFFSET,_SEMANTIC_VOCAB_SIZE,_AUDIO_END_TOKEN_ID
    sys.path.insert(0,str(ROOT.parent))
    from app import generator
    from app.lora_runtime import LoRANetwork
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong physical GPU')
    if total not in (0,10,30,60):raise ValueError('Declared short checkpoints: 10/30/60; zero is audit only')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    with locked(folder/'train.lock'):
        source=ROOT/'analysis/reward_preference_20260908';manifest=read(source/'manifest.json');pairs=read(source/'preferences.json')
        families={f['family']:f for f in manifest['families']}
        if any(families[p['family']]['split']!='train' for p in pairs):raise ValueError('Development leakage into training')
        initial=manifest['preference']['initial'];seed=71
        recipe=dict(method='positive-semantic-imitation-merged-v1',initial=initial,learning_rate=lr,
                    beta_AdamW=[.9,.999],weight_decay=0.,gradient_norm=1.,seed=seed,
                    selected_positions=list(range(0,500,16)),frames=500,kl_weight=.2,prompt_weight=.2,
                    sampling='equal family permutation, one chosen training take per update',
                    source_manifest_sha256=sha(source/'manifest.json'),preferences_sha256=sha(source/'preferences.json'),
                    tokens={p['chosen']:sha(source/'tokens'/f"{p['chosen']}.pt") for p in pairs},
                    source_sha256={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('merged_forward.py'))},
                    physical_gpu=gpu,checkpoint_steps=[10,30,60],full_audio_likelihood=False)
        immutable(folder/'recipe.json',recipe)
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(seed);random.seed(seed)
        began=time.monotonic()
        lm=AutoModelForCausalLM.from_pretrained(str(MODEL/'language_model'),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval().requires_grad_(False)
        lm.config.use_cache=False;pipe=SimpleNamespace(language_model=lm)
        network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
        weights=load_file(initial['checkpoint'])
        for key in weights:
            if key.endswith('.lora_up.weight'):weights[key]=weights[key]*.5
        network.load_state_dict(weights,strict=True);audit_adapter(network,lm)
        wrapper=MergedForward(network);load_seconds=time.monotonic()-began
        rows={p['chosen']:torch.load(source/'tokens'/f"{p['chosen']}.pt",map_location='cpu',weights_only=True) for p in pairs}
        current_style=None
        def styles(family):
            nonlocal current_style
            if current_style==family:return
            components=manifest['style_components'][family]
            nets=[]
            with torch.random.fork_rng(devices=[0]):
                for comp in components:nets.append((generator._slider_network(pipe,'cuda:0',comp,attach=False),comp['multiplier']))
            wrapper.styles(nets);current_style=family
        def forward(row,frames=500):
            embeds=torch.cat((row['prompt_embeds'],row['frame_embeds'][:,:frames]),1).to('cuda:0')
            boundary=row['prompt_embeds'].shape[1];h=generation_hidden(lm,embeds)
            positions=torch.arange(0,frames,16,device=h.device)
            logp=guided_log_probs(h[:,boundary+positions],lm.lm_head.weight,offset=_AUDIO_CODE_OFFSET,
                                 vocabulary=_SEMANTIC_VOCAB_SIZE,eos=_AUDIO_END_TOKEN_ID)
            target=row['tokens'][positions.cpu()].to(h.device).long()
            mean=logp.gather(-1,target[:,None]).mean()
            return logp,h[:,:boundary],mean,h
        def export(step):
            path=folder/f'reward-ce-merged-imitation_step{step}.safetensors'
            tensors={k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()}
            save_file(tensors,str(path))
            metadata=read(Path(initial['checkpoint']).with_suffix('.json'))
            metadata.update(steps=step,prompts_file=str(source/'manifest.json'),weights_sha256=sha(path),
                reward=dict(name='reward-ce-merged-imitation',method=recipe['method'],recipe_sha256=digest(recipe),
                    numerical_forward='pristine plus all style/reward deltas summed on CPU FP32, single BF16 cast',
                    interpretation='experimental semantic positive imitation; evaluate through frozen development game',
                    initial=initial,full_audio_likelihood=False,new_optimizer='AdamW for changed objective'))
            write(path.with_suffix('.json'),metadata)
            loaded=load_file(str(path));assert all(torch.equal(loaded[k],v) for k,v in tensors.items())
            return path
        completed=0;history=[];order=[];cursor=0;sampler=torch.Generator().manual_seed(seed)
        optimizer=torch.optim.AdamW(network.parameters(),lr=lr,betas=(.9,.999),weight_decay=0.)
        def save():
            save_tensor(folder/'state.pt',dict(recipe_sha256=digest(recipe),network=cpu(network.state_dict()),optimizer=cpu(optimizer.state_dict()),
                completed=completed,history=history,order=order,cursor=cursor,sampler=sampler.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate()))
        try:
            if not (folder/'live-forward-audit.json').exists():
                pair=next(p for p in pairs if manifest['style_components'][p['family']]);row=rows[pair['chosen']]
                styles(pair['family']);wrapper.attach()
                with torch.no_grad():logp,_,_,expected=forward(row,128)
                wrapper.detach();path=export(0)
                generator._merge_sliders(pipe,'cuda:0',manifest['style_components'][pair['family']]+[component(path,1.)])
                with torch.no_grad():actual_logp,_,_,actual=forward(row,128)
                equal=torch.equal(expected,actual);difference=float((expected.float()-actual.float()).abs().max())
                generator._merge_sliders(pipe,'cuda:0',[]);wrapper.attach()
                optimizer.zero_grad(set_to_none=True);_,_,mean,_=forward(row,128);(-mean).backward()
                grad=sum(float(p.grad.abs().sum()) for p in network.parameters() if p.grad is not None)
                optimizer.step();changed=any(not torch.equal(v.cpu(),weights[k]) for k,v in network.state_dict().items())
                base_ok=wrapper.base_unchanged() and all(p.grad is None for p in lm.parameters())
                network.load_state_dict(weights,strict=True);optimizer=torch.optim.AdamW(network.parameters(),lr=lr,betas=(.9,.999),weight_decay=0.)
                result=dict(passed=equal and grad>0 and changed and base_ok,exact_hidden_output=equal,max_abs_hidden_error=difference,
                            exact_guided_policy=torch.equal(logp,actual_logp),gradient_absolute_sum=grad,adapter_update_changed_weights=changed,
                            base_weights_frozen=base_ok,physical_gpu=gpu,family=pair['family'],frames=128,
                            audit_update_discarded=True,load_seconds=load_seconds,new_clips=0)
                immutable(folder/'live-forward-audit.json',result)
                if not result['passed']:raise RuntimeError('Live merged forward/gradient audit failed')
            else:
                if not read(folder/'live-forward-audit.json')['passed']:raise RuntimeError('Recorded live audit failed')
                wrapper.attach()
            if total==0:return
            # References are specific to this numerical forward and initial policy.
            ref_path=folder/'reference.pt'
            if ref_path.exists():
                value=torch.load(ref_path,map_location='cpu',weights_only=True)
                if value['recipe_sha256']!=digest(recipe):raise ValueError('Incompatible numerical reference cache')
                reference=value['rows']
            else:
                reference={}
                with torch.no_grad():
                    for pair in pairs:
                        styles(pair['family']);logp,prompt,_,_=forward(rows[pair['chosen']])
                        reference[pair['chosen']]=dict(logp=logp.cpu(),prompt=prompt.cpu())
                        print('REFERENCE '+pair['chosen'],flush=True)
                save_tensor(ref_path,dict(recipe_sha256=digest(recipe),rows=reference))
            if (folder/'state.pt').exists():
                value=torch.load(folder/'state.pt',map_location='cpu',weights_only=True)
                if value['recipe_sha256']!=digest(recipe):raise ValueError('Full-state recipe changed')
                network.load_state_dict(value['network'],strict=True);optimizer.load_state_dict(value['optimizer'])
                completed=value['completed'];history=value['history'];order=value['order'];cursor=value['cursor'];sampler.set_state(value['sampler'])
                torch.set_rng_state(value['torch_rng']);torch.cuda.set_rng_state(value['cuda_rng']);random.setstate(value['python_rng'])
            stop=False
            def cancel(*args):
                nonlocal stop
                stop=True
            signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});lm.train()
            while completed<total and not stop:
                start=time.monotonic()
                if cursor>=len(order):order=torch.randperm(len(pairs),generator=sampler).tolist();cursor=0
                pair=pairs[order[cursor]];cursor+=1;styles(pair['family']);row=rows[pair['chosen']]
                optimizer.zero_grad(set_to_none=True);logp,prompt,mean,_=forward(row)
                ref=reference[pair['chosen']];kl=reference_kl(logp,ref['logp']);anchor=ref['prompt'].to(prompt).float()
                preservation=(prompt.float()-anchor).square().mean()/anchor.square().mean().clamp_min(1e-8)
                loss=-pair['weight']*mean+.2*kl+.2*preservation
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite imitation objective')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();completed+=1
                metric=dict(step=completed,family=pair['family'],loss=float(loss.detach()),selected_semantic_mean=float(mean.detach()),
                            reference_kl=float(kl.detach()),prompt_loss=float(preservation.detach()),gradient_norm=float(grad),seconds=time.monotonic()-start)
                history.append(metric)
                with (folder/'updates.jsonl').open('a') as log:log.write(__import__('json').dumps(metric)+'\n')
                write(folder/'status.json',dict(state='training',actual_updates=completed,total=total,load_seconds=load_seconds,updated_unix=time.time()))
                print('UPDATE '+str(metric),flush=True);save()
                if completed in (10,30,60):export(completed);save_tensor(folder/f'state-step{completed}.pt',torch.load(folder/'state.pt',map_location='cpu',weights_only=True))
                if float(kl.detach())>.5:raise FloatingPointError('Predeclared gross policy drift stop')
            save()
            if stop:raise SystemExit(130)
            path=export(completed);audit_adapter(network,lm)
            write(folder/'status.json',dict(state='checkpoint_ready',actual_updates=completed,checkpoint=str(path),
                        optimizer_seconds=sum(v['seconds'] for v in history),load_seconds=load_seconds))
        except BaseException as exc:
            if total:save()
            write(folder/'error.json',dict(error=repr(exc),actual_updates=completed));raise
        finally:
            wrapper.detach();generator._merge_sliders(pipe,'cuda:0',[])
            exact=wrapper.base_unchanged()
            write(folder/f'off-restoration-{total}.json',dict(exact=exact,physical_gpu=gpu))
            if not exact:raise RuntimeError('Base changed during imitation training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--total',type=int,required=True);p.add_argument('--lr',type=float,default=1e-5)
    a=p.parse_args();train(a.folder,a.gpu,a.total,a.lr)
