"""Full-batch clipped semantic-plus-residual surrogate on frozen parent rollouts.

Eight-code targets require exact recovery and a passed actual-host gradient audit.
This remains a code-policy surrogate and is not a waveform likelihood.
"""
import argparse
import os
from pathlib import Path
import random
import signal
import sys
import time

from .core import ROOT, read, write, immutable, digest, sha, locked
from .merged_forward import MergedForward
from .policy_objective import selected_frame_logp,clipped_loss
from .residual_policy import log_probs as residual_log_probs,selected_mean as residual_selected_mean


def train(folder,gpu,total,lr=1e-6):
    import torch
    from diffusers import ModularPipeline
    from safetensors.torch import load_file,save_file
    from ..reward_sliders.specs import MODEL,save_tensor
    from ..reward_sliders.data import generation_hidden
    from ..reward_preference.objective import guided_log_probs,reference_kl
    from ..reward_sliders.train import audit_adapter
    from ..gan_v2.state import cpu
    from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CODE_OFFSET,_SEMANTIC_VOCAB_SIZE,_AUDIO_END_TOKEN_ID
    sys.path.insert(0,str(ROOT.parent))
    from app import generator
    from app.lora_runtime import LoRANetwork
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong physical GPU')
    if total not in (10,30,60):raise ValueError('Declared short checkpoints: 10/30/60')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    with locked(folder/'train.lock'):
        source=folder;manifest=read(source/'manifest.json');episodes=read(source/'episodes.json')
        pairs=[dict(e,chosen=e['id']) for e in episodes]
        collection=read(folder/'policy-parent.json')
        if len(pairs)!=8 or read(folder/'prepare-status.json')['verified_targets']!=8:raise ValueError('Prepare all eight parent rollouts first')
        families={f['family']:f for f in manifest['families']}
        if any(families[p['family']]['split']!='train' for p in pairs):raise ValueError('Development leakage into training')
        initial=dict(checkpoint=collection['checkpoint'],checkpoint_sha256=collection['checkpoint_sha256'],coefficient=1.,name='matched-parent');seed=71
        if sha(initial['checkpoint'])!=initial['checkpoint_sha256']:raise ValueError('Matched parent changed')
        if not read(ROOT/'analysis/reward_game_v1_20260908/training/residual-imitation-v1/live-loss-audit.json')['passed']:raise ValueError('A passed actual residual-loss audit is required')
        if len(list((folder/'tokens').glob('*.pt')))!=len(pairs):raise ValueError('Recover every declared training target before training')
        recovery=read(folder/'recovery-recipe.json')
        if recovery['manifest_sha256']!=sha(source/'manifest.json') or recovery['episodes_sha256']!=sha(source/'episodes.json'):
            raise ValueError('Training sources changed since code recovery')
        for path,value in recovery['source_hashes'].items():
            if sha(path)!=value:raise ValueError('Code recovery implementation changed')
        for pair in pairs:
            target=folder/'tokens'/f"{pair['chosen']}.pt";audit=read(target.with_suffix('.json'))
            if not audit['passed'] or audit['sha256']!=sha(target):raise ValueError('Recovered target no longer matches its exact replay audit')
        recipe=dict(method='clipped-semantic-residual-full-batch-v1',initial=initial,learning_rate=lr,
                    beta_AdamW=[.9,.999],weight_decay=0.,gradient_norm=1.,seed=seed,
                    selected_positions=list(range(0,500,16)),frames=500,kl_weight=.2,residual_kl_weight=.2,prompt_weight=.2,residual_objective_weight=1.,
                    sampling='all eight saved rollouts per optimizer update, two signed examples per family; mean eight losses',policy_clip=.2,min_family_ce_gap=.05,
                    policy_probability='sum of selected semantic and seven residual log probabilities per frame; no sampler truncation',
                    advantages={e['id']:e['advantage'] for e in episodes},
                    source_manifest_sha256=sha(source/'manifest.json'),preferences_sha256=sha(source/'episodes.json'),
                    tokens={p['chosen']:sha(folder/'tokens'/f"{p['chosen']}.pt") for p in pairs},
                    source_sha256={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('merged_forward.py'),Path(__file__).with_name('residual_policy.py'),Path(__file__).with_name('policy_objective.py'))},
                    physical_gpu=gpu,checkpoint_steps=[10,30,60],full_audio_likelihood=False)
        immutable(folder/'recipe.json',recipe)
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(seed);random.seed(seed)
        began=time.monotonic()
        pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        for name in ('language_model','rvq_depth_decoder'):
            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');lm=pipe.language_model;depth=pipe.rvq_depth_decoder
        lm.eval().requires_grad_(False);depth.eval().requires_grad_(False);lm.config.use_cache=False
        network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
        weights=load_file(initial['checkpoint'])
        # Parent already contains its effective scale; load at unit multiplier.
        network.load_state_dict(weights,strict=True);audit_adapter(network,lm)
        wrapper=MergedForward(network);load_seconds=time.monotonic()-began
        rows={p['chosen']:torch.load(folder/'tokens'/f"{p['chosen']}.pt",map_location='cpu',weights_only=True) for p in pairs}
        for row in rows.values():
            if (row['signature']['recovery_sha256']!=digest(recovery) or row.get('codebooks')!=8 or
                not row.get('exact_feedback_reconstruction') or row['codes'].shape!=(500,8) or
                not torch.equal(row['tokens'],row['codes'][:,0])):raise ValueError('Invalid eight-code target alignment')
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
            codes=row['codes'][positions.cpu()].to(h.device)
            residual=residual_log_probs(pipe,h[:,boundary+positions],codes)
            return logp,residual,h[:,:boundary],mean,residual_selected_mean(residual,codes)
        def export(step):
            path=folder/f'reward-ce-balanced-policy_step{step}.safetensors'
            tensors={k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()}
            save_file(tensors,str(path))
            metadata=read(Path(initial['checkpoint']).with_suffix('.json'))
            metadata.update(steps=step,prompts_file=str(source/'manifest.json'),weights_sha256=sha(path),
                reward=dict(name='reward-ce-balanced-policy',method=recipe['method'],recipe_sha256=digest(recipe),
                    numerical_forward='pristine plus all style/reward deltas summed on CPU FP32, single BF16 cast',
                    interpretation='clipped parent-rollout code-policy surrogate; untruncated probabilities, not exact sampler or waveform likelihood',
                    initial=initial,full_audio_likelihood=False,new_optimizer='AdamW for full-batch signed clipped parent-rollout objective',policy_parent_sha256=sha(folder/'policy-parent.json')))
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
            wrapper.attach()
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
                        styles(pair['family']);logp,residual,prompt,_,_=forward(rows[pair['chosen']])
                        reference[pair['chosen']]=dict(logp=logp.cpu(),residual_logp=residual.cpu(),prompt=prompt.cpu())
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
                optimizer.zero_grad(set_to_none=True)
                metrics=[]
                # Each real optimizer update includes both signed outcomes from
                # every family; backward releases each graph before the next.
                for pair in pairs:
                    styles(pair['family']);row=rows[pair['chosen']]
                    logp,residual,prompt,mean,residual_mean=forward(row)
                    ref=reference[pair['chosen']];kl=reference_kl(logp,ref['logp']);residual_kl=reference_kl(residual,ref['residual_logp']);anchor=ref['prompt'].to(prompt).float()
                    preservation=(prompt.float()-anchor).square().mean()/anchor.square().mean().clamp_min(1e-8)
                    code_targets=row['codes'][::16].to(logp.device)
                    current_frame=selected_frame_logp(logp,residual,code_targets)
                    reference_frame=selected_frame_logp(ref['logp'].to(logp),ref['residual_logp'].to(residual),code_targets)
                    policy_loss=clipped_loss(current_frame,reference_frame,pair['advantage'],.2)
                    loss=policy_loss+.2*kl+.2*residual_kl+.2*preservation
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-batch clipped objective')
                    if max(float(kl.detach()),float(residual_kl.detach()))>.5:raise FloatingPointError('Predeclared gross policy drift stop')
                    (loss/len(pairs)).backward()
                    metrics.append(dict(family=pair['family'],rollout=pair['chosen'],advantage=pair['advantage'],policy_loss=float(policy_loss.detach()),loss=float(loss.detach()),
                        selected_semantic_mean=float(mean.detach()),selected_residual_mean=float(residual_mean.detach()),
                        residual_reference_kl=float(residual_kl.detach()),reference_kl=float(kl.detach()),prompt_loss=float(preservation.detach())))
                    del logp,residual,prompt,loss,policy_loss,current_frame,reference_frame
                grad=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();completed+=1
                metric=dict(step=completed,gradient_norm=float(grad),seconds=time.monotonic()-start,examples=len(metrics),family_count=len({p['family'] for p in metrics}),rollouts=metrics)
                for key in ('policy_loss','loss','selected_semantic_mean','selected_residual_mean','residual_reference_kl','reference_kl','prompt_loss'):
                    metric[key]=sum(m[key] for m in metrics)/len(metrics)
                history.append(metric)
                with (folder/'updates.jsonl').open('a') as log:log.write(__import__('json').dumps(metric)+'\n')
                write(folder/'status.json',dict(state='training',actual_updates=completed,total=total,load_seconds=load_seconds,updated_unix=time.time()))
                print('UPDATE '+str(metric),flush=True);save()
                if completed in (10,30,60):export(completed);save_tensor(folder/f'state-step{completed}.pt',torch.load(folder/'state.pt',map_location='cpu',weights_only=True))
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
            exact=wrapper.base_unchanged() and all(p.grad is None for p in depth.parameters())
            write(folder/f'off-restoration-{total}.json',dict(exact=exact,physical_gpu=gpu))
            if not exact:raise RuntimeError('Base changed during clipped policy training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--total',type=int,required=True);p.add_argument('--lr',type=float,default=1e-6)
    a=p.parse_args();train(a.folder,a.gpu,a.total,a.lr)
