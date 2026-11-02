"""Recover chosen training codebooks and audit the richer loss on the real host."""
import argparse
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from .core import DEFAULT_HOME,ROOT,read,write,immutable,sha,digest,locked
from .replay_residual import recover
from .residual_policy import log_probs,selected_mean


def prepare(home,folder,gpu,limit):
    import torch
    from diffusers import ModularPipeline
    from ..reward_sliders.specs import MODEL,save_tensor
    from ..reward_sliders.data import generation_hidden
    from .setup import verify
    from .resources import gpu_lease
    from .merged_forward import MergedForward
    from app import generator
    from app.lora_runtime import LoRANetwork
    from safetensors.torch import load_file
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong physical GPU')
    folder=Path(folder);source=ROOT/'analysis/reward_preference_20260908'
    with locked(folder/'prepare.lock'),gpu_lease(home,gpu):
        game,_=verify(home);manifest=read(source/'manifest.json');pairs=read(source/'preferences.json')
        families={f['family']:f for f in manifest['families']};development={c['family']['family'] for c in game['cases']}
        observations={v['observation']['id']:v['observation'] for g in (0,1) for v in read(source/f'replay-gpu{g}.json')}
        if any(p['family'] in development or families[p['family']]['split']!='train' for p in pairs):raise ValueError('Training split overlap')
        paths=[Path(__file__),Path(__file__).with_name('replay_residual.py'),Path(__file__).with_name('residual_policy.py')]
        import inspect
        from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
        paths.extend([Path(__file__).with_name('merged_forward.py'),Path(inspect.getfile(MiniMaxMusic3RVQDepthDecoder)),Path(inspect.getfile(generation_hidden))])
        signature=dict(method='chosen-eight-codebook-recovery-v1',source_manifest_sha256=sha(source/'manifest.json'),
                       preferences_sha256=sha(source/'preferences.json'),frames=500,physical_gpu=gpu,
                       source_hashes={str(p):sha(p) for p in paths},chosen=[p['chosen'] for p in pairs])
        immutable(folder/'recovery-recipe.json',signature)
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(71)
        began=time.monotonic();pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        for name in ('language_model','rvq_depth_decoder'):
            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');lm=pipe.language_model;decoder=pipe.rvq_depth_decoder
        lm.eval().requires_grad_(False);decoder.eval().requires_grad_(False);load_seconds=time.monotonic()-began
        wrapper=None
        try:
            for pair in pairs[:limit]:
                observation=observations[pair['chosen']];path=folder/'tokens'/f"{pair['chosen']}.pt"
                target_signature=dict(recovery_sha256=digest(signature),trajectory_sha256=observation['trajectory_sha256'],
                                      audio_sha256=observation['audio_sha256'],semantic_target_sha256=sha(source/'tokens'/path.name))
                if path.exists():
                    if torch.load(path,map_location='cpu',weights_only=True)['signature']!=target_signature:raise ValueError('Target cache identity changed')
                    continue
                for label in ('audio','trajectory'):
                    if sha(observation[label])!=observation[label+'_sha256']:raise ValueError('Capture corrupted')
                generator._merge_sliders(pipe,'cuda:0',manifest['style_components'][pair['family']])
                started=time.monotonic();trajectory=torch.load(observation['trajectory'],map_location='cpu',weights_only=True)
                recovered=recover(pipe,trajectory,500)
                previous=torch.load(source/'tokens'/path.name,map_location='cpu',weights_only=True)
                for key in ('tokens','prompt_embeds','frame_embeds'):
                    if not torch.equal(recovered[key],previous[key]):raise ValueError('Eight-code recovery differs from verified semantic history')
                save_tensor(path,dict(signature=target_signature,**recovered))
                immutable(path.with_suffix('.json'),dict(passed=True,sha256=sha(path),source=pair['chosen'],
                    codebooks=8,frames=500,verified_feedback_frames=501,seconds=time.monotonic()-started,new_audio=0))
                print('RECOVERED '+pair['chosen']+'; all eight codebooks; 501 exact feedback frames',flush=True)
            audit=folder/'live-loss-audit.json'
            if not audit.exists():
                pair=pairs[0];row=torch.load(folder/'tokens'/f"{pair['chosen']}.pt",map_location='cpu',weights_only=True)
                generator._merge_sliders(pipe,'cuda:0',[])
                network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
                weights=load_file(manifest['preference']['initial']['checkpoint'])
                weights={k:v*.5 if k.endswith('.lora_up.weight') else v for k,v in weights.items()}
                network.load_state_dict(weights,strict=True);wrapper=MergedForward(network)
                nets=[(generator._slider_network(pipe,'cuda:0',c,attach=False),c['multiplier']) for c in manifest['style_components'][pair['family']]]
                wrapper.styles(nets);wrapper.attach();lm.config.use_cache=False
                lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                embeds=torch.cat((row['prompt_embeds'],row['frame_embeds'][:,:128]),1).to('cuda:0');boundary=row['prompt_embeds'].shape[1]
                hidden=generation_hidden(lm,embeds)[:,boundary:][:,::16];codes=row['codes'][:128:16].to('cuda:0')
                probabilities=log_probs(pipe,hidden,codes)
                # Runtime's exact prefix-by-prefix teacher forcing at one position.
                from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CODE_OFFSET
                with torch.no_grad():
                    h=hidden[:,0];target=codes[:1].repeat(2,1)
                    sequence=[decoder.projection(h)[:,None],decoder.projection(lm.model.embed_tokens(target[:,0]+_AUDIO_CODE_OFFSET))[:,None]]
                    reference=[]
                    for index in range(7):
                        result=decoder(torch.cat(sequence,dim=1))[:,-1];logits=decoder.audio_heads[index](result).float()
                        reference.append((logits[1]+1.5*(logits[0]-logits[1])).log_softmax(-1))
                        if index<6:
                            embed=decoder.audio_embeddings(target[:,index+1]+index*decoder.audio_heads[index].out_features)
                            sequence.append(decoder.projection(embed)[:,None])
                    reference=torch.stack(reference);actual=log_probs(pipe,hidden[:,:1],codes[:1])[0]
                    kl=float((reference.exp()*(reference-actual)).sum(-1).mean());maximum=float((reference-actual).abs().max())
                loss=-selected_mean(probabilities,codes);loss.backward()
                grad=sum(float(p.grad.abs().sum()) for p in network.parameters() if p.grad is not None)
                optimizer=torch.optim.AdamW(network.parameters(),lr=1e-5);optimizer.step()
                changed=any(not torch.equal(v.cpu(),weights[k]) for k,v in network.state_dict().items())
                base_ok=wrapper.base_unchanged() and all(p.grad is None for p in lm.parameters())
                depth_ok=all(p.grad is None for p in decoder.parameters())
                result=dict(passed=kl<.001 and grad>0 and changed and base_ok and depth_ok,
                    residual_policy_kl_vs_sequential_prefix=kl,max_abs_log_probability_error=maximum,predeclared_kl_limit=.001,
                    gradient_absolute_sum=grad,adapter_update_changed_weights=changed,base_weights_frozen=base_ok,
                    depth_decoder_frozen=depth_ok,audit_update_discarded=True,new_clips=0,load_seconds=load_seconds,
                    full_waveform_likelihood=False,physical_gpu=gpu)
                immutable(audit,result);network.load_state_dict(weights,strict=True)
                print('LIVE AUDIT '+str(result),flush=True)
                if not result['passed']:raise ValueError('Actual residual objective audit failed')
            elif not read(audit)['passed']:raise ValueError('Saved actual residual objective audit failed')
            write(folder/'status.json',dict(state='targets_prepared' if limit==len(pairs) else 'pilot_audited',
                verified_targets=len(list((folder/'tokens').glob('*.pt'))),required_targets=len(pairs),actual_updates=0,new_clips=0))
        finally:
            if wrapper is not None:wrapper.detach()
            generator._merge_sliders(pipe,'cuda:0',[])
            exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in generator._merge_state('cuda:0').pristine.items())
            write(folder/f'off-restoration-limit{limit}.json',dict(exact=exact,physical_gpu=gpu))
            if not exact:raise RuntimeError('Base restoration failed')


if __name__=='__main__':
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser();p.add_argument('--home',default=str(DEFAULT_HOME));p.add_argument('--folder',required=True)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--limit',type=int,choices=(1,24),default=1)
    a=p.parse_args();prepare(a.home,a.folder,a.gpu,a.limit)
