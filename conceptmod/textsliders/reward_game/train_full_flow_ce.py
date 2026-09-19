"""Unselected direct CE through all thirty flow steps within each chunk.

Conditioning and reference overlap between chunks remain fixed. Native ordinary
rank-8 attention export and the existing reward procedure are preserved.
"""
import argparse
from pathlib import Path
import os
import signal
import sys
import time
from .core import ROOT,read,write,immutable,sha,digest,locked,IntegrityError
from .acoustic_artifact import STRUCTURE,checkpoint
from .acoustic_tail import decode,decode_piece
from .full_flow import replay as replay_latents,SCOPE
from .ce_wave_gradient import value_and_gradient
from .cached_merged_forward import CachedMergedForward as MergedForward


def train(home,folder,game_home,full_audit_folder,cache_audit_folder,total=8,lr=.001):
    import torch
    from diffusers import ModularPipeline
    from safetensors.torch import save_file
    from ..reward_sliders.specs import MODEL,RewardSpec,save_tensor,validate_families
    from ..reward_sliders.reward import CEReward
    from ..gan_v2.state import cpu
    from .setup import verify
    sys.path.insert(0,str(ROOT.parent));from app import generator
    from app.lora_runtime import LoRANetwork
    home=Path(home).resolve();folder=Path(folder).resolve();game,manifest=verify(home)
    game_home=Path(game_home).resolve();acoustic_game,_=verify(game_home)
    if read(game_home/'game.json').get('parent_home')!=str(home):raise IntegrityError('Acoustic game belongs to a different parent campaign')
    off_audit=game_home/'audit/off-pcm-v1/result.json'
    if not off_audit.exists() or not read(off_audit)['passed']:raise IntegrityError('Acoustic game Off PCM audit must pass before training')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise IntegrityError('Acoustic training owns physical GPU 1')
    if total!=8:raise IntegrityError('This full-within-chunk-flow pilot has exactly eight updates')
    collection=read(folder/'collection-recipe.json');captures=read(folder/'captures.json');status=read(folder/'collection-status.json')
    if status['state']!='complete' or status['verified_captures']!=8 or len(captures)!=8:raise IntegrityError('All eight acoustic captures must be verified before updates')
    audit=Path(collection['previous_tail_audit_folder'])
    full_audit_folder=Path(full_audit_folder).resolve();cache_audit_folder=Path(cache_audit_folder).resolve()
    full_result=read(full_audit_folder/'gradient-result.json');full_protocol=read(full_audit_folder/'gradient-protocol.json')
    cache_result=read(cache_audit_folder/'result.json');cache_protocol=read(cache_audit_folder/'protocol.json')
    if collection['retained_flow_steps']!=30 or not full_result['passed'] or not full_result['off_waveform_exact']:
        raise IntegrityError('Actual complete within-chunk flow gradient and native waveform audit must pass')
    if not cache_result['passed'] or cache_protocol['kind']!='attention' or cache_result['protocol_sha256']!=digest(cache_protocol):
        raise IntegrityError('Actual attention cache equivalence must pass')
    if full_protocol['collection_recipe_sha256']!=sha(folder/'collection-recipe.json') or full_protocol['cache_audit_result_sha256']!=sha(cache_audit_folder/'result.json'):
        raise IntegrityError('Full-flow audit belongs to different capture/cache evidence')
    for path,h in {**full_protocol['source_hashes'],**cache_protocol['sources']}.items():
        if sha(path)!=h:raise IntegrityError('Full-flow or cache audit source changed')
    if not read(audit/'gradient-result.json')['passed'] or not read(folder/'collection-off-restoration.json')['exact']:raise IntegrityError('Acoustic prerequisites incomplete')
    for path,h in collection['source_hashes'].items():
        if sha(path)!=h:raise IntegrityError('Acoustic capture implementation changed')
    for name,h in collection['audit_hashes'].items():
        if sha(audit/name)!=h:raise IntegrityError('Acoustic audit changed')
    by_id={r['id']:r for r in captures};cases=collection['cases'];families={c['family']['family']:c['family'] for c in cases}
    if len(families)!=4 or any(f['split']!='train' for f in families.values()):raise IntegrityError('Acoustic data must contain four training families')
    validate_families(list(families.values()))
    for c in cases:
        r=by_id[c['id']]
        if r['status']!='complete' or r['recipe_sha256']!=digest(collection) or r['source_observation_sha256']!=c['source_observation_sha256']:raise IntegrityError('Acoustic target identity changed')
        if sha(r['capture'])!=r['capture_sha256'] or sha(r['audio'])!=r['audio_sha256']:raise IntegrityError('Acoustic target bytes changed')
    immutable(folder/'training-prompts.json',dict(families=list(families.values()),capture_collection_sha256=sha(folder/'collection-recipe.json')))
    sources=[Path(__file__),Path(__file__).with_name('acoustic_tail.py'),Path(__file__).with_name('merged_forward.py'),Path(__file__).with_name('cached_merged_forward.py'),Path(__file__).with_name('full_flow.py'),Path(__file__).with_name('ce_wave_gradient.py'),Path(__file__).with_name('acoustic_artifact.py')]
    recipe=dict(method='direct-ce-full-within-chunk-flow-attention-v1',initial='zero acoustic LoRA; fresh Kaiming down factors and zero up factors',
        structure=STRUCTURE,seed=9131,learning_rate=lr,optimizer='AdamW',betas=[.9,.999],weight_decay=0.,gradient_norm=1.,
        sampling='one full captured waveform per update; all four families once at seed 1103, then once at seed 3301; no selected or rejected takes',
        gradient_scope=SCOPE,
        objective='negative frozen 20-second CE plus 10 times mean per-chunk normalized latent reconstruction error',latent_fidelity_weight=10.,
        hard_stops=dict(max_chunk_relative_latent_l2=.1,ce_range=[0.,10.],gradient_forward_wave_max_error=1e-6),
        checkpoint_steps=[8],collection_recipe_sha256=sha(folder/'collection-recipe.json'),captures_sha256=sha(folder/'captures.json'),
        previous_tail_gradient_audit_sha256=sha(audit/'gradient-result.json'),full_flow_gradient_audit_sha256=sha(full_audit_folder/'gradient-result.json'),cache_audit_sha256=sha(cache_audit_folder/'result.json'),reward_spec=game['reward_spec'],physical_gpu=1,
        acoustic_benchmark_sha256=digest(acoustic_game),host_energy_by_kind=acoustic_game['host_energy_by_kind'],
        source_hashes={str(p.resolve()):sha(p) for p in sources},full_generation_gradient=False)
    immutable(folder/'recipe.json',recipe)
    with locked(folder/'train.lock'):
        torch.set_num_threads(4);torch.cuda.set_device(0);began=time.monotonic()
        pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        for name in ('transformer','vocoder'):
            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');tf=pipe.transformer;vocoder=pipe.vocoder
        tf.eval().requires_grad_(False);vocoder.eval().requires_grad_(False);tf.enable_gradient_checkpointing()
        torch.manual_seed(9131)
        network=LoRANetwork(tf,rank=8,alpha=8.,multiplier=1.,target_replace=['MiniMaxMusic3Attention'],prefix='lora_unet',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
        if len(network.unet_loras)!=144:raise IntegrityError('Unexpected acoustic attention topology')
        wrapper=MergedForward(network);optimizer=torch.optim.AdamW(network.parameters(),lr=lr,betas=(.9,.999),weight_decay=0.)
        scorer=CEReward(RewardSpec(**game['reward_spec']));scorer.load();load_seconds=time.monotonic()-began
        # Explicit family order comes from the frozen collection, not CE labels.
        family_order=list(dict.fromkeys(c['family']['family'] for c in cases))
        ordered=[next(c for c in cases if c['family']['family']==family and c['seed']==seed) for seed in (1103,3301) for family in family_order]
        completed=0;history=[]
        def save():
            save_tensor(folder/'state.pt',dict(recipe_sha256=digest(recipe),network=cpu(network.state_dict()),optimizer=cpu(optimizer.state_dict()),completed=completed,
                history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        def export(step):
            path=folder/f'reward-ce-full-flow_step{step}.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()},str(path))
            write(path.with_suffix('.json'),dict(STRUCTURE,weights_sha256=sha(path),prompts_file=str(folder/'training-prompts.json'),steps=step,
                reward=dict(name='reward-ce-full-flow',method=recipe['method'],recipe_sha256=digest(recipe),interpretation=recipe['gradient_scope'],full_generation_gradient=False)))
            checkpoint(path,1.);return path
        try:
            wrapper.attach()
            if (folder/'state.pt').exists():
                state=torch.load(folder/'state.pt',map_location='cpu',weights_only=True)
                if state['recipe_sha256']!=digest(recipe):raise IntegrityError('Acoustic optimizer resume recipe changed')
                network.load_state_dict(state['network'],strict=True);optimizer.load_state_dict(state['optimizer']);completed=state['completed'];history=state['history']
                torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
            stop=False
            def cancel(*args):
                nonlocal stop
                stop=True
            signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
            while completed<total and not stop:
                began=time.monotonic();case=ordered[completed%8];row=by_id[case['id']]
                capture=torch.load(row['capture'],map_location='cpu',weights_only=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    latents,_=replay_latents(tf,capture);wave=decode(vocoder,latents,capture['latent_hop_length'])
                ce,wave_gradient=value_and_gradient(scorer,wave,capture['sampling_rate']);current_wave=wave.detach().cpu();del wave,latents
                if not 0<=ce<=10:raise FloatingPointError('Acoustic training CE outside the frozen valid range')
                offset=0;penalty_value=0.;worst_latent=0.;wave_error=0.
                for index in range(len(capture['chunks'])):
                    latents,_=replay_latents(tf,capture,chunk_indices=(index,));latent=latents[0]
                    reference=capture['chunks'][index]['latent'].to(latent)
                    fidelity=(latent.float()-reference.float()).square().mean()/reference.float().square().mean().clamp_min(1e-8)
                    relative=float(fidelity.detach().sqrt());worst_latent=max(worst_latent,relative)
                    if relative>.1:raise FloatingPointError('Predeclared acoustic latent drift stop')
                    piece=decode_piece(vocoder,latent,index,len(capture['chunks']),capture['latent_hop_length']);length=piece.shape[-1]
                    difference=float((piece.detach().cpu()-current_wave[...,offset:offset+length]).abs().max());wave_error=max(wave_error,difference)
                    if difference>1e-6:raise IntegrityError('Acoustic gradient-mode forward differs from ordinary values')
                    penalty=10.*fidelity/len(capture['chunks'])
                    # This dot product supplies the exact local CE derivative;
                    # its scalar value is not the CE training objective.
                    local=-(piece*wave_gradient[...,offset:offset+length].to(piece)).sum()+penalty
                    if not torch.isfinite(local):raise FloatingPointError('Nonfinite acoustic objective')
                    local.backward();penalty_value+=float(penalty.detach());offset+=length
                    del latents,latent,piece,reference,fidelity,penalty,local
                if offset!=current_wave.shape[-1]:raise IntegrityError('Acoustic gradient chunks failed to tile the waveform')
                grad=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
                if not float(grad)>0:raise FloatingPointError('No acoustic adapter gradient')
                optimizer.step();completed+=1
                metric=dict(step=completed,family=case['family']['family'],seed=case['seed'],ce=ce,reference_ce=case['reference_ce'],objective=-ce+penalty_value,
                    latent_penalty=penalty_value,worst_chunk_relative_latent_l2=worst_latent,gradient_forward_wave_max_error=wave_error,gradient_norm=float(grad),seconds=time.monotonic()-began)
                history.append(metric)
                with (folder/'updates.jsonl').open('a') as file:file.write(__import__('json').dumps(metric)+'\n')
                write(folder/'status.json',dict(state='training',actual_updates=completed,total=total,load_seconds=load_seconds,updated_unix=time.time()))
                print('FULL FLOW UPDATE',metric,flush=True);save()
                if completed==8:export(completed);save_tensor(folder/f'state-step{completed}.pt',torch.load(folder/'state.pt',map_location='cpu',weights_only=True))
            save()
            if stop:raise SystemExit(130)
            path=export(completed)
            write(folder/'status.json',dict(state='checkpoint_ready',actual_updates=completed,checkpoint=str(path),load_seconds=load_seconds,optimizer_seconds=sum(r['seconds'] for r in history)))
        except BaseException as exc:
            save();write(folder/'error.json',dict(error=repr(exc),actual_updates=completed));raise
        finally:
            wrapper.detach();generator._merge_sliders(pipe,'cuda:0',[])
            exact=wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters()) and all(p.grad is None for p in vocoder.parameters())
            write(folder/f'off-restoration-{total}.json',dict(exact=exact,physical_gpu=1))
            if not exact:raise IntegrityError('Acoustic training changed frozen base/vocoder weights')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--game-home',required=True);p.add_argument('--full-audit-folder',required=True);p.add_argument('--cache-audit-folder',required=True);p.add_argument('--total',type=int,default=8);p.add_argument('--lr',type=float,default=.001);a=p.parse_args()
    train(a.home,a.folder,a.game_home,a.full_audit_folder,a.cache_audit_folder,a.total,a.lr)
