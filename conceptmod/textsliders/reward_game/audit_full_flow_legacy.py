"""Declared legacy-forward complete within-chunk flow gradient audit; one discarded update."""
import argparse
from pathlib import Path
import os
import sys
import time
import signal
import shutil
from .core import ROOT,read,write,immutable,sha,digest,locked,IntegrityError,Store
from .resources import gpu_lease
from .setup import verify
from .acoustic_tail import decode,decode_piece
from .full_flow import replay as replay_latents,SCOPE
from .ce_wave_gradient import value_and_gradient
from .merged_forward import MergedForward


def audit(home,folder,collection_folder,previous_audit_folder):
    import torch
    import numpy as np
    import soundfile as sf
    from diffusers import ModularPipeline
    from safetensors.torch import save_file,load_file
    from ..reward_sliders.specs import MODEL,RewardSpec
    from ..reward_sliders.reward import CEReward
    sys.path.insert(0,str(ROOT.parent))
    from app import generator
    from app.lora_runtime import LoRANetwork
    home=Path(home).resolve();folder=Path(folder).resolve();game,manifest=verify(home)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise IntegrityError('Acoustic gradient audit owns physical GPU 1')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    collection_folder=Path(collection_folder).resolve();previous_audit_folder=Path(previous_audit_folder).resolve()
    previous_result=read(previous_audit_folder/'gradient-result.json');previous_protocol=read(previous_audit_folder/'gradient-protocol.json')
    if not previous_result['passed'] or previous_protocol['target_replace']!=['MiniMaxMusic3Attention']:
        raise IntegrityError('Existing ordinary attention gradient audit must pass first')
    for p,h in previous_protocol['source_hashes'].items():
        if sha(p)!=h:raise IntegrityError('Audited ordinary-forward source changed')
    collection=read(collection_folder/'collection-recipe.json');rows=read(collection_folder/'captures.json')
    status=read(collection_folder/'collection-status.json')
    if status['state']!='complete' or status['verified_captures']!=8 or collection['retained_flow_steps']!=30:
        raise IntegrityError('All eight full-flow captures must pass before the gradient audit')
    if not read(collection_folder/'collection-off-restoration.json')['exact']:raise IntegrityError('Collection failed Off restoration')
    for p,h in collection['source_hashes'].items():
        if sha(p)!=h:raise IntegrityError('Collection source changed')
    case=collection['cases'][0];row=next(r for r in rows if r['id']==case['id'])
    if row['status']!='complete' or row['recipe_sha256']!=digest(collection) or row['source_observation_sha256']!=case['source_observation_sha256']:
        raise IntegrityError('Full-flow capture identity changed')
    capture_file=Path(row['capture'])
    if sha(capture_file)!=row['capture_sha256'] or sha(case['source_observation'])!=case['source_observation_sha256']:
        raise IntegrityError('Full-flow capture or original observation changed')
    source=read(case['source_observation'])
    if sha(source['audio'])!=source['audio_sha256']:raise IntegrityError('Original Off audio changed')
    source_paths=[Path(__file__),Path(__file__).with_name('acoustic_tail.py'),Path(__file__).with_name('ce_wave_gradient.py'),
                  Path(__file__).with_name('audit_ce_gradient_v3.py'),Path(__file__).with_name('merged_forward.py'),Path(__file__).with_name('full_flow.py')]
    protocol=dict(capture_sha256=sha(capture_file),previous_audit_result_sha256=sha(previous_audit_folder/'gradient-result.json'),collection_recipe_sha256=sha(collection_folder/'collection-recipe.json'),capture_record=row,source_hashes={str(p.resolve()):sha(p) for p in source_paths},
        numerical_forward='ordinary pristine FP32 delta merge with one BF16 cast',physical_gpu=1,rank=8,alpha=8,
        target_replace=['MiniMaxMusic3Attention'],prefix='lora_unet',delimiter='-',seed=9131,
        gradient=SCOPE,full_generation_gradient=False,
        required=dict(zero_forward_exact=True,gradient_forward_wave_max_error=1e-6,ce_error=1e-5,finite_nonzero_factor_gradient=True,
                      actual_one_update_changes_factors=True,nonzero_complete_waveform_matches_ordinary_merge_exactly=True,base_vocoder_frozen=True,off_exact=True),
        budget=dict(new_audio_generated=0,discarded_optimizer_updates=1),audit_learning_rate=1e-6)
    immutable(folder/'gradient-protocol.json',protocol)
    for path,expected in protocol['source_hashes'].items():
        archived=folder/'sources'/expected/Path(path).name
        archived.parent.mkdir(parents=True,exist_ok=True)
        if not archived.exists():shutil.copy2(path,archived)
        if sha(archived)!=expected:raise IntegrityError('Archived full-flow audit source differs')
    if (folder/'gradient-result.json').exists():return read(folder/'gradient-result.json')
    if (folder/'gradient-attempt.json').exists():raise IntegrityError('Interrupted acoustic gradient audit retained; explicit amendment required')
    with locked(folder/'gradient.lock'),gpu_lease(home,1):
        immutable(folder/'gradient-attempt.json',dict(started_unix=time.time(),discarded_update_budget=1))
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(9131);began=time.monotonic()
        pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        for name in ('transformer','vocoder'):
            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');tf=pipe.transformer;vocoder=pipe.vocoder
        tf.eval().requires_grad_(False);vocoder.eval().requires_grad_(False)
        torch.manual_seed(9131)
        network=LoRANetwork(tf,rank=8,alpha=8.,multiplier=1.,target_replace=['MiniMaxMusic3Attention'],prefix='lora_unet',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
        if len(network.unet_loras)!=144:raise IntegrityError('Unexpected acoustic attention topology')
        wrapper=MergedForward(network);capture=torch.load(capture_file,map_location='cpu',weights_only=True)
        raw,rate=sf.read(source['audio'],dtype='float32',always_2d=True)
        before={k:v.detach().cpu().clone() for k,v in network.state_dict().items()};updates=0
        try:
            wrapper.attach()
            with torch.no_grad():
                latents,checks=replay_latents(tf,capture,record_checks=True);wave=decode(vocoder,latents,capture['latent_hop_length'])
            exact=np.array_equal(wave[0].T.cpu().numpy(),raw) and all(c['exact'] for c in checks)
            if not exact:raise IntegrityError('Zero acoustic adapter changed reference waveform')
            scorer=CEReward(RewardSpec(**game['reward_spec']));ce,wave_gradient=value_and_gradient(scorer,wave,rate)
            if abs(ce-source['reward']['scalar'])>1e-5:raise IntegrityError('Differentiable CE differs from the frozen score')
            del scorer,latents,wave
            tf.enable_gradient_checkpointing()
            network.zero_grad(set_to_none=True);offset=0;wave_max_error=0.;parts=[]
            for index in range(len(capture['chunks'])):
                latents,_=replay_latents(tf,capture,chunk_indices=(index,));piece=decode_piece(vocoder,latents[0],index,len(capture['chunks']),capture['latent_hop_length'])
                length=piece.shape[-1];reference=torch.from_numpy(raw[offset:offset+length].T.copy())[None].to(piece)
                difference=float((piece.detach()-reference).abs().max());wave_max_error=max(wave_max_error,difference)
                if difference>1e-6:raise IntegrityError('Gradient-mode acoustic values differ from deployed zero forward')
                piece.backward(-wave_gradient[...,offset:offset+length].to(piece))
                parts.append(dict(chunk=index,samples=length,wave_max_error=difference));offset+=length
                del piece,latents,reference
            if offset!=len(raw):raise IntegrityError('Acoustic chunks did not tile the waveform')
            grads=[p.grad for p in network.parameters() if p.grad is not None]
            if len(grads)!=sum(p.requires_grad for p in network.parameters()):raise IntegrityError('Missing complete-flow factor gradient')
            norm=sum(float(g.abs().sum()) for g in grads)
            if not grads or norm==0 or not all(torch.isfinite(g).all() for g in grads):raise IntegrityError('No finite nonzero acoustic adapter gradient')
            gradnorm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
            optimizer=torch.optim.AdamW(network.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=0.);optimizer.step();updates=1
            write(folder/'discarded-update.json',dict(actual_updates=1,used_for_candidate=False,gradient_absolute_sum=norm,gradient_norm=float(gradnorm)))
            tensors={k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()};path=folder/'discarded-full-flow-probe.safetensors';save_file(tensors,str(path))
            changed=any(not torch.equal(tensors[k],before[k]) for k in tensors)
            with torch.no_grad():
                probe_latents,_=replay_latents(tf,capture)
                differentiable=decode(vocoder,probe_latents,capture['latent_hop_length']).detach().cpu()
            del probe_latents
            wrapper.detach()
            comp=dict(kind='transformer',rank=8,alpha=8,target_replace=['MiniMaxMusic3Attention'],prefix='lora_unet',delimiter='-',train_method='full',
                      unit_scale=1.,weights=str(path),mtime=path.stat().st_mtime,multiplier=1.)
            generator._merge_sliders(pipe,'cuda:0',[comp])
            with torch.no_grad():
                probe_latents,_=replay_latents(tf,capture)
                ordinary=decode(vocoder,probe_latents,capture['latent_hop_length']).detach().cpu()
            del probe_latents
            merged_exact=torch.equal(ordinary,differentiable)
            generator._merge_sliders(pipe,'cuda:0',[])
            with torch.no_grad():
                probe_latents,_=replay_latents(tf,capture)
                off_wave=decode(vocoder,probe_latents,capture['latent_hop_length'])
            off_wave_exact=np.array_equal(off_wave[0].T.cpu().numpy(),raw)
            del off_wave,probe_latents
            base_frozen=wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters()) and all(p.grad is None for p in vocoder.parameters())
            result=dict(passed=changed and merged_exact and base_frozen and off_wave_exact,zero_waveform_exact=exact,gradient_wave_max_error=wave_max_error,
                ce=ce,cached_ce=source['reward']['scalar'],factor_gradient_absolute_sum=norm,gradient_norm=float(gradnorm),changed_factors=changed,
                nonzero_complete_waveform_matches_ordinary_merge=merged_exact,off_waveform_exact=off_wave_exact,base_and_vocoder_frozen=base_frozen,adapter_modules=len(network.unet_loras),tensor_count=len(tensors),
                parts=parts,seconds=time.monotonic()-began,discarded_optimizer_updates=updates,new_audio_generated=0,
                interpretation='Engineering feasibility for all thirty within-chunk flow steps with fixed conditioning/reference overlap and BF16 cast autograd; no music-improvement evidence')
            immutable(folder/'gradient-result.json',result);Store(home).event('full_flow_gradient_audit',passed=result['passed'],discarded_optimizer_updates=updates)
            if not result['passed']:raise IntegrityError('Acoustic gradient/merge audit failed')
            return result
        finally:
            wrapper.detach();generator._merge_sliders(pipe,'cuda:0',[])
            exact=wrapper.base_unchanged();write(folder/'gradient-off-restoration.json',dict(exact=exact,physical_gpu=1,discarded_optimizer_updates=updates))
            if not exact:raise IntegrityError('Acoustic gradient audit changed base weights')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--collection-folder',required=True);p.add_argument('--previous-audit-folder',required=True);a=p.parse_args()
    print(__import__('json').dumps(audit(a.home,a.folder,a.collection_folder,a.previous_audit_folder),indent=2),flush=True)
