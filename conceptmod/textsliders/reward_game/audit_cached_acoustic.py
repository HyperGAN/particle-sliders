"""Bounded full-host audit for a separate first-order merged-weight cache.

This does not train a candidate or change an active recipe. It compares both
implementations on one existing complete capture and one frozen checkpoint.
"""
import argparse
import gc
import os
from pathlib import Path
import signal
import shutil
import time

from .core import read,write,immutable,sha,digest,locked,IntegrityError,Store
from .resources import gpu_lease
from .setup import verify
from .acoustic_tail import replay_latents,decode,decode_piece
from .ce_wave_gradient import value_and_gradient
from .merged_forward import MergedForward
from .cached_merged_forward import CachedMergedForward


def audit(home,folder,probe,kind='ff'):
    import soundfile as sf
    import torch
    from diffusers import ModularPipeline
    from safetensors.torch import load_file,save_file
    from ..reward_sliders.specs import MODEL,RewardSpec
    from ..reward_sliders.reward import CEReward
    from app import generator
    from app.lora_runtime import LoRANetwork
    if kind=='ff':
        from .ff_artifact import checkpoint,STRUCTURE
        from .ff_network import make,check_frozen_attention
        capture_name='acoustic-ff-v1'
    elif kind=='attention':
        from .acoustic_artifact import checkpoint,STRUCTURE
        capture_name='acoustic-tail-v1'
        def make(tf):
            return LoRANetwork(tf,rank=8,alpha=8.,multiplier=1.,target_replace=['MiniMaxMusic3Attention'],
                prefix='lora_unet',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
        def check_frozen_attention(network,before):return True
    else:raise IntegrityError('Unknown acoustic audit kind')
    home=Path(home).resolve();folder=Path(folder).resolve();probe=Path(probe).resolve()
    game,manifest=verify(home);candidate=checkpoint(probe,1.)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise IntegrityError('Cache audit owns physical GPU 1')
    capture_folder=home/'audit'/capture_name
    if not read(capture_folder/'gradient-result.json')['passed']:raise IntegrityError('Ordinary acoustic gradient audit must pass first')
    capture_file=capture_folder/'acoustic-tail.pt'
    if sha(capture_file)!=read(capture_folder/'capture-result.json')['capture_sha256']:
        raise IntegrityError('Original capture changed')
    source=read(read(capture_folder/'capture-protocol.json')['source_observation'])
    if sha(source['audio'])!=source['audio_sha256']:raise IntegrityError('Original Off recording changed')
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in (
        'cached_merged_forward.py','merged_forward.py','ff_network.py','ff_artifact.py','acoustic_artifact.py',
        'acoustic_tail.py','ce_wave_gradient.py','resources.py')]
    protocol=dict(version=1,kind=kind,probe=candidate,capture_sha256=sha(capture_file),
        original_audio_sha256=source['audio_sha256'],reward_spec=game['reward_spec'],physical_gpu=1,
        sources={str(p.resolve()):sha(p) for p in sources},
        scope='Full five-chunk captured waveform, final two ordinary flow steps, nonzero frozen acoustic probe; first-order CE derivative only',
        required=dict(zero_waveform_exact=True,nonzero_waveform_exact=True,all_factor_gradients_exact=True,
            nonzero_waveform_matches_native_merge=True,frozen_attention_base_vocoder=True,off_waveform_exact=True),
        budget=dict(new_audio_generated=0,optimizer_updates=0,legacy_full_wave_gradients=1,cached_full_wave_gradients=1))
    immutable(folder/'protocol.json',protocol)
    for path,expected in protocol['sources'].items():
        archived=folder/'sources'/expected/Path(path).name
        archived.parent.mkdir(parents=True,exist_ok=True)
        if not archived.exists():shutil.copy2(path,archived)
        if sha(archived)!=expected:raise IntegrityError('Archived cache audit source differs')
    if (folder/'result.json').exists():return read(folder/'result.json')
    if (folder/'attempt.json').exists():raise IntegrityError('Interrupted cache audit retained; explicit amendment required')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    with locked(folder/'audit.lock'),gpu_lease(home,1):
        immutable(folder/'attempt.json',dict(started_unix=time.time(),protocol_sha256=digest(protocol)))
        torch.set_num_threads(4);torch.cuda.set_device(0);began=time.monotonic()
        pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        for name in ('transformer','vocoder'):
            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');tf=pipe.transformer;vocoder=pipe.vocoder
        tf.eval().requires_grad_(False);vocoder.eval().requires_grad_(False);tf.enable_gradient_checkpointing()
        torch.manual_seed(9131);network=make(tf)
        capture=torch.load(capture_file,map_location='cpu',weights_only=True)
        raw,rate=sf.read(source['audio'],dtype='float32',always_2d=True)
        reference_wave=torch.from_numpy(raw.T.copy())[None]
        state=load_file(str(probe));zero_state={k:v.clone() for k,v in state.items()}
        for key in zero_state:
            if key.endswith('.lora_up.weight'):zero_state[key].zero_()
        expected_grads=None;expected_wave=None;wave_gradient=None;records=[];wrapper=None
        load_seconds=time.monotonic()-began
        try:
            for label,implementation in (('legacy',MergedForward),('cached',CachedMergedForward)):
                stage_began=time.monotonic();network.load_state_dict(zero_state,strict=True)
                wrapper=implementation(network).attach()
                with torch.no_grad():
                    latents,checks=replay_latents(tf,capture,record_checks=True)
                    zero_wave=decode(vocoder,latents,capture['latent_hop_length']).cpu()
                if not torch.equal(zero_wave,reference_wave) or not all(row['exact'] for row in checks):
                    raise IntegrityError(f'{label} zero waveform differs from original recording')
                del latents,zero_wave
                zero_seconds=time.monotonic()-stage_began
                network.load_state_dict(state,strict=True)
                forward_began=time.monotonic()
                with torch.no_grad():
                    latents,_=replay_latents(tf,capture)
                    wave=decode(vocoder,latents,capture['latent_hop_length'])
                nonzero_forward_seconds=time.monotonic()-forward_began
                ce_gradient_seconds=0.
                if label=='legacy':
                    expected_wave=wave.detach().cpu().clone()
                    ce_began=time.monotonic()
                    scorer=CEReward(RewardSpec(**game['reward_spec']))
                    ce,wave_gradient=value_and_gradient(scorer,wave,rate)
                    del scorer
                    ce_gradient_seconds=time.monotonic()-ce_began
                elif not torch.equal(wave.detach().cpu(),expected_wave):
                    raise IntegrityError('Cached nonzero waveform differs from legacy')
                del latents,wave
                network.zero_grad(set_to_none=True);offset=0;pieces=[];gradient_began=time.monotonic()
                for index in range(len(capture['chunks'])):
                    latents,_=replay_latents(tf,capture,chunk_indices=(index,))
                    piece=decode_piece(vocoder,latents[0],index,len(capture['chunks']),capture['latent_hop_length'])
                    length=piece.shape[-1]
                    if not torch.equal(piece.detach().cpu(),expected_wave[...,offset:offset+length]):
                        raise IntegrityError(f'{label} gradient-mode waveform differs')
                    piece.backward(-wave_gradient[...,offset:offset+length].to(piece))
                    pieces.append(dict(chunk=index,samples=length,gradient_wave_exact=True));offset+=length
                    del piece,latents
                if offset!=expected_wave.shape[-1]:raise IntegrityError('Gradient pieces failed to tile the waveform')
                factor_gradient_seconds=time.monotonic()-gradient_began
                grads={name:p.grad.detach().cpu().clone() for name,p in network.named_parameters() if p.grad is not None}
                expected_names={name for name,p in network.named_parameters() if p.requires_grad}
                if grads.keys()!=expected_names:raise IntegrityError('Missing or unexpected trainable factor gradients')
                if not grads or not all(torch.isfinite(g).all() for g in grads.values()) or not any(g.count_nonzero() for g in grads.values()):
                    raise IntegrityError('No finite nonzero factor gradient')
                save_file(grads,str(folder/f'{label}-gradients.safetensors'))
                if label=='legacy':expected_grads=grads
                else:
                    if grads.keys()!=expected_grads.keys():raise IntegrityError('Gradient support changed')
                    differences={name:dict(exact=torch.equal(value,expected_grads[name]),
                        max_absolute_error=float((value-expected_grads[name]).abs().max())) for name,value in grads.items()}
                    immutable(folder/'gradient-comparison.json',differences)
                    if not all(row['exact'] for row in differences.values()):raise IntegrityError('Cached factor gradients differ')
                frozen=check_frozen_attention(network,state) and wrapper.base_unchanged()
                if not frozen or any(p.grad is not None for p in tf.parameters()) or any(p.grad is not None for p in vocoder.parameters()):
                    raise IntegrityError('Frozen host or attention state changed')
                records.append(dict(implementation=label,seconds=time.monotonic()-stage_began,parts=pieces,
                    zero_forward_seconds=zero_seconds,nonzero_forward_seconds=nonzero_forward_seconds,
                    ce_gradient_seconds=ce_gradient_seconds,factor_gradient_seconds=factor_gradient_seconds,
                    zero_wave_exact=True,nonzero_wave_exact=True,gradient_tensors=len(grads),
                    gradient_file_sha256=sha(folder/f'{label}-gradients.safetensors'),frozen_attention_base_vocoder=True))
                write(folder/'progress.json',dict(state='auditing',completed=records))
                print('CACHE AUDIT',records[-1],flush=True)
                wrapper.detach();wrapper=None;network.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            component=dict(STRUCTURE,weights=str(probe),mtime=probe.stat().st_mtime,multiplier=1.)
            generator._merge_sliders(pipe,'cuda:0',[component])
            with torch.no_grad():
                latents,_=replay_latents(tf,capture);native_wave=decode(vocoder,latents,capture['latent_hop_length']).cpu()
            if not torch.equal(native_wave,expected_wave):raise IntegrityError('Cached nonzero wave differs from ordinary native merge')
            del latents,native_wave
            generator._merge_sliders(pipe,'cuda:0',[])
            with torch.no_grad():
                latents,_=replay_latents(tf,capture);off_wave=decode(vocoder,latents,capture['latent_hop_length']).cpu()
            if not torch.equal(off_wave,reference_wave):raise IntegrityError('Off waveform differs after cache audit')
            result=dict(passed=True,protocol_sha256=digest(protocol),records=records,ce=ce,load_seconds=load_seconds,
                seconds=time.monotonic()-began,all_factor_gradients_exact=True,native_nonzero_wave_exact=True,off_wave_exact=True,
                new_audio_generated=0,optimizer_updates=0,
                interpretation='Engineering equivalence at one frozen nonzero acoustic probe and one captured trajectory; no music-improvement result. Phase timings are observed once in legacy-then-cached order; CE is computed once and shared, so total times are not an equal-work speed comparison.')
            immutable(folder/'result.json',result);Store(home).event('cached_acoustic_gradient_audit',passed=True,kind=kind,optimizer_updates=0)
            return result
        finally:
            if wrapper is not None:wrapper.detach()
            generator._merge_sliders(pipe,'cuda:0',[])
            write(folder/'cleanup.json',dict(completed_unix=time.time(),native_merge_cleared=True,new_audio_generated=0,optimizer_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--probe',required=True);p.add_argument('--kind',choices=('ff','attention'),default='ff');a=p.parse_args()
    print(__import__('json').dumps(audit(a.home,a.folder,a.probe,a.kind),indent=2),flush=True)
