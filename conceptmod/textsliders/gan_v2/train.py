"""Train the versioned recipe, with explicit migration and resumable schedules.

Run as python -m conceptmod.textsliders.gan_v2.train. GPU visibility is supplied
by the caller; the research launcher assigns physical GPU 1.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import signal
import shutil
import sys

import torch

from .. import train_lm_slider_music3 as legacy
from ..lora import LoRANetwork
from .critic import SpanCritic, pad_sequences
from .data import ROOT, prepare_rows, StudentForward, fixture_manifest, sha, validate_prompts
from .engine import GANEngine, Recipe
from .optimization import EffectiveEMA
from . import state


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def arm_settings(name,*,origin,horizon,diagnostics_every):
    recipe=Recipe(name=name,schedule_origin=origin,schedule_horizon=horizon,diagnostics_every=diagnostics_every)
    critic=dict(width=128,layers=2,heads=4,conditioned=False,ordered=False,normalized_features=False)
    if name in ('baseline','fm','fm_normalized','fm_capped','decay'):
        recipe=replace(recipe,guard_weight=0.,policy_weight=0.,paired_fraction=0.,
            schedule='delayed_cosine' if name=='decay' else 'constant',
            control_fm=name in ('fm','fm_capped'),control_effective=False,clip_mode='value')
        critic['normalized_features']=name in ('fm','fm_normalized')
    elif name=='safeguards':
        # Full-space and policy guards with the old critic topology.
        critic['normalized_features']=True
    elif name in ('representation','repaired'):
        critic.update(conditioned=True,ordered=True,normalized_features=True)
        if name=='representation':recipe=replace(recipe,schedule='constant')
    else:raise ValueError('Unknown experiment arm')
    return recipe,critic


def export(run,engine,ema,*,rank,alpha,metadata,prompts,run_signature):
    from safetensors.torch import save_file
    step=engine.completed
    base=dict(steps=step,rank=rank,alpha=alpha,kind='language_model',target_replace=['Qwen3Attention'],
        prefix='lora_te',delimiter='-',train_method='full',unit_scale=1.,
        plus_label=metadata.get('plus_label','On'),minus_label=metadata.get('minus_label','Off'),
        prompts_file=str(Path(prompts).resolve()),recommended_range=[0.,1.],
        gan_v2=dict(recipe=asdict(engine.recipe),signature_digest=state.digest(run_signature),
                    quality_status='research_candidate_requires_audio_validation'))
    path=run/f'{run.name}_step{step}.safetensors'
    tensors={k:v.detach().cpu().contiguous() for k,v in engine.network.state_dict().items()}
    save_file(tensors,str(path));write_json(path.with_suffix('.json'),base)
    outputs={'live':str(path)}
    if ema is not None:
        if ema.step!=step:ema.update(engine.network,step)
        averaged=run/f'{run.name}_ema_step{step}.safetensors'
        save_file(ema.inference_state(),str(averaged))
        sidecar=dict(base,rank=ema.rank,alpha=float(ema.rank),
                     effective_ema=dict(decay=ema.decay,rank=ema.rank,last_compression_relative_error=ema.last_error,
                         interpretation='EMA of effective delta weights, periodically observed and rank-compressed; validate audio separately.'))
        write_json(averaged.with_suffix('.json'),sidecar);outputs['ema']=str(averaged)
    write_json(run/'latest.json',dict(step=step,checkpoints=outputs))
    return outputs


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--arm',choices=['baseline','fm','fm_normalized','fm_capped','decay','safeguards','representation','repaired'],default='repaired')
    sources=p.add_mutually_exclusive_group()
    sources.add_argument('--source-state',type=Path)
    sources.add_argument('--resume',type=Path)
    p.add_argument('--model-dir',type=Path,default=legacy.DEFAULT_MODEL)
    p.add_argument('--schedule-origin',type=int,default=600)
    p.add_argument('--schedule-horizon',type=int,default=900)
    p.add_argument('--until',type=int,default=900,help='Resource stop; never moves the fixed annealing horizon')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--history-seeds',type=int,nargs='+',default=[7])
    p.add_argument('--frames',type=int,default=250)
    p.add_argument('--policy-stride',type=int,default=4)
    p.add_argument('--cache',type=Path,default=ROOT/'cache/endreg')
    p.add_argument('--save-every',type=int,default=30)
    p.add_argument('--diagnostics-every',type=int,default=25)
    p.add_argument('--ema-rank',type=int,default=32)
    p.add_argument('--ema-every',type=int,default=25)
    p.add_argument('--no-ema',action='store_true')
    p.add_argument('--recipe-overrides',type=Path,help='JSON overrides, included in the exact resume signature')
    return p.parse_args(argv)


def train(args):
    if not torch.cuda.is_available():raise RuntimeError('Use the assigned physical GPU via CUDA_VISIBLE_DEVICES=1')
    if min(args.rank,args.batch,args.save_every,args.ema_every)<1:raise ValueError('Invalid positive training setting')
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device('cuda:0')
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'state.pt').exists() and not args.resume:raise ValueError('Existing run requires explicit resume or a fresh directory')
    rows,metadata=legacy._load_rows(args.prompts);validate_prompts(rows)
    recipe,critic_config=arm_settings(args.arm,origin=args.schedule_origin,horizon=args.schedule_horizon,
                                     diagnostics_every=args.diagnostics_every)
    if args.recipe_overrides:
        overrides=json.loads(args.recipe_overrides.read_text())
        recipe=replace(recipe,**overrides)
    recipe.validate()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_dir/'tokenizer'),local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained(str(args.model_dir/'language_model'),
        torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache=False
    prepared=prepare_rows(lm,tokenizer,rows,model_dir=args.model_dir,cache_dir=args.cache,device=device,
        frames=args.frames,seeds=args.history_seeds,policy_stride=args.policy_stride)
    network=LoRANetwork(lm,multiplier=1.,rank=args.rank,alpha=args.rank,delimiter='-',
        target_replace=['Qwen3Attention'],prefix='lora_te',train_method='full').to(device)
    if not network.unet_loras:raise RuntimeError('No attention adapters attached')
    network.requires_grad_(True)
    critic=SpanCritic(prepared[0]['real'].shape[-1],**critic_config).to(device)
    calibration,mask=pad_sequences([row['real'].to(device) for row in prepared])
    critic.calibrate_input_scale(calibration,mask)
    del calibration,mask
    engine=GANEngine(network,critic,StudentForward(lm,network,device),prepared,recipe)
    sampler=state.RowSampler(len(prepared),args.batch,args.seed)
    ema=None if args.no_ema else EffectiveEMA(rank=args.ema_rank)
    source=None
    if args.source_state:
        source=state.initialize_from_legacy(args.source_state,engine,reuse_critic=not critic_config['conditioned'])
    elif args.resume:
        source=torch.load(args.resume,map_location='cpu',weights_only=True)['signature']['source']
    identity={'path':str(args.model_dir.resolve()),'files':{str(p.relative_to(args.model_dir)):
        {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted((args.model_dir/'language_model').glob('*.safetensors'))}}
    run_signature=state.signature(recipe,critic_config,fixture_manifest(prepared),rank=args.rank,alpha=float(args.rank),
        ema_config=None if ema is None else dict(rank=args.ema_rank,decay=ema.decay,every=args.ema_every),
        model_identity=identity,source=source)
    # Batch/sampler and frame geometry must also be immutable on resume.
    run_signature['data']={'batch':args.batch,'seed':args.seed,'frames':args.frames,'policy_stride':args.policy_stride,
        'history_seeds':args.history_seeds,'prompts_sha256':sha(args.prompts)}
    history=state.restore(args.resume,engine,sampler,ema,run_signature) if args.resume else []
    if args.until<=engine.completed:raise ValueError('Requested endpoint must advance the loaded state')
    if ema is not None and ema.step is None:ema.update(network,engine.completed)
    write_json(run/'manifest.json',run_signature)
    provenance=run/'provenance';provenance.mkdir(exist_ok=True)
    for path,expected in run_signature['sources'].items():
        original=Path(path)
        destination=provenance/original.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        if sha(destination)!=expected:raise RuntimeError('Source changed while capturing the run')
    write_json(run/'invocation.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        argv=sys.argv,source=source))
    # Verify scale zero in the SAME shape after attaching/restoring adapters.
    legacy._set_scale(network,0.)
    check=prepared[0];prompt=check['prompt_embeds'].to(device)
    frames=None if check['frame_embeds'] is None else check['frame_embeds'].to(device)
    with torch.no_grad():
        _,_,hidden=legacy._forward_teacher_forced(lm,prompt,frames)
        from .data import gather
        actual=gather(hidden[:,:prompt.shape[1]],check['span_mask'])
        if not torch.equal(actual.cpu(),check['neutral_span']):raise RuntimeError('Scale-zero geometry is not identical to its teacher')
    legacy._set_scale(network,1.)
    stop_requested=False
    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    old_handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    log=run/f'train-from-{engine.completed}.jsonl'
    if log.exists():raise ValueError('Existing attempt log; resume into a fresh run directory')
    try:
        with log.open('w') as handle:
            while engine.completed<args.until and not stop_requested:
                row=engine.update(sampler.next());history.append(row)
                if ema is not None and engine.completed%args.ema_every==0:ema.update(network,engine.completed)
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                print(f"Update {engine.completed}: loss={sum(row['losses'].values()):.5f} "
                      f"FM grad factor={row['fm_gradient']['factor']:.3f} "
                      f"effective step={row['effective_limit']['actual_norm']:.5f}",flush=True)
                if engine.completed%args.save_every==0:
                    export(run,engine,ema,rank=args.rank,alpha=float(args.rank),metadata=metadata,prompts=args.prompts,run_signature=run_signature)
                    state.save(run/'state.pt',engine,sampler,ema,run_signature,history)
                    state.save(run/f'step{engine.completed}_state.pt',engine,sampler,ema,run_signature,history)
            outputs=export(run,engine,ema,rank=args.rank,alpha=float(args.rank),metadata=metadata,prompts=args.prompts,run_signature=run_signature)
            state.save(run/'state.pt',engine,sampler,ema,run_signature,history)
            write_json(run/'status.json',dict(status='paused' if stop_requested else 'budget_complete',
                completed=engine.completed,outputs=outputs,convergence='not_inferred_from_training_budget'))
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    return outputs


if __name__=='__main__':
    train(parse_args())
