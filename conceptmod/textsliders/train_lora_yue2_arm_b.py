#!/usr/bin/env python3
"""Locked Music Arm B on YuE2, fresh prompts/histories throughout."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders import yue2_arm_b as game
from conceptmod.textsliders.train_lora_yue2_fresh import RowSampler,fresh_history,write_json
from conceptmod.textsliders.yue2_uni import _cpu
from conceptmod.textsliders.yue2_backend import YuE2Backend,YuE2Slider,file_digest,DEFAULT_MODEL


def source_hashes():
    names=['train_lora_yue2_arm_b.py','yue2_arm_b.py','train_lora_yue2_fresh.py',
           'yue2_uni.py','yue2_backend.py','lora.py','lm_adv.py','slider_targets.py',
           'train_lm_slider_music3.py']
    paths=[ROOT/'conceptmod/textsliders'/n for n in names]
    paths += [ROOT/'analysis/slider2d'/n for n in ['adv.py','grad_regularizers.py']]
    return {str(p.relative_to(ROOT)):file_digest(p) for p in paths}


def train(args):
    rows,meta=game.load_prompts(args.prompts_file)
    batch=game.RECIPE['adv_batch']
    if len(rows)<batch or len(rows)%batch:
        raise ValueError('Arm B needs a multiple of four rows for distinct balanced batches')
    args.save_dir.mkdir(parents=True,exist_ok=True)
    with (args.save_dir/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train_locked(args,rows,meta)


def _train_locked(args,rows,meta):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    batch=game.RECIPE['adv_batch']
    run=args.save_dir; path=run/'state.pt'
    settings=dict(recipe=game.RECIPE,name=args.name,rows=rows,metadata=meta,
        seed=args.seed,seed_start=args.seed_start,frames=args.train_tokens,
        max_seq_len=args.max_seq_len,history_backend=args.history_backend,
        checkpointing=not args.no_checkpointing,rank=8,alpha=8.,dummy=args.dummy,
        model_id=args.model_id,sources=source_hashes())
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True) if path.exists() else None
    if saved and saved['signature']['settings']!=settings:
        raise ValueError('Resume recipe, prompts, model settings, or source differs')
    if not saved and (run/'manifest.json').exists():
        raise FileExistsError('Existing run has no recovery state; use a fresh save_dir')
    backend=YuE2Backend(args.model_id,device=args.device,dummy=args.dummy)
    signature=dict(settings=settings,model=backend.identity)
    if saved and saved['signature']!=signature: raise ValueError('Resume base weights differ')
    fixed=saved['prepared'] if saved else game.prepare(backend,rows,meta,args.train_tokens,args.max_seq_len)
    network=YuE2Slider(backend.model,rank=8,alpha=8.)
    critic,g,d=game.build_game(backend,network,fixed)
    device=next(backend.model.parameters()).device
    sampler=RowSampler(len(rows),args.seed)
    completed=0; history=[]
    if saved:
        network.load_state_dict(saved['network'],strict=True)
        critic.load_state_dict(saved['critic'],strict=True)
        g.load_state_dict(saved['g_optimizer']);d.load_state_dict(saved['d_optimizer'])
        sampler.load_state_dict(saved['sampler'])
        completed=saved['completed'];history=saved['history']
        torch.set_rng_state(saved['rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if [h['step'] for h in history]!=list(range(1,completed+1)):
        raise ValueError('Non-contiguous saved update history')
    seeds={a['seed'] for h in history for a in h['histories']}
    if len(seeds)!=completed*batch:raise ValueError('Repeated saved history seeds')
    metadata=dict(backend='yue2',recipe=game.RECIPE['name'],recipe_settings=signature,
        model_id=args.model_id,model_identity=backend.identity,dummy=args.dummy,
        rows=rows,prompt_metadata=meta,cot='off',recommended_range=[-1,1],
        validation_status='experimental',teacher_rms=float(critic.input_scale))
    write_json(run/'manifest.json',signature)
    write_json(run/'teacher-audit.json',dict(rows=[{k:r[k] for k in ('guard_applied','target_shift')} for r in fixed],
        teacher_rms=float(critic.input_scale),batch=batch,rank=8,alpha=8.,recipe=game.RECIPE))
    start=time.monotonic();start_step=completed;stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    old={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def status(phase,**extra):
        write_json(run/'status.json',dict(status=phase,completed=completed,total=args.steps,until=args.until,
            elapsed_sec=time.monotonic()-start,start_step=start_step,gpu=os.getenv('CUDA_VISIBLE_DEVICES'),**extra))
    def save():
        if any(not torch.isfinite(v).all() for v in [*network.state_dict().values(),*critic.state_dict().values()]):
            raise FloatingPointError('Non-finite checkpoint')
        blob=dict(signature=signature,prepared=fixed,completed=completed,history=history,
            network=_cpu(network.state_dict()),critic=_cpu(critic.state_dict()),
            g_optimizer=_cpu(g.state_dict()),d_optimizer=_cpu(d.state_dict()),
            sampler=sampler.state_dict(),rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        temporary=path.with_suffix('.pt.tmp');torch.save(blob,temporary);temporary.replace(path)
        record=dict(metadata,step=completed,fresh_histories=len(seeds),
            row_counts=dict(Counter(i for h in history for i in h['rows'])))
        # Save the export atomically; preserve all older milestone exports.
        export=run/f'{args.name}_last.safetensors';tmp=run/f'{args.name}_pending.safetensors'
        network.save(tmp,record);tmp.replace(export);tmp.with_suffix('.json').replace(export.with_suffix('.json'))
        if completed and (completed%args.save_every==0 or completed==args.until):
            pinned=run/f'state-step{completed}.pt'
            if not pinned.exists():
                os.link(path,pinned)
                os.link(export,run/f'{args.name}_step{completed}.safetensors')
                write_json(run/f'{args.name}_step{completed}.json',record)
        status('paused' if stop else 'complete' if completed>=args.steps else 'checkpoint_ready' if completed>=args.until else 'training',
               fresh_histories=len(seeds),row_counts=record['row_counts'])
    try:
        if completed>=args.until:
            status('complete' if completed>=args.steps else 'checkpoint_ready');return
        if not saved:save()
        with (run/f'updates-from-{completed}-{time.time_ns()}.jsonl').open('w') as log:
            while completed<args.until and not stop:
                step=completed+1;indices=[sampler.next() for _ in range(batch)]
                current=[];audits=[];begin=time.monotonic()
                assert len(set(indices))==batch
                for draw,index in enumerate(indices):
                    seed=args.seed_start+(step-1)*batch+draw
                    status('sampling',next_step=step,row=index,next_seed=seed,draw=draw)
                    row,tokens,audit=fresh_history(backend,network,rows[index],fixed[index],args.train_tokens,
                        seed,args.history_backend=='cuda_graph')
                    if seed in seeds:raise ValueError('Repeated sampling seed')
                    current.append(row);audits.append(audit)
                    hp=run/'histories'/f'step{step}-draw{draw}.pt';hp.parent.mkdir(exist_ok=True)
                    temp=hp.with_suffix('.pt.tmp')
                    torch.save(dict(row=index,audit=audit,tokens=tokens,end_teacher=row['end_teacher']),temp);temp.replace(hp)
                sampling=time.monotonic()-begin
                metrics=game.update(backend,network,critic,g,d,current,step=step,checkpointing=not args.no_checkpointing)
                completed=step;seeds.update(a['seed'] for a in audits)
                record=dict(metrics,step=step,rows=indices,histories=audits,
                    sampling_seconds=sampling,step_seconds=time.monotonic()-begin)
                history.append(record);log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                write_json(run/'progress.json',record);print(json.dumps(record),flush=True)
                if completed%args.save_every==0 and completed!=args.until:save()
            save()
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='metal-yue2-arm-b')
    p.add_argument('--prompts_file',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml')
    p.add_argument('--save_dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--until',type=int)
    p.add_argument('--save_every',type=int,default=100)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--seed_start',type=int,default=4000001)
    p.add_argument('--train_tokens',type=int,default=250)
    p.add_argument('--max_seq_len',type=int,default=1024)
    p.add_argument('--history_backend',choices=('eager','cuda_graph'),default='cuda_graph')
    p.add_argument('--model_id',default=DEFAULT_MODEL)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dummy',action='store_true')
    p.add_argument('--no_checkpointing',action='store_true')
    a=p.parse_args(argv);a.until=a.steps if a.until is None else a.until
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.name):p.error('Invalid checkpoint name')
    if not 1<=a.until<=a.steps or min(a.save_every,a.train_tokens,a.max_seq_len)<1:p.error('Invalid budget')
    if not 0<=a.seed<2**63 or not 0<=a.seed_start<=2**63-a.steps*game.RECIPE['adv_batch']:p.error('Invalid seeds')
    return a

if __name__=='__main__':train(parse_args())
