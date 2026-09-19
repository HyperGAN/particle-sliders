#!/usr/bin/env python3
"""Pure unipolar YuE2 RpGAN on shuffled prompt states, with no audio sampling."""
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
from conceptmod.textsliders.train_lora_yue2_fresh import RowSampler,write_json
from conceptmod.textsliders.yue2_uni import _cpu
from conceptmod.textsliders.yue2_backend import YuE2Backend,YuE2Slider,file_digest,DEFAULT_MODEL


def source_hashes():
    names=['train_lora_yue2_arm_b.py','yue2_arm_b.py','train_lora_yue2_fresh.py',
           'yue2_uni.py','yue2_backend.py','lora.py','lm_adv.py','slider_targets.py',
           'train_lm_slider_music3.py','unipolar_gan.py','yue2_gan_plus_neu.py',
           'particle_bridge_gan.py','yue2_particle_bridge.py','distribution_probe.py']
    paths=[ROOT/'conceptmod/textsliders'/n for n in names]
    paths += [ROOT/'analysis/slider2d'/n for n in ['adv.py','grad_regularizers.py']]
    return {str(p.relative_to(ROOT)):file_digest(p) for p in paths}


def run_recipe(args, selected):
    """Pin an explicit LR-only experiment without mutating either recipe."""
    recipe = dict(selected.RECIPE)
    particle = args.recipe == 'particle_bridge'
    for key in ('end_weight', 'pole_weight', 'cover_weight', 'fm_weight',
                'lyrichold_weight', 'plan_weight', 'anchor_weight'):
        if recipe[key] != 0:
            raise ValueError(f'GAN-only training requires {key}=0')
    if recipe['adv_weight'] != 1:
        raise ValueError('Adversarial weight must be one')
    if particle:
        if recipe['parts'] != 128 or recipe['vicreg_weight'] != 1 or recipe['particle_dim'] != 4:
            raise ValueError('Particle bridge requires exactly 128x4 particles and particle VIC weight 1')
        if args.propose_only_c9_g4x or args.propose_only_lr_scale is not None:
            raise ValueError('Particle bridge pins the reference constant learning rates')
        recipe.update(critic=args.critic, critic_config=critic_config(args),
                      adv_batch=int(args.adv_batch), noise_decay_steps=int(args.steps))
        from conceptmod.textsliders import particle_bridge_gan as particle_game
        particle_game.REFERENCE['batch_size'] = int(args.adv_batch)
    elif recipe['parts'] != 0 or recipe['vicreg_weight'] != 0:
        raise ValueError('GAN-only training requires no particles and vicreg_weight=0')
    if args.propose_only_c9_g4x:
        if args.recipe != 'unipolar_gan':
            raise ValueError('c9_g4x applies only to unipolar_gan')
        recipe.update(g_lr=.002, d_lr=.003, ablation='c9_g4x',
                      propose_only=True, merge_to_trainer=False)
    if args.propose_only_lr_scale is not None:
        scale = args.propose_only_lr_scale
        if not 0 < scale <= 1 or args.propose_only_c9_g4x:
            raise ValueError('Native LR scale must be in (0, 1] and cannot combine with c9_g4x')
        recipe.update(g_lr=recipe['g_lr'] * scale, d_lr=recipe['d_lr'] * scale,
                      ablation='native_lr_scale', lr_scale=scale,
                      propose_only=True, merge_to_trainer=False)
    return recipe


def critic_config(args):
    """Explicit, JSON-stable critic architecture for provenance and resume."""
    values = {
        'patch': args.critic_patch,
        'width': args.critic_width,
        'layers': args.critic_layers,
        'heads': args.critic_heads,
        'tokens': args.critic_tokens,
        'queries': args.critic_queries,
        'rank': args.critic_rank,
        'score_bound': args.critic_score_bound,
    }
    return {key: value for key, value in values.items() if value is not None}


def apply_run_lrs(g, d, recipe):
    for optimizer, key in ((g, 'g_lr'), (d, 'd_lr')):
        for group in optimizer.param_groups:
            group['lr'] = recipe['particle_lr'] if group.get('role') == 'particles' else recipe[key]
            # The +/0 schedule reads initial_lr on every update. An override
            # must set its base too, or step one silently undoes the override.
            if 'initial_lr' in group:
                group['initial_lr'] = recipe[key]


def train(args):
    selected = game
    if args.recipe == 'gan_plus_neu':
        from conceptmod.textsliders import yue2_gan_plus_neu as selected
    elif args.recipe == 'particle_bridge':
        from conceptmod.textsliders import yue2_particle_bridge as selected
    run_recipe(args, selected)  # Reject forbidden losses before loading weights.
    rows,meta=selected.load_prompts(args.prompts_file)
    batch=selected.RECIPE['adv_batch']
    if args.recipe == 'particle_bridge':
        if len(rows) < 2: raise ValueError('Particle bridge needs at least two prompt rows')
    elif len(rows)<batch or len(rows)%batch:
        raise ValueError('Arm B needs a multiple of four rows for distinct balanced batches')
    args.save_dir.mkdir(parents=True,exist_ok=True)
    with (args.save_dir/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train_locked(args,rows,meta,selected)


def _train_locked(args,rows,meta,game):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    run=args.save_dir; path=run/'state.pt'
    recipe=run_recipe(args,game)
    batch=int(recipe['adv_batch'])
    print(json.dumps(dict(recipe=recipe,seed=args.seed,steps=args.steps,device=args.device,adv_batch=batch)),flush=True)
    particle = args.recipe == 'particle_bridge'
    settings=dict(recipe=recipe,name=args.name,rows=rows,metadata=meta,
        seed=args.seed,max_seq_len=args.max_seq_len,
        checkpointing=not args.no_checkpointing,rank=8,alpha=8.,dummy=args.dummy,
        model_id=args.model_id,sources=source_hashes())
    if args.recipe == 'gan_plus_neu':
        settings['schedule_horizon'] = args.steps
    if particle:
        settings['sample_seeds'] = int(args.sample_seeds)
        settings['history_tokens'] = int(args.history_tokens)
        settings['adv_batch'] = int(args.adv_batch)
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True) if path.exists() else None
    if saved and saved['signature']['settings']!=settings:
        raise ValueError('Resume recipe, prompts, model settings, or source differs')
    if not saved and (run/'manifest.json').exists():
        raise FileExistsError('Existing run has no recovery state; use a fresh save_dir')
    backend=YuE2Backend(args.model_id,device=args.device,dummy=args.dummy)
    signature=dict(settings=settings,model=backend.identity)
    if saved and saved['signature']!=signature: raise ValueError('Resume base weights differ')
    prepared_path = run/'prepared.pt'
    prepared_key = dict(rows=rows, metadata=meta, seed=args.seed, max_seq_len=args.max_seq_len,
                        sample_seeds=int(getattr(args,'sample_seeds',1)),
                        history_tokens=int(getattr(args,'history_tokens',0)),
                        model_id=args.model_id, dummy=args.dummy, recipe=args.recipe)
    if particle:
        if saved:
            fixed = saved['prepared']
        elif prepared_path.exists():
            cached = torch.load(prepared_path, map_location='cpu', weights_only=True)
            if cached.get('key') != prepared_key:
                raise ValueError('Cached prepared seed bank does not match this run')
            fixed = cached['prepared']
            print(json.dumps(dict(reused_prepared=len(fixed))), flush=True)
        else:
            fixed = game.prepare(
                backend, rows, meta, args.max_seq_len,
                sample_seeds=args.sample_seeds, history_tokens=args.history_tokens, seed=args.seed)
            temporary = prepared_path.with_suffix('.pt.tmp')
            torch.save(dict(key=prepared_key, prepared=fixed), temporary)
            temporary.replace(prepared_path)
    else:
        fixed=saved['prepared'] if saved else game.prepare(backend,rows,meta,args.max_seq_len)
    if particle:
        from conceptmod.textsliders import particle_bridge_gan as particle_game
    network=(game.ParticleSlider if particle else YuE2Slider)(backend.model,rank=8,alpha=8.)
    if particle: torch.manual_seed(args.seed + 1000)
    build = dict(critic=args.critic, critic_config=critic_config(args)) if particle else {}
    critic,g,d=game.build_game(backend,network,fixed,**build)
    apply_run_lrs(g,d,recipe)
    device=next(backend.model.parameters()).device
    source_count = len(fixed)
    sampler=particle_game.BridgeSampler(source_count,args.seed) if particle else RowSampler(len(rows),args.seed)
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
    if particle:
        if any(len(h[key]) != batch or any(not 0 <= i < source_count for i in h[key])
               for h in history for key in ('d_rows','g_rows')):
            raise ValueError('Invalid saved particle-bridge prompt batches')
        ema = {k:v.to(device) for k,v in saved['ema'].items()} if saved else particle_game.initialize_ema(network)
        critic.noise_decay_steps = int(recipe['noise_decay_steps'])
    elif any(len(h['rows'])!=batch or len(set(h['rows']))!=batch for h in history):
        raise ValueError('Invalid saved prompt batches')
    metadata=dict(backend='yue2',recipe=game.RECIPE['name'],recipe_settings=signature,
        model_id=args.model_id,model_identity=backend.identity,dummy=args.dummy,
        rows=rows,prompt_metadata=meta,cot='off',recommended_range=[0,1],
        polarity='unipolar',trained_scales=game.RECIPE['trained_scales'],zero_behavior='exact_base_by_adapter_scale',
        validation_status='experimental',teacher_rms=float(critic.input_scale))
    write_json(run/'manifest.json',signature)
    teacher_audit=dict(rows=[{k:r[k] for k in ('guard_applied','target_shift')} for r in fixed],
        teacher_rms=float(critic.input_scale),batch=batch,rank=8,alpha=8.,recipe=recipe)
    if particle:
        std=critic.target_std.detach().float().cpu()
        teacher_audit['normalization']=dict(
            training_rows=len(rows),dimensions=std.numel(),
            mode=getattr(critic,'normalization','unknown'),
            edit_rms=float(critic.edit_rms) if hasattr(critic,'edit_rms') else None,
            noise_start=float(getattr(critic,'noise_start',1.)),
            edit_noise_ratio=float(particle_game.REFERENCE['edit_noise_ratio']),
            edit_rms_target=float(particle_game.REFERENCE['edit_rms_target']),
            noise_hold_ratio=float(particle_game.REFERENCE['noise_hold_ratio']),
            noise_hold=float(critic.edit_rms)*float(particle_game.REFERENCE['noise_hold_ratio']),
            noise_decay_steps=int(critic.noise_decay_steps),
            noise_floor=float(particle_game.REFERENCE['noise_floor']),
            std_min=float(std.min()),std_median=float(std.median()),std_max=float(std.max()),
            floor_count=int((std<=particle_game.REFERENCE['target_std_floor']).sum()))
        print(json.dumps(dict(
            paired_edit_norm=teacher_audit['normalization']['mode'],
            edit_rms=teacher_audit['normalization']['edit_rms'],
            noise_start=teacher_audit['normalization']['noise_start'],
            edit_noise_ratio=teacher_audit['normalization']['edit_noise_ratio'],
            noise_decay_steps=teacher_audit['normalization']['noise_decay_steps'],
            noise_floor=teacher_audit['normalization']['noise_floor'],
        )), flush=True)
    write_json(run/'teacher-audit.json',teacher_audit)
    probe_mod=None
    if particle and not args.no_probe and int(args.sample_seeds)>1 and int(args.history_tokens)>0:
        from conceptmod.textsliders import distribution_probe as probe_mod
    start=time.monotonic();start_step=completed;stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    old={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def status(phase,**extra):
        write_json(run/'status.json',dict(status=phase,completed=completed,total=args.steps,until=args.until,
            elapsed_sec=time.monotonic()-start,start_step=start_step,gpu=os.getenv('CUDA_VISIBLE_DEVICES'),**extra))
    def run_probe():
        if probe_mod is None or not completed:
            return
        if completed%args.save_every!=0 and completed!=args.until:
            return
        if not probe_mod.needs_probe(run, completed):
            return
        status('probing',probe_step=completed)
        probe_mod.probe_checkpoint(
            run,backend,network,rows,fixed,step=completed,state=ema,
            max_seq_len=args.max_seq_len,history_tokens=args.history_tokens,
            probe_per_template=args.probe_seeds,listening_per_template=args.probe_listening_seeds,
            seed=args.seed,model_id=args.model_id,dummy=args.dummy,stride=args.save_every,
            critic_scale=critic.target_std.detach())
    def save():
        if any(not torch.isfinite(v).all() for v in [*network.state_dict().values(),*critic.state_dict().values()]):
            raise FloatingPointError('Non-finite checkpoint')
        blob=dict(signature=signature,prepared=fixed,completed=completed,history=history,
            network=_cpu(network.state_dict()),critic=_cpu(critic.state_dict()),
            g_optimizer=_cpu(g.state_dict()),d_optimizer=_cpu(d.state_dict()),
            sampler=sampler.state_dict(),rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        if particle:
            if any(not torch.isfinite(v).all() for v in ema.values()): raise FloatingPointError('Non-finite EMA')
            blob['ema'] = _cpu(ema)
        temporary=path.with_suffix('.pt.tmp');torch.save(blob,temporary);temporary.replace(path)
        counted = [i for h in history for i in (h['d_rows']+h['g_rows'] if particle else h['rows'])]
        record=dict(metadata,step=completed,prompt_draws=len(counted),
            row_counts=dict(Counter(counted)))
        # Save the export atomically; preserve all older milestone exports.
        export=run/f'{args.name}_last.safetensors';tmp=run/f'{args.name}_pending.safetensors'
        if particle:
            network.save(tmp,dict(record,weights_kind='ema'),state=ema)
        else: network.save(tmp,record)
        tmp.replace(export);tmp.with_suffix('.json').replace(export.with_suffix('.json'))
        if particle:
            live=run/f'{args.name}_live_last.safetensors'
            network.save(tmp,dict(record,weights_kind='live'))
            tmp.replace(live);tmp.with_suffix('.json').replace(live.with_suffix('.json'))
        if completed and (completed%args.save_every==0 or completed==args.until):
            pinned=run/f'state-step{completed}.pt'
            if not pinned.exists():
                os.link(path,pinned)
                os.link(export,run/f'{args.name}_step{completed}.safetensors')
                write_json(run/f'{args.name}_step{completed}.json',dict(record,weights_kind='ema') if particle else record)
                if particle:
                    os.link(live,run/f'{args.name}_live_step{completed}.safetensors')
                    write_json(run/f'{args.name}_live_step{completed}.json',dict(record,weights_kind='live'))
        run_probe()
        status('paused' if stop else 'complete' if completed>=args.steps else 'checkpoint_ready' if completed>=args.until else 'training',
               prompt_draws=record['prompt_draws'],row_counts=record['row_counts'])
    try:
        run_probe()
        if completed>=args.until:
            status('complete' if completed>=args.steps else 'checkpoint_ready');return
        if not saved:save()
        with (run/f'updates-from-{completed}-{time.time_ns()}.jsonl').open('w') as log:
            while completed<args.until and not stop:
                step=completed+1
                indices=list(range(source_count)) if particle else [sampler.next() for _ in range(batch)]
                begin=time.monotonic()
                assert particle or len(set(indices))==batch
                status('training',next_step=step,rows=indices,sources=source_count)
                if particle:
                    current=[dict(src, ids=src.get('train_ids', src['prefix'])) for src in fixed]
                else:
                    current=[dict(fixed[index],ids=fixed[index]['prefix']) for index in indices]
                extra={'total_steps':args.steps} if args.recipe=='gan_plus_neu' else {}
                if particle: extra['sampler'] = sampler
                metrics=game.update(backend,network,critic,g,d,current,step=step,
                    checkpointing=not args.no_checkpointing,**extra)
                if particle: particle_game.update_ema(ema,network)
                completed=step
                record=dict(metrics,step=step,rows=indices,step_seconds=time.monotonic()-begin,
                    g_lr=g.param_groups[0]['lr'],d_lr=d.param_groups[0]['lr'])
                if particle: record['particle_lr'] = g.param_groups[1]['lr']
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
    p.add_argument('--recipe',choices=['unipolar_gan','gan_plus_neu','particle_bridge'],default='unipolar_gan')
    p.add_argument('--propose_only_c9_g4x',action='store_true',
        help='Explicit LR-only trial: G 0.002 / D 0.003; production defaults unchanged')
    p.add_argument('--propose_only_lr_scale',type=float,
        help='Explicit native transfer: scale both learning rates, retaining the recipe ratio and schedule')
    p.add_argument('--name',default='metal-yue2-arm-b')
    p.add_argument('--prompts_file',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml')
    p.add_argument('--save_dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--until',type=int)
    p.add_argument('--save_every',type=int,default=100)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--max_seq_len',type=int,default=1024)
    p.add_argument('--model_id',default=DEFAULT_MODEL)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dummy',action='store_true')
    p.add_argument('--no_checkpointing',action='store_true')
    p.add_argument('--critic',choices=['mlp','sn_mlp','patch','mix','gmix','bottleneck','lowrank','hybrid','query','bquery'],
        default='mlp',help='Particle-bridge discriminator architecture')
    p.add_argument('--critic_patch',type=int)
    p.add_argument('--critic_width',type=int)
    p.add_argument('--critic_layers',type=int)
    p.add_argument('--critic_heads',type=int)
    p.add_argument('--critic_tokens',type=int)
    p.add_argument('--critic_queries',type=int)
    p.add_argument('--critic_rank',type=int)
    p.add_argument('--critic_score_bound',type=float)
    p.add_argument('--sample_seeds',type=int,default=256,
        help='Particle bridge: distinct continuation seeds per prompt template')
    p.add_argument('--history_tokens',type=int,default=32,
        help='Particle bridge: generated history tokens per sample seed (0 with --sample_seeds 1 = legacy)')
    p.add_argument('--adv_batch',type=int,default=64,
        help='Particle bridge: D/G draw size per update (smaller keeps unique forwards cheap)')
    p.add_argument('--no_probe',action='store_true',
        help='Particle bridge: skip the held-out distribution probe after EMA milestones')
    p.add_argument('--probe_seeds',type=int,default=32,
        help='Held-out continuation seeds per prompt template; never added to training')
    p.add_argument('--probe_listening_seeds',type=int,default=4,
        help='Seeds per template reserved for final listening and excluded from the probe')
    a=p.parse_args(argv);a.until=a.steps if a.until is None else a.until
    if a.recipe=='particle_bridge' and (a.propose_only_c9_g4x or a.propose_only_lr_scale is not None):
        p.error('Particle bridge pins the reference constant learning rates')
    critic_values=(a.critic_patch,a.critic_width,a.critic_layers,a.critic_heads,
        a.critic_tokens,a.critic_queries,a.critic_rank)
    if a.recipe!='particle_bridge' and (a.critic!='mlp' or any(v is not None for v in critic_values)
            or a.critic_score_bound is not None):
        p.error('Critic architecture flags require --recipe particle_bridge')
    if a.recipe!='particle_bridge' and (a.sample_seeds!=256 or a.history_tokens!=32 or a.adv_batch!=64):
        p.error('sample_seeds/history_tokens/adv_batch require --recipe particle_bridge')
    if a.recipe!='particle_bridge' and (a.no_probe or a.probe_seeds!=32 or a.probe_listening_seeds!=4):
        p.error('Probe flags require --recipe particle_bridge')
    if a.probe_seeds<1 or a.probe_listening_seeds<1: p.error('Probe seed counts must be positive')
    if any(v is not None and v < 1 for v in critic_values):p.error('Critic sizes must be positive')
    if a.critic_score_bound is not None and a.critic_score_bound < 0:p.error('critic_score_bound must be nonnegative')
    if a.sample_seeds < 1: p.error('sample_seeds must be >= 1')
    if a.sample_seeds > 1 and a.history_tokens < 1:
        p.error('history_tokens must be >= 1 when sample_seeds > 1')
    if a.history_tokens < 0: p.error('history_tokens must be >= 0')
    if a.adv_batch < 2: p.error('adv_batch must be >= 2')
    if a.propose_only_c9_g4x and a.recipe!='unipolar_gan':p.error('c9_g4x requires --recipe unipolar_gan')
    if a.propose_only_lr_scale is not None and (not 0<a.propose_only_lr_scale<=1 or a.propose_only_c9_g4x):
        p.error('Native LR scale must be in (0, 1] and cannot combine with c9_g4x')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.name):p.error('Invalid checkpoint name')
    if not 1<=a.until<=a.steps or min(a.save_every,a.max_seq_len)<1:p.error('Invalid budget')
    if not 0<=a.seed<2**63:p.error('Invalid seed')
    return a

if __name__=='__main__':train(parse_args())
