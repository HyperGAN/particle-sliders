"""Stable staged scorecards with subprocess-isolated physical GPU work."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import (Store, read, write, immutable, digest, sha, locked, checkpoint,
                   candidate_identity, generation_identity, scorecard, markdown, IntegrityError, ROOT)
from .setup import verify


def dispatch(home, jobs):
    from .resources import gpu_lease
    store=Store(home)
    def batch(gpu):
        selected=[j for j in jobs if j['case']['physical_gpu']==gpu]
        if not selected: return
        key=digest(selected);folder=store.home/'jobs';path=folder/(key+'.json');immutable(path,selected)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HOME=str(ROOT.parent/'.cache/huggingface'),
                 HF_HUB_OFFLINE='1',PYTHONPATH=str(ROOT.parent)+':'+str(ROOT),
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with gpu_lease(home,gpu):
            with (folder/(key+f'-gpu{gpu}.log')).open('a') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','conceptmod.textsliders.reward_game.worker',
                    '--home',str(store.home),'--jobs',str(path),'--gpu',str(gpu)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:
                    code=process.wait()
                except BaseException:
                    process.terminate()
                    try:process.wait(timeout=120)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise
                if code: raise RuntimeError(f'GPU {gpu} worker exited {code}; original job and log retained: {path}')
    # No model or CUDA contexts are shared between threads; each starts a process.
    workers=int(os.environ.get('REWARD_GAME_MAX_GPU_WORKERS','1'))
    if workers not in (1,2):raise ValueError('Use one or two isolated GPU workers')
    store.event('render_concurrency',workers=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(batch,gpu) for gpu in sorted({j['case']['physical_gpu'] for j in jobs})]
        errors=[]
        for f in futures:
            try:f.result()
            except Exception as exc:errors.append(repr(exc))
        if errors: raise RuntimeError('; '.join(errors))


def engineering_audit(home):
    store=Store(home);game,manifest=verify(home)
    case=next(c for c in game['cases'] if c['physical_gpu']==1)
    candidate=checkpoint(None,0.);identity=generation_identity(game,manifest,case,candidate)
    dispatch(home,[dict(case=case,candidate=candidate,key=digest(identity),engineering_audit=True)])
    return read(store.home/'audit/off-pcm-v1/result.json')


def evaluate(home, path, multiplier=1., stage='auto', tag='manual', runner=dispatch, max_new_clips=16):
    store=Store(home);game,manifest=verify(home);candidate=checkpoint(path,multiplier)
    if stage not in ('auto','4','8','16'): raise ValueError('Stage must be auto, 4, 8 or 16')
    signature=dict(benchmark_sha256=digest(game),candidate=candidate_identity(candidate),stage=stage,max_new_clips=max_new_clips)
    run_id='eval-'+digest(signature)[:20];folder=store.home/'evaluations'/run_id
    with locked(folder/'run.lock'):
        immutable(folder/'identity.json',signature)
        # Human tags are aliases and never alter the underlying scorecard identity.
        store.event('evaluation_requested',run_id=run_id,tag=tag,candidate=candidate,forced_stage=stage!='auto')
        if (folder/'scorecard.json').exists():
            saved=read(folder/'scorecard.json')
            for key in saved['render_keys']:
                if store.observation(key) is None:raise IntegrityError('Completed scorecard is missing a cached observation')
            store.event('evaluation_reused',run_id=run_id,new_clips=0,cache_hits=len(saved['render_keys']))
            return saved
        if not (folder/'candidate.json').exists(): immutable(folder/'candidate.json',candidate)
        initial_path=folder/'initial.json'
        if not initial_path.exists():
            write(initial_path,dict(started_unix=time.time(),existing_keys=[p.parent.name for p in (store.home/'renders').glob('*/observation.json')]))
        initial=read(initial_path);existing=set(initial['existing_keys'])
        observations={};keys={};worker_error=None;card=None
        for n in ([4,8,16] if stage=='auto' else [int(stage)]):
            jobs=[]
            for case in game['cases']:
                if case['id'] not in game['stages'][str(n)]:continue
                identity=generation_identity(game,manifest,case,candidate);key=digest(identity);keys[case['id']]=key
                row=store.observation(key)
                if row is None:jobs.append(dict(case=case,candidate=candidate,key=key))
                else:observations[case['id']]=row
            if jobs:
                new_so_far=sum(k not in existing and bool(observations.get(i,{}).get('audio_sha256')) for i,k in keys.items())
                if new_so_far+len(jobs)>max_new_clips:
                    store.event('attempt_render_budget_exhausted',run_id=run_id,stage=n,budget=max_new_clips)
                    if card is None:raise ValueError('Render budget is smaller than the first declared stage')
                    card=dict(card,decision='attempt_budget_exhausted',advance=False,next_action='register_next_attempt')
                    break
                if runner is dispatch:
                    audit=store.home/'audit/off-pcm-v1/result.json'
                    if not audit.exists() or not read(audit)['passed']:
                        raise IntegrityError('Run audit-off successfully before generating new candidate observations')
                store.status('evaluating',run_id=run_id,stage=n,research_complete=False)
                immutable(folder/f'jobs-stage{n}-{digest(jobs)[:12]}.json',jobs)
                try:runner(home,jobs)
                except Exception as exc:worker_error=repr(exc)
            for ident,key in keys.items():
                row=store.observation(key)
                if row is not None:observations[ident]=row
            card=scorecard(game,observations,n)
            engineering=[r for r in observations.values() if r['status'] in ('engineering_error','scoring_error','running','rendered','interrupted')]
            pending=[ident for ident in game['stages'][str(n)] if ident not in observations]
            card.update(run_id=run_id,candidate=candidate_identity(candidate),benchmark_sha256=digest(game),
                        forced_stage=stage!='auto',early_rejected=stage=='auto' and not card['advance'] and n<16,
                        render_keys=list(keys.values()),new_clips=sum(k not in existing and bool(observations.get(i,{}).get('audio_sha256')) for i,k in keys.items()),
                        cache_hits=sum(k in existing for k in keys.values()),
                        elapsed_seconds=time.time()-initial['started_unix'],
                        render_seconds=sum(r.get('timing',{}).get('total_seconds',0) for i,r in observations.items() if keys[i] not in existing))
            if engineering or pending or worker_error:
                card.update(decision='engineering_failure' if engineering or worker_error else 'incomplete',advance=False,
                            early_rejected=False,worker_error=worker_error)
                write(folder/'partial-scorecard.json',card)
                (folder/'scorecard.md').write_text(markdown(card))
                store.event('evaluation_incomplete',run_id=run_id,decision=card['decision'],worker_error=worker_error)
                store.status(card['decision'],run_id=run_id,research_complete=False)
                return card
            if not (folder/f'stage{n}.json').exists(): immutable(folder/f'stage{n}.json',card)
            if not card['advance']:break
        immutable(folder/'scorecard.json',card)
        (folder/'scorecard.md').write_text(markdown(card))
        store.event('evaluation_finished',run_id=run_id,decision=card['decision'],advance=card['advance'],
                    new_clips=card['new_clips'],render_seconds=card['render_seconds'])
        store.status('candidate_decided',run_id=run_id,decision=card['decision'],research_complete=False)
        leaderboard(home)
        return card


def leaderboard(home):
    store=Store(home);groups={str(n):[] for n in (4,8,16)};partial=[]
    for folder in (store.home/'evaluations').glob('*'):
        if (folder/'scorecard.json').exists():
            card=read(folder/'scorecard.json');groups[str(card['stage'])].append(card)
        elif (folder/'partial-scorecard.json').exists():partial.append(read(folder/'partial-scorecard.json'))
        elif (folder/'initial.json').exists():partial.append(inspect(home,folder.name))
    def rank(card):
        off=card['comparisons']['off'];original=card['comparisons']['v1-original']
        return (card['advance'],off['wins'],off['worst_delta'] if off['worst_delta'] is not None else -float('inf'),
                off['equal_family_gain'] if off['equal_family_gain'] is not None else -float('inf'),
                original['equal_family_gain'] if original['equal_family_gain'] is not None else -float('inf'))
    for rows in groups.values():rows.sort(key=rank,reverse=True)
    game=read(read(store.home/'game.json')['spec'])
    result=dict(game_id=game['game_id'],development_only=True,default='off',stages=groups,incomplete=partial,
                references=read(Path(read(store.home/'game.json')['spec']).parent/'reference-scorecards.json')['stages'])
    write(store.home/'leaderboard.json',result)
    lines=['Development leaderboard. Off remains eligible. Fresh confirmation is required.', '']
    for n,rows in groups.items():
        lines += [f'{n}-case completed stages:', '', '| Run | Wins vs Off | Worst | Mean gain | Decision |','|---|---:|---:|---:|---|']
        for card in rows:
            off=card['comparisons']['off']
            lines.append(f"| {card['run_id']} | {off['wins']}/{n} | {off['worst_delta']} | {off['equal_family_gain']} | {card['decision']} |")
        lines.append('')
    (store.home/'leaderboard.md').write_text('\n'.join(lines))
    from .listen import build
    result['listening_page']=build(home,result)
    write(store.home/'leaderboard.json',result)
    return result


def inspect(home,run_id):
    if Path(run_id).name!=run_id:raise ValueError('Expected a run ID')
    folder=Path(home)/'evaluations'/run_id
    if (folder/'scorecard.json').exists():card=read(folder/'scorecard.json')
    elif (folder/'partial-scorecard.json').exists():card=read(folder/'partial-scorecard.json')
    else:
        game,manifest=verify(home);candidate=read(folder/'candidate.json');identity=read(folder/'identity.json')
        scheduled=[int(p.name.split('stage')[1].split('-')[0].split('.')[0]) for p in folder.glob('jobs-stage*.json')]
        stage=max(scheduled,default=4 if identity['stage']=='auto' else int(identity['stage']))
        store=Store(home);rows={};keys=[]
        for case in game['cases']:
            if case['id'] not in game['stages'][str(stage)]:continue
            key=digest(generation_identity(game,manifest,case,candidate));keys.append(key)
            row=store.observation(key)
            if row is not None:rows[case['id']]=row
        card=scorecard(game,rows,stage)
        initial=read(folder/'initial.json');existing=set(initial['existing_keys'])
        card.update(run_id=run_id,decision='in_progress',advance=False,render_keys=keys,benchmark_sha256=digest(game),
                    new_clips=sum(r['key'] not in existing and bool(r.get('audio_sha256')) for r in rows.values()),
                    cache_hits=sum(key in existing for key in keys),elapsed_seconds=time.time()-initial['started_unix'],
                    forced_stage=identity['stage']!='auto',early_rejected=False)
    return dict(card,candidate_provenance=read(folder/'candidate.json'),
                identity=read(folder/'identity.json'),
                observation_paths=[str(Path(home)/'renders'/key/'observation.json') for key in card['render_keys']])


def compare(home,left,right):
    a,b=inspect(home,left),inspect(home,right)
    if a['benchmark_sha256']!=b['benchmark_sha256']:raise IntegrityError('Different game versions')
    by_id={r['case']:r for r in b['rows']}
    pairs=[dict(case=r['case'],family=r['family'],delta=r['ce']-by_id[r['case']]['ce']) for r in a['rows'] if r['case'] in by_id]
    families={}
    for r in pairs:families.setdefault(r['family'],[]).append(r['delta'])
    return dict(left=left,right=right,matched_cases=len(pairs),left_stage=a['stage'],right_stage=b['stage'],
                wins=sum(p['delta']>0 for p in pairs),equal_family_gain=sum(sum(v)/len(v) for v in families.values())/len(families) if families else None,
                pairs=pairs,interpretation='Only shared valid development cases; stages remain separate')
