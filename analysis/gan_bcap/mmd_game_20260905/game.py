"""Persistent MMD research game: fixed judge, comparable scores, saved rounds.

The coding agent chooses and implements each experiment. This tool preserves
the rules, evaluates checkpoints, records scores and carries the best result
between sessions. It never launches training or changes the deployed catalog.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from analysis.gan_bcap.mmd_game_20260905.family_bookkeeping import (
    ALL_FAMILIES, RESEARCH_FAMILIES, round_result, validate_budget)


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def identifier(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}',value):raise ValueError('Use a short lowercase ID with digits, underscores or hyphens')
    return value


@contextmanager
def locked(home):
    home.mkdir(parents=True,exist_ok=True)
    with (home/'game.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def champions(state):
    eligible=[r for r in state['entries'].values() if r['eligible']]
    def best(rows):return max(rows,key=lambda r:r['score'])['id'] if rows else None
    return dict(overall=best(eligible),mmd=best([r for r in eligible if r['family']=='mmd']))


def publish(home,state):
    state['champions']=champions(state)
    state['updated_utc']=now()
    write(home/'state.json',state)
    pending=[r for r in read(home/'opponents.json')['opponents'] if r['sha256'] not in {e['weights_sha256'] for e in state['entries'].values()}]
    lines=['# MMD game leaderboard','',
        'Fixed render-heuristic-v2; two prompts × seeds 7/23, 20 seconds, GPU 1. '
        'These are repeated development fixtures. The score is a heuristic, not a human-quality certificate.','',
        f"Overall leader: **{state['champions']['overall'] or 'not measured yet'}**. "
        f"MMD personal best: **{state['champions']['mmd'] or 'not measured yet'}**.",
        f"Active round: {state['active_round'] or 'none'}. Completed rounds: {len(state['completed_rounds'])}. "
        f"Historical opponents still awaiting this board: {len(pending)}.",'',
        '| Rank | Entry | Family | Score ↑ | Worst clip | Eligible | Round |',
        '| ---: | --- | --- | ---: | ---: | --- | --- |']
    ordered=sorted(state['entries'].values(),key=lambda r:(not r['eligible'],-r['score'],r['id']))
    for i,r in enumerate(ordered,1):
        lines.append(f"| {i} | {r['id']} | {r['family']} | {r['score']:.6f} | {r['worst_score']:.6f} | {r['eligible']} | {r['round'] or 'bootstrap'} |")
    lines+=['','Historical scores from different fixtures are listed only in `opponents.json`; they are not leaderboard entries. '
        'Each entry retains its component scores, all four examples, control hashes and score reports. '
        'A higher mean may coexist with individual regressions; those remain visible.','',
        '## Next opponents to evaluate','']
    lines += [f"- `{r['id']}` — `{r['weights']}`" for r in pending[:8]] or ['All inventoried opponents have comparable measurements.']
    lines+=['','## Completed rounds','']
    for r in state['completed_rounds']:
        lines.append(f"- {r['id']}: {r['outcome']}; next experiment: {r['next_experiment']}")
    (home/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')


def init_game(home):
    from analysis.gan_bcap.autonomous_audio import RULE,CONCEPTS
    with locked(home):
        if (home/'protocol.json').exists():
            if not (home/'state.json').exists() or not (home/'opponents.json').exists():raise ValueError('Incomplete initialization; preserve the files and repair before continuing')
            print('Already initialized; existing rules and history preserved.');return
        prompts=ROOT/'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml'
        sources=[ROOT/'analysis/gan_bcap/autonomous_audio.py',ROOT/'analysis/gan_bcap/render_v2.py',
            ROOT/'slider_selection/features.py',ROOT/'scripts/lm_score.py']
        protocol=dict(version='mmd-game-1',created_utc=now(),concept='gender',
            goal='Improve MMD-trained checkpoints until they beat the strongest comparable entries; preserve each result for the next round.',
            prompts=str(prompts),prompts_sha256=sha(prompts),rows=[0,1],seeds=[7,23],duration=20.0,
            gpu=1,scales=[0.,1.],plus_label='Female',minus_label='Off',rule=RULE,concepts=CONCEPTS,
            scorer_sha256=sha(sources[0]),sources={str(p):sha(p) for p in sources},
            round_default=dict(candidates=1,additional_update_attempts=150,meaning='Starting budget, not a convergence or stopping claim.'),
            ranking='Highest mean eligible render-heuristic-v2 across the exact four fixtures. Retain every clip and component.',
            confirmation='New leaders should be checked on additional prompts/seeds and longer audio. Record that evidence separately; do not mix fixture sets in this board.',
            diversity='Paired deterministic hidden regression has no general distributional anti-collapse guarantee. Report diversity as unmeasured until actually tested.')
        opponents=[];seen=set()
        def add(name,path,historical=None,priority=20):
            path=Path(path).resolve()
            if str(path) in seen:return
            if not path.exists():raise FileNotFoundError(path)
            seen.add(str(path));opponents.append(dict(id=identifier(name),weights=str(path),sha256=sha(path),
                historical=historical,priority=priority))
        add('gan_original_600',ROOT/'models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_last.safetensors',priority=0)
        for report in sorted((ROOT/'analysis/gan_bcap/paired_continuation_20260905').glob('fast-results-*.json')):
            for r in read(report)['ranking']:
                name=('gan_fm_cap_' if '/fm_capped-' in r['checkpoint'] else 'gan_bounded_')+str(r['steps'])
                priority=1 if name=='gan_fm_cap_1950' else 2 if name=='gan_bounded_1200' else 20
                add(name,r['checkpoint'],dict(score=r['heuristic_score'],report=str(report),
                    scope='One prompt, two seeds, GPU 0; not comparable with this board.'),priority)
        for r in read(ROOT/'analysis/gan_bcap/v2_20260905/final-results.json')['candidates']:
            add('gan_v2_'+r['id'],r['path'],dict(score=r['score'],scope='Two prompts, four seeds; not this board.'),10)
        add('learned_energy_conditional_720',ROOT/'models/conditional-energy-research-20260905/conditional-energy-research-20260905_step720.safetensors',priority=12)
        add('learned_energy_cfg_720',ROOT/'models/conditional-energy-cfg-research-20260905-attempt2/conditional-energy-cfg-research-20260905-attempt2_step720.safetensors',priority=11)
        write(home/'protocol.json',protocol)
        write(home/'opponents.json',dict(scope='Known comparable concept checkpoints inventoried at game creation; add later discoveries without discarding history.',
            opponents=sorted(opponents,key=lambda r:(r['priority'],r['id']))))
        state=dict(version=1,protocol_sha256=sha(home/'protocol.json'),active_round=None,completed_rounds=[],entries={},controls={})
        publish(home,state)
        print(f'Initialized {home}; {len(opponents)} historical opponents, none falsely ranked on mismatched fixtures.')


def load_game(home):
    protocol=read(home/'protocol.json');state=read(home/'state.json')
    if sha(home/'protocol.json')!=state['protocol_sha256']:raise ValueError('Locked game protocol changed')
    if sha(protocol['prompts'])!=protocol['prompts_sha256']:raise ValueError('Evaluation prompts changed')
    for name,value in protocol['sources'].items():
        if sha(name)!=value:raise ValueError(f'Frozen evaluation source changed: {name}')
    return protocol,state


def validate_records(protocol,weights,reports,controls):
    from analysis.gan_bcap.autonomous_audio import score_components
    weights=Path(weights).resolve();weight_sha=sha(weights)
    expected={digest([protocol['prompts_sha256'],row,seed,protocol['duration']]) for row in protocol['rows'] for seed in protocol['seeds']}
    records=[]
    for report in reports:
        blob=read(report)
        if blob.get('status')!='complete' or blob.get('rule')!=protocol['rule'] or blob.get('source_sha256')!=protocol['scorer_sha256'] or blob.get('concept')!=protocol['concept'] or blob.get('concepts')!=protocol['concepts']:
            raise ValueError('Report does not use the complete, frozen scorer and concept definitions')
        records += [r for r in blob['records'] if Path(r['checkpoint']['path']).resolve()==weights]
    if len(records)!=len(expected) or {r['fixture'] for r in records}!=expected:
        raise ValueError('Require exactly one clip per locked prompt/seed fixture; missing, duplicate or different fixtures')
    new_controls=dict(controls)
    for r in records:
        if r['checkpoint']['sha256']!=weight_sha:raise ValueError('Checkpoint changed since evaluation')
        if r['seed'] not in protocol['seeds']:raise ValueError('Unexpected seed')
        for key in ('candidate','baseline','positive_reference'):
            if sha(r[key]['audio'])!=r[key]['sha256']:raise ValueError('Audio changed since scoring')
        folder=Path(r['candidate']['audio']).parent.parent
        provenance=read(folder/'game-provenance.json')
        if provenance['gpu']!=protocol['gpu'] or provenance['protocol_sha256']!=protocol['_sha256']:
            raise ValueError('Rendering does not belong to this GPU and game protocol')
        spec=read(folder/'render_spec.json')
        if (spec['prompts_sha256']!=protocol['prompts_sha256'] or spec['row'] not in protocol['rows'] or
            spec['seeds']!=protocol['seeds'] or spec['duration']!=protocol['duration'] or
            spec['scales']!=protocol['scales'] or spec['seed_retries']!=0):raise ValueError('Render specification differs from the board')
        if spec.get('renderer_sha256')!=protocol['sources'][str(ROOT/'analysis/gan_bcap/render_v2.py')]:
            raise ValueError('Render used a different renderer')
        if r['checkpoint'] not in spec.get('checkpoints',[]):raise ValueError('Checkpoint is not in the render specification')
        if digest([spec['prompts_sha256'],spec['row'],r['seed'],spec['duration']])!=r['fixture']:raise ValueError('Score fixture does not match rendered input')
        score,components=score_components(r['candidate'],r['baseline'],r['positive_reference'])
        if not math.isfinite(score) or not math.isclose(score,r['heuristic_score'],rel_tol=0,abs_tol=1e-9):raise ValueError('Score does not recompute under the fixed rule')
        if components!=r['components']:raise ValueError('Score components changed')
        eligible=r['candidate']['rms']>=.02*max(r['baseline']['rms'],1e-12)
        if r['eligible']!=eligible:raise ValueError('Eligibility changed')
        identity={k:r[k]['sha256'] for k in ('baseline','positive_reference')}
        if r['fixture'] in controls and identity!=controls[r['fixture']]['hashes']:
            raise ValueError('Candidate did not use the same control audio as the existing board')
        new_controls[r['fixture']]=dict(hashes=identity,paths={k:r[k]['audio'] for k in identity})
    return records,new_controls


def register(home,args):
    with locked(home):
        protocol,state=load_game(home)
        protocol['_sha256']=state['protocol_sha256']
        name=identifier(args.id)
        if name in state['entries']:raise ValueError('Entry ID already exists; history is immutable')
        if args.round and args.round!=state['active_round']:raise ValueError('Candidate round is not active')
        if args.family not in ALL_FAMILIES:raise ValueError('Unknown method family')
        if state['active_round'] and args.family in RESEARCH_FAMILIES and args.round!=state['active_round']:raise ValueError('Assign new research candidates to the active round')
        records,controls=validate_records(protocol,args.weights,args.scores,state['controls'])
        archive=home/'entries'/name;archive.mkdir(parents=True,exist_ok=True)
        intent=dict(id=name,family=args.family,round=args.round,label=args.label or name,
            weights=str(Path(args.weights).resolve()),weights_sha256=sha(args.weights),
            reports_sha256=[sha(source) for source in args.scores])
        intent_path=archive/'registration.json'
        if intent_path.exists() and read(intent_path)!=intent:
            raise ValueError('Interrupted registration has different inputs; preserve it and use a new ID')
        write(intent_path,intent)
        saved=[]
        for i,source in enumerate(args.scores):
            dest=archive/f'score-{i}.json'
            if Path(source).resolve()!=dest.resolve():shutil.copyfile(source,dest)
            saved.append(dict(path=str(dest),sha256=sha(dest)))
        entry=dict(id=name,label=args.label or name,family=args.family,round=args.round,
            weights=str(Path(args.weights).resolve()),weights_sha256=sha(args.weights),
            score=statistics.mean(r['heuristic_score'] for r in records),
            worst_score=min(r['heuristic_score'] for r in records),eligible=all(r['eligible'] for r in records),
            components={k:statistics.mean(r['components'][k] for r in records) for k in protocol['rule']['weights']},
            records=records,reports=saved,created_utc=now(),diversity='not measured by this four-clip score')
        if (archive/'entry.json').exists():entry['created_utc']=read(archive/'entry.json')['created_utc']
        write(archive/'entry.json',entry)
        state['entries'][name]=entry;state['controls']=controls
        publish(home,state)
        print(json.dumps(dict(entry=name,score=entry['score'],champions=state['champions']),indent=2))


def evaluate(home,args):
    with locked(home):protocol,state=load_game(home)
    if args.id in state['entries']:
        entry=state['entries'][args.id]
        if (entry['weights_sha256']!=sha(args.weights) or entry['family']!=args.family or
            entry['round']!=args.round):raise ValueError('Entry ID already belongs to different inputs')
        print('Already scored; existing entry retained.');return
    identifier(args.id)
    if args.round and args.round!=state['active_round']:raise ValueError('Round is not active')
    if args.family not in ALL_FAMILIES:raise ValueError('Unknown method family')
    if state['active_round'] and args.family in RESEARCH_FAMILIES and args.round!=state['active_round']:
        raise ValueError('Assign new research candidates to the active round')
    from analysis.gan_bcap.lm_evaluate import checkpoint_metadata
    metadata=checkpoint_metadata(Path(args.weights).resolve())[0]
    if metadata['plus_label']!=protocol['plus_label'] or metadata['minus_label']!=protocol['minus_label']:
        raise ValueError('This board is for the declared concept and polarity labels')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(protocol['gpu']),HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    reports=[]
    work=home/'evaluations'/args.id;work.mkdir(parents=True,exist_ok=True)
    for row in protocol['rows']:
        folder=ROOT/'eval/listen/mmd-game-20260905'/args.id/f'row-{row}'
        leaf=folder/f'{Path(args.weights).stem}-s'
        copies=[]
        for seed in protocol['seeds']:
            fixture=digest([protocol['prompts_sha256'],row,seed,protocol['duration']])
            if fixture not in state['controls']:continue
            dest=Path(str(leaf)+str(seed));dest.mkdir(parents=True,exist_ok=True)
            for key,source in state['controls'][fixture]['paths'].items():
                source=Path(source);target=dest/source.name
                if sha(source)!=state['controls'][fixture]['hashes'][key]:raise ValueError('Control template changed')
                if target.exists() and sha(target)!=sha(source):raise ValueError('Existing control differs')
                if not target.exists():shutil.copyfile(source,target)
                copies.append(dict(source=str(source),destination=str(target),sha256=sha(target)))
        folder.mkdir(parents=True,exist_ok=True)
        write(folder/'game-provenance.json',dict(protocol_sha256=state['protocol_sha256'],gpu=protocol['gpu'],control_reuse=copies))
        report=work/f'scores-row-{row}.json'
        commands=[
            [sys.executable,str(ROOT/'analysis/gan_bcap/render_v2.py'),'--weights',str(Path(args.weights).resolve()),
             '--out',str(folder),'--prompts',protocol['prompts'],'--row',str(row),
             '--seeds',*[str(s) for s in protocol['seeds']],'--duration',str(protocol['duration'])],
            [sys.executable,str(ROOT/'analysis/gan_bcap/autonomous_audio.py'),'--folders',str(folder),
             '--concept',protocol['concept'],'--output',str(report)]]
        for name,command in zip(('render','score'),commands):
            log=work/f'{name}-row-{row}.log'
            print(f'{name}, row {row}: {log}',flush=True)
            with log.open('a') as stream:subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        reports.append(report)
    args.scores=reports
    register(home,args)


def begin(home,args):
    with locked(home):
        protocol,state=load_game(home)
        if state['active_round']:raise ValueError(f"Resume active round {state['active_round']} instead of discarding it")
        if state['champions']['mmd'] is None or not any(r['family']=='reference' for r in state['entries'].values()):
            raise ValueError('First score the seed MMD and at least one reference on the locked board')
        name=f"round-{len(state['completed_rounds'])+1:04d}"
        parent=args.parent or state['champions']['mmd']
        if parent and parent not in state['entries']:raise ValueError('Parent must be a scored entry ID')
        family=getattr(args,'family','mmd')
        validate_budget(family,args.budget)
        card=dict(id=name,status='active',started_utc=now(),hypothesis=args.hypothesis,change=args.change,
            parent=parent,family=family,initial_champions=state['champions'],
            initial_best_scores={k:state['entries'][v]['score'] if v else None for k,v in state['champions'].items()},
            additional_update_attempts=args.budget)
        write(home/'rounds'/name/'round.json',card)
        state['active_round']=name;publish(home,state)
        print(json.dumps(card,indent=2))


def finish(home,args):
    with locked(home):
        protocol,state=load_game(home)
        name=identifier(args.round)
        if state['active_round']!=name:raise ValueError('Round is not active')
        card=read(home/'rounds'/name/'round.json')
        result=round_result(state['entries'],name,card['initial_best_scores'],state['champions'])
        if not result['candidates'] and not args.failed:raise ValueError('Finish requires a measured research candidate or an explicit failed experiment')
        notes=Path(args.notes_file).read_text()
        if not notes.strip():raise ValueError('Keep a concrete experiment note')
        card.update(status='complete',finished_utc=now(),notes=notes,
            next_experiment=args.next_experiment,**result)
        write(home/'rounds'/name/'round.json',card)
        state['completed_rounds'].append(card);state['active_round']=None;publish(home,state)
        print(json.dumps(dict(round=name,outcome=result['outcome'],champions=state['champions'],next_experiment=args.next_experiment),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,default=HERE)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('init');commands.add_parser('status')
    p=commands.add_parser('begin')
    p.add_argument('--hypothesis',required=True);p.add_argument('--change',required=True)
    p.add_argument('--parent');p.add_argument('--budget',type=int,default=150)
    p.add_argument('--family',choices=RESEARCH_FAMILIES,default='mmd')
    for command in ('register','evaluate'):
        p=commands.add_parser(command)
        p.add_argument('--id',required=True);p.add_argument('--weights',type=Path,required=True)
        p.add_argument('--family',choices=ALL_FAMILIES,required=True)
        p.add_argument('--round');p.add_argument('--label')
        if command=='register':p.add_argument('--scores',type=Path,nargs='+',required=True)
    p=commands.add_parser('finish')
    p.add_argument('--round',required=True);p.add_argument('--notes-file',type=Path,required=True)
    p.add_argument('--next-experiment',required=True);p.add_argument('--failed',action='store_true')
    args=parser.parse_args();home=args.home.resolve()
    if args.command=='init':init_game(home)
    elif args.command=='status':
        with locked(home):_,state=load_game(home);publish(home,state)
        print((home/'LEADERBOARD.md').read_text())
    else:globals()[args.command](home,args)


if __name__=='__main__':main()
