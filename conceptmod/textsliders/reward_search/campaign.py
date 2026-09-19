"""Run a bounded series of scored experiments and retain every declared result."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import torch

from ..reward_sliders.specs import sha,digest,read_json,write_json,save_tensor
from ..reward_sliders.experiment import verify
from ..reward_sliders.capture import pooled
from ..reward_sliders.directions import RewardTeacher,fit_direction,rank_agreement,choose_dev,causal_gate,paired_statistics
from .setup import DEFAULT_RUN,SOURCE,arm,freeze
from .resources import start_worker,stop_worker,active,worker_unit
from . import queue


def observations(run,stage):
    return [read_json(p) for p in sorted((Path(run)/'stages'/stage/'observations').glob('*.json'))]


def status(run,stage,**extra):
    write_json(Path(run)/'status.json',dict(stage=stage,updated_unix=time.time(),**extra))
    from .report import report
    report(run)


def stage(run,name,families,seeds,arms,capture=False):
    protocol=dict(name=name,families=[f['family'] for f in families],seeds=seeds,arms=arms,capture=capture)
    path=Path(run)/'stages'/name/'protocol.json'
    if path.exists() and read_json(path)!=protocol:raise ValueError('Stage protocol changed: '+name)
    write_json(path,protocol)
    queue.add(run,name,[dict(family=f,seed=seed,arms=arms,capture=capture) for f in families for seed in seeds])


def wait_stage(run,name):
    while True:
        count=queue.counts(run,name)
        status(run,name,bundles=count)
        if not count:raise ValueError('Empty stage '+name)
        if count.get('failed'):raise RuntimeError(f'{name} has a failed render bundle; retained without reroll')
        if not count.get('pending') and not count.get('running'):return observations(run,name)
        for gpu in (0,1):
            state=subprocess.check_output(['systemctl','--user','show',worker_unit(gpu),'-p','ActiveState','--value'],text=True).strip()
            if state=='failed':raise RuntimeError(f'GPU {gpu} worker failed; inspect its log before continuing')
        time.sleep(10)


def select(rows,arms,expected=16):
    result=choose_dev(rows,arms,expected_pairs=expected)
    result['comparisons']={a['name']:paired_statistics(rows,a['name']) for a in arms if a['name']!='off'}
    return result


def fit_teachers(run,m):
    reused=read_json(run/'reused-training.json')
    rows=[]
    for reference in reused:
        if sha(reference['source'])!=reference['source_sha256']:raise ValueError('Reused training observation changed')
        rows.append(reference['observation'])
    rows+=observations(run,'training-capture')
    if len(rows)!=96 or len({r['family'] for r in rows})!=24:raise ValueError('Expanded capture is incomplete')
    features={}
    for row in rows:
        if row['status']!='complete' or sha(row['trajectory'])!=row['trajectory_sha256']:raise ValueError('Invalid capture')
        features[row['id']]=pooled(torch.load(row['trajectory'],map_location='cpu',weights_only=True))
    write_json(run/'training-observations.json',rows)
    teachers={layer:fit_direction(rows,features,layer) for layer in m['search']['teacher_layers']}
    write_json(run/'direction-fit.json',{str(layer):dict(training=rank_agreement(rows,features,t,'train'),
               residual_norm=t.residual_norm,families=t.fitting_families) for layer,t in teachers.items()})
    save_tensor(run/'pooled.pt',features)
    arms=[dict(name='off')]
    for layer,t in teachers.items():
        path=run/'teachers'/f'layer{layer}.pt';save_tensor(path,asdict(t))
        for strength in m['search']['teacher_strengths']:
            arms.append(dict(name=f'new-layer{layer}-s{strength:g}',teacher=str(path),teacher_sha256=sha(path),
                             layer=layer,coefficient=strength))
    previous=SOURCE/'teacher.pt';old=torch.load(previous,map_location='cpu',weights_only=True)
    arms.append(dict(name='v1-teacher',teacher=str(previous),teacher_sha256=sha(previous),
                     layer=old['layer'],coefficient=old['coefficient']))
    return arms


def train_students(run,m):
    (run/'training-active').touch()
    stop_worker(run,1)
    if active(worker_unit(0)):stop_worker(run,0)
    pending=list(m['search']['student_arms']);running={};results={};handles=[]
    try:
        while pending or running:
            available=[1]+([0] if (run/'audit/gpu0-borrowed.json').exists() else [])
            for gpu in available:
                if gpu in running or not pending:continue
                name=pending.pop(0);folder=run/'students'/name
                if (folder/'status.json').exists() and read_json(folder/'status.json')['stage']=='complete':
                    results[name]=0;continue
                folder.mkdir(parents=True,exist_ok=True)
                log=(folder/'process.log').open('a');handles.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                process=subprocess.Popen([sys.executable,'-u','-m','conceptmod.textsliders.reward_search.train',
                    '--run-dir',str(run),'--arm',name,'--gpu',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT)
                running[gpu]=(name,process)
                print(f'TRAIN {name} on physical GPU {gpu}',flush=True)
            for gpu,(name,process) in list(running.items()):
                code=process.poll()
                if code is not None:
                    results[name]=code;del running[gpu]
                    print(f'TRAIN {name} exited {code}',flush=True)
            status(run,'training_students',running={str(g):n for g,(n,_) in running.items()},finished=results)
            if pending or running:time.sleep(10)
    finally:
        for name,process in running.values():
            process.terminate()
        for name,process in running.values():
            try:process.wait(timeout=300)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for handle in handles:handle.close()
        (run/'training-active').unlink(missing_ok=True)
    write_json(run/'training-outcomes.json',results)
    # A failed training arm remains recorded. Only its fully audited earlier exports are eligible.
    start_worker(run,1)
    if (run/'audit/gpu0-borrowed.json').exists():start_worker(run,0)
    return results


def campaign(run):
    run=Path(run);m=freeze(run);search=m['search']
    lock=(run/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={str(Path(__file__).with_name(name)):sha(Path(__file__).with_name(name))
             for name in ('campaign.py','train.py','data.py','resources.py')}
    frozen=run/'controller-sources.json'
    if frozen.exists() and read_json(frozen)!=sources:raise ValueError('Controller or trainer changed; declare an amendment')
    write_json(frozen,sources)
    for path,expected in sources.items():
        target=run/'provenance'/Path(path).relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==expected
    (run/'experiment-done').unlink(missing_ok=True)
    def cancel(signum,frame):raise KeyboardInterrupt('Campaign cancellation requested')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    dev=[f for f in m['families'] if f['split']=='dev']
    final=[f for f in m['families'] if f.get('group')=='final']
    reference=search['reference'];v1=arm(reference['checkpoint'],1.,'v1-original')
    selected_student=None;candidate=v1
    try:
        start_worker(run,1)
        rows=wait_stage(run,'v1-strength')
        arms=read_json(run/'stages/v1-strength/protocol.json')['arms']
        selection=select(rows,arms)
        if (selection['selected']['name']=='v1-m1.25' and
            selection['mean_ce']['v1-m1.25']-selection['mean_ce']['v1-m1']>=.02):
            extension=[dict(name='off'),arm(reference['checkpoint'],1.25,'v1-m1.25'),
                       arm(reference['checkpoint'],1.5,'v1-m1.5')]
            stage(run,'v1-boundary',dev,search['dev_seeds'],extension)
            extra=wait_stage(run,'v1-boundary');extended=select(extra,extension)
            write_json(run/'v1-boundary-selection.json',extended)
            if extended['selected']['name']=='v1-m1.5':selection['selected']=extended['selected']
        write_json(run/'v1-selection.json',selection)
        calibrated=selection['selected']
        candidate=calibrated
        wait_stage(run,'training-capture')
        teacher_arms=fit_teachers(run,m)
        stage(run,'teacher-development',dev,search['dev_seeds'],teacher_arms)
        teacher_rows=wait_stage(run,'teacher-development')
        teacher_selection=select(teacher_rows,teacher_arms)
        write_json(run/'teacher-selection.json',teacher_selection)
        if teacher_selection['selected']['name']!='off':
            chosen=teacher_selection['selected'];values=torch.load(chosen['teacher'],map_location='cpu',weights_only=True)
            values['coefficient']=chosen['coefficient'];teacher=RewardTeacher(**values)
            save_tensor(run/'teacher.pt',asdict(teacher))
            rng=torch.Generator().manual_seed(m['pilot']['random_direction_seed'])
            direction=torch.randn(teacher.direction.shape,generator=rng);direction/=direction.norm()
            save_tensor(run/'random-control.pt',asdict(replace(teacher,direction=direction)))
            causal_arms=[dict(name='off')]
            for name,coefficient in [('positive',teacher.coefficient),('reversed',-teacher.coefficient),('random',teacher.coefficient)]:
                path=run/('random-control.pt' if name=='random' else 'teacher.pt')
                causal_arms.append(dict(name=name,teacher=str(path),teacher_sha256=sha(path),
                                       coefficient=coefficient,layer=teacher.layer))
            families=[f for f in m['families'] if f.get('group')=='causal']
            stage(run,'causal',families,search['causal_seeds'],causal_arms)
            causal_rows=wait_stage(run,'causal');gate=causal_gate(causal_rows)
            gate.update(teacher_sha256=sha(run/'teacher.pt'),manifest_sha256=digest(m))
            write_json(run/'causal-result.json',gate)
            if gate['passed']:
                train_students(run,m)
                student_arms=[dict(name='off'),dict(candidate,name='v1-calibrated')]
                if calibrated['name']=='off':student_arms=[dict(name='off'),v1]
                for recipe in search['student_arms']:
                    folder=run/'students'/recipe
                    for step in search['student_checkpoints']:
                        path=folder/f'reward-ce-v2-{recipe}_step{step}.safetensors'
                        diagnostics=folder/f'diagnostics-step{step}.json'
                        if not path.exists() or not diagnostics.exists():continue
                        metrics=read_json(diagnostics)
                        if any(r['prompt_relative_rms']>2. or r['generation_relative_error']>20. for r in metrics):continue
                        for strength in search['student_strengths']:
                            student_arms.append(arm(path,strength,f'{recipe}-step{step}-m{strength:g}'))
                stage(run,'student-development',dev,search['dev_seeds'],student_arms)
                student_rows=wait_stage(run,'student-development');student_selection=select(student_rows,student_arms)
                write_json(run/'student-selection.json',student_selection)
                selected_student=student_selection['selected']
                if selected_student['name']!='off':candidate=selected_student
                else:candidate=dict(name='off')
        # Final controls are fixed now, before the first evaluation on these eight new families.
        final_arms=[dict(name='off'),v1]
        if calibrated['name']!='off' and calibrated['coefficient']!=1.:
            final_arms.append(dict(calibrated,name='v1-calibrated'))
        if candidate.get('checkpoint') and not any(a.get('checkpoint')==candidate['checkpoint'] and
                                                   a.get('coefficient')==candidate['coefficient'] for a in final_arms):
            final_arms.append(dict(candidate,name='candidate'))
        candidate_name=next((a['name'] for a in final_arms if a.get('checkpoint')==candidate.get('checkpoint') and
                             a.get('coefficient')==candidate.get('coefficient')),'off')
        write_json(run/'final-candidate.json',dict(candidate=candidate,comparison_name=candidate_name,
                   selected_before_final_audio=True,original_v1_control_retained=True))
        stage(run,'final',final,search['final_seeds'],final_arms)
        final_rows=wait_stage(run,'final')
        comparisons={a['name']:dict(vs_off=paired_statistics(final_rows,a['name']),
                    vs_original_v1=paired_statistics(final_rows,a['name'],'v1-original'))
                     for a in final_arms if a['name'] not in ('off','v1-original')}
        original=paired_statistics(final_rows,'v1-original')
        effect=comparisons.get(candidate_name,{}).get('vs_original_v1')
        effect_off=comparisons.get(candidate_name,{}).get('vs_off')
        supported=all(bool(e and e['valid_pairs']==32 and e['family_bootstrap_95'][0]>0)
                      for e in (effect,effect_off))
        result=dict(candidate=candidate,candidate_name=candidate_name,original_v1_vs_off=original,
            comparisons=comparisons,fresh_improvement_supported=supported,
            decision='fresh_ce_improvement' if supported else 'no_confirmed_gain_over_v1',
            primary='equal-family CE on the first 20 seconds',completion_gate=False,
            production_registry_changed=False,default='off')
        write_json(run/'results.json',result)
        status(run,'complete',decision=result['decision'])
        print('RESULT '+result['decision'],flush=True)
    except BaseException as error:
        status(run,'error',error=repr(error));raise
    finally:
        stop_worker(run,1)
        if active(worker_unit(0)):stop_worker(run,0)
        (run/'experiment-done').touch()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();campaign(args.run_dir)
