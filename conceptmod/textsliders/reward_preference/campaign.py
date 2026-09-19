"""Bounded training rounds; weak screens do not trigger a large evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..reward_sliders.specs import read_json,write_json,sha
from ..reward_search.setup import arm
from ..reward_search.resources import studio,keep_settings,restore_keep,wait_studio,OVERRIDE,CONTENTS
from .setup import DEFAULT_RUN,freeze
from .objective import screen_summary


def report(run,stage,**details):
    write_json(run/'status.json',dict(stage=stage,updated_unix=time.time(),**details))
    text=['# Reward preference continuation','',f'Current stage: **{stage}**.','',
        'The user stopped the larger half-strength evaluation to prioritize training for more consistent wins. '
        'This experiment continues the original rank-8 LoRA weights, initialized at their effective half strength. '
        'It uses a new preference optimizer; the unsuccessful activation teacher is not used.','',
        'Twenty-four same-prompt high/low CE pairs reuse 48 existing training histories and their scores. '
        'Semantic codes must reproduce 501 saved feedback frames exactly. No new training audio is generated. '
        'The differentiable objective favors the higher-scoring semantic sequence relative to the initial LoRA, '
        'with reference-policy KL and prompt preservation. It is a semantic-policy surrogate; it does not '
        'differentiate the audio scorer or claim the full audio likelihood.','',
        'Two learning rates train for 30 and 60 updates, with four comparisons per checkpoint. '
        'Step 120 is allowed only after a 3/4-win screen with mean CE gain at least .02 and no loss worse than -.30. '
        'Selection ranks wins, then the worst regression, then mean CE; the initial half-strength LoRA remains eligible. '
        'Only a promising new candidate receives four fresh Off/candidate pairs (eight clips). '
        'There are at most 36 new audio clips across this entire bounded experiment.','',
        '[Protocol](manifest.json) · [Training preferences](preferences.json) · [Validation](audit/tests.json)','',
        'Method reference: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290). '
        'This music-specific adaptation uses mean semantic log probabilities and omits sampling top-k and residual/acoustic likelihoods.','']
    if (run/'screens.json').exists():
        text+=['| Checkpoint | Wins vs Off | Mean CE gain | Worst delta | Pass |','|---|---:|---:|---:|---|']
        for name,s in read_json(run/'screens.json').items():
            if 'wins' in s:text.append(f'| {name} | {s["wins"]}/4 | {s["mean_ce_gain"]:+.3f} | {s["worst_ce_delta"]:+.3f} | {s["passed"]} |')
        text+=['','These are reused development comparisons. They cannot establish shipping readiness.','']
    if (run/'results.json').exists():
        r=read_json(run/'results.json');text+=[f'Outcome: **{r["decision"]}**.','', '[Recorded results](results.json)','']
    if details.get('error'):text+=['Execution error: '+str(details['error']),'']
    text+=['Studio playback stays available during the run. GPU 0 returns to one studio worker afterward. '
           'Original weights and production slider registration remain unchanged.','']
    (run/'README.md').write_text('\n'.join(text))


def children(run,jobs,label):
    processes=[];handles=[];outcomes={}
    try:
        for gpu,module,arguments in jobs:
            path=run/'logs'/f'{label}-gpu{gpu}.log';path.parent.mkdir(exist_ok=True)
            handle=path.open('a');handles.append(handle)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            command=[sys.executable,'-u','-m','conceptmod.textsliders.reward_preference.'+module,
                     '--run-dir',str(run),'--gpu',str(gpu),*arguments]
            process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            processes.append((gpu,process));print('START '+label+' GPU '+str(gpu),flush=True)
        while any(p.poll() is None for _,p in processes):
            report(run,label,running_gpus=[g for g,p in processes if p.poll() is None]);time.sleep(10)
        outcomes={gpu:p.returncode for gpu,p in processes}
    finally:
        for _,p in processes:
            if p.poll() is None:p.terminate()
        for _,p in processes:
            if p.poll() is None:
                try:p.wait(timeout=60)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
    write_json(run/'audit'/f'{label}-processes.json',{str(g):code for g,code in outcomes.items()})
    return outcomes


def render_jobs(run,name,jobs):
    for gpu in (0,1):
        path=run/'jobs'/f'{name}-gpu{gpu}.json';items=[j for j in jobs if j['gpu']==gpu]
        if path.exists() and read_json(path)!=items:raise ValueError('Declared audio jobs changed')
        write_json(path,items)
    result=children(run,[(gpu,'render',['--stage',name]) for gpu in (0,1) if any(j['gpu']==gpu for j in jobs)],'render-'+name)
    if any(result.values()):raise RuntimeError('Small audio screen failed; outputs retained without reroll')
    rows=[read_json(p) for p in sorted((run/'renders'/name/'observations').glob('*.json'))]
    if len(rows)!=len(jobs):raise ValueError('Missing audio result')
    for r in rows:
        assert r['status']=='complete' and sha(r['audio'])==r['audio_sha256']
        job=next(j for j in jobs if j['family']['family']==r['family'] and j['seed']==r['seed'] and j['arm']==r['arm'])
        assert r['provenance']['physical_gpu']==job['gpu']
    return rows


def campaign(run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    lock=(run/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=freeze(run);settings=m['preference'];borrowed=False;snapshot=None
    def cancel(*_):raise KeyboardInterrupt('User interrupted preference continuation')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    controls_list=read_json(run/'controls.json')
    for control in controls_list:assert sha(control['source'])==control['sha256']
    controls={(r['observation']['family'],r['observation']['seed']):r['observation'] for r in controls_list}
    families={f['family']:f for f in m['families']}
    def dev_jobs(treatment):
        return [dict(family=families[family],seed=seed,arm=treatment,gpu=row['provenance']['physical_gpu'])
                for (family,seed),row in controls.items()]
    try:
        # Both GPUs were authorized. Wait for a genuinely empty studio queue before borrowing GPU 0.
        while True:
            state=studio()
            if not state['active'] and not state['queued'] and not state['keep']['enabled']:
                if OVERRIDE.exists():raise RuntimeError('Another run still owns the studio GPU override')
                snapshot=keep_settings(state['keep']);write_json(run/'audit/studio-before.json',dict(keep=snapshot,time_unix=time.time()))
                state=studio()
                if state['active'] or state['queued'] or state['keep']['enabled']:continue
                OVERRIDE.write_text(CONTENTS);borrowed=True
                subprocess.run(['systemctl','--user','daemon-reload'],check=True)
                subprocess.run(['systemctl','--user','restart','music-studio.service'],check=True)
                wait_studio();restore_keep(snapshot)
                if studio()['loaded']:raise RuntimeError('Studio retained GPU 0')
                write_json(run/'audit/gpu0-borrowed.json',dict(time_unix=time.time(),queue_drained=True,playback_available=True))
                break
            report(run,'waiting_for_studio_queue');time.sleep(10)
        replay=children(run,[(g,'replay',[]) for g in (0,1)],'recover_saved_histories')
        if any(replay.values()):raise RuntimeError('Saved history did not reproduce; semantic targets not inferred')
        initial=render_jobs(run,'incumbent',dev_jobs(settings['initial']))
        screens={'incumbent':screen_summary(initial,controls)};checkpoints={'incumbent':settings['initial']}
        write_json(run/'screens.json',screens)
        eligible=list(settings['learning_rates'])
        for total in settings['checkpoints']:
            if not eligible:break
            gpu_for={name:index for index,name in enumerate(eligible)}
            outcomes=children(run,[(gpu_for[name],'train',['--arm',name,'--total',str(total)]) for name in eligible],f'train-{total}')
            next_eligible=[]
            treatments={}
            for name in eligible:
                if outcomes[gpu_for[name]]!=0:continue
                checkpoint=run/'students'/name/f'reward-ce-preference-{name}_step{total}.safetensors'
                label=f'{name}-{total}';treatment=arm(checkpoint,1.,label)
                treatments[name]=treatment
            # Keep each renderer resident while it compares both learning-rate arms.
            round_rows=render_jobs(run,f'round-{total}',[job for treatment in treatments.values() for job in dev_jobs(treatment)]) if treatments else []
            for name,treatment in treatments.items():
                label=treatment['name'];rows=[r for r in round_rows if r['arm']['name']==label]
                metrics=screen_summary(rows,controls)
                initial_by_key={(r['family'],r['seed']):r for r in initial}
                metrics['vs_initial_half']=screen_summary(rows,initial_by_key)
                screens[label]=metrics;checkpoints[label]=treatment;write_json(run/'screens.json',screens)
                if total==30 or metrics['passed']:next_eligible.append(name)
            eligible=next_eligible
        def rank(label):
            x=screens[label];return (x.get('wins',-1),x.get('worst_ce_delta',-100),x.get('mean_ce_gain',-100))
        promising=[name for name,x in screens.items() if name!='incumbent' and x['passed'] and rank(name)>rank('incumbent')]
        selected=max(promising,key=rank) if promising else 'incumbent'
        write_json(run/'selection.json',dict(selected=selected,arm=checkpoints[selected],selected_before_confirmation=True))
        result=dict(selected=selected,checkpoint=checkpoints[selected],screens=screens,
                    decision='no_promising_continuation' if selected=='incumbent' else 'promising_development_continuation',
                    production_registry_changed=False,completion_gate=False)
        if selected!='incumbent':
            jobs=[]
            for index,f in enumerate(f for f in m['families'] if f.get('group')=='preference-confirm'):
                for treatment in (dict(name='off'),dict(checkpoints[selected],name='candidate')):
                    jobs.append(dict(family=f,seed=9701,arm=treatment,gpu=index%2))
            final=render_jobs(run,'confirmation',jobs)
            base={(r['family'],r['seed']):r for r in final if r['arm']['name']=='off'}
            measured=screen_summary([r for r in final if r['arm']['name']=='candidate'],base)
            measured['interpretation']='four fresh pairs; early confirmation only, not a robust quality claim'
            result.update(confirmation=measured,decision='small_fresh_gain' if measured['passed'] else 'continuation_not_confirmed')
        # Audit every new score and export; raw audio and original weights stay unchanged.
        observed=[read_json(p) for p in (run/'renders').glob('*/observations/*.json')]
        assert len(observed)<=settings['max_new_audio']
        for row in observed:
            assert row['status']=='complete' and sha(row['audio'])==row['audio_sha256']==row['reward']['audio_sha256']
            assert abs(sum(w['axes']['CE'] for w in row['reward']['windows'])/2-row['reward']['scalar'])<1e-9
        import torch
        from safetensors.torch import load_file
        for path in (run/'students').glob('*/*.safetensors'):
            sidecar=read_json(path.with_suffix('.json'));weights=load_file(str(path))
            assert sha(path)==sidecar['weights_sha256'] and len(weights)==432
            assert all(torch.isfinite(v).all() for v in weights.values())
            assert sum(k.endswith('.alpha') for k in weights)==144
            assert all(float(v)==8 for k,v in weights.items() if k.endswith('.alpha'))
        assert sha(settings['initial']['checkpoint'])==settings['initial']['checkpoint_sha256']
        write_json(run/'audit/final-integrity.json',dict(passed=True,new_audio=len(observed),
            original_weights_unchanged=True,training_replay_audio=0,source_and_target_identity_verified=True))
        write_json(run/'results.json',result);report(run,'complete',decision=result['decision'])
    except BaseException as error:
        report(run,'error',error=repr(error));raise
    finally:
        if borrowed:
            try:snapshot=keep_settings(studio()['keep'])
            except Exception:pass
            if OVERRIDE.exists():
                if OVERRIDE.read_text()!=CONTENTS:raise RuntimeError('Studio override was changed by another owner')
                OVERRIDE.unlink()
            subprocess.run(['systemctl','--user','daemon-reload'],check=True)
            subprocess.run(['systemctl','--user','restart','music-studio.service'],check=True)
            wait_studio()
            if snapshot:restore_keep(snapshot)
            write_json(run/'audit/studio-restored.json',dict(time_unix=time.time(),gpu=0,workers=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=p.parse_args();campaign(args.run_dir)
