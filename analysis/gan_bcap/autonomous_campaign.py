"""Durable GPU-1 pilot search, automatic suggestion, catalog training and gallery.

The user explicitly authorized automatic metric-based suggestions and the full
LM catalog. Every rendered seed is fixed; no audio is silently retried or hidden.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PY=sys.executable
HERE=ROOT/'analysis/gan_bcap'
WORK=HERE/'autonomous_campaign_20260905'
GALLERY=ROOT/'eval/listen/gan-autonomous-20260905'
STATE=WORK/'state.json'
sys.path.insert(0,str(ROOT))
from analysis.gan_bcap.autonomous_audio import RULE, summarize


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def search_progress(pilot,minimum_gain=.05):
    initial=[r['score'] for r in pilot[:3] if r.get('eligible',True)]
    if not initial:raise ValueError('No eligible initial reference')
    best=max(initial);misses=0
    for row in pilot[3:]:
        if row.get('eligible',True) and row['score']>best+minimum_gain:
            best=row['score'];misses=0
        else:misses+=1
    return best,misses


def reuse_audio(source,destination,weights,prompts,seeds,duration,row=0):
    source=Path(source);destination=Path(destination)
    spec=read(source/'render_spec.json')
    if (spec['prompts_sha256'],spec['row'],spec['duration'],spec['scales'])!=(sha(prompts),row,duration,[0.,1.]):
        raise ValueError('Reference fixture differs')
    destination.mkdir(parents=True,exist_ok=True)
    records=read(destination/'reference_reuse.json') if (destination/'reference_reuse.json').exists() else {}
    for weight in weights:
        weight=Path(weight).resolve()
        previous=[r for r in spec['checkpoints'] if Path(r['path'])==weight]
        if not previous:continue
        if previous[0]['sha256']!=sha(weight):raise ValueError('Reference checkpoint changed')
        for seed in seeds:
            if seed not in spec['seeds']:continue
            src=source/f'{weight.stem}-s{seed}';dst=destination/src.name;dst.mkdir(exist_ok=True)
            wavs=list(src.glob('*.wav'))
            if len(wavs)!=4:raise ValueError('Reference ladder incomplete')
            for wav in wavs:
                target=dst/wav.name
                if target.exists() and sha(target)!=sha(wav):raise ValueError('Existing audio differs')
                if not target.exists():shutil.copyfile(wav,target)
                records[str(target.relative_to(ROOT))]=dict(source=str(wav.relative_to(ROOT)),sha256=sha(wav))
    write(destination/'reference_reuse.json',records)


class Campaign:
    def __init__(self):
        WORK.mkdir(parents=True,exist_ok=True);GALLERY.mkdir(parents=True,exist_ok=True)
        self.catalog=read(HERE/'autonomous_catalog_20260905.json')['sliders']
        self.plan=read(HERE/'stepcap_only_plan_20260905.json')
        self.state=read(STATE) if STATE.exists() else dict(
            status='running',phase='Waiting for the first listening comparison',started_utc=now(),
            pilot=[],catalog={},events=[],rule=RULE,
            search=dict(interval=150,patience=3,minimum_gain=.05,initial_ceiling=2400,
                        definition='Stop after three evaluated extensions without a new mean heuristic high by at least 0.05. The initial ceiling automatically extends if gains continue; storage availability is checked separately.'),
            authorization='User requested automatic metric-based continuation, all sliders, and an impressive presentation without further intervention.')
        if self.state['status'] == 'interrupted':
            self.state['status'] = 'running'
        self.save()

    def save(self):
        self.state['updated_utc']=now();write(STATE,self.state)
        write(GALLERY/'status.json',self.state)
        self.gallery()

    def event(self,text):
        print(now(),text,flush=True)
        self.state['events'].append(dict(time=now(),message=text));self.save()

    def gallery(self):
        esc=html.escape
        ready=[r for r in self.state['catalog'].values() if r.get('status')=='complete']
        cards=[]
        for item in self.catalog:
            row=self.state['catalog'].get(item['id'],{})
            body=f'<span class="tag">{esc(row.get("status","queued"))}</span>'
            if row.get('off_audio'):
                off=os.path.relpath(row['off_audio'],GALLERY);on=os.path.relpath(row['on_audio'],GALLERY)
                body+=f'<div class="pair"><div><small>Off</small><audio controls preload="none" src="{esc(off)}"></audio></div><div><small>{esc(item["label"])}</small><audio controls preload="none" src="{esc(on)}"></audio></div></div>'
                body+=f'<p class="detail">{row["selected_steps"]} updates · fixed seed 7 · 20 seconds</p>'
                body+=f'<a href="{esc(os.path.relpath(row["download"],GALLERY))}">Download adapter</a>'
                if row.get('long_audio'):
                    body+=f'<details><summary>Longer example · seed 101</summary><audio controls preload="none" src="{esc(os.path.relpath(row["long_audio"],GALLERY))}"></audio></details>'
            elif row.get('error'):
                body+=f'<p>{esc(row["error"])}</p>'
            cards.append(f'<article><h2>{esc(item["label"])}</h2><p>{esc(item["description"])}</p>{body}</article>')
        pilot=self.state.get('pilot',[])
        points=''
        if pilot:
            scores=[r['score'] for r in pilot];lo=min(scores)-.1;hi=max(scores)+.1
            points=' '.join(f'{35+i*630/max(1,len(pilot)-1):.1f},{145-(r["score"]-lo)/(hi-lo)*110:.1f}' for i,r in enumerate(pilot))
        suggestion=self.state.get('suggestion')
        hint=(f'Suggested recipe: {suggestion["steps"]} updates, '+('step limit 2' if suggestion['maximum'] else 'original update rule')+f', lyric hold {suggestion["lyric_hold_weight"]:g}.' if suggestion else 'Searching the training curve before choosing the catalog settings.')
        page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Music / GAN slider study</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101115;color:#f2eee6;font:16px system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:48px 28px}header{border-bottom:1px solid #34353b;padding-bottom:30px}.eyebrow{letter-spacing:.18em;text-transform:uppercase;color:#c0a6ff;font-size:12px}h1{font-size:clamp(38px,6vw,72px);letter-spacing:-.045em;line-height:1.05;margin:20px 0}p{color:#aaaab4;line-height:1.6}.status{display:flex;gap:24px;align-items:center;margin:24px 0}.count{font-size:36px;font-weight:700;color:#b7f4c3}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;margin-top:30px}article{background:#1b1c22;border:1px solid #34353b;border-radius:18px;padding:25px}h2{font-size:25px;margin:0}a{color:#c0a6ff}.tag{display:inline-block;background:#30313a;border-radius:20px;padding:5px 10px;font-size:12px;margin-bottom:20px}audio{width:100%;height:34px;margin:9px 0 16px}small,.detail{font-size:12px;color:#aaaab4}details{margin-top:20px}summary{cursor:pointer;color:#c0a6ff}.search{margin:28px 0;padding:22px;border:1px solid #34353b;border-radius:18px}svg{width:100%;max-width:720px;height:160px}footer{margin-top:35px;font-size:12px}</style>
<main><header><div class="eyebrow">Music research / repaired GAN</div><h1>A new range<br>of musical character.</h1><p>Matched audio, saved checkpoints, and an automatic training study across the studio’s 28 LM controls.</p></header>'''
        page+=f'<div class="status"><span class="count">{len(ready)} / 28</span><div><strong>Controls ready to compare</strong><p id="phase">{esc(self.state["phase"])}</p></div></div>'
        page+=f'<section class="search"><strong>{esc(hint)}</strong><p>The search score combines concept match, predicted enjoyment and production quality, and lyric diagnostics. It is an automatic suggestion, not a proven musical optimum.</p>'
        if points:
            marks=''
            for i,row in enumerate(pilot):
                x=35+i*630/max(1,len(pilot)-1);y=145-(row['score']-lo)/(hi-lo)*110
                marks+=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#c0a6ff"/><text x="{x:.1f}" y="{y-10:.1f}" fill="#f2eee6" text-anchor="middle" font-size="11">{row["score"]:.2f}</text><text x="{x:.1f}" y="166" fill="#aaaab4" text-anchor="middle" font-size="11">{row["steps"]}</text>'
            page+=f'<p class="detail">Pilot heuristic score by completed training updates</p><svg viewBox="0 0 700 170" role="img" aria-label="Measured pilot search scores"><polyline points="{points}" fill="none" stroke="#c0a6ff" stroke-width="3"/>{marks}</svg>'
        page+='<p><a href="pilot.html">Listen to the training curve and inspect score components</a> · <a href="status.json">Full measurements and progress</a></p></section>'
        page+='<div class="grid">'+''.join(cards)+'</div><footer>First renders are retained. Each Off/On pair uses the same prompt, lyrics and seed. Additional examples and score components are recorded in the study artifacts.</footer></main>'
        version=f'{len(ready)}:{len(pilot)}:{suggestion["steps"] if suggestion else 0}'
        page+='<script>const initialVersion='+json.dumps(version)+';document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()})));setInterval(()=>fetch("status.json").then(r=>r.json()).then(s=>{document.querySelector("#phase").textContent=s.phase;const n=Object.values(s.catalog).filter(r=>r.status==="complete").length;const version=n+":"+s.pilot.length+":"+(s.suggestion?s.suggestion.steps:0);if(version!==initialVersion&&Array.from(document.querySelectorAll("audio")).every(a=>a.paused))location.reload();}).catch(()=>{}),15000)</script></html>'
        temp=GALLERY/'index.html.tmp';temp.write_text(page);temp.replace(GALLERY/'index.html')

    def run(self,args,log,gpu=True):
        log=WORK/log
        self.state['phase']=str(log.stem).replace('-',' ');self.save()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if gpu else '',HF_HUB_OFFLINE='1',
            HF_HOME=str(ROOT.parent/'.cache/huggingface'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if gpu:
            while True:
                used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used<512:break
                self.state['phase']=f'Waiting for GPU 1 ({used} MiB occupied)';self.save();time.sleep(15)
        with log.open('a') as handle:
            handle.write('\n'+json.dumps(args)+'\n');handle.flush()
            proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            self.state['child']=dict(pid=proc.pid,command=args,log=str(log),gpu=gpu);self.save()
            while proc.poll() is None:
                self.state['heartbeat_utc']=now();write(STATE,self.state);time.sleep(10)
        self.state.pop('child',None);self.save()
        if proc.returncode:raise RuntimeError(f'{log.name}: process exited {proc.returncode}')

    def render(self,weights,out,prompts,seeds=(7,23),duration=20):
        out=Path(out)
        if (out/'render_spec.json').exists():
            spec=read(out/'render_spec.json')
            if (spec['prompts_sha256'],spec['seeds'],spec['duration'],
                [(c['path'],c['sha256']) for c in spec['checkpoints']])!=(
                    sha(prompts),list(seeds),duration,[(str(Path(w).resolve()),sha(w)) for w in weights]):
                raise ValueError('Existing rendering uses different fixtures or weights')
            complete=all(len(list((out/f'{Path(w).stem}-s{s}').glob('*.wav')))==4 for w in weights for s in seeds)
            if complete:return
        self.run([PY,'-u',str(HERE/'render_steps.py'),'--weights',*map(str,weights),
            '--out',str(out),'--prompts',str(prompts),'--seeds',*map(str,seeds),'--duration',str(duration)],
            out.name+'.log')

    def score(self,folders,concept,out):
        if Path(out).exists() and read(out).get('status')=='complete':
            stored=read(out)
            if stored['rule']!=RULE or stored['source_sha256']!=sha(HERE/'autonomous_audio.py') or stored['concept']!=concept:
                raise ValueError('Existing score uses a different protocol')
            return stored
        self.run([PY,'-u',str(HERE/'autonomous_audio.py'),'--folders',*map(str,folders),
                  '--concept',concept,'--output',str(out)],Path(out).stem+'.log',gpu=False)
        return read(out)

    def train_stage(self,job,base_name,model_root,target,builder,source=None):
        """Resume an interrupted stage into a fresh name, retaining its evidence."""
        import torch
        jobs=self.state.setdefault('jobs',{})
        info=jobs.setdefault(job,dict(name=base_name,attempt=0,source=str(source) if source else None))
        for _ in range(3):
            name=info['name'];folder=Path(model_root)/name
            weights=folder/f'{name}_last.safetensors';state=folder/f'{name}_state.pt'
            if weights.exists() and weights.with_suffix('.json').exists():
                if read(weights.with_suffix('.json'))['steps']!=target:
                    raise ValueError('Completed stage has an unexpected update count')
                info.update(status='complete',weights=str(weights),state=str(state));self.save()
                return weights,state
            if (folder/f'{name}_train.jsonl').exists():
                recovery=None
                if state.exists():
                    completed=torch.load(state,map_location='cpu',weights_only=True,mmap=True)['completed_updates']
                    if completed<target:recovery=state
                if recovery is None:
                    snapshots=sorted(folder.glob('*_step*_state.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
                    for saved in snapshots:
                        if torch.load(saved,map_location='cpu',weights_only=True,mmap=True)['completed_updates']<target:
                            recovery=saved;break
                info['attempt']+=1
                info['name']=base_name+f'-resume{info["attempt"]}'
                if recovery:info['source']=str(recovery)
                name=info['name']
                self.event(f'Resuming {job} into {name}; previous artifacts retained.')
            resume=Path(info['source']) if info.get('source') else None
            try:
                self.run(builder(name,resume),job+f'-attempt{info["attempt"]}.log')
                completed_folder=Path(model_root)/name
                output=completed_folder/f'{name}_last.safetensors'
                if output.exists() and output.with_suffix('.json').exists() and read(output.with_suffix('.json'))['steps']==target:
                    saved=completed_folder/f'{name}_state.pt'
                    info.update(status='complete',weights=str(output),state=str(saved));self.save()
                    return output,saved
            except RuntimeError as exc:
                info['last_error']=str(exc);self.save();continue
        raise RuntimeError(f'{job}: automatic recovery attempts exhausted')

    def pilot(self):
        if self.state.get('suggestion'):return
        primary=ROOT/self.plan['primary']['out'];heldout=ROOT/self.plan['heldout']['out']
        while not all((p/'verification.json').exists() for p in [primary,heldout]):
            self.state['phase']='Rendering the first step-limit-only comparison';self.save();time.sleep(15)
        initial=self.score([primary,heldout],'gender',WORK/'pilot-initial.json')
        initial_rows=initial['ranking']
        allowed=[r for r in initial_rows if 'smoke-stepcap2-only-' in r['checkpoint'] or 'smoke-steps600-' in r['checkpoint']]
        candidates={r['checkpoint']:r for r in initial_rows}
        if not self.state['pilot']:
            self.state['pilot']=[dict(steps=r['steps'],score=r['heuristic_score'],checkpoint=r['checkpoint'],eligible=r['eligible']) for r in sorted(allowed,key=lambda x:x['steps'])]
            self.save()
        latest=900
        name=self.plan['run'];statepath=ROOT/'models/gan-bcap-repair'/name/f'{name}_state.pt'
        best,misses=search_progress(self.state['pilot'])
        for entry in self.state['pilot'][3:]:
            latest=entry['steps'];statepath=Path(entry['state'])
        while misses<3:
            if shutil.disk_usage(ROOT).free < 15*1024**3:
                raise RuntimeError('Search paused with less than 15 GiB free; saved states and measurements are retained.')
            if latest>=self.state['search']['initial_ceiling']:
                self.state['search']['initial_ceiling']+=1500
                self.event('Measured gains continue; extending the search budget automatically.')
            target=latest+150;name=f'smoke-stepcap2-only-{target}-s7-20260905'
            folder=ROOT/'models/gan-bcap-repair'/name;weights=folder/f'{name}_last.safetensors'
            weights,nextstate=self.train_stage(f'pilot-train-{target}',name,ROOT/'models/gan-bcap-repair',target,
                lambda actual,resume:[PY,'-u',str(HERE/'parameter_step_limit.py'),'--max-update-norm','2',
                    '--state',str(resume),'--name',actual,'--lyric-hold-weight','0','--steps',str(target),
                    '--save-every','50','--memory-fraction','.48'],source=statepath)
            folders=[]
            for group in ['primary','heldout']:
                p=self.plan[group];out=ROOT/'eval/listen'/f'gan-auto-pilot-{target}-{group}-20260905'
                base=ROOT/self.plan['primary']['weights'][0]
                prompts=ROOT/p['prompts']
                reuse_audio(ROOT/p['out'],out,[base],prompts,[7,23],20)
                self.render([base,weights],out,prompts);folders.append(out)
            measured=self.score(folders,'gender',WORK/f'pilot-{target}.json')
            current=next(r for r in measured['ranking'] if r['checkpoint']==str(weights))
            candidates[str(weights)]=current
            score=current['heuristic_score']
            if current['eligible'] and score>best+.05:best=score;misses=0
            else:misses+=1
            latest=target;statepath=nextstate
            self.state['pilot'].append(dict(steps=target,score=score,checkpoint=str(weights),state=str(statepath),misses=misses,eligible=current['eligible']))
            self.event(f'Pilot {target}: heuristic {score:.3f}; {misses}/3 intervals without a new high.')
        # Reload measurements to include extensions from a resumed controller.
        for report in WORK.glob('pilot-[0-9]*.json'):
            if read(report).get('status')=='complete':
                for row in read(report)['ranking']:candidates[row['checkpoint']]=row
        eligible=[r for r in candidates.values() if r['eligible']]
        if not eligible:raise RuntimeError('No nonsilent measured pilot candidate')
        winner=sorted(eligible,key=lambda r:(-r['heuristic_score'],r['steps']))[0]
        path=winner['checkpoint'];hold=.001 if 'lyrichold-w0p001-stepcap2' in path else 0.
        maximum=2. if ('stepcap2' in path) else 0.
        self.state['suggestion']=dict(steps=winner['steps'],maximum=maximum,lyric_hold_weight=hold,
            checkpoint=path,score=winner['heuristic_score'],pilot_search_stopped_at=latest,
            plateau_observed=misses>=3,search_budget_reached=False,
            basis='Highest mean fixed rendered-audio heuristic among the measured eligible pilot candidates. Subjective preference remains unproven.',
            warmup_updates=min(600,winner['steps']))
        write(WORK/'suggestion.json',self.state['suggestion'])
        self.event(f'Automatic suggestion saved: {winner["steps"]} updates, step limit {maximum:g}, lyric hold {hold:g}.')

    def confirm_suggestion(self):
        """Confirm the two leading recipes on arrangements/seeds not used to search."""
        if self.state.get('confirmation',{}).get('status')=='complete':return
        import yaml
        from analysis.gan_bcap.lm_evaluate import make_heldout_rows
        candidates={}
        for file in [WORK/'pilot-initial.json',*WORK.glob('pilot-[0-9]*.json')]:
            report=read(file)
            if report.get('status')!='complete':continue
            for row in report['ranking']:candidates[row['checkpoint']]=row
        leaders=sorted([r for r in candidates.values() if r['eligible']],key=lambda r:(-r['heuristic_score'],r['steps']))[:2]
        if len(leaders)!=2:raise RuntimeError('Need two eligible candidates for confirmation')
        spec=yaml.safe_load((ROOT/self.plan['primary']['prompts']).read_text())
        training=spec['rows'] if isinstance(spec,dict) else spec
        examples=make_heldout_rows(training[0],4)[1:3]
        folders=[]
        for index,example in enumerate(examples):
            prompts=WORK/f'confirmation-{index}.yaml'
            if not prompts.exists():prompts.write_text(yaml.safe_dump([example],sort_keys=False,allow_unicode=True))
            out=ROOT/'eval/listen'/f'gan-auto-confirm-{index}-20260905'
            self.render([r['checkpoint'] for r in leaders],out,prompts,seeds=(101,303))
            folders.append(out)
        measured=self.score(folders,'gender',WORK/'confirmation.json')
        combined=[]
        for row in measured['ranking']:
            prior=candidates[row['checkpoint']]
            combined.append(dict(checkpoint=row['checkpoint'],steps=row['steps'],
                eligible=prior['eligible'] and row['eligible'],
                search_score=prior['heuristic_score'],confirmation_score=row['heuristic_score'],
                combined_score=(prior['heuristic_score']+row['heuristic_score'])/2))
        eligible=[r for r in combined if r['eligible']]
        if not eligible:raise RuntimeError('No eligible candidate after confirmation')
        winner=sorted(eligible,key=lambda r:(-r['combined_score'],r['steps']))[0]
        self.state['pre_confirmation_suggestion']=dict(self.state['suggestion'])
        path=winner['checkpoint']
        self.state['suggestion'].update(steps=winner['steps'],maximum=2. if 'stepcap2' in path else 0.,
            lyric_hold_weight=.001 if 'lyrichold-w0p001-stepcap2' in path else 0.,
            checkpoint=path,score=winner['combined_score'],warmup_updates=min(600,winner['steps']),
            basis='Highest combined mean heuristic among the two pilot leaders, using four search examples and four additional arrangement/seed examples. This remains a proxy-based suggestion.')
        self.state['confirmation']=dict(status='complete',ranking=combined,folders=list(map(str,folders)),
            seeds=[101,303],scope='Arrangements and seeds not used in the audio search. Their use to confirm/select makes these development validation, not an untouched final test.')
        write(WORK/'suggestion.json',self.state['suggestion'])
        self.event(f'Confirmation complete; catalog suggestion is {winner["steps"]} updates.')

    def catalog_axis(self,item):
        import yaml
        sid=item['id'];suggestion=self.state['suggestion'];steps=suggestion['steps']
        self.state['catalog'][sid]=dict(status='training');self.save()
        prompts=ROOT/item['prompts']
        if sha(prompts)!=item['prompts_sha256']:raise ValueError('Catalog prompt changed after audit')
        prefix=f'{sid}-gan-warmup-s7';warm=ROOT/'models/gan-autonomous-20260905'/prefix
        warmsteps=min(600,steps);warmweights=warm/f'{prefix}_last.safetensors'
        warmweights,warmstate=self.train_stage(f'{sid}-warmup',prefix,ROOT/'models/gan-autonomous-20260905',warmsteps,
            lambda actual,resume:[PY,'-u',str(HERE/'catalog_train.py'),'--name',actual,'--prompts',str(prompts),
                '--steps',str(warmsteps),'--maximum','0','--save-every','150']+
                (['--resume-state',str(resume)] if resume else []))
        if steps>600:
            name=f'{sid}-gan-selected-s7';run=ROOT/'models/gan-autonomous-20260905'/name
            weights=run/f'{name}_last.safetensors'
            weights,_=self.train_stage(f'{sid}-continuation',name,ROOT/'models/gan-autonomous-20260905',steps,
                lambda actual,resume:[PY,'-u',str(HERE/'catalog_train.py'),'--name',actual,'--prompts',str(prompts),
                    '--steps',str(steps),'--maximum',str(suggestion['maximum']),
                    '--lyric-hold-weight',str(suggestion['lyric_hold_weight']),
                    '--resume-state',str(resume),'--save-every','150'],source=warmstate)
        else:weights=warmweights
        # Inspect the suggested endpoint and the nearest preceding exported checkpoint.
        alternatives=[weights]
        previous=weights.parent/f'{weights.stem.removesuffix("_last")}_step{steps-150}.safetensors'
        if previous.exists():alternatives.insert(0,previous)
        if warmweights not in alternatives and steps>600:alternatives.insert(0,warmweights)
        early=warmweights.parent/f'{warmweights.stem.removesuffix("_last")}_step300.safetensors'
        if early.exists() and early not in alternatives:alternatives.insert(0,early)
        if sid == 'gender' and Path(suggestion['checkpoint']) not in alternatives:
            alternatives.append(Path(suggestion['checkpoint']))
        raw=yaml.safe_load(prompts.read_text());rows=raw['rows'] if isinstance(raw,dict) else raw
        original=rows[0]
        fresh=dict(original);fresh['lyrics']='[verse]\nThe kettle clicks beside the window\nA paper boat rests by the door\n[chorus]\nWe count the stairs and carry daylight\nAcross the quiet kitchen floor'
        folders=[]
        for label,example in [('original',original),('new-lyrics',fresh)]:
            fixture=WORK/f'{sid}-{label}.yaml'
            if not fixture.exists():fixture.write_text(yaml.safe_dump([example],sort_keys=False,allow_unicode=True))
            out=ROOT/'eval/listen'/f'gan-auto-{sid}-{label}-20260905'
            self.render(alternatives,out,fixture);folders.append(out)
        report=self.score(folders,sid,WORK/f'{sid}-scores.json')
        eligible=[r for r in report['ranking'] if r['eligible']]
        if not eligible:raise RuntimeError('No nonsilent measured candidate')
        winner=eligible[0];chosen=Path(winner['checkpoint'])
        final=GALLERY/'adapters'/sid;final.mkdir(parents=True,exist_ok=True)
        download=final/f'{sid}.safetensors';shutil.copyfile(chosen,download)
        # Preserve accurate source topology and actual selected step in the portable sidecar.
        sys.path.insert(0,str(HERE))
        from lm_evaluate import checkpoint_metadata
        metadata,_,_=checkpoint_metadata(chosen)
        metadata=dict(metadata,steps=winner['steps'],weights=str(download),checkpoint='automatic suggestion',
                      automatic_selection=dict(rule=RULE,source=str(chosen),source_sha256=sha(chosen),metrics=winner))
        write(download.with_suffix('.json'),metadata)
        sub=folders[0]/f'{chosen.stem}-s7'
        row=dict(status='checking longer audio',selected_steps=winner['steps'],score=winner['heuristic_score'],
            source_weights=str(chosen),download=str(download),off_audio=str(sub/'01_slider_neutral_base_zero.wav'),
            on_audio=str(next(sub.glob('02_slider_*_plus1.wav'))),measurements=str(WORK/f'{sid}-scores.json'))
        self.state['catalog'][sid]=row;self.save()
        longout=ROOT/'eval/listen'/f'gan-auto-{sid}-long-20260905'
        self.render([chosen],longout,WORK/f'{sid}-new-lyrics.yaml',seeds=(101,),duration=60)
        longsub=longout/f'{chosen.stem}-s101'
        row['long_audio']=str(next(longsub.glob('02_slider_*_plus1.wav')))
        row['status']='complete';self.state['catalog'][sid]=row
        self.event(f'{item["label"]}: selected {winner["steps"]} updates; matched examples and longer seed-101 render complete.')

    def main(self):
        self.pilot()
        self.confirm_suggestion()
        for item in self.catalog:
            if self.state['catalog'].get(item['id'],{}).get('status')=='complete':continue
            try:self.catalog_axis(item)
            except Exception as exc:
                self.state['catalog'][item['id']]=dict(status='failed',error=str(exc))
                self.event(f'{item["label"]}: recorded failure and continuing other controls: {exc}')
        done=sum(r.get('status')=='complete' for r in self.state['catalog'].values())
        self.state.update(status='complete' if done==len(self.catalog) else 'completed with failures',
            phase=f'{done} of {len(self.catalog)} controls completed',completed_utc=now())
        self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT);WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=Campaign()
        if args.prepare_only:return
        try:campaign.main()
        except Exception as exc:
            campaign.state.update(status='interrupted',phase=str(exc));campaign.save();raise


if __name__=='__main__':main()
