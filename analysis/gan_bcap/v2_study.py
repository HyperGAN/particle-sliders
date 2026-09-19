"""Bounded architecture ablations and matched audio; the catalog stays paused."""
from __future__ import annotations

from datetime import datetime,timezone
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders.gan_v2.data import sha
from conceptmod.textsliders.gan_v2.state import code_fingerprints

WORK=ROOT/'analysis/gan_bcap/v2_20260905'
PAGE=ROOT/'eval/listen/gan-v2-20260905'
SOURCE=ROOT/'models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_state.pt'
INCUMBENT=SOURCE.with_name('smoke-steps600-s7-20260904_last.safetensors')
ORIGINAL=ROOT/'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml'
FIXTURES=WORK/'fixtures'
ARMS=['baseline','fm_normalized','fm_capped','decay','repaired']
TARGET=660


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def publish(status):
    PAGE.mkdir(parents=True,exist_ok=True)
    cards=[]
    for arm,result in status.get('runs',{}).items():
        run=Path(result['directory']);rows=[]
        for log in sorted(run.glob('train-from-*.jsonl')):
            rows.extend(json.loads(line) for line in log.read_text().splitlines() if line.strip())
        if rows:
            last=rows[-1]
            cards.append(f'<tr><td>{html.escape(arm)}</td><td>{last["step"]}</td>'
                         f'<td>{last["effective_limit"]["actual_norm"]:.4f}</td><td>{last["lr_scale"]:.4f}</td></tr>')
    audio=[]
    for folder in sorted(PAGE.glob('prompt-*')):
        spec_path=folder/'render_spec.json'
        if not spec_path.exists():continue
        spec=json.loads(spec_path.read_text())
        for checkpoint in spec['checkpoints']:
            stem=Path(checkpoint['path']).stem
            clips=[]
            for seed in spec['seeds']:
                files=list((folder/f'{stem}-s{seed}').glob('02_slider_*_plus1.wav'))
                if files:
                    relative=files[0].relative_to(PAGE)
                    clips.append(f'<p>Seed {seed}<audio controls preload="none" src="{html.escape(str(relative))}"></audio></p>')
            if clips:audio.append(f'<details><summary>{folder.name} · {html.escape(stem)}</summary>'+''.join(clips)+'</details>')
    text='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music GAN repair study</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#151820;color:#e8edf3;font:17px system-ui}a{color:#a5d6ff}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #39414f;text-align:left}details{padding:12px;border:1px solid #39414f;margin:12px 0}audio{display:block;width:100%;max-width:700px}p{line-height:1.6}</style>'''
    text+='<h1>Music GAN repair study</h1><p>'+html.escape(status.get('phase','Preparing'))+'</p>'
    text+='<p>Controlled changes from the saved 600 checkpoint. Baseline, normalized FM, capped FM and decay use the same four prompts. The combined repaired candidate also uses broader training data. All first samples are retained; no musical optimum is claimed.</p>'
    text+='<table><tr><th>Run</th><th>Updates</th><th>Effective adapter step</th><th>Learning-rate fraction</th></tr>'+''.join(cards)+'</table>'
    text+='<p><a href="data.json">Study data</a> · <a href="../gan-convergence-20260905/index.html">Earlier convergence study</a></p>'+''.join(audio)
    text+='<script>setTimeout(()=>location.reload(),60000)</script>'
    (PAGE/'index.html').write_text(text);write(PAGE/'data.json',status)


def phase(status,text):
    status.update(phase=text,updated_utc=datetime.now(timezone.utc).isoformat())
    write(WORK/'study-status.json',status);publish(status);print(text,flush=True)


def command(arguments,log,*,gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if gpu else '',HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu:
        used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>512:raise RuntimeError(f'GPU 1 is occupied ({used} MiB); no process was terminated')
    with Path(log).open('a') as handle:
        subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'study.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(version=1,source=dict(path=str(SOURCE),sha256=sha(SOURCE)),
            source_files=code_fingerprints(),harness_sha256=sha(__file__),
            arms=ARMS,origin=600,horizon=TARGET,endpoint=TARGET,
            original_prompts_sha256=sha(ORIGINAL),fixture_manifest_sha256=sha(FIXTURES/'manifest.json'),
            evaluation_rows=[0,1],seeds=[7,23,101,303],duration=20,
            renderer_sha256=sha(ROOT/'analysis/gan_bcap/render_v2.py'),
            scorer_sha256=sha(ROOT/'analysis/gan_bcap/autonomous_audio.py'),
            gates_sha256=sha(ROOT/'analysis/gan_bcap/audio_v2.py'),
            scope='Bounded controlled screening, not a final convergence or catalog-promotion decision',
            combined_arm='repaired additionally changes critic context/order, paired FM, full-width and policy guards, functional update cap and training support',
            catalog='paused throughout')
        path=WORK/'study-manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Declared architecture experiment changed')
        write(path,manifest)
        status=json.loads((WORK/'study-status.json').read_text()) if (WORK/'study-status.json').exists() else dict(status='running',runs={})
        status['status']='running'
        try:
            for arm in ARMS:
                prior=status['runs'].get(arm)
                if prior and prior.get('complete'):continue
                name=f'{arm}-660-20260905'
                run=ROOT/'models/gan-v2'/name
                resume=None
                known=Path(prior['directory']) if prior else run
                if (known/'state.pt').exists():
                    import torch
                    saved=torch.load(known/'state.pt',map_location='cpu',weights_only=True)
                    if saved['completed']==TARGET:
                        result=json.loads((known/'latest.json').read_text())
                        if result['step']!=TARGET or not all(Path(p).exists() for p in result['checkpoints'].values()):
                            raise RuntimeError('Completed state is missing its inference artifacts')
                        status['runs'][arm]=dict(directory=str(known),complete=True,**result)
                        phase(status,f'Recovered completed {arm} from its saved state');continue
                    resume=known/'state.pt'
                    del saved
                if run.exists():
                    attempt=2
                    while (ROOT/'models/gan-v2'/f'{name}-attempt{attempt}').exists():attempt+=1
                    run=ROOT/'models/gan-v2'/f'{name}-attempt{attempt}'
                status['runs'][arm]=dict(directory=str(run),complete=False)
                phase(status,f'Training {arm}: controlled continuation to {TARGET}')
                prompts=FIXTURES/'train.yaml' if arm=='repaired' else ORIGINAL
                args=['-m','conceptmod.textsliders.gan_v2.train','--prompts',prompts,'--run-dir',run,'--arm',arm,
                    '--schedule-origin','600','--schedule-horizon',str(TARGET),'--until',str(TARGET),
                    '--save-every','30','--ema-every','15','--diagnostics-every','15']
                args+=['--resume',resume] if resume else ['--source-state',SOURCE]
                if arm!='repaired':args+=['--no-ema']
                command(args,WORK/f'{run.name}.log',gpu=True)
                result=json.loads((run/'latest.json').read_text())
                if result['step']!=TARGET:raise RuntimeError('Training was interrupted before its declared endpoint')
                status['runs'][arm].update(complete=True,**result)
                phase(status,f'Completed {arm} at {TARGET}; checkpoint retained for matched audio')
            weights=[str(INCUMBENT)]+[status['runs'][arm]['checkpoints']['live'] for arm in ARMS]
            weights.append(status['runs']['repaired']['checkpoints']['ema'])
            score_files=[]
            for row in manifest['evaluation_rows']:
                folder=PAGE/f'prompt-{row:02d}'
                phase(status,f'Rendering reserved prompt {row+1}/2, four matched seeds per candidate')
                command(['analysis/gan_bcap/render_v2.py','--weights',*weights,'--out',folder,
                    '--prompts',FIXTURES/'evaluation.yaml','--row',str(row),'--seeds','7','23','101','303','--duration','20'],
                    WORK/f'render-v2-{row:02d}.log',gpu=True)
                scores=WORK/f'scores-v2-{row:02d}.json';score_files.append(scores)
                if not scores.exists() or json.loads(scores.read_text()).get('status')!='complete':
                    phase(status,f'Measuring audio components for reserved prompt {row+1}/2')
                    command(['analysis/gan_bcap/autonomous_audio.py','--folders',folder,'--concept','gender','--output',scores],
                        WORK/f'score-v2-{row:02d}.log',gpu=False)
            phase(status,'Measuring diversity and per-prompt regression flags')
            command(['analysis/gan_bcap/audio_v2.py','--scores',*score_files,'--output',WORK/'screening.json'],WORK/'screening.log',gpu=False)
            status.update(status='complete',screening=str(WORK/'screening.json'),catalog='paused',
                conclusion='Architecture screening complete; two prompts do not establish convergence or validate overall musical quality.')
            phase(status,'Completed controlled training and matched audio screening; catalog remains paused')
            subprocess.run(['systemctl','--user','start','music-gan-convergence-audio-20260905.service'],check=True)
        except Exception as error:
            status.update(status='failed',error=str(error));phase(status,'Architecture study stopped; partial artifacts retained')
            raise


if __name__=='__main__':main()
