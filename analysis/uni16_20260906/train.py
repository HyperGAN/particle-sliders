"""Persistent, resumable uni16 campaign using the exact winning two-stage recipe."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(WORK), str(ROOT), str(ROOT.parent)]
from build import sha, validate_document

MODELS = ROOT/'models/uni16-gan-v1'
PAGE = ROOT/'eval/listen/uni16-gan-v1'
PY = '/home/mikkel/anaconda3/envs/minimax-music3/bin/python'
WINNER = ROOT/'models/gan-v2/baseline-660-20260905'
ORIGINAL = ROOT/'models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_state.pt'
PRESENTATION_FILES = {'report.py', 'patch_listen_page.py'}


class CampaignBlocked(RuntimeError):
    """A shared prerequisite stopped the queue before another model job."""


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def warmup_args(item, run, resume=None):
    args=['--name',run.name,'--prompts_file',item['train_prompts'],'--save_dir',str(run),
          '--lm_target','faithful_plus_neu_lyric','--pole_mode','hidden',
          '--rank','8','--alpha','8','--lr','5e-4','--steps','600','--seed','7',
          '--no-early_stop','--endreg_weight','1','--save_every','150','--device','0',
          '--adv_arch','tx','--adv_in','scaled','--adv_readout','mean_last','--adv_condition','none',
          '--adv_weight','1','--fm_weight','1','--fm_mode','batch','--pole_weight','0',
          '--lyrichold_weight','0','--adv_reg_coeff','1','--adv_reg_kappa','1','--adv_batch','4',
          '--gan_beta1','0','--gan_lr_schedule','constant','--grad_account','--parts','0','--save_training_state']
    if resume: args+=['--resume_state',str(resume)]
    return args


def bounded_args(item, run, source, resume=None):
    args=['-m','conceptmod.textsliders.gan_v2.train','--prompts',item['train_prompts'],
          '--run-dir',str(run),'--arm','baseline','--schedule-origin','600',
          '--schedule-horizon','660','--until','660','--save-every','30',
          '--diagnostics-every','15','--ema-every','15','--no-ema']
    return args+(['--resume',str(resume)] if resume else ['--source-state',str(source)])


def signed_files():
    from conceptmod.textsliders.gan_v2.state import code_fingerprints
    paths=set(map(Path,code_fingerprints()))
    paths.update(p for p in WORK.glob('*.py') if p.name not in PRESENTATION_FILES)
    paths.update(WORK.glob('*.json'))
    paths.update((WORK/'prompts').glob('*.yaml'))
    for name in ('lm_particles.py','slider_targets.py','generate_listen.py'):
        paths.add(ROOT/'conceptmod/textsliders'/name)
    paths.update(ROOT/'analysis/gan_bcap'/name for name in ('render_v2.py','autonomous_audio.py','lm_evaluate.py'))
    paths.update([ROOT/'slider_selection/features.py',ROOT/'scripts/lm_score.py',ROOT.parent/'app/rewriter.py'])
    paths-=set(WORK/name for name in ('manifest.json','status.json','validation.json','studio-catalog.json'))
    return {str(p.resolve()):sha(p) for p in sorted(paths)}


def prepare():
    import torch
    from conceptmod.textsliders import train_lm_slider_music3 as legacy, lm_gan_state
    from conceptmod.textsliders.gan_v2.train import arm_settings
    catalog=read(WORK/'catalog.json')
    assert len(catalog['sliders'])==16
    for item in catalog['sliders']:
        for split in ('train','eval'):
            validate_document(Path(item[split+'_prompts']),item,split=='eval')
    winning=read(WINNER/'manifest.json')
    for path,expected in winning['sources'].items():
        if sha(path)!=expected: raise ValueError(f'Winning source changed: {path}')
    old=torch.load(ORIGINAL,map_location='cpu',weights_only=True,mmap=True)
    item=catalog['sliders'][0]
    args=legacy.parse_args(warmup_args(item,MODELS/'preflight'))
    rows,meta=legacy._load_rows(Path(item['train_prompts']))
    signature=lm_gan_state.signature(args,rows,meta)
    if signature['settings']!=old['signature']['settings']:
        difference={k:(v,old['signature']['settings'].get(k)) for k,v in signature['settings'].items()
                    if v!=old['signature']['settings'].get(k)}
        raise ValueError(f'Warmup recipe differs from the winning source: {difference}')
    if signature['sources']!=old['signature']['sources']:raise ValueError('Original warmup sources changed')
    recipe,critic=arm_settings('baseline',origin=600,horizon=660,diagnostics_every=15)
    if asdict(recipe)!=winning['recipe'] or critic!=winning['critic']:
        raise ValueError('Bounded continuation differs from winner')
    manifest=dict(version='uni16-gan-v1',created_utc=now(),gpu=1,
        source_files=signed_files(),warmup_settings=signature['settings'],bounded_recipe=asdict(recipe),
        winner=dict(path=str(WINNER),manifest_sha256=sha(WINNER/'manifest.json'),
                    warmup_state_sha256=sha(ORIGINAL)),
        initialization='Fresh zero-output LoRA per slider; no pretrained concept weights reused',
        intervention='Original 0 to 600, then bounded baseline 601 to 660 with own complete state',
        differences_from_winner=['New prompt pairs and original longer lyrics; optimization code and recipe unchanged'],
        evaluation=dict(rows=[0,1],seeds=[7,23],duration=20,checkpoints=[600,660],
            reserved_rows=[2,3],reserved_seeds=[101,303],
            description='New lyrics and arrangements; matched off and target-caption controls; first samples retained'),
        selection='660 is the requested endpoint; score 600 alongside it. Do not automatically promote or claim all genres win.',
        sources=[
            'https://github.com/MiniMax-AI/MiniMax-Music3/blob/main/skills/music-caption-rewriter/SKILL.md',
            'https://luminatedata.com/blog/luminate-2026-midyear-report-trends-in-music-television-film/',
            'https://www.ifpi.org/ifpis-global-study-finds-were-listening-to-more-music-in-more-ways-than-ever/'])
    if (WORK/'manifest.json').exists():
        prior=read(WORK/'manifest.json');manifest['created_utc']=prior['created_utc']
        for key in ('amendments','presentation_files'):
            if key in prior:manifest[key]=prior[key]
        if prior!=manifest:raise ValueError('Frozen campaign changed; use a new version')
    else:
        for path,expected in manifest['source_files'].items():
            src=Path(path); dst=WORK/'provenance'/src.relative_to(ROOT.parent)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            assert sha(dst)==expected
        write(WORK/'manifest.json',manifest)
    print('Preflight passed: all prompts clean; both training stages exactly match winning settings and sources.',flush=True)
    return manifest


def verify_frozen():
    for path,expected in read(WORK/'manifest.json')['source_files'].items():
        if not Path(path).is_file() or sha(path)!=expected:
            raise CampaignBlocked(f'Frozen campaign file changed: {path}')


def audit_checkpoint(weights, state, target, phase):
    import torch
    from safetensors.torch import load_file
    meta=read(weights.with_suffix('.json'))
    blob=torch.load(state,map_location='cpu',weights_only=True,mmap=True)
    completed=blob['completed_updates'] if phase=='warmup' else blob['completed']
    if meta['steps']!=target or completed!=target:raise ValueError('Incomplete checkpoint')
    values=load_file(str(weights))
    expected=blob['modules']['lora'] if phase=='warmup' else blob['network']
    if values.keys()!=expected.keys():raise ValueError('Checkpoint topology mismatch')
    for key,value in values.items():
        if not torch.isfinite(value).all() or not torch.equal(value,expected[key]):
            raise ValueError(f'Nonfinite or mismatched checkpoint tensor: {key}')
    from app.rewriter import _artist_name_hit
    if _artist_name_hit('',json.dumps(meta)):raise ValueError('Name validation rejected checkpoint metadata')
    if phase=='bounded':
        if meta['gan_v2']['recipe']!=read(WORK/'manifest.json')['bounded_recipe']:
            raise ValueError('Wrong bounded recipe')
        if [r['step'] for r in blob['history']]!=list(range(601,661)):
            raise ValueError('Bounded continuation has an unexpected history')
    result=dict(steps=target,tensors=len(values),finite=True,state_export_identical=True,
                weights_sha256=sha(weights),state_sha256=sha(state),checked_utc=now())
    write(weights.parent/'checkpoint-audit.json',result)
    return result


class Campaign:
    def __init__(self):
        self.catalog=read(WORK/'catalog.json')
        self.status=read(WORK/'status.json') if (WORK/'status.json').exists() else dict(
            status='running',phase='Preparing',sliders={},created_utc=now())

    def save(self, phase=None):
        if phase:self.status['phase']=phase;print(phase,flush=True)
        self.status['updated_utc']=now()
        write(WORK/'status.json',self.status)
        try:
            import report
            importlib.reload(report)
            report.publish(copy.deepcopy(self.catalog),copy.deepcopy(self.status),PAGE)
        except Exception as exc:
            print(f'Listening page refresh failed; training continues: {exc}',flush=True)

    def command(self, command, log, gpu=True):
        verify_frozen()
        if shutil.disk_usage(ROOT).free<5*1024**3:
            raise CampaignBlocked('Less than 5 GiB available; checkpoints retained')
        if gpu:
            while True:
                used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used',
                         '--format=csv,noheader,nounits'],text=True).strip())
                if used<512:break
                self.save(f'Waiting for GPU 1 ({used} MiB in use)');time.sleep(15)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1' if gpu else '',HF_HUB_OFFLINE='1',
            HF_HOME='/ml2/music/.cache/huggingface',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTHONPATH=str(ROOT),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        write(log.with_suffix('.command.json'),dict(argv=list(map(str,command)),gpu=1 if gpu else None,started_utc=now()))
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as handle:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            self.status['child_pid']=child.pid;self.status['current_log']=str(log);self.save()
            while child.poll() is None:
                time.sleep(15);self.save()
        self.status.pop('child_pid',None)
        if child.returncode:raise RuntimeError(f'Command exited {child.returncode}: {log}')

    def stage(self, item, phase, source=None):
        info=self.status['sliders'].setdefault(item['id'],{})
        stage=info.setdefault(phase,dict(status='pending',attempt=0))
        target=600 if phase=='warmup' else 660
        if stage.get('status')=='complete':
            if sha(stage['weights'])!=stage['audit']['weights_sha256']:raise ValueError('Completed weights changed')
            return Path(stage['weights']),Path(stage['state'])
        verify_frozen()
        resume=None
        if stage.get('directory'):
            run=Path(stage['directory'])
            state=run/(f'{run.name}_state.pt' if phase=='warmup' else 'state.pt')
            weights=run/(f'{run.name}_last.safetensors' if phase=='warmup' else f'{run.name}_step660.safetensors')
            if state.exists():
                import torch
                blob=torch.load(state,map_location='cpu',weights_only=True,mmap=True)
                completed=blob['completed_updates'] if phase=='warmup' else blob['completed']
                del blob
                if completed==target and weights.exists():
                    audit=audit_checkpoint(weights,state,target,phase)
                    stage.update(status='complete',weights=str(weights),state=str(state),audit=audit);self.save()
                    return weights,state
                if completed>=target:raise ValueError('Completed state is missing its inference export')
                resume=state
        stage['attempt']+=1
        run=MODELS/f'{item["id"]}-{phase}{target}-a{stage["attempt"]:02d}'
        if run.exists():raise ValueError(f'Refusing to overwrite {run}')
        stage.update(status='running',directory=str(run),resume=str(resume) if resume else None)
        info.pop('error',None)
        info['status']='training';self.save(f'Training {item["label"]}: {phase} to {target}')
        args=([str(ROOT/'conceptmod/textsliders/train_lm_slider_music3.py')]+warmup_args(item,run,resume)
              if phase=='warmup' else bounded_args(item,run,source,resume))
        self.command([PY,'-u',*args],WORK/'logs'/f'{run.name}.log')
        state=run/(f'{run.name}_state.pt' if phase=='warmup' else 'state.pt')
        weights=run/(f'{run.name}_last.safetensors' if phase=='warmup' else f'{run.name}_step660.safetensors')
        audit=audit_checkpoint(weights,state,target,phase)
        stage.update(status='complete',weights=str(weights),state=str(state),audit=audit);self.save()
        return weights,state

    def evaluate(self,item,weights):
        info=self.status['sliders'][item['id']]
        if info.get('status')=='complete':return
        folders=[]
        for row in [0,1]:
            folder=PAGE/item['id']/f'row-{row}';folders.append(folder)
            marker=folder/'render-complete.json'
            if marker.exists():continue
            self.save(f'Rendering {item["label"]}: reserved prompt {row+1}, seeds 7 and 23')
            self.command([PY,'-u',ROOT/'analysis/gan_bcap/render_v2.py','--weights',*weights,
                '--out',folder,'--prompts',item['eval_prompts'],'--row',str(row),
                '--seeds','7','23','--duration','20'],WORK/'logs'/f'{item["id"]}-render-{row}.log')
            write(marker,dict(completed_utc=now()))
        score=WORK/'scores'/f'{item["id"]}.json'
        if not score.exists() or read(score).get('status')!='complete':
            self.save(f'Scoring {item["label"]}: all matched clips, including lyric preservation')
            self.command([PY,'-u',WORK/'score.py','--id',item['id'],'--folders',*folders,
                          '--output',score],WORK/'logs'/f'{item["id"]}-score.log',gpu=False)
        info.update(status='complete',score_report=str(score),ranking=read(score)['ranking']);self.save()

    def blocked(self, exc, item=None):
        self.status.update(status='blocked',blocked_reason=str(exc))
        self.status.pop('child_pid',None)
        if item is not None:
            info=self.status['sliders'].setdefault(item['id'],{})
            info.update(status='blocked',error=str(exc))
            for phase in ('warmup','bounded'):
                if info.get(phase,{}).get('status')=='running':info[phase]['status']='blocked'
        self.save(f'Queue paused: {exc}; completed checkpoints and queued jobs retained')
        return 1

    def run(self):
        try:verify_frozen()
        except CampaignBlocked as exc:return self.blocked(exc)
        self.status.pop('blocked_reason',None)
        self.status['status']='running';self.save('Running the 16-slider training and listening queue')
        # Failures stay explicit while independent sliders continue. A service
        # restart resumes saved states into new attempt directories.
        for item in self.catalog['sliders']:
            try:
                warm,state=self.stage(item,'warmup')
                final,_=self.stage(item,'bounded',state)
                self.evaluate(item,[warm,final])
            except CampaignBlocked as exc:
                return self.blocked(exc,item)
            except Exception as exc:
                self.status['sliders'].setdefault(item['id'],{}).update(status='failed',error=str(exc))
                self.save(f'{item["label"]} failed: {exc}')
        all_done=all(self.status['sliders'].get(x['id'],{}).get('status')=='complete' for x in self.catalog['sliders'])
        if all_done:
            registry=dict(root=str(ROOT/'models'),transformer_energy=1.5,transformer_energy_max=3.,
                          language_model_energy=2.,language_model_energy_max=4.,sliders=[])
            for item in self.catalog['sliders']:
                weights=Path(self.status['sliders'][item['id']]['bounded']['weights'])
                registry['sliders'].append(dict(id=item['id'],label_minus='Off',label_plus=item['label'],
                    description=item['description'],range=1.,unipolar=True,
                    components=[dict(weights=str(weights.relative_to(ROOT/'models')),ratio=1.)]))
            write(WORK/'studio-catalog.json',registry)
            self.save('All 16 trained and individually scored; rendering the declared style combinations')
            try:
                self.command([PY,'-u',WORK/'combos.py'],WORK/'logs/combos.log')
                self.status['combos']='rendered_requires_listening'
            except CampaignBlocked as exc:
                return self.blocked(exc)
            except Exception as exc:
                self.status['combos']='failed';self.status['combo_error']=str(exc);all_done=False
        self.status['status']='complete' if all_done else 'incomplete'
        self.save('Training and listening package complete; new catalog prepared for review' if all_done
                  else 'Queue finished with explicit failures; successful weights retained')
        return 0 if all_done else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare();Campaign().save('Validated and ready to train');return 0
        return Campaign().run()


if __name__=='__main__':
    raise SystemExit(main())
