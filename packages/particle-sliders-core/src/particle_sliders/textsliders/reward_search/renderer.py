"""GPU-selectable research host, preserving the frozen v1 generator and hooks."""
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import time

import torch

from ..reward_sliders.specs import WORKSPACE, Observation, RewardSpec, digest, sha, read_json, write_json, save_tensor
from ..reward_sliders.render import ResearchRenderer, studio_snapshot, pcm_sha
from ..reward_sliders.reward import CEReward
from ..reward_sliders.directions import RewardTeacher
from ..reward_sliders.evaluate import component
from ..reward_sliders.experiment import verify
from . import queue


class SearchRenderer(ResearchRenderer):
    def __init__(self, run, manifest, gpu):
        if gpu not in (0,1) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):
            raise RuntimeError('Expose exactly the assigned physical GPU')
        self.run,self.manifest,self.gpu=Path(run),manifest,gpu
        self.device='cuda:0'
        sys.path.insert(0,str(WORKSPACE))
        from app import generator
        self.host=generator
        torch.set_num_threads(4);torch.cuda.set_device(0)
        self.ownership=studio_snapshot()
        if gpu==0 and self.ownership['loaded']:
            raise RuntimeError('GPU 0 remains owned by the studio')
        started=time.monotonic()
        self.pipe=generator._ensure_loaded(self.device)
        self.load_seconds=time.monotonic()-started
        if generator._apply_mode(self.device)!='merge':raise RuntimeError('Use ordinary merged adapters')
        lm=self.pipe.language_model
        if len(lm.model.layers)!=36 or lm.config.hidden_size!=4096:raise RuntimeError('Unexpected composer topology')
        self.pipe.set_progress_bar_config(disable=True)

    def observe(self, family, seed, arm, scorer, teacher=None, capture=False, extra=None, full_song=False):
        if full_song:raise ValueError('This follow-up has a fixed 20-second quality objective')
        ident=f"{family['family']}-s{seed}-{arm['name']}"
        path=self.run/'observations'/f'{ident}.json'
        provenance=dict(manifest_sha256=digest(self.manifest),family_sha256=digest(family),
            physical_gpu=self.gpu,gpu_name=torch.cuda.get_device_name(),components=self.components(family,extra),
            sampler=self.manifest['sampler'],base_sha256=digest(self.manifest['model_hashes']),
            capture_spec_sha256=digest(self.manifest['capture_spec']),studio=self.ownership)
        if path.exists():
            old=read_json(path)
            compatible=[provenance['manifest_sha256'],*self.manifest.get('compatible_manifest_sha256',[])]
            if old['arm']!=arm or old['provenance']['manifest_sha256'] not in compatible:
                raise ValueError('Observation identity changed')
            for label in ('audio','trajectory'):
                if old.get(label) and sha(old[label])!=old[label+'_sha256']:raise ValueError('Corrupted '+label)
            if old['status']=='running':
                old.update(status='failed',error='Interrupted output retained without reroll');write_json(path,old)
            return old
        row=Observation(ident,family['family'],family['split'],seed,
                        digest({k:family[k] for k in ('caption','lyrics','style_multipliers')}),arm,
                        status='running',provenance=provenance).json()
        write_json(path,row)
        print('RENDER '+ident,flush=True)
        try:
            audio=self.run/'audio'/f'{ident}.wav'
            timing,trajectory=self.generate(family,seed,audio,teacher=teacher,capture=capture,extra=extra)
            row.update(audio=str(audio),audio_sha256=sha(audio),timing=timing)
            if trajectory is not None:
                target=self.run/'trajectories'/f'{ident}.pt';save_tensor(target,trajectory)
                row.update(trajectory=str(target),trajectory_sha256=sha(target))
            reward=scorer.measure(audio)
            row.update(reward=reward,status='complete' if reward['valid'] else 'failed',
                       error=reward.get('error'))
        except Exception as error:
            row.update(status='failed',error=repr(error))
        write_json(path,row)
        print(f"RESULT {ident} {row['status']} CE={(row.get('reward') or {}).get('scalar')}",flush=True)
        return row

    def parity(self, family, root):
        path=Path(root)/'audit'/f'gpu{self.gpu}-capture-parity.json'
        if path.exists():
            assert read_json(path)['passed'];return
        plain=path.with_name(f'gpu{self.gpu}-plain.wav')
        captured=path.with_name(f'gpu{self.gpu}-capture.wav')
        self.generate(family,8137,plain,capture=False)
        _,trajectory=self.generate(family,8137,captured,capture=True)
        equal=pcm_sha(plain)==pcm_sha(captured)
        write_json(path,dict(passed=equal,physical_gpu=self.gpu,seed=8137,
            plain_sha256=sha(plain),capture_sha256=sha(captured),pcm_sha256=pcm_sha(plain),
            cfg_branches=int(trajectory['frame_embeds'].shape[0]),
            feedback_positions=int(trajectory['frame_embeds'].shape[1])))
        if not equal:raise RuntimeError('Live capture parity failed')


def worker(run,gpu):
    run=Path(run);manifest=read_json(run/'manifest.json');verify(manifest)
    stop=False
    def cancel(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    renderer=SearchRenderer(run,manifest,gpu)
    scorer=CEReward(RewardSpec(**manifest['reward_spec']))
    renderer.parity(next(f for f in manifest['families'] if f['split']=='train'),run)
    try:
        while not stop and not (run/f'stop-gpu{gpu}').exists():
            job=queue.claim(run,gpu)
            if job is None:
                time.sleep(3);continue
            renderer.run=run/'stages'/job['stage']
            errors=[]
            try:
                verify(manifest)
                for arm in job['payload']['arms']:
                    teacher=None;extra=None
                    if arm.get('teacher'):
                        if sha(arm['teacher'])!=arm['teacher_sha256']:raise ValueError('Teacher changed')
                        values=torch.load(arm['teacher'],map_location='cpu',weights_only=True)
                        values['coefficient']=arm['coefficient'];teacher=RewardTeacher(**values)
                    if arm.get('checkpoint'):
                        if sha(arm['checkpoint'])!=arm['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                        extra=component(arm['checkpoint'],arm['coefficient'])
                    row=renderer.observe(job['payload']['family'],job['payload']['seed'],arm,scorer,
                                         teacher=teacher,extra=extra,capture=job['payload'].get('capture',False))
                    if row['status']!='complete':errors.append(row['id']+': '+str(row['error']))
                queue.finish(run,job['id'],'; '.join(errors) or None)
            except Exception as error:
                queue.finish(run,job['id'],repr(error));raise
    finally:
        renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
        exact=all(torch.equal(m.weight.detach().cpu(),p)
                  for m,p in renderer.host._merge_state(renderer.device).pristine.items())
        write_json(run/'audit'/f'gpu{gpu}-off-restoration.json',dict(exact=exact))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    worker(args.run_dir,args.gpu)
