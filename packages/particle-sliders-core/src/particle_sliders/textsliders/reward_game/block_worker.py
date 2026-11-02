"""A batch of declared jobs on exactly one physical GPU, using frozen rendering."""
import argparse
import os
from pathlib import Path
import signal
import time

from .core import Store, read, write, sha, digest, IntegrityError, locked
from .setup import verify
from .block_artifact import checkpoint
from .block_identity import generation_identity


def run(home, jobs_path, gpu):
    import torch
    from .block_renderer import AcousticBlockRenderer as SearchRenderer
    from ..reward_sliders.reward import CEReward
    from ..reward_sliders.specs import RewardSpec
    from .block_artifact import component
    from ..reward_sliders.render import pcm_sha
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu): raise ValueError('Wrong physical GPU')
    def cancel(signum, frame): raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    store=Store(home);game,manifest=verify(home);manifest=dict(manifest,host_energy_by_kind=game['host_energy_by_kind']);jobs=read(jobs_path);renderer=None
    scorer=CEReward(RewardSpec(**game['reward_spec']));reward_hash=digest(game['reward_spec'])
    try:
        for job in jobs:
            case=job['case'];candidate=checkpoint(job['candidate']['path'],job['candidate']['multiplier'])
            identity=generation_identity(game,manifest,case,candidate)
            key=digest(identity)
            if key != job['key'] or case['physical_gpu'] != gpu: raise IntegrityError('Worker job identity mismatch')
            previous=store.observation(key)
            if previous and not job.get('engineering_audit'): continue
            if renderer is None: renderer=SearchRenderer(store.home,manifest,gpu)
            extra=component(candidate['path'],candidate['multiplier']) if candidate['path'] else None
            if job.get('engineering_audit'):
                folder=store.home/'audit'/'off-pcm-v1'
                with locked(folder/'claim.lock'):
                    result_path=folder/'result.json'
                    if result_path.exists():
                        if not read(result_path)['passed']: raise IntegrityError('Recorded engineering parity failed')
                        continue
                    if (folder/'attempt.json').exists(): raise IntegrityError('Interrupted PCM audit retained; explicit amendment required')
                    write(folder/'attempt.json',dict(job=job,started_unix=time.time()))
                    path=folder/'off-rerender.wav'
                    timing,_=renderer.generate(case['family'],case['seed'],path,duration=game['duration_seconds'])
                    saved=case['controls']['off']['audio'];equal=pcm_sha(path)==pcm_sha(saved)
                    write(result_path,dict(passed=equal,physical_gpu=gpu,case=case['id'],timing=timing,
                                          audio=str(path),audio_sha256=sha(path),pcm_sha256=pcm_sha(path),
                                          reference_pcm_sha256=pcm_sha(saved),new_clips=1,independent_quality_observation=False))
                    store.event('off_pcm_audit',passed=equal,new_clips=1)
                    if not equal: raise IntegrityError('Off decoded PCM differs from frozen control')
                continue
            with store.claim(key) as folder:
                path=folder/'audio.wav'
                row=dict(key=key,identity=identity,status='running',audio=str(path),audio_sha256=None,
                         physical_gpu=gpu,started_unix=time.time(),reward=None)
                write(folder/'observation.json',row)
                try:
                    timing,_=renderer.generate(case['family'],case['seed'],path,extra=extra,duration=game['duration_seconds'])
                    row.update(status='rendered',audio_sha256=sha(path),timing=timing)
                    write(folder/'observation.json',row)
                    reward=store.score(row,reward_hash)
                    if reward is None: reward=store.save_score(scorer.measure(path),reward_hash)
                    row.update(reward=reward,status='complete' if reward['valid'] else 'invalid_output',error=reward.get('error'))
                    if not reward['valid'] and not any(message in reward.get('error','') for message in
                        ('Silent waveform','Empty or nonfinite waveform','Short-screen output','Nonfinite scorer result','CE outside declared range')):
                        row['status']='scoring_error'
                except BaseException as exc:
                    row.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'engineering_error',error=repr(exc))
                    if path.exists(): row['audio_sha256']=sha(path)
                    write(folder/'observation.json',row);store.event('case_failed',key=key,status=row['status'],error=row['error'])
                    raise
                finally:
                    row['finished_unix']=time.time();write(folder/'observation.json',row)
                store.event('case_finished',key=key,status=row['status'],ce=(row.get('reward') or {}).get('scalar'),
                            new_clips=1,render_seconds=row.get('timing',{}).get('total_seconds',0))
                print(f"CASE {case['id']} {row['status']} CE={(row.get('reward') or {}).get('scalar')}",flush=True)
    finally:
        if renderer is not None:
            renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
            exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in renderer.host._merge_state(renderer.device).pristine.items())
            write(store.home/'audit'/f'off-restoration-{Path(jobs_path).stem}-gpu{gpu}.json',dict(exact=exact,physical_gpu=gpu))
            if not exact: raise IntegrityError('Base weights failed exact Off restoration')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--jobs',required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();run(a.home,a.jobs,a.gpu)
