"""One retained engineering rerender of an existing training Off recording."""
import argparse
from pathlib import Path
import os
import signal
import time
from .core import read,write,immutable,sha,locked,IntegrityError,Store
from .setup import verify
from .resources import gpu_lease
from .acoustic_tail import TailCapture,replay_latents,decode


def capture(home,folder,observation):
    import torch
    import numpy as np
    import soundfile as sf
    from ..reward_search.renderer import SearchRenderer
    from ..reward_sliders.render import pcm_sha
    from ..reward_sliders.specs import save_tensor
    home=Path(home).resolve();folder=Path(folder).resolve();game,manifest=verify(home);row=read(observation)
    family=next(f for f in manifest['families'] if f['family']==row['family'])
    if family['split']!='train' or row['arm']['name']!='off' or row['provenance']['physical_gpu']!=1:raise IntegrityError('Acoustic audit requires a training Off control on GPU 1')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or sha(row['audio'])!=row['audio_sha256']:raise IntegrityError('Wrong GPU or modified reference')
    protocol=dict(source_observation=str(Path(observation).resolve()),source_observation_sha256=sha(observation),reference_audio_sha256=sha(row['audio']),
        reference_pcm_sha256=pcm_sha(row['audio']),family=family,seed=row['seed'],physical_gpu=1,
        source_hashes={str(p.resolve()):sha(p) for p in (Path(__file__),Path(__file__).with_name('acoustic_tail.py'))},
        budget=dict(new_engineering_clips=1,optimizer_updates=0),required=dict(capture_pcm_exact=True,tail_latents_exact=True,tail_pcm_exact=True),
        gradient_scope='last two of 30 flow steps per chunk; earlier states and overlap fixed')
    immutable(folder/'capture-protocol.json',protocol)
    if (folder/'capture-result.json').exists():return read(folder/'capture-result.json')
    if (folder/'capture-attempt.json').exists():raise IntegrityError('Interrupted acoustic capture retained; explicit amendment required')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    renderer=None
    with locked(folder/'capture.lock'),gpu_lease(home,1):
        immutable(folder/'capture-attempt.json',dict(started_unix=time.time(),new_engineering_clip_budget=1))
        output=folder/'off-capture.wav';timing={}
        try:
            renderer=SearchRenderer(folder,manifest,1);output=folder/'off-capture.wav'
            with TailCapture(renderer.pipe) as collector:
                timing,_=renderer.generate(family,row['seed'],output,duration=game['duration_seconds'])
            data=collector.result();path=folder/'acoustic-tail.pt';save_tensor(path,data)
            captured_exact=pcm_sha(output)==protocol['reference_pcm_sha256']
            write(folder/'capture-observation.json',dict(audio=str(output),audio_sha256=sha(output),timing=timing,
                physical_gpu=1,reference_audio=row['audio'],capture_exact=captured_exact,new_clips=1,
                independent_quality_observation=False,trajectory=str(path),trajectory_sha256=sha(path)))
            if not captured_exact:raise IntegrityError('Acoustic capture changed decoded PCM')
            with torch.no_grad():
                latents,checks=replay_latents(renderer.pipe.transformer,data,record_checks=True)
                audio=decode(renderer.pipe.vocoder,latents,data['latent_hop_length'])
            expected,rate=sf.read(output,dtype='float32',always_2d=True)
            exact_wave=np.array_equal(audio[0].T.cpu().numpy(),expected)
            result=dict(passed=all(c['exact'] for c in checks) and exact_wave,capture_pcm_exact=captured_exact,tail_latent_checks=checks,tail_pcm_exact=exact_wave,
                sample_rate=rate,audio_sha256=sha(output),capture_sha256=sha(path),new_clips=1,optimizer_updates=0,
                interpretation='Numerical engineering parity only; reference acoustic prefix and overlap are fixed, no candidate or quality gain claimed')
            immutable(folder/'capture-result.json',result);Store(home).event('acoustic_tail_capture_audit',passed=result['passed'],new_engineering_clips=1)
            if not result['passed']:raise IntegrityError('Truncated acoustic replay failed exact parity')
            return result
        finally:
            if output.exists() and not (folder/'capture-observation.json').exists():
                write(folder/'capture-observation.json',dict(audio=str(output),audio_sha256=sha(output),timing=timing,
                    physical_gpu=1,reference_audio=row['audio'],new_clips=1,status='incomplete_engineering_audit',independent_quality_observation=False))
            if renderer is not None:
                renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
                exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in renderer.host._merge_state(renderer.device).pristine.items())
                write(folder/'capture-off-restoration.json',dict(exact=exact,physical_gpu=1))
                if not exact:raise IntegrityError('Acoustic capture failed Off restoration')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--observation',required=True)
    a=p.parse_args();print(__import__('json').dumps(capture(a.home,a.folder,a.observation),indent=2),flush=True)
