"""Borrow GPU 0 after the studio queue drains, preserving playback and settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from ..reward_sliders.specs import WORKSPACE, write_json, read_json
from .setup import DEFAULT_RUN

OVERRIDE=Path.home()/'.config/systemd/user/music-studio.service.d/20-reward-search.conf'
CONTENTS='[Service]\nEnvironment="CUDA_VISIBLE_DEVICES="\nEnvironment=MUSIC_WORKERS=1\n'


def studio():
    with urllib.request.urlopen('http://127.0.0.1:7860/api/studio',timeout=15) as response:
        return json.load(response)


def keep_settings(value):
    return {k:value.get(k) for k in ('enabled','prompt','duration','sliders','energy','randomize_sliders')}


def restore_keep(settings):
    request=urllib.request.Request('http://127.0.0.1:7860/api/keep',
        data=json.dumps(settings).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request,timeout=15) as response:json.load(response)


def wait_studio():
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        try:return studio()
        except Exception:time.sleep(2)
    raise RuntimeError('Studio did not respond after restart')


def worker_unit(gpu):return f'music-reward-search-gpu{gpu}-20260908.service'


def active(unit):
    return subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()=='active'


def start_worker(run,gpu):
    unit=worker_unit(gpu)
    if active(unit):return
    (Path(run)/f'stop-gpu{gpu}').unlink(missing_ok=True)
    exists=subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','--value'],text=True).strip()!='not-found'
    if exists:
        # A stopped transient unit can disappear between the query and the start.
        # Resetting a failed transient unit also releases its definition, so start it directly.
        started=subprocess.run(['systemctl','--user','start',unit],capture_output=True,text=True)
        if started.returncode==0:return
        if 'not found' not in started.stderr:started.check_returncode()
    subprocess.run(['systemd-run','--user','--unit='+unit.removesuffix('.service'),
        '--description=Reward CE search rendering on physical GPU '+str(gpu),
        '--property=WorkingDirectory='+str(WORKSPACE/'sliders-conceptmod'),'--property=TimeoutStopSec=1200',
        '--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),'--setenv=HF_HOME='+str(WORKSPACE/'.cache/huggingface'),
        '--setenv=HF_HUB_CACHE='+str(WORKSPACE/'.cache/huggingface/hub'),'--setenv=HF_HUB_OFFLINE=1',
        '--setenv=PYTHONPATH='+str(WORKSPACE)+':'+str(WORKSPACE/'sliders-conceptmod'),
        '--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',sys.executable,'-u','-m',
        'conceptmod.textsliders.reward_search.renderer','--run-dir',str(run),'--gpu',str(gpu)],check=True)


def stop_worker(run,gpu):
    (Path(run)/f'stop-gpu{gpu}').touch()
    unit=worker_unit(gpu)
    loaded=subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','--value'],text=True).strip()
    if loaded!='not-found':subprocess.run(['systemctl','--user','stop',unit],check=True)


def monitor(run):
    run=Path(run);borrowed=False
    snapshot=run/'audit/studio-settings-before.json'
    try:
        if OVERRIDE.exists():
            if OVERRIDE.read_text()!=CONTENTS:raise RuntimeError('GPU override belongs to another configuration')
            borrowed=True
        while not (run/'experiment-done').exists():
            if not borrowed:
                state=studio()
                if not state['active'] and not state['queued'] and not state['keep']['enabled']:
                    write_json(snapshot,dict(keep=keep_settings(state['keep']),time_unix=time.time(),
                                            active_count=0,queued_count=0))
                    # Recheck immediately before a restart that would clear the in-memory queue.
                    state=studio()
                    if state['active'] or state['queued'] or state['keep']['enabled']:
                        time.sleep(10);continue
                    OVERRIDE.parent.mkdir(parents=True,exist_ok=True);OVERRIDE.write_text(CONTENTS)
                    borrowed=True
                    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
                    # Keep the transient studio unit referenced throughout the transition.
                    subprocess.run(['systemctl','--user','restart','music-studio.service'],check=True)
                    wait_studio();restore_keep(keep_settings(state['keep']))
                    if studio()['loaded']:raise RuntimeError('Studio did not release GPU 0')
                    write_json(run/'audit/gpu0-borrowed.json',dict(time_unix=time.time(),
                        studio_queue_drained=True,playback_available=True,rendering_paused=True))
                    print('Studio queue drained; GPU 0 borrowed for the experiment. Playback remains available.',flush=True)
                    if not (run/'training-active').exists():start_worker(run,0)
            time.sleep(10)
    finally:
        if borrowed:
            stop_worker(run,0)
            # Training processes are owned by the campaign, which writes experiment-done after they exit.
            settings=None
            try:settings=keep_settings(studio()['keep'])
            except Exception:
                if snapshot.exists():settings=read_json(snapshot)['keep']
            if OVERRIDE.exists():
                if OVERRIDE.read_text()!=CONTENTS:raise RuntimeError('GPU override changed; refusing to remove another configuration')
                OVERRIDE.unlink()
            subprocess.run(['systemctl','--user','daemon-reload'],check=True)
            subprocess.run(['systemctl','--user','restart','music-studio.service'],check=True)
            wait_studio()
            if settings:restore_keep(settings)
            write_json(run/'audit/studio-restored.json',dict(time_unix=time.time(),physical_gpu=0,workers=1))
            print('Studio restored to GPU 0 with one worker.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();monitor(args.run_dir)
