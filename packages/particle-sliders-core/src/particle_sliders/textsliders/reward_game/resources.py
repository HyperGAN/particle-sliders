"""Per-game GPU leases; only the owning run removes its studio override."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .core import Store, BusyError, locked, write, digest


def studio():
    with urllib.request.urlopen('http://127.0.0.1:7860/api/studio', timeout=15) as response:
        return json.load(response)


def systemctl(*args):
    return subprocess.run(['systemctl', '--user', *args], check=True, capture_output=True, text=True)


def wait_studio():
    for _ in range(90):
        try: return studio()
        except (OSError, ValueError): time.sleep(2)
    raise RuntimeError('Studio unavailable after restart')


def keep_settings(state):
    return {k:state['keep'].get(k) for k in ('enabled','prompt','duration','sliders','energy','randomize_sliders')}


def restore_keep(settings):
    request=urllib.request.Request('http://127.0.0.1:7860/api/keep',data=json.dumps(settings).encode(),
                                   headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request,timeout=15) as response:json.load(response)


@contextmanager
def gpu_lease(home, gpu):
    store=Store(home)
    with locked(Path('/tmp')/f'music-reward-game-physical-gpu{gpu}.lock'):
        if gpu == 1:
            began=time.monotonic();store.event('gpu_lease_acquired',physical_gpu=gpu)
            try:yield
            finally:store.event('gpu_lease_released',physical_gpu=gpu,elapsed_seconds=time.monotonic()-began)
            return
        folder=Path.home()/'.config/systemd/user/music-studio.service.d'
        override=folder/'30-reward-game-lease.conf'
        content=f'# Owner: {store.home}\n[Service]\nEnvironment="CUDA_VISIBLE_DEVICES="\nEnvironment=MUSIC_WORKERS=1\n'
        borrowed=False
        try:
            if override.exists():
                if override.read_text()!=content: raise BusyError('Studio lease belongs to another run')
                # An interrupted parent can recover only its own exact lease.
                borrowed=True
            else:
                if (folder/'20-reward-search.conf').exists(): raise BusyError('Another research run holds the studio override')
                while True:
                    state=studio()
                    if not state['active'] and not state['queued'] and not state['keep']['enabled']:
                        break
                    store.status('waiting_for_studio_queue', research_complete=False, physical_gpu=0)
                    time.sleep(5)
                snapshot=dict(state=state,unit=systemctl('cat','music-studio.service').stdout,time_unix=time.time())
                settings=keep_settings(state)
                write(store.home/'resources'/f'studio-before-{time.time_ns()}.json',snapshot)
                write(store.home/'resources'/'studio-lease-settings.json',settings)
                state=studio()
                if state['active'] or state['queued'] or state['keep']['enabled']:
                    raise BusyError('Studio received new work; retry after it drains')
                folder.mkdir(parents=True,exist_ok=True)
                with override.open('x') as stream: stream.write(content)
                borrowed=True
                systemctl('daemon-reload');systemctl('restart','music-studio.service')
                state=wait_studio()
                if state['loaded']: raise RuntimeError('Studio failed to release GPU 0')
                restore_keep(settings)
                store.event('gpu0_borrowed', playback_available=True)
            yield
        finally:
            if borrowed:
                if not override.exists() or override.read_text()!=content:
                    raise BusyError('Studio lease changed; refusing to remove another owner')
                try:settings=keep_settings(studio())
                except Exception:
                    from .core import read
                    settings=read(store.home/'resources'/'studio-lease-settings.json')
                override.unlink();systemctl('daemon-reload');systemctl('restart','music-studio.service')
                state=wait_studio();restore_keep(settings);state=studio();store.event('studio_restored',state=state)
                write(store.home/'resources'/'studio-restored.json',dict(state=state,time_unix=time.time()))
