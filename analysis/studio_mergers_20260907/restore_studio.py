"""Restore the original two-GPU studio only after its live queue becomes idle."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

WORK = Path(__file__).resolve().parent
OVERRIDE = Path('/run/user/1000/systemd/user/music-studio.service.d/90-merger-screen.conf')
EXPECTED = '[Service]\nEnvironment="CUDA_VISIBLE_DEVICES=0" "MUSIC_WORKERS=1"\n'
URL = 'http://127.0.0.1:7860'


def read_state():
    return json.load(urllib.request.urlopen(URL+'/api/studio', timeout=10))


def write(data):
    temporary = WORK/'restore-status.json.tmp'
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(WORK/'restore-status.json')


def busy(s):
    return bool(s['active'] or s['queued'] or s['keep']['enabled'])


def render_running():
    status = subprocess.check_output(['systemctl', '--user', 'show',
        'studio-mergers-render-20260907.service', '-p', 'ActiveState', '--value'], text=True).strip()
    return status in ('active', 'activating', 'reloading', 'deactivating')


def main():
    while True:
        if not OVERRIDE.exists() or OVERRIDE.read_text() != EXPECTED:
            write(dict(status='stopped_configuration_changed', explanation='The study no longer owns the GPU override; no restart performed.'))
            return
        if render_running():
            write(dict(status='waiting_for_study', checked=time.time()))
            time.sleep(10)
            continue
        try:
            s = read_state()
        except OSError as exc:
            write(dict(status='waiting_for_studio', error=str(exc)))
            time.sleep(10)
            continue
        if busy(s):
            write(dict(status='waiting_for_idle', active=len(s['active']), queued=len(s['queued']),
                       keep_enabled=s['keep']['enabled'], checked=time.time()))
            time.sleep(10)
            continue
        # Check again immediately before restart, rather than relying on the
        # earlier poll. This service never cancels jobs or disables Keep.
        time.sleep(1)
        s = read_state()
        if busy(s):
            continue
        processes = subprocess.check_output([
            'nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'
        ], text=True).strip()
        if processes:
            write(dict(status='waiting_for_gpu1', pids=processes, checked=time.time()))
            time.sleep(10)
            continue
        (WORK/'studio-before-restore.json').write_text(json.dumps(s, indent=2)+'\n')
        write(dict(status='restarting_idle_studio', started=time.time()))
        OVERRIDE.unlink()
        subprocess.run(['systemctl','--user','daemon-reload'], check=True)
        subprocess.run(['systemctl','--user','restart','music-studio.service'], check=True)
        keep = {k:v for k,v in s['keep'].items()
                if k in ('enabled','prompt','duration','sliders','energy','randomize_sliders')}
        for _ in range(90):
            try:
                req = urllib.request.Request(URL+'/api/keep', data=json.dumps(keep).encode(),
                                             headers={'Content-Type':'application/json'}, method='POST')
                json.load(urllib.request.urlopen(req, timeout=5))
                restored = read_state()
                assert restored['workers'] == 2
                assert all(restored['keep'].get(k) == v for k,v in keep.items())
                (WORK/'studio-restored.json').write_text(json.dumps(restored, indent=2)+'\n')
                write(dict(status='restored', workers=2, keep_settings_preserved=True, finished=time.time()))
                print('Two-GPU studio restored after queue became idle.', flush=True)
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError('Studio did not respond after restart; preserved settings are in studio-before-restore.json')


if __name__ == '__main__':
    main()
