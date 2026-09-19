"""Queue matched studio recuts on both existing workers and archive their outputs."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from common import WORK, OUTPUT, PUBLIC, sha, write, api, verify


def refresh(state):
    from report import build
    build(state, OUTPUT)


def main():
    import numpy as np
    import soundfile as sf
    lock = (WORK / 'render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads((WORK / 'screen.json').read_text())
    verify(spec)
    state = json.loads((WORK / 'renders.json').read_text())
    assert state['screen_sha256'] == sha(WORK / 'screen.json')
    if state['status'] == 'complete':
        print('Already complete', flush=True)
        return
    by_id = {j['id']: j for j in spec['jobs']}
    saved = json.loads((WORK / 'studio-before.json').read_text())['keep']
    state.update(status='running', pid=os.getpid())
    state.setdefault('started', time.time())
    write(WORK / 'renders.json', state)
    owned_keep = None
    try:
        for jid in spec['submission_order']:
            if jid in state['records']:
                continue
            job = by_id[jid]
            current = api('/api/studio')
            if current['keep']['enabled'] or current['workers'] != 2:
                raise RuntimeError('Studio configuration changed during submission')
            payload = {k: job[k] for k in ('title', 'lyrics', 'caption', 'seed', 'sliders', 'energy')}
            payload.update(prompt='Studio combination and energy study 20260907 cut ' + jid,
                           duration=job['requested_duration'], count=1)
            # Persist intent before POST; never silently retry an ambiguous submission.
            write(WORK / 'submission-pending.json', dict(id=jid, payload=payload, started=time.time()))
            result = api('/api/generate', payload)
            state['records'][jid] = dict(id=jid, label=job['label'], status='queued',
                studio_job_id=result['job_id'], submitted=time.time(), request=payload)
            write(WORK / 'renders.json', state)
            (WORK / 'submission-pending.json').unlink()
            owned_keep = {**current['keep'], 'prompt': payload['prompt'],
                          'duration': payload['duration'], 'sliders': payload['sliders'],
                          'energy': payload['energy']}
            print('QUEUED', jid, result['job_id'], flush=True)
    finally:
        # Generate updates Keep's remembered controls. Restore them immediately,
        # without restarting workers or affecting the frozen queued requests.
        if owned_keep is not None:
            current = api('/api/studio')
            if current['keep'] == owned_keep:
                payload = {k: saved[k] for k in ('enabled', 'prompt', 'duration', 'sliders',
                                                'energy', 'randomize_sliders')}
                actual = api('/api/keep', payload)['keep']
                assert actual == saved
                write(WORK / 'restore-status.json', dict(status='restored',
                    keep_settings_preserved=True, workers=2, restarted=False, time=time.time()))
            else:
                write(WORK / 'restore-status.json', dict(status='newer_user_settings_preserved',
                    explanation='Live Keep settings changed after the last study submission; left intact.'))
    refresh(state)
    while True:
        pending = False
        changed = False
        for job in spec['jobs']:
            rec = state['records'][job['id']]
            if rec['status'] in ('complete', 'error'):
                continue
            current = api('/api/jobs/' + rec['studio_job_id'])
            rec.update(studio_job=current, device=current.get('device'),
                       stage=current.get('stage'), progress=current.get('progress'))
            if current['status'] in ('error', 'cancelled'):
                rec.update(status='error', error=current.get('error') or current['status'])
                print('ERROR', job['id'], rec['error'], flush=True)
                changed = True
            elif current['status'] == 'ready':
                song = api('/api/songs/' + current['song_id'])
                for field in ('title', 'lyrics', 'caption', 'seed', 'sliders', 'energy'):
                    if song.get(field) != job[field]:
                        raise ValueError(f'Stored input mismatch: {job["id"]} {field}')
                if current.get('rewrite') is not False:
                    raise ValueError('Study unexpectedly used a rewrite')
                source = WORK.parents[2] / 'library' / song['id'] / 'song.wav'
                dest = OUTPUT / (job['id'] + '.wav')
                if not dest.exists():
                    os.link(source, dest)
                if sha(source) != sha(dest):
                    raise ValueError('Archive disagrees with library audio')
                data, rate = sf.read(dest, dtype='float32', always_2d=True)
                if not len(data) or not np.isfinite(data).all():
                    raise ValueError('Empty or nonfinite output')
                excerpt = OUTPUT / (job['id'] + '-20s.wav')
                sf.write(excerpt, data[:20*rate], rate, subtype='PCM_16')
                for path in (dest, excerpt):
                    published = PUBLIC / path.name
                    if not published.exists():
                        os.link(path, published)
                rec.update(status='complete', song=song, completed=time.time(),
                    audio=str(dest), excerpt=str(excerpt), audio_sha256=sha(dest),
                    excerpt_sha256=sha(excerpt), source_audio=str(source),
                    inspection=dict(seconds=len(data)/rate, rate=rate, channels=data.shape[1],
                        rms=float(np.sqrt(np.mean(data.astype('float64')**2))),
                        peak=float(np.abs(data).max()),
                        clipped_fraction=float(np.mean(np.abs(data) >= .999)),
                        short=len(data) < 20*rate, finite=True))
                print(f'DONE {job["id"]} {rec["device"]} {len(data)/rate:.2f}s', flush=True)
                changed = True
            else:
                rec['status'] = current['status']
                pending = True
        state['updated'] = time.time()
        write(WORK / 'renders.json', state)
        if changed:
            refresh(state)
        if not pending:
            break
        time.sleep(10)
    verify(spec)
    state.update(status='complete' if all(r['status'] == 'complete' for r in state['records'].values())
                 else 'complete_with_errors', finished=time.time())
    state['devices_used'] = sorted({r.get('device') for r in state['records'].values() if r.get('device')})
    write(WORK / 'renders.json', state)
    write(WORK / 'studio-after.json', api('/api/studio'))
    refresh(state)
    print(state['status'], state['devices_used'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (WORK / 'render-error.txt').write_text(traceback.format_exc())
        raise
