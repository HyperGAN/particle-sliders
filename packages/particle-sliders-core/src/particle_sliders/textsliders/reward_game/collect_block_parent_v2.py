"""Capture the fixed block parent on the unchanged balanced training cohort."""
import argparse
from pathlib import Path
import os
import signal
import time
from .core import read, write, immutable, sha, digest, locked, IntegrityError, Store
from .block_artifact import checkpoint, component


def run(home, folder):
    import numpy as np
    import soundfile as sf
    import torch
    from .resources import gpu_lease
    from .setup import verify
    from .block_renderer import AcousticBlockRenderer
    from .acoustic_tail import TailCapture, replay_latents, decode
    from ..reward_sliders.reward import CEReward
    from ..reward_sliders.specs import RewardSpec, save_tensor, validate_families
    home = Path(home).resolve(); folder = Path(folder).resolve()
    game, manifest = verify(home/'block-v1'); recipe = read(folder/'collection-recipe.json')
    manifest['host_energy_by_kind'] = dict(game['host_energy_by_kind'])
    if checkpoint(recipe['parent']['path'], 1.) != recipe['parent']:
        raise IntegrityError('Frozen block parent changed')
    for path, expected in recipe['source_hashes'].items():
        if sha(path) != expected:
            raise IntegrityError('Frozen parent collection source changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise IntegrityError('Parent capture owns physical GPU 1')
    families = {c['family']['family']: c['family'] for c in recipe['cases']}
    validate_families(list(families.values()))
    if len(recipe['cases']) != 8 or len(families) != 4 or any(f['split'] != 'train' for f in families.values()):
        raise IntegrityError('Parent captures must use the frozen balanced training cohort')
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM, cancel); signal.signal(signal.SIGINT, cancel)
    renderer = None; results = []; started = time.monotonic()
    with locked(folder/'collection.lock'), gpu_lease(home, 1):
        try:
            scorer = CEReward(RewardSpec(**game['reward_spec']))
            for case in recipe['cases']:
                ident = case['id']; record = folder/'capture-records'/f'{ident}.json'
                if sha(case['source_observation']) != case['source_observation_sha256'] or sha(case['reference_audio']) != case['reference_audio_sha256']:
                    raise IntegrityError('Matched Off evidence changed')
                if record.exists():
                    row = read(record)
                    if row['status'] != 'complete':
                        raise IntegrityError('Incomplete parent capture retained without reroll')
                    if any(sha(row[k]) != row[k+'_sha256'] for k in ('audio', 'capture')):
                        raise IntegrityError('Saved parent capture bytes changed')
                    results.append(row); continue
                if renderer is None:
                    renderer = AcousticBlockRenderer(folder, manifest, 1)
                arm = dict(name='block-parent', checkpoint=recipe['parent']['path'], multiplier=1.)
                capture_path = folder/'captures'/ident/'acoustic-tail.pt'
                row = dict(id=ident, status='running', family=case['family']['family'], seed=case['seed'],
                    recipe_sha256=digest(recipe), source_observation_sha256=case['source_observation_sha256'],
                    capture=str(capture_path), reused_exact_capture=False, started_unix=time.time())
                write(record, row)
                try:
                    with TailCapture(renderer.pipe) as collector:
                        observation = renderer.observe(case['family'], case['seed'], arm, scorer,
                            extra=component(recipe['parent']['path'], 1.))
                    if observation['status'] != 'complete':
                        raise IntegrityError('Invalid parent training waveform')
                    capture = collector.result(); save_tensor(capture_path, capture)
                    row.update(audio=observation['audio'], audio_sha256=observation['audio_sha256'],
                        capture_sha256=sha(capture_path), observation_sha256=sha(folder/'observations'/f"{observation['id']}.json"),
                        parent_ce=observation['reward']['scalar'], reference_ce=case['reference_ce'])
                    with torch.no_grad():
                        latents, checks = replay_latents(renderer.pipe.transformer, capture, record_checks=True)
                        wave = decode(renderer.pipe.vocoder, latents, capture['latent_hop_length'])
                    raw, rate = sf.read(row['audio'], dtype='float32', always_2d=True)
                    if rate != capture['sampling_rate'] or not all(c['exact'] for c in checks) or not np.array_equal(wave[0].T.cpu().numpy(), raw):
                        raise IntegrityError('Parent tail replay differs from its original raw waveform')
                    row.update(status='complete', tail_replay_checks=checks, exact_waveform_replay=True)
                    del latents, wave, capture
                except BaseException as exc:
                    row.update(status='failed', error=repr(exc)); raise
                finally:
                    row['finished_unix'] = time.time(); write(record, row)
                results.append(row)
                write(folder/'collection-status.json', dict(state='collecting', verified_captures=len(results), optimizer_updates=0))
                Store(home).event('block_parent_training_capture_finished', case=ident, parent_ce=row['parent_ce'], off_ce=case['reference_ce'], new_training_clips=1)
                print('BLOCK PARENT CAPTURE', ident, row['parent_ce']-case['reference_ce'], flush=True)
            immutable(folder/'captures.json', results)
            status = dict(state='complete', verified_captures=8, new_training_clips=8, new_independent_families=0,
                          new_engineering_clips=0, optimizer_updates=0, seconds=time.monotonic()-started)
            write(folder/'collection-status.json', status); return status
        finally:
            if renderer is not None:
                renderer.host._merge_sliders(renderer.pipe, renderer.device, [])
                exact = all(torch.equal(m.weight.detach().cpu(), p) for m, p in renderer.host._merge_state(renderer.device).pristine.items())
                write(folder/'collection-off-restoration.json', dict(exact=exact, physical_gpu=1))
                if not exact:
                    raise IntegrityError('Parent capture failed exact Off restoration')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--home', required=True); p.add_argument('--folder', required=True)
    a = p.parse_args(); print(__import__('json').dumps(run(a.home, a.folder), indent=2))
