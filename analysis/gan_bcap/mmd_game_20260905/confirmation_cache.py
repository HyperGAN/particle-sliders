"""Reuse verified reference WAVs on exactly matching additional fixtures."""
import json
from pathlib import Path
import shutil

from conceptmod.textsliders.gan_v2.data import sha


def reuse_confirmation(source, destination, plan, entries, row, renderer):
    source, destination = Path(source), Path(destination)
    summary = json.loads((source/'summary.json').read_text())
    old = summary['plan']
    if summary['status'] != 'complete':
        raise ValueError('Only completed confirmation evidence may be reused')
    for key in ('prompts_sha256', 'seeds', 'duration', 'gpu'):
        if old[key] != plan[key]:
            raise ValueError(f'Confirmation cache differs: {key}')
    if row not in old['rows'] or old['gpu'] != 1:
        raise ValueError('Source row/GPU differs')
    spec = json.loads((source/f'row-{row}/render_spec.json').read_text())
    required = dict(prompts_sha256=plan['prompts_sha256'], row=row, seeds=plan['seeds'],
                    duration=plan['duration'], scales=[0., 1.], seed_retries=0,
                    renderer_sha256=sha(renderer))
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError('Source rendering specification differs')
    copied = []

    def copy_sample(sample, folder, kind):
        path = Path(sample['audio'])
        expected = sample['sha256']
        if sha(path) != expected:
            raise ValueError('Cached audio changed')
        target = folder/path.name
        folder.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != expected:
                raise ValueError('Existing destination differs from cached audio')
        else:
            shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError('Cached audio copy failed')
        copied.append(dict(source=str(path), destination=str(target), sha256=expected, kind=kind))

    for seed in plan['seeds']:
        records = [r for r in summary['records'] if r['row'] == row and r['seed'] == seed]
        if not records:
            raise ValueError('Missing source confirmation seed')
        for key in ('baseline', 'positive_reference'):
            if len({r[key]['sha256'] for r in records}) != 1:
                raise ValueError('Source controls are inconsistent')
        for entry in entries:
            if sha(entry['weights']) != entry['weights_sha256']:
                raise ValueError('Current checkpoint changed')
            folder = destination/f"{Path(entry['weights']).stem}-s{seed}"
            copy_sample(records[0]['baseline'], folder, 'slider_off')
            copy_sample(records[0]['positive_reference'], folder, 'positive_caption')
            matches = [r for r in records if r['checkpoint']['path'] == entry['weights']
                       and r['checkpoint']['sha256'] == entry['weights_sha256']]
            if len(matches) > 1:
                raise ValueError('Duplicate source checkpoint/seed')
            if matches:
                if not any(c['path'] == entry['weights'] and c['sha256'] == entry['weights_sha256']
                           for c in spec['checkpoints']):
                    raise ValueError('Source checkpoint not in rendering specification')
                copy_sample(matches[0]['candidate'], folder, 'identical_checkpoint')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'reused-audio.json').write_text(json.dumps(dict(source=str(source), row=row,
        copied=copied, policy='Exact bytes only; matching fixtures, renderer and checkpoint hashes'), indent=2)+'\n')
    return copied
