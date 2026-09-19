#!/usr/bin/env python3
"""Verify rendered artifacts and comparison controls; never assign audio quality."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Players(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'audio':
            self.sources.append(dict(attrs)['src'])


def verify(folder):
    spec = json.loads((folder / 'render_spec.json').read_text())
    assert sha(Path(spec['prompts'])) == spec['prompts_sha256']
    records = []
    controls = {}
    for checkpoint in spec['checkpoints']:
        weight = Path(checkpoint['path'])
        assert sha(weight) == checkpoint['sha256']
        for seed in spec['seeds']:
            sub = folder / f'{weight.stem}-s{seed}'
            meta = json.loads((sub / 'checkpoint.json').read_text())
            assert meta['sha256'] == checkpoint['sha256']
            assert meta['steps'] == checkpoint['steps'] and meta['seed'] == seed
            paths = sorted(sub.glob('*.wav'))
            assert len(paths) == 4, sub
            hashes = {}
            for path in paths:
                audio, rate = sf.read(path, dtype='float32', always_2d=True)
                assert audio.size > 0 and np.isfinite(audio).all(), path
                digest = sha(path)
                hashes[path.name[:2]] = digest
                records.append(dict(path=str(path.relative_to(folder)), sha256=digest,
                                    seconds=len(audio) / rate, sample_rate=rate,
                                    channels=audio.shape[1], peak=float(np.abs(audio).max()),
                                    rms=float(np.sqrt(np.mean(audio.astype('float64') ** 2)))))
            assert hashes['01'] == hashes['04'], sub
            identity = (hashes['01'], hashes['03'])
            if seed in controls:
                assert controls[seed] == identity, sub
            controls[seed] = identity
    reused = folder / 'reference_reuse.json'
    reuse_count = 0
    if reused.exists():
        for destination, source in json.loads(reused.read_text()).items():
            assert sha(ROOT / destination) == source['sha256']
            assert sha(ROOT / source['source']) == source['sha256']
            reuse_count += 1
    players = Players()
    players.feed((folder / 'index.html').read_text())
    assert len(players.sources) == (len(spec['checkpoints']) + 2) * len(spec['seeds'])
    for source in players.sources:
        assert (folder / source).is_file(), source
    report = dict(checkpoints=len(spec['checkpoints']), wav_files=len(records),
                  unique_audio_hashes=len({r['sha256'] for r in records}),
                  verified_reused_files=reuse_count, audio_players=len(players.sources),
                  all_samples_finite=True, matched_reference_hashes=True,
                  seed_retries=spec['seed_retries'], records=records,
                  limitation='Artifact integrity and matching controls only; this is not a quality or preference score.')
    (folder / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    verify(parser.parse_args().folder)
