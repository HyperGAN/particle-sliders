"""Audit frozen inputs, saved studio settings, audio and public playback."""
import collections
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.request
from common import WORK, OLD, PUBLIC, OUTPUT, URL, sha, write, verify
from report import assignments


def main():
    import numpy as np
    import soundfile as sf
    spec = json.loads((WORK / 'screen.json').read_text())
    renders = json.loads((WORK / 'renders.json').read_text())
    measured = json.loads((WORK / 'measurements.json').read_text())
    verify(spec)
    assert renders['status'] == measured['status'] == 'complete'
    assert renders['screen_sha256'] == measured['screen_sha256'] == sha(WORK / 'screen.json')
    expected = {j['id'] for j in spec['jobs']}
    assert expected == set(renders['records']) == set(measured['records']) and len(expected) == 30
    for path, digest in measured['source_sha256'].items():
        assert sha(path) == digest, path
    old_measurements = json.loads((OLD / 'measurements.json').read_text())
    scorer_provenance = next(iter(old_measurements['records'].values()))['measurement_provenance']
    assert len({j['caption'] for j in spec['jobs']}) == len({j['lyrics'] for j in spec['jobs']}) == 1
    device_counts = collections.Counter()
    pairs = []
    served = []
    short = []
    hashes = []
    for job in spec['jobs']:
        rec = renders['records'][job['id']]
        m = measured['records'][job['id']]
        assert rec['status'] == m['status'] == 'complete'
        assert rec['studio_job']['status'] == 'ready' and rec['studio_job']['rewrite'] is False
        for field in ('lyrics', 'caption', 'title', 'sliders', 'energy', 'seed'):
            assert rec['song'][field] == job[field], (job['id'], field)
            assert rec['studio_job'][field] == job[field], (job['id'], field)
        assert m['excerpt_sha256'] == rec['excerpt_sha256']
        assert m['measurement_provenance'] == scorer_provenance
        assert all(math.isfinite(v) for v in (*m['aesthetics'].values(), *m['concepts'].values()))
        assert rec['inspection']['finite']
        device_counts[rec['device']] += 1
        audio, rate = sf.read(rec['audio'], dtype='float32', always_2d=True)
        excerpt, erate = sf.read(rec['excerpt'], dtype='float32', always_2d=True)
        assert rate == erate and len(excerpt) == min(len(audio), 20*rate)
        assert excerpt.shape[1] == audio.shape[1]
        assert float(np.max(np.abs(excerpt - audio[:len(excerpt)]))) <= 1/32768 + 1e-7
        if len(audio) < rate*20:
            short.append(job['id'])
        assert sha(rec['source_audio']) == rec['audio_sha256']
        hashes.append(rec['audio_sha256'])
        for field in ('audio', 'excerpt'):
            path = Path(rec[field])
            assert sha(path) == rec[field+'_sha256']
            assert sha(PUBLIC / path.name) == rec[field+'_sha256']
            req = urllib.request.Request(URL + '/studio-combo-energy-20260907/' + path.name,
                                         headers={'Range': 'bytes=0-63'})
            with urllib.request.urlopen(req, timeout=10) as response:
                assert response.status == 206 and response.read() == path.read_bytes()[:64]
            served.append(path.name)
    assert set(device_counts) == {'cuda:0', 'cuda:1'}
    assert len(set(hashes)) == 30, 'Unexpected duplicate audio across distinct conditions'
    groups = assignments(spec)
    assert len(groups) == 12
    for group in groups:
        jobs = group['jobs']
        assert len(jobs) == 3 and len({j['seed'] for j in jobs}) == 1
        if group['axis'] == 'energy':
            assert all(j['share'] == .5 for j in jobs)
            assert {j['energy']['language_model'] for j in jobs} == {2.4, 2.8, 3.2}
        else:
            assert all(j['energy']['language_model'] == 2.8 for j in jobs)
            assert {j['share'] for j in jobs} == {.4, .5, .6}
    key = json.loads((WORK / 'blind-key.json').read_text())
    assert [{letter: j['id'] for letter, j in g['letters'].items()} for g in key] == [
        {chr(65+i): j['id'] for i, j in enumerate(g['jobs'])} for g in groups]
    with urllib.request.urlopen(URL + '/studio-combo-energy-20260907/', timeout=10) as response:
        page = response.read()
    assert page == (PUBLIC / 'index.html').read_bytes() == (OUTPUT / 'index.html').read_bytes()
    assert page.count(b'<audio ') == 36 and b'30/30 recordings complete' in page
    restore = json.loads((WORK / 'restore-status.json').read_text())
    assert restore['status'] in ('restored', 'newer_user_settings_preserved')
    # Confirm the generator itself logged the expected pair multipliers and
    # seed on the device reported by each completed studio job.
    journal = subprocess.check_output(['journalctl', '--user', '-u', 'music-studio.service',
        '--since', '@' + str(int(renders['started'])-2), '-o', 'cat', '--no-pager'], text=True)
    current = {}
    observed = []
    for line in journal.splitlines():
        match = re.search(r'app.generator: sliders on (cuda:\d+): (.*)', line)
        if match and '->' in match[2]:
            components = []
            for part in match[2].split(', '):
                name, value = part.rsplit('->', 1)
                components.append((name, float(value)))
            current[match[1]] = sorted(components)
        match = re.search(r'app.generator: generating 20\.0s audio .*seed=(\d+)\) on (cuda:\d+)', line)
        if match:
            observed.append(dict(seed=int(match[1]), device=match[2],
                                 components=current.get(match[2])))
    confirmed = []
    for job in spec['jobs']:
        expected_components = sorted((Path(c['weights']).name, float(c['multiplier']))
                                     for c in job['lora_components'])
        candidates = [x for x in observed if x['seed'] == job['seed']
                      and x['device'] == renders['records'][job['id']]['device']
                      and x['components'] is not None
                      and len(x['components']) == len(expected_components)
                      and all(a[0] == b[0] and math.isclose(a[1], b[1], rel_tol=5e-6, abs_tol=1e-7)
                              for a, b in zip(x['components'], expected_components))]
        assert len(candidates) == 1, ('Generator log match', job['id'], candidates)
        confirmed.append(dict(id=job['id'], **candidates[0]))
    write(WORK / 'generator-events.json', dict(events=confirmed,
        limitation='Logs verify resolved multipliers and seeds; they are not an independent dump of merged host weights.'))
    write(WORK / 'audit.json', dict(status='passed', screen_sha256=sha(WORK / 'screen.json'),
        records=30, measured=30, public_audio_urls_checked=len(served), blind_triplets=12,
        studio_device_counts=dict(device_counts), distinct_audio_hashes=len(set(hashes)),
        fixed_sheet=True, frozen_inputs_verified=True, stored_inputs_verified=True,
        excerpts_match=True, short_recordings=short, studio_settings=restore['status'],
        generator_events_confirmed=len(confirmed), scorer_matches_previous_study=True))
    print('PASSED: 30 renders, 30 measurements, 60 range-served audio URLs, 12 blind triplets.', flush=True)


if __name__ == '__main__':
    main()
