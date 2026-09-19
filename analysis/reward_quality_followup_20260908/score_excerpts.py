"""Post hoc fixed-duration quality diagnostic, preserving the original full-song study."""
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict
import json
import time

from conceptmod.textsliders.reward_sliders.specs import RewardSpec, digest, sha, read_json, write_json
from conceptmod.textsliders.reward_sliders.reward import CEReward
from conceptmod.textsliders.reward_sliders.directions import paired_statistics


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'reward_sliders_20260907'
original_manifest = read_json(SOURCE/'transfer/manifest.json')
selected = original_manifest['transfer']['adapter']
assert sha(selected['checkpoint']) == selected['checkpoint_sha256']
original_paths = sorted((SOURCE/'transfer/observations').glob('*.json'))
original_rows = [read_json(p) for p in original_paths]
assert len(original_rows) == 48 and all(r['status'] in ('complete','failed') for r in original_rows)
families = {f['family']:f for f in original_manifest['families']}
expected = {f'{family}-s{seed}-{arm}' for family in families
            for seed in original_manifest['transfer']['seeds'] for arm in ('off','lora')}
assert {r['id'] for r in original_rows} == expected
spec = RewardSpec(**original_manifest['reward_spec'])
assert CEReward.provenance() == spec.hashes
protocol = dict(
    scope='post hoc first-20-second quality diagnostic; no full-song completion claim',
    reason='User prioritizes improving CE wins and deprioritizes baseline completion failures',
    selected=selected, source_manifest_sha256=sha(SOURCE/'transfer/manifest.json'),
    source_observation_sha256={str(p):sha(p) for p in original_paths},
    rule='all 48 original outputs, two disjoint 10-second windows in first 20 seconds; one RMS 0.1 copy',
    reward_spec=asdict(spec), original_full_song_statuses_preserved=True,
    checkpoint_or_strength_selection=False, new_generations=0,
    dependencies={str(p):sha(p) for p in [Path(__file__),
        Path(__import__('conceptmod.textsliders.reward_sliders.reward',fromlist=['x']).__file__),
        Path(__import__('conceptmod.textsliders.reward_sliders.directions',fromlist=['x']).__file__)]})
protocol_path = ROOT/'protocol.json'
if protocol_path.exists():
    assert digest(read_json(protocol_path)) == digest(protocol), 'Frozen diagnostic protocol changed'
else:
    write_json(protocol_path,protocol)
scorer = CEReward(spec)
records=[]
for original in original_rows:
    assert sha(original['audio']) == original['audio_sha256']
    destination=ROOT/'observations'/f"{original['id']}.json"
    if destination.exists():
        row=read_json(destination)
        assert row['protocol_sha256'] == digest(protocol)
        assert row['reward']['audio_sha256'] == original['audio_sha256']
    else:
        reward=scorer.measure(original['audio'],full_song=False)
        row=dict(id=original['id'],family=original['family'],split='transfer',seed=original['seed'],
                 cell_hash=original['cell_hash'],arm=original['arm'],
                 status='complete' if reward['valid'] else 'failed',reward=reward,
                 original_generation_status=original['status'],original_generation_error=original['error'],
                 audio=original['audio'],audio_sha256=original['audio_sha256'],
                 scope='first 20 seconds only',protocol_sha256=digest(protocol))
        write_json(destination,row)
    records.append(row)
    write_json(ROOT/'status.json',dict(stage='scoring_excerpts',completed=len(records),total=48,updated_unix=time.time()))
    print(f"{len(records)}/48 {row['id']} CE={row['reward']['scalar']}",flush=True)
statistics=paired_statistics(records,'lora')
groups={}
for attribute in ('voice','genre'):
    groups[attribute]={}
    for value in sorted({f[attribute] for f in families.values()}):
        ids={f['family'] for f in families.values() if f[attribute]==value}
        groups[attribute][value]=paired_statistics([r for r in records if r['family'] in ids],'lora')
pairs=defaultdict(dict)
for row in records:
    if row['status']=='complete':
        pairs[(row['family'],row['seed'])][row['arm']['name']]=row['reward']['scalar']
deltas=[dict(family=family,seed=seed,off=p['off'],lora=p['lora'],delta=p['lora']-p['off'])
        for (family,seed),p in pairs.items() if set(p)=={'off','lora'}]
result=dict(statistics=statistics,subgroups=groups,pairs=deltas,selected=selected,
            original_full_song_failures=sum(r['status']=='failed' for r in original_rows),
            excerpt_failures=sum(r['status']=='failed' for r in records),
            limitation='Post hoc excerpt diagnostic; endpoint completion and full-song preservation are not evaluated',
            protocol_sha256=digest(protocol))
assert sha(selected['checkpoint']) == selected['checkpoint_sha256']
for path,expected_hash in protocol['source_observation_sha256'].items():
    assert sha(path)==expected_hash
write_json(ROOT/'results.json',result)
lines=['# Reward quality follow-up','',
       'Post hoc quality-only assessment requested after the original full-song completion study. '
       'All 48 existing outputs are scored over their first 20 seconds with the original short-screen protocol. '
       'No new audio, training, checkpoint selection or original observation changes.','',
       f"Valid excerpt pairs: {statistics['valid_pairs']}/24. Mean CE change: {statistics['mean_delta']:+.4f}. "
       f"Seed win rate: {statistics['seed_win_rate']:.1%}. Family bootstrap 95% interval: {statistics['family_bootstrap_95']}.",'',
       f"The original study still has {result['original_full_song_failures']} incomplete full-song outputs. "
       'This diagnostic evaluates the beginning of each take and does not change that result.','',
       '| Family | Voice | Arrangement | LoRA minus Off CE |','|---|---|---|---:|']
for family,delta in statistics['family_deltas'].items():
    f=families[family]
    lines.append(f"| {family} | {f['voice']} | {f['genre']} | {delta:+.4f} |")
lines += ['', '[Frozen diagnostic protocol](protocol.json) · [All pair results](results.json)','']
(ROOT/'README.md').write_text('\n'.join(lines))
write_json(ROOT/'status.json',dict(stage='complete',completed=48,total=48,updated_unix=time.time()))
print(json.dumps(statistics),flush=True)
