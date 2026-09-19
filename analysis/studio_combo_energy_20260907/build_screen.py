"""Freeze a 30-clip cross-shaped local sweep, before any new scoring."""
import json
import random
import subprocess
from pathlib import Path
from common import WORK, OLD, OUTPUT, PUBLIC, APP_ROOT, PAIRS, sha, write, api, verify
from app import sliders
from app.rewriter import validate_package


def main():
    assert not (WORK / 'screen.json').exists(), 'Screen already frozen'
    old = json.loads((OLD / 'screen.json').read_text())
    fixture = next(j for j in old['jobs'] if j['id'] == '00')
    assert sha(sliders.REGISTRY_PATH) == old['registry_sha256']
    conditions = [dict(key='e24', energy=2.4, share=.5),
                  dict(key='center', energy=2.8, share=.5),
                  dict(key='e32', energy=3.2, share=.5),
                  dict(key='balance40', energy=2.8, share=.4),
                  dict(key='balance60', energy=2.8, share=.6)]
    seeds = [707, 909]
    jobs = []
    for pair in PAIRS:
        for seed in seeds:
            for c in conditions:
                settings = [dict(id=pair[0], scale=c['share']),
                            dict(id=pair[1], scale=round(1-c['share'], 2))]
                energy = dict(transformer=sliders.catalog()['energy']['transformer']['default'],
                              language_model=c['energy'])
                jid = f'{len(jobs):02d}'
                package = validate_package(dict(title='Blend study ' + jid,
                    lyrics=fixture['lyrics'], caption=fixture['caption']))
                comps = sliders.resolve(settings, host_energy=energy)
                assert len(comps) == 2 and all(x['kind'] == 'language_model' for x in comps)
                assert abs(sum(x['multiplier'] for x in comps) - c['energy']) < 1e-6
                jobs.append(dict(id=jid, label=f'{" + ".join(pair)}; seed {seed}; {c["key"]}',
                    pair=list(pair), seed=seed, condition=c['key'], share=c['share'],
                    sliders=settings, energy=energy, requested_duration=20,
                    lora_components=comps, **package))
    order = []
    # Complete energy triplets first. Shuffle each wave independently of quality.
    for keys in [('e24', 'center', 'e32'), ('balance40', 'balance60')]:
        wave = [j['id'] for j in jobs if j['condition'] in keys]
        random.Random('combo-energy-20260907-' + '-'.join(keys)).shuffle(wave)
        order.extend(wave)
    source_paths = [APP_ROOT / 'app' / f for f in
                    ('generator.py', 'lora_runtime.py', 'sliders.py', 'server.py', 'store.py', 'rewriter.py')]
    source_paths += [WORK / f for f in ('common.py', 'build_screen.py', 'render.py')]
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', 'music-studio.service',
                                     '-p', 'MainPID', '--value'], text=True))
    selected_env = {}
    for raw in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
        key, _, val = raw.partition(b'=')
        if key.decode(errors='replace') in ('CUDA_VISIBLE_DEVICES', 'MUSIC_WORKERS',
              'MUSIC_HOST', 'MUSIC_PORT', 'LATHE_SLIDER_APPLY', 'MUSIC_CPU_OFFLOAD'):
            selected_env[key.decode()] = val.decode()
    assert selected_env.get('CUDA_VISIBLE_DEVICES') == '0,1'
    assert selected_env.get('LATHE_SLIDER_APPLY', 'merge') == 'merge'
    live = api('/api/studio')
    assert live['workers'] == 2 and set(live['loaded']) == {'cuda:0', 'cuda:1'}
    assert not live['queued'] and not live['active'] and not live['keep']['enabled']
    spec = dict(schema=1, physical_gpus=[0, 1], execution='existing studio API workers',
        studio_pid=pid, studio_environment=selected_env, seeds=seeds, conditions=conditions,
        fixture=dict(source_screen=str(OLD / 'screen.json'), source_screen_sha256=sha(OLD / 'screen.json'),
                     source_job='00', description='Existing electronic song and lyrics; two fresh seeds, shared across all conditions.'),
        registry_sha256=old['registry_sha256'], checkpoints=old['checkpoints'],
        source_sha256={str(p): sha(p) for p in source_paths}, jobs=jobs, submission_order=order,
        protocol=dict(method='Ordinary full-delta addition through the production studio.',
          design='Three pairs × two fresh seeds × five conditions. Energy at 50/50; balance at E2.8.',
          limitation='No energy-by-balance interaction grid, new song, three-control mix, or universal optimum is established.',
          comparisons='Blind energy and balance triplets share the center audio. Letters randomized separately per group.',
          metrics='Frozen CE, PQ, PC, CU, both concept CLAP margins and lyric diagnostics; no fitted aggregate or automatic winner.',
          audio='20-second target with studio end grace; first 20 seconds for existing metric comparability; full recordings retained.',
          strength='Experimental two-slider strengths; not a new solo default.'))
    assert len(jobs) == 30 and len(set(order)) == 30
    verify(spec)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write(WORK / 'screen.json', spec)
    write(WORK / 'studio-before.json', live)
    write(WORK / 'renders.json', dict(schema=1, screen_sha256=sha(WORK / 'screen.json'),
        status='prepared', physical_gpus=[0, 1], output=str(OUTPUT), records={}))
    from report import build
    build(json.loads((WORK / 'renders.json').read_text()), OUTPUT)
    print('Frozen 30 new renders; 3 pairs, 2 shared fresh seeds, 5 settings. Studio has two idle workers.')


if __name__ == '__main__':
    main()
