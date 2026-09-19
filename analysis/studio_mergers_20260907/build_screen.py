"""Freeze six blind three-method groups, with ordinary-energy-2 references."""
import copy
import json
import os
import yaml
from common import WORK, OLD, APP_ROOT, OUTPUT, PAIRS, sha, write
from app import sliders
from app.rewriter import _artist_name_hit


def main():
    assert not (WORK/'screen.json').exists(), 'Screen already frozen'
    old = json.loads((OLD/'screen.json').read_text())
    renders = json.loads((OLD/'renders.json').read_text())
    assert renders['screen_sha256'] == sha(OLD/'screen.json')
    for path, digest in old['source_sha256'].items():
        assert sha(path) == digest, path
    assert sha(sliders.REGISTRY_PATH) == old['registry_sha256']
    assert sha(old['fixture']['path']) == old['fixture']['sha256']
    rows = yaml.safe_load(open(old['fixture']['path']))['rows']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    jobs, reused = [], {}
    for fixture, row_index, seed in [('familiar',2,101),('confirmation',3,303)]:
        row = rows[row_index]
        for pair in PAIRS:
            key = '+'.join(pair)
            settings = [dict(id=i,scale=.5) for i in pair]
            for method in ('linear', 'ties', 'knots_ties', 'linear_e2'):
                energy = 2. if method == 'linear_e2' else 2.8
                components = sliders.resolve(settings, host_energy={'language_model':energy})
                if method in ('ties','knots_ties'):
                    path = WORK/'artifacts'/key/method/'manifest.json'
                    manifest = json.loads(path.read_text())
                    assert manifest['source_components'] == components
                    components = [dict(weights=str(path),mtime=path.stat().st_mtime,
                                       multiplier=1.,kind='language_model',format='dense_merge',sha256=sha(path))]
                job = dict(id=f'{len(jobs):02d}',label=f'{fixture}: {key}, {method}',
                           fixture=fixture,pair=list(pair),method=method,
                           sliders=settings, energy={'language_model':energy},
                           seed=seed,requested_duration=20,caption=row['neutral'],lyrics=row['lyrics'],
                           lora_components=components)
                if _artist_name_hit('',job['caption']+'\n'+job['lyrics']):
                    raise ValueError('Fixture failed name validation')
                if fixture == 'familiar' and method in ('linear','linear_e2'):
                    candidates = [j for j in old['jobs'] if all(j[k] == job[k] for k in (
                        'sliders','energy','seed','requested_duration','caption','lyrics','lora_components'))]
                    assert len(candidates) == 1
                    old_id = candidates[0]['id']
                    rec = copy.deepcopy(renders['records'][old_id])
                    assert rec['status'] == 'complete' and rec['apply_mode'] == 'merge'
                    for field in ('audio','excerpt'):
                        assert sha(rec[field]) == rec[field+'_sha256']
                        dest = OUTPUT/(f'{job["id"]}-reused'+('.wav' if field=='audio' else '-20s.wav'))
                        assert not dest.exists()
                        os.link(rec[field],dest)
                        rec[field]=str(dest)
                    job['reuse'] = dict(screen=str(OLD/'screen.json'),screen_sha256=sha(OLD/'screen.json'),job_id=old_id)
                    rec.update(id=job['id'],label=job['label'],reuse=job['reuse'])
                    reused[job['id']] = rec
                jobs.append(job)
    spec = dict(schema=1,physical_gpu=1,fixture=old['fixture'],
                fixtures=[dict(name='familiar',row=2,seed=101),dict(name='confirmation',row=3,seed=303)],
                registry_sha256=old['registry_sha256'],checkpoints=old['checkpoints'],
                source_sha256={**old['source_sha256'],**{str(WORK/n):sha(WORK/n) for n in (
                    'common.py','mergers.py','dense_runtime.py','build_artifacts.py','build_screen.py')}},
                methods=dict(linear='ordinary sum, energy 2.8; 1.4 per adapter',
                    ties='TIES 30% per projection, disjoint mean, each output projection norm matched to linear E2.8',
                    knots_ties='Joint-SVD alignment then TIES 30% per aligned projection, output norm matched to linear E2.8',
                    linear_e2='ordinary sum, energy 2.0 reference'),
                interpretation='Method screen, not density optimization. Frobenius matching controls parameter strength, not perceived strength. Human preference determines next steps. No production defaults changed.',
                jobs=jobs)
    assert len(jobs)==24 and len(reused)==6
    write(WORK/'screen.json',spec)
    write(WORK/'renders.json',dict(schema=1,screen_sha256=sha(WORK/'screen.json'),status='running',
          physical_gpu=1,output=str(OUTPUT),records=reused))
    print('Frozen 24 clips: 6 verified reused, 18 new renders.')


if __name__ == '__main__':
    main()
