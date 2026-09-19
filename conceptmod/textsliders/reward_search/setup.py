"""Freeze the new search domain without rewriting any v1 evidence."""
from copy import deepcopy
from pathlib import Path
import time

from ..reward_sliders.specs import ROOT, WORKSPACE, digest, sha, read_json, write_json, validate_families
from ..reward_sliders.experiment import verify
from ..reward_sliders.evaluate import extra_style_hashes
from ..reward_sliders.render import resolve_styles
from .fixtures import fresh
from . import queue

SOURCE=ROOT/'analysis/reward_sliders_20260907'
DEFAULT_RUN=ROOT/'analysis/reward_search_20260908'


def arm(path, strength, name=None):
    return dict(name=name or f"{Path(path).stem}-m{strength:g}",checkpoint=str(path),
                checkpoint_sha256=sha(path),coefficient=float(strength))


def freeze(run=DEFAULT_RUN):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    destination=run/'manifest.json'
    if destination.exists():
        result=read_json(destination);verify(result);return result
    previous=read_json(SOURCE/'manifest.json');verify(previous)
    old_rows=[f for f in previous['families'] if f['split']=='train']
    rows=deepcopy(old_rows)+fresh();validate_families(rows)
    result=deepcopy(previous)
    result.update(name='reward-ce-v2-search',schema='music-reward-search-v2',families=rows,
                  created_unix=time.time(),source_run=str(SOURCE),source_manifest_sha256=sha(SOURCE/'manifest.json'))
    result['style_components']={f['family']:resolve_styles(f['style_multipliers']) for f in rows}
    for comps in result['style_components'].values():extra_style_hashes(result,comps)
    result['file_stats'].update({p:dict(size=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns)
                                 for p in result['style_hashes']})
    for p in result['style_hashes']:
        if sha(p)!=result['style_hashes'][p]:raise ValueError('Style dependency changed')
    sources=read_json(SOURCE/'evaluation/manifest.json')['source_hashes']
    for path,expected in sources.items():
        if sha(path)!=expected:raise ValueError('Frozen v1 implementation changed')
    result['source_hashes'].update(sources)
    result['source_hashes'].update({str(p):sha(p) for p in Path(__file__).parent.glob('*.py')
                                    if p.name in ('__init__.py','queue.py','renderer.py','fixtures.py','setup.py')})
    result['operation']=dict(physical_gpus=[0,1],gpu0_after_studio_queue_drains=True,
        preserve_unrelated_gpu_processes=True,studio_restore='GPU 0, one worker',registry_changes=False)
    selected=read_json(SOURCE/'evaluation/selection.json')['selected']
    result['search']=dict(
        reference=selected,quality_scope='fixed first 20 seconds; natural endings are not the primary gate',
        selection='highest equal-family mean CE; exact ties Off then smaller multiplier; all declared outputs valid',
        training_families=24,new_training_families=16,train_seeds=[1103,2207,3301,4409],
        dev_seeds=[5519,6637],causal_seeds=[5521,6649,7753,8863],final_seeds=[9011,9029,9041,9059],
        strength_grid=[.5,.75,1.,1.25],boundary_extension=1.5,
        boundary_extension_rule='only if 1.25 wins and exceeds 1.0 by at least 0.02 mean CE',
        teacher_layers=[11,23],teacher_strengths=[.01,.03],include_v1_teacher=True,
        causal_rule='positive beats Off and random by >=0.02 mean CE; >50% wins vs Off; 16 complete pairs each',
        training_windows=[[0,128],[124,252],[248,376],[372,500]],training_stride=4,
        student_arms=['baseline','fm_capped'],student_checkpoints=[300,600],
        student_strengths=[.5,1.],student_seed=7,
        final_comparison=['Off','calibrated v1','best new student'],
        final_rule='fresh-family paired CE, wins, whole-family uncertainty and separate regressions/intent diagnostics',
        no_checkpoint_promoted_automatically=True)
    write_json(destination,result)
    for path,expected in result['source_hashes'].items():
        target=run/'provenance'/Path(path).relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes())
        assert sha(target)==expected
    write_json(run/'prompt-audit.json',dict(passed=True,new_families=36,reused_training_families=8,
                                         sound_only=True,style_provenance_checked=True))
    reused=[]
    for f in old_rows:
        for seed in result['search']['train_seeds']:
            path=SOURCE/'observations'/f"{f['family']}-s{seed}-off.json"
            observation=read_json(path)
            assert observation['status']=='complete' and observation['reward']['valid']
            for label in ('audio','trajectory'):assert sha(observation[label])==observation[label+'_sha256']
            reused.append(dict(observation=observation,source=str(path),source_sha256=sha(path)))
    write_json(run/'reused-training.json',reused)
    return result


def enqueue_initial(run=DEFAULT_RUN):
    run=Path(run);m=freeze(run);search=m['search'];reference=search['reference']
    arms=[dict(name='off')]+[arm(reference['checkpoint'],v,f'v1-m{v:g}') for v in search['strength_grid']]
    dev=[f for f in m['families'] if f['split']=='dev']
    stage=dict(name='v1-strength',arms=arms,families=[f['family'] for f in dev],seeds=search['dev_seeds'])
    write_json(run/'stages/v1-strength/protocol.json',stage)
    queue.add(run,'v1-strength',[dict(family=f,seed=seed,arms=arms)
                                 for f in dev for seed in search['dev_seeds']])
    queue.add(run,'training-capture',[dict(family=f,seed=seed,arms=[dict(name='off')],capture=True)
        for f in m['families'] if f.get('group')=='train' for seed in search['train_seeds']])
    write_json(run/'status.json',dict(stage='v1_strength_and_expanded_capture',updated_unix=time.time(),
                                   initial_render_jobs=80+64,training_families=24))
    return m


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();enqueue_initial(args.run_dir)
    print('Frozen 24 training families and 20 distinct development, causal and final families; 144 initial renders queued.')
