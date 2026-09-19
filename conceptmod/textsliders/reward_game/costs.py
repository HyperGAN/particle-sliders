"""Rebuild observed campaign costs without treating wall segments as GPU utilization."""
import json
from pathlib import Path
import time

from .core import DEFAULT_HOME, read, write


def report(home):
    home=Path(home)
    rows=[read(p) for p in (home/'renders').glob('*/observation.json')]
    new=[r for r in rows if not r.get('imported_reference') and r.get('audio_sha256')]
    development=[r for r in new if r['identity']['case']['family']['split']=='dev']
    training=[r for r in new if r['identity']['case']['family']['split']=='train']
    captures=[read(p) for p in (home/'training').glob('*/observations/*.json')]
    captures=[r for r in captures if r.get('audio_sha256')]
    confirmation=[read(p) for p in (home/'confirmation').glob('*/observations/*.json')]
    confirmation=[r for r in confirmation if r.get('audio_sha256')]
    composition=[read(p) for p in (home/'composition').glob('*/observations/*.json')]
    composition=[r for r in composition if r.get('audio_sha256')]
    engineering=[]
    off=home/'audit/off-pcm-v1/result.json'
    if off.exists():engineering.append(read(off))
    engineering.extend(read(p) for p in (home/'training').glob('*/capture-parity.json'))
    engineering.extend(read(p) for p in (home/'audit').glob('*/capture-observation.json'))
    engineering.extend(read(p) for p in (home/'training').glob('*/collection-observations/*.json') if read(p).get('audio_sha256') and read(p).get('new_clips',0))
    updates=[json.loads(line) for p in (home/'training').glob('*/updates.jsonl') for line in p.read_text().splitlines() if line]
    batches=[]
    by_key={r['key']:r for r in new}
    for path in (home/'jobs').glob('*.json'):
        jobs=read(path)
        if not isinstance(jobs,list):continue
        observations=[by_key[j['key']] for j in jobs if j.get('key') in by_key]
        if observations:
            timing=next((r['timing'] for r in observations if r.get('timing')),None)
            if timing:batches.append(dict(batch=path.stem,physical_gpu=observations[0]['physical_gpu'],seconds=timing['pipeline_load_seconds']))
    replay=[read(p) for p in (home/'training').glob('*/tokens/*.json')]
    leases=[json.loads(line) for line in (home/'ledger.jsonl').read_text().splitlines()]
    result=dict(updated_unix=time.time(),development_clips=len(development),training_clips=len(training)+len(captures),
        engineering_clips=sum(r['new_clips'] for r in engineering),confirmation_clips=len(confirmation),composition_clips=len(composition),optimizer_updates=len(updates),
        completed_full_batch_rollout_forward_backward_evaluations=sum(r.get('examples',0) for r in updates),
        optimizer_seconds=sum(r['seconds'] for r in updates),
        quality_render_seconds=sum(r.get('timing',{}).get('total_seconds',0) for r in new+captures+confirmation+composition),
        engineering_render_seconds=sum(r.get('timing',{}).get('total_seconds',0) for r in engineering),
        completed_batch_loads=batches,verified_residual_histories=len(replay),verified_target_artifacts=len(replay),
        reused_exact_target_artifacts=sum(bool(r.get('reused_exact_target')) for r in replay),
        residual_replay_seconds=sum(r.get('seconds',0) for r in replay),
        measured_gpu1_lease_seconds=sum(r['elapsed_seconds'] for r in leases if r['kind']=='gpu_lease_released' and r['physical_gpu']==1),
        discarded_actual_host_audit_updates=(0 if read(home/'game.json').get('parent_home') else 2)+sum(read(p).get('actual_updates',0) for p in (home/'audit').glob('*/discarded-update.json')),
        complete_accounting=False,
        limitations='Observed render/load/update/replay wall segments and instrumented completed GPU-1 leases. Earlier lease intervals, active work and some numerical-audit loads are not covered. Segments overlap; do not sum them as GPU utilization. Target artifact counts include verified cache reuse and are not unique independent histories.')
    result['new_audio_generated']=sum(result[k] for k in ('development_clips','training_clips','engineering_clips','confirmation_clips','composition_clips'))
    additive=('new_audio_generated','development_clips','training_clips','engineering_clips','confirmation_clips','composition_clips',
        'optimizer_updates','optimizer_seconds','completed_full_batch_rollout_forward_backward_evaluations','quality_render_seconds',
        'engineering_render_seconds','verified_residual_histories','verified_target_artifacts','reused_exact_target_artifacts',
        'residual_replay_seconds','measured_gpu1_lease_seconds','discarded_actual_host_audit_updates')
    result['own_campaign']={k:result[k] for k in additive};result['subcampaigns']={}
    for child_name in ('acoustic-v1','joint-v1','ff-v1','block-v1'):
        child=home/child_name
        if (child/'game.json').exists() and Path(read(child/'game.json').get('parent_home','')).resolve()==home.resolve():
            result['subcampaigns'][child_name]=report(child)
            for k in additive:result[k]+=result['subcampaigns'][child_name][k]
    write(home/'audit/cumulative-cost-current.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--home',default=str(DEFAULT_HOME));a=p.parse_args()
    print(json.dumps(report(a.home),indent=2))
