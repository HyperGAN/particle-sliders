"""Balanced frozen Off acoustic states, with exact duplicate-audio accounting."""
import argparse
from pathlib import Path
import os
import signal
import time
from .core import ROOT,read,write,immutable,sha,digest,locked,IntegrityError,Store
from .setup import verify

FAMILIES=('search-train-04','search-train-05','search-train-02','search-train-03')
SEEDS=(1103,3301)


def freeze(home,folder,audit_folder):
    from ..reward_sliders.render import pcm_sha
    home=Path(home).resolve();folder=Path(folder).resolve();audit_folder=Path(audit_folder).resolve();game,manifest=verify(home)
    if not read(audit_folder/'capture-result.json')['passed'] or not read(audit_folder/'gradient-result.json')['passed']:
        raise IntegrityError('Actual acoustic capture and gradient prerequisites must pass')
    rows=[]
    for family in FAMILIES:
        f=next(f for f in manifest['families'] if f['family']==family)
        if f['split']!='train':raise IntegrityError('Acoustic training must not use development families')
        for seed in SEEDS:
            ident=f'{family}-s{seed}-off';file=ROOT/'analysis/reward_search_20260908/stages/training-capture/observations'/f'{ident}.json';r=read(file)
            if r['provenance']['physical_gpu']!=1 or r['status']!='complete' or sha(r['audio'])!=r['audio_sha256']:raise IntegrityError('Invalid Off training reference')
            if r['provenance']['components']!=manifest['style_components'][family]:raise IntegrityError('Off capture used different style components')
            rows.append(dict(id=ident,family=f,seed=seed,source_observation=str(file),source_observation_sha256=sha(file),reference_audio=r['audio'],reference_audio_sha256=r['audio_sha256'],reference_pcm_sha256=pcm_sha(r['audio']),reference_ce=r['reward']['scalar']))
    reuse=read(audit_folder/'capture-protocol.json');reuse_id=read(reuse['source_observation'])['id']
    if reuse_id not in {r['id'] for r in rows}:raise IntegrityError('Engineering capture not in the frozen training bank')
    sources=[Path(__file__),Path(__file__).with_name('acoustic_tail.py')]
    recipe=dict(method='balanced-Off-acoustic-state-capture-v1',cases=rows,physical_gpu=1,source_manifest_sha256=sha(game['source_manifest']),
        source_hashes={str(p.resolve()):sha(p) for p in sources},reused_audit_folder=str(audit_folder),reused_case=reuse_id,
        audit_hashes={f:sha(audit_folder/f) for f in ('capture-result.json','gradient-result.json','acoustic-tail.pt')},
        budget=dict(new_engineering_clips=7,new_independent_training_clips=0,optimizer_updates=0),
        scalar_labels='Existing Off CE values are reused only after exact decoded-PCM equality; no selection among seeds')
    immutable(folder/'collection-recipe.json',recipe);immutable(folder/'manifest.json',manifest)
    Store(home).event('acoustic_training_collection_frozen',folder=str(folder),recipe_sha256=digest(recipe),new_engineering_clip_budget=7)
    return recipe


def run(home,folder):
    import torch
    from .resources import gpu_lease
    from .acoustic_tail import TailCapture,replay_latents
    from ..reward_search.renderer import SearchRenderer
    from ..reward_sliders.render import pcm_sha
    from ..reward_sliders.specs import save_tensor
    home=Path(home).resolve();folder=Path(folder).resolve();game,manifest=verify(home);recipe=read(folder/'collection-recipe.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or sha(game['source_manifest'])!=recipe['source_manifest_sha256']:raise IntegrityError('Wrong physical GPU or changed training manifest')
    for p,h in recipe['source_hashes'].items():
        if sha(p)!=h:raise IntegrityError('Acoustic collection source changed')
    audit=Path(recipe['reused_audit_folder'])
    for f,h in recipe['audit_hashes'].items():
        if sha(audit/f)!=h:raise IntegrityError('Reused acoustic audit changed')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel);renderer=None;results=[]
    with locked(folder/'collection.lock'),gpu_lease(home,1):
        try:
            for case in recipe['cases']:
                ident=case['id'];record=folder/'collection-observations'/f'{ident}.json'
                if sha(case['source_observation'])!=case['source_observation_sha256'] or sha(case['reference_audio'])!=case['reference_audio_sha256']:raise IntegrityError('Training Off reference changed')
                if record.exists():
                    r=read(record)
                    if r['status']!='complete':raise IntegrityError('Incomplete acoustic capture retained without reroll')
                    if any(sha(r[k])!=r[k+'_sha256'] for k in ('audio','capture')):raise IntegrityError('Corrupted saved acoustic capture')
                    results.append(r);continue
                if ident==recipe['reused_case']:
                    if read(audit/'capture-protocol.json')['source_observation_sha256']!=case['source_observation_sha256']:raise IntegrityError('Reused acoustic capture belongs to a different reference')
                    r=dict(id=ident,status='complete',family=case['family']['family'],seed=case['seed'],source_observation_sha256=case['source_observation_sha256'],
                        recipe_sha256=digest(recipe),audio=str(audit/'off-capture.wav'),audio_sha256=sha(audit/'off-capture.wav'),capture=str(audit/'acoustic-tail.pt'),capture_sha256=sha(audit/'acoustic-tail.pt'),
                        reference_ce=case['reference_ce'],reused_exact_capture=True,new_clips=0,independent_quality_observation=False)
                    write(record,r);results.append(r);continue
                if renderer is None:renderer=SearchRenderer(folder,manifest,1)
                target=folder/'captures'/ident;path=target/'off.wav';capture_file=target/'acoustic-tail.pt'
                r=dict(id=ident,status='running',family=case['family']['family'],seed=case['seed'],source_observation_sha256=case['source_observation_sha256'],recipe_sha256=digest(recipe),
                    audio=str(path),capture=str(capture_file),new_clips=0,started_unix=time.time(),reused_exact_capture=False,independent_quality_observation=False)
                write(record,r)
                try:
                    with TailCapture(renderer.pipe) as collector:
                        timing,_=renderer.generate(case['family'],case['seed'],path,duration=game['duration_seconds'])
                    data=collector.result();save_tensor(capture_file,data)
                    r.update(audio_sha256=sha(path),capture_sha256=sha(capture_file),timing=timing,new_clips=1)
                    if pcm_sha(path)!=case['reference_pcm_sha256']:raise IntegrityError('Acoustic training capture changed Off PCM')
                    with torch.no_grad():_,checks=replay_latents(renderer.pipe.transformer,data,record_checks=True)
                    if not all(c['exact'] for c in checks):raise IntegrityError('Acoustic training tail replay differs')
                    r.update(status='complete',tail_replay_checks=checks,reference_ce=case['reference_ce'])
                except BaseException as exc:
                    r.update(status='failed',error=repr(exc))
                    if path.exists():r.update(audio_sha256=sha(path),new_clips=1)
                    if capture_file.exists():r['capture_sha256']=sha(capture_file)
                    raise
                finally:r['finished_unix']=time.time();write(record,r)
                results.append(r);print('ACOUSTIC CAPTURE',ident,r['status'],flush=True)
                Store(home).event('acoustic_training_capture_finished',case=ident,new_engineering_clips=r['new_clips'])
            immutable(folder/'captures.json',results)
            write(folder/'collection-status.json',dict(state='complete',verified_captures=len(results),new_engineering_clips=sum(r['new_clips'] for r in results),new_independent_training_clips=0,optimizer_updates=0))
            return results
        finally:
            if renderer is not None:
                renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
                exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in renderer.host._merge_state(renderer.device).pristine.items())
                write(folder/'collection-off-restoration.json',dict(exact=exact,physical_gpu=1))
                if not exact:raise IntegrityError('Acoustic collection changed base weights')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--audit-folder');p.add_argument('action',choices=('freeze','run'));a=p.parse_args()
    result=freeze(a.home,a.folder,a.audit_folder) if a.action=='freeze' else run(a.home,a.folder)
    print(__import__('json').dumps(result,indent=2),flush=True)
