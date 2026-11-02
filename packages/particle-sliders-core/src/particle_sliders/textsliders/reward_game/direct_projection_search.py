"""Direct CE selection over predeclared projection gains on training families only."""
import argparse
from pathlib import Path
import shutil
import signal
import time

from .core import Store, read, write, immutable, sha, digest, checkpoint, generation_identity, IntegrityError, locked, STRUCTURE
from .evaluate import dispatch
from .setup import verify
from .gains import training_objective
from .projection_gains import fold


def choose(rows):
    eligible=[r for r in rows if r['valid'] and r['objective']>0]
    return max(eligible,key=lambda r:(r['objective'],-sum((g-1.)**2 for g in r['gains']))) if eligible else None


def training_cases(game,manifest,proposal):
    from ..reward_sliders.specs import digest as legacy_digest
    from app.rewriter import _artist_name_hit
    development={c['family']['family'] for c in game['cases']}
    cases=[]
    for entry in proposal['training_cases']:
        family=entry['family'];row=entry['off_observation'];p=row['provenance'];reward=row['reward']
        if family['family'] in development or family['split']!='train':raise IntegrityError('Development family in training selection')
        if _artist_name_hit('',__import__('json').dumps(family)):raise IntegrityError('Prohibited fixture provenance')
        components=manifest['style_components'][family['family']]
        structural=lambda items:[{k:v for k,v in c.items() if k!='mtime'} for c in items]
        if (row['status']!='complete' or not reward['valid'] or row['arm']!={'name':'off'} or
            row['family']!=family['family'] or row['seed']!=entry['seed'] or
            p['manifest_sha256'] not in [legacy_digest(manifest),*manifest.get('compatible_manifest_sha256',[])] or
            p['family_sha256']!=legacy_digest(family) or p['physical_gpu']!=entry['physical_gpu'] or
            p['base_sha256']!=legacy_digest(manifest['model_hashes']) or p['sampler']!=game['sampler'] or
            structural(p['components'])!=structural(components) or
            reward['reward_spec_sha256']!=legacy_digest(game['reward_spec']) or
            sha(row['audio'])!=row['audio_sha256'] or reward['audio_sha256']!=row['audio_sha256']):
            raise IntegrityError('Training Off control identity mismatch')
        cases.append(dict(id=f"{family['family']}-s{entry['seed']}",family=family,seed=entry['seed'],
                          physical_gpu=entry['physical_gpu'],style_components=components,off=row))
    if len(cases)!=4 or len({c['family']['family'] for c in cases})!=4:raise IntegrityError('Pilot requires four distinct training families')
    return cases


def run(home,folder,proposal_path,runner=dispatch):
    from safetensors.torch import load_file,save_file
    import torch
    home=Path(home);folder=Path(folder);store=Store(home);started=time.time()
    with locked(folder/'method.lock'):
        game,manifest=verify(home);proposal=read(proposal_path)
        for path,h in proposal['sources'].items():
            if sha(path)!=h:raise IntegrityError('Direct-search source changed')
        immutable(folder/'proposal.json',proposal)
        cases=training_cases(game,manifest,proposal)
        parent=proposal['parent_checkpoint'];base=checkpoint(parent['checkpoint'],1.)
        if base['weights_sha256']!=parent['checkpoint_sha256']:raise IntegrityError('Parent changed')
        if proposal['sampler']!=game['sampler'] or proposal['reward_spec']!=game['reward_spec']:raise IntegrityError('Pilot reward or sampler changed')
        weights=load_file(base['path']);meta=read(Path(base['path']).with_suffix('.json'))
        jobs=[];candidates=[]
        for index,gains in enumerate(proposal['changed_variables']['projection_gains']):
            path=folder/f'proposal-{index}.safetensors'
            if not path.exists():
                save_file(fold(weights,gains),str(path))
                sidecar={k:meta[k] for k in STRUCTURE}
                sidecar.update(plus_label='CE',minus_label='Off',unipolar=True,recommended_range=[0.,1.],
                               prompts_file=meta['prompts_file'],weights_sha256=sha(path),steps=0,
                               reward=dict(method='direct-projection-gain-search-v1',parent_sha256=base['weights_sha256'],
                                           projection_gains=gains,local_optimizer_updates=0,proposal_sha256=digest(proposal),
                                           interpretation='Training-side CE parameter selection; unconfirmed experimental adapter'))
                immutable(path.with_suffix('.json'),sidecar)
            candidate=checkpoint(path,1.)
            exported=load_file(str(path));expected=fold(weights,gains)
            if any(not torch.equal(exported[k],v) for k,v in expected.items()):raise IntegrityError('Exported projection gains differ from declaration')
            candidates.append(dict(index=index,gains=gains,candidate=candidate,keys=[]))
            for case in cases:
                key=digest(generation_identity(game,manifest,case,candidate));candidates[-1]['keys'].append(key)
                jobs.append(dict(case={k:v for k,v in case.items() if k!='off'},candidate=candidate,key=key))
        if len(jobs)!=proposal['budget']['new_training_clips']:raise IntegrityError('Declared pilot render budget changed')
        immutable(folder/'generation-jobs.json',jobs)
        initial_path=folder/'initial.json'
        if not initial_path.exists():immutable(initial_path,dict(started_unix=started,existing_keys=[j['key'] for j in jobs if store.observation(j['key'])]))
        initial=read(initial_path)
        pending=[j for j in jobs if store.observation(j['key']) is None]
        if pending:
            store.event('training_selection_started',method='direct-projection-gain-search-v1',folder=str(folder),scheduled=len(pending))
            runner(home,pending)
        results=[];observations=[]
        for candidate in candidates:
            rows=[store.observation(k) for k in candidate['keys']];observations.extend(rows)
            if any(not r or r['status'] not in ('complete','invalid_output') for r in rows):raise IntegrityError('Incomplete or failed training render retained')
            valid=all(r['status']=='complete' and r['reward']['valid'] for r in rows)
            deltas=[r['reward']['scalar']-c['off']['reward']['scalar'] for r,c in zip(rows,cases)] if valid else []
            results.append(dict(**candidate,valid=valid,deltas=deltas,objective=training_objective(deltas) if valid else None))
        selected=choose(results)
        new_clips=sum(bool(r and r.get('audio_sha256')) and r['key'] not in initial['existing_keys'] for r in observations)
        result=dict(method='direct-projection-gain-search-v1',actual_updates=0,results=results,
                    selected_index=selected['index'] if selected else None,selected_off=selected is None,
                    new_clips=new_clips,cache_hits=len(initial['existing_keys']),elapsed_seconds=time.time()-initial['started_unix'],
                    render_seconds=sum(r.get('timing',{}).get('total_seconds',0) for r in observations if r['key'] not in initial['existing_keys']),
                    development_observations_used_for_training=0)
        if (folder/'selection.json').exists():result['elapsed_seconds']=read(folder/'selection.json')['elapsed_seconds']
        immutable(folder/'selection.json',result);immutable(folder/'audio-observations.json',observations)
        if selected:
            for suffix in ('.safetensors','.json'):
                source=Path(selected['candidate']['path']).with_suffix(suffix);target=folder/('selected'+suffix)
                if not target.exists():shutil.copy2(source,target)
                if sha(target)!=sha(source):raise IntegrityError('Selected export differs')
        state='checkpoint_ready' if selected else 'no_candidate_selected'
        write(folder/'status.json',dict(state=state,actual_updates=0,selection=str(folder/'selection.json'),
                                       selection_sha256=sha(folder/'selection.json'),new_training_clips=new_clips))
        store.event('training_selection_finished',folder=str(folder),selected_off=selected is None,new_clips=new_clips,
                    render_seconds=result['render_seconds'],actual_updates=0)
        print(__import__('json').dumps(result,indent=2),flush=True)
        return result


if __name__=='__main__':
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--proposal',required=True)
    a=p.parse_args();run(a.home,a.folder,a.proposal)
