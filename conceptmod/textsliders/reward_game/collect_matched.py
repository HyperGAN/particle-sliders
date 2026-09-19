"""Collect a frozen eight-pair current-policy pilot; every outcome is retained."""
import argparse
from pathlib import Path
import signal
import time

from .core import Store, read, write, immutable, sha, digest, checkpoint, locked, IntegrityError
from .setup import verify
from .direct_search import training_cases
from .resources import gpu_lease


def collect(home, folder, proposal_path):
    import torch
    from ..reward_search.renderer import SearchRenderer
    from ..reward_sliders.reward import CEReward
    from ..reward_sliders.specs import RewardSpec
    from ..reward_sliders.evaluate import component
    from ..reward_sliders.render import pcm_sha
    home=Path(home);folder=Path(folder);proposal=read(proposal_path);store=Store(home)
    with locked(folder/'collection.lock'),gpu_lease(home,1):
        game,manifest=verify(home)
        for path,value in proposal['sources'].items():
            if sha(path)!=value:raise IntegrityError('Collection source changed')
        immutable(folder/'collection-recipe.json',proposal);immutable(folder/'manifest.json',manifest)
        parent=checkpoint(proposal['parent_checkpoint'],1.)
        if parent['weights_sha256']!=proposal['parent_sha256']:raise IntegrityError('Collection parent changed')
        entries=proposal['training_cases'];cases=[]
        for seed in (1103,3301):
            cases.extend(training_cases(game,manifest,dict(training_cases=[v for v in entries if v['seed']==seed])))
        if len(cases)!=8 or proposal['budget']['new_training_clips']!=8:raise IntegrityError('Eight matched takes required')
        for case in cases:
            off=case['off']
            if sha(off['trajectory'])!=off['trajectory_sha256']:raise IntegrityError('Off trajectory changed')
        immutable(folder/'cases.json',cases)
        arm=dict(name='matched-parent',checkpoint=parent['path'],checkpoint_sha256=parent['weights_sha256'],coefficient=1.)
        scorer=CEReward(RewardSpec(**game['reward_spec']));renderer=None;began=time.monotonic()
        try:
            renderer=SearchRenderer(folder,manifest,1);extra=component(parent['path'],1.)
            rows=[]
            for index,case in enumerate(cases):
                row=renderer.observe(case['family'],case['seed'],arm,scorer,capture=True,extra=extra)
                rows.append(row)
                write(folder/'collection-status.json',dict(state='collecting',completed=len(rows),scheduled=8))
                if row['status']!='complete' or not row['reward']['valid'] or not row.get('trajectory'):
                    raise IntegrityError('Invalid matched take retained; no replacement draw')
                # One additional plain render is an engineering comparison, not another quality draw.
                audit=folder/'capture-parity.json'
                if index==0 and not audit.exists():
                    marker=folder/'capture-parity-attempt.json'
                    if marker.exists():raise IntegrityError('Interrupted parity audit requires an explicit amendment')
                    immutable(marker,dict(case=case['id'],parent_sha256=parent['weights_sha256']))
                    plain=folder/'capture-parity.wav'
                    timing,_=renderer.generate(case['family'],case['seed'],plain,extra=extra,duration=game['duration_seconds'])
                    equal=pcm_sha(plain)==pcm_sha(row['audio'])
                    immutable(audit,dict(passed=equal,plain_audio=str(plain),plain_sha256=sha(plain),
                        captured_sha256=row['audio_sha256'],pcm_sha256=pcm_sha(plain),timing=timing,
                        independent_quality_observation=False,new_clips=1))
                    if not equal:raise IntegrityError('Capture changes matched parent audio')
                if audit.exists() and not read(audit)['passed']:raise IntegrityError('Saved capture parity failed')
            pairs=[]
            for case,row in zip(cases,rows):
                off=case['off'];delta=row['reward']['scalar']-off['reward']['scalar']
                chosen,rejected=(row,off) if delta>0 else (off,row)
                pairs.append(dict(family=case['family']['family'],seed=case['seed'],chosen=chosen['id'],
                    rejected=rejected['id'],chosen_observation=chosen,rejected_observation=rejected,
                    chosen_is_parent=delta>0,ce_delta_parent_vs_off=delta,ce_gap=abs(delta),
                    weight=min(2.,max(.25,abs(delta)))))
            immutable(folder/'preferences.json',pairs)
            result=dict(state='pairs_frozen',pairs=8,parent_wins=sum(p['chosen_is_parent'] for p in pairs),
                mean_parent_ce_delta=sum(p['ce_delta_parent_vs_off'] for p in pairs)/8,
                new_training_clips=8,new_engineering_clips=1,actual_updates=0,
                render_seconds=sum(r['timing']['total_seconds'] for r in rows),load_seconds=renderer.load_seconds,
                elapsed_seconds=time.monotonic()-began,preferences_sha256=sha(folder/'preferences.json'))
            write(folder/'collection-status.json',result);store.event('matched_collection_finished',folder=str(folder),**result)
            print('FROZEN PAIRS '+str(result),flush=True)
        finally:
            if renderer is not None:
                renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
                exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in renderer.host._merge_state(renderer.device).pristine.items())
                write(folder/'collection-off-restoration.json',dict(exact=exact,physical_gpu=1))
                if not exact:raise IntegrityError('Collection base restoration failed')


if __name__=='__main__':
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--proposal',required=True)
    a=p.parse_args();collect(a.home,a.folder,a.proposal)
