"""Prepare a conditional next-method pilot using all eight saved parent draws.

This module is not launched by the projection search. A separate registered
attempt must choose and freeze the policy objective before it is run.
"""
import argparse
from pathlib import Path
import signal
import time

from .core import read,write,immutable,sha,digest,locked,IntegrityError
from .setup import verify
from .resources import gpu_lease
from .policy_objective import balanced_advantages
from .replay_residual import recover


def prepare(home,folder,source):
    import torch
    from diffusers import ModularPipeline
    from ..reward_sliders.specs import MODEL,save_tensor
    from ..reward_sliders.evaluate import component
    from app import generator
    folder=Path(folder);source=Path(source)
    with locked(folder/'prepare.lock'),gpu_lease(home,1):
        game,original_manifest=verify(home);collection=read(source/'collection-recipe.json');manifest=read(source/'manifest.json')
        if digest(manifest)!=digest(original_manifest):raise IntegrityError('Rollout manifest differs from frozen host')
        if not read(source/'capture-parity.json')['passed']:raise IntegrityError('Parent capture parity failed')
        parent=collection['parent_checkpoint']
        if sha(parent)!=collection['parent_sha256']:raise IntegrityError('Saved rollout policy changed')
        rows=[read(p) for p in sorted((source/'observations').glob('*.json'))]
        if len(rows)!=8 or any(r['status']!='complete' or not r['reward']['valid'] or
            r['arm']['checkpoint_sha256']!=collection['parent_sha256'] or r['arm']['coefficient']!=1. for r in rows):
            raise IntegrityError('Need all eight valid draws from the exact frozen parent')
        if any(r['provenance']['physical_gpu']!=1 or r['split']!='train' for r in rows):raise IntegrityError('Wrong rollout GPU or split')
        episodes=[dict(id=r['id'],family=r['family'],seed=r['seed'],ce=r['reward']['scalar'],observation=r) for r in rows]
        if len({r['family'] for r in episodes})!=4 or {r['family'] for r in episodes}&{c['family']['family'] for c in game['cases']}:
            raise IntegrityError('Need four independent training families')
        advantages=balanced_advantages(episodes)
        for row in episodes:row['advantage']=advantages[row['id']]
        immutable(folder/'episodes.json',episodes);immutable(folder/'manifest.json',manifest)
        immutable(folder/'policy-parent.json',dict(checkpoint=parent,checkpoint_sha256=collection['parent_sha256'],coefficient=1.,
            collection_recipe=str(source/'collection-recipe.json'),collection_sha256=sha(source/'collection-recipe.json')))
        paths=[Path(__file__),Path(__file__).with_name('replay_residual.py'),Path(__file__).with_name('residual_policy.py'),
               Path(__file__).with_name('merged_forward.py'),Path(__file__).with_name('policy_objective.py')]
        signature=dict(method='eight-parent-policy-targets-v1',episodes_sha256=sha(folder/'episodes.json'),
            manifest_sha256=sha(folder/'manifest.json'),parent_sha256=sha(parent),frames=500,physical_gpu=1,
            source_hashes={str(p):sha(p) for p in paths})
        immutable(folder/'recovery-recipe.json',signature)
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(71)
        pipe=None;began=time.monotonic();load_seconds=0.
        try:
            for episode in episodes:
                row=episode['observation'];path=folder/'tokens'/f"{row['id']}.pt"
                target_signature=dict(recovery_sha256=digest(signature),audio_sha256=row['audio_sha256'],trajectory_sha256=row['trajectory_sha256'])
                for label in ('audio','trajectory'):
                    if sha(row[label])!=row[label+'_sha256']:raise IntegrityError('Parent rollout changed')
                if path.exists():
                    if read(path.with_suffix('.json'))['sha256']!=sha(path):raise IntegrityError('Policy target cache changed')
                    continue
                previous=source/'tokens'/path.name
                if previous.exists():
                    audit=read(previous.with_suffix('.json'))
                    if not audit['passed'] or audit['sha256']!=sha(previous):raise IntegrityError('Existing exact target changed')
                    recovered=torch.load(previous,map_location='cpu',weights_only=True)
                    if any(recovered['signature'][k]!=target_signature[k] for k in ('audio_sha256','trajectory_sha256')):
                        raise IntegrityError('Existing target has another source rollout')
                    recovered.pop('signature');seconds=0.;reuse=dict(path=str(previous),sha256=sha(previous))
                else:
                    if pipe is None:
                        start=time.monotonic();pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
                        for name in ('language_model','rvq_depth_decoder'):
                            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
                        pipe.to('cuda:0');pipe.language_model.eval().requires_grad_(False);pipe.rvq_depth_decoder.eval().requires_grad_(False)
                        load_seconds=time.monotonic()-start
                    components=list(manifest['style_components'][row['family']])+[component(parent,1.)]
                    generator._merge_sliders(pipe,'cuda:0',components)
                    start=time.monotonic();trajectory=torch.load(row['trajectory'],map_location='cpu',weights_only=True)
                    recovered=recover(pipe,trajectory,500);seconds=time.monotonic()-start;reuse=None
                save_tensor(path,dict(signature=target_signature,**recovered))
                immutable(path.with_suffix('.json'),dict(passed=True,sha256=sha(path),source=row['id'],verified_feedback_frames=501,
                    codebooks=8,frames=500,seconds=seconds,reused_exact_target=reuse,new_audio=0))
                print('POLICY TARGET '+row['id']+(' reused exact cache' if reuse else ' 501 exact replay frames'),flush=True)
            write(folder/'prepare-status.json',dict(state='targets_prepared',verified_targets=8,new_audio=0,actual_updates=0,
                load_seconds=load_seconds,elapsed_seconds=time.monotonic()-began,active_families=len({e['family'] for e in episodes if e['advantage']})))
        finally:
            if pipe is not None:
                generator._merge_sliders(pipe,'cuda:0',[])
                exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in generator._merge_state('cuda:0').pristine.items())
                write(folder/'prepare-off-restoration.json',dict(exact=exact,physical_gpu=1))
                if not exact:raise IntegrityError('Parent policy preparation base restoration failed')


if __name__=='__main__':
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();prepare(a.home,a.folder,a.source)
