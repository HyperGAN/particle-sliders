"""Recover chosen matched-policy targets under each take's actual source policy."""
import argparse
from pathlib import Path
import signal
import time

from .core import ROOT, read, write, immutable, sha, digest, locked, IntegrityError
from .resources import gpu_lease
from .setup import verify
from .replay_residual import recover


def prepare(home,folder):
    import torch
    from diffusers import ModularPipeline
    from ..reward_sliders.specs import MODEL, save_tensor
    from ..reward_sliders.evaluate import component
    from app import generator
    folder=Path(folder)
    with locked(folder/'prepare.lock'),gpu_lease(home,1):
        verify(home);collection=read(folder/'collection-recipe.json');manifest=read(folder/'manifest.json')
        pairs=read(folder/'preferences.json');status=read(folder/'collection-status.json')
        if len(pairs)!=8 or status['state']!='pairs_frozen' or sha(folder/'preferences.json')!=status['preferences_sha256']:
            raise IntegrityError('Freeze all eight matched labels before code recovery')
        if not read(folder/'capture-parity.json')['passed']:raise IntegrityError('Captured policy lacks audio parity')
        for path,value in collection['sources'].items():
            if sha(path)!=value:raise IntegrityError('Collection implementation changed')
        paths=[Path(__file__),Path(__file__).with_name('replay_residual.py'),Path(__file__).with_name('residual_policy.py'),Path(__file__).with_name('merged_forward.py')]
        signature=dict(method='matched-eight-codebook-recovery-v1',source_manifest_sha256=sha(folder/'manifest.json'),
            preferences_sha256=sha(folder/'preferences.json'),collection_sha256=sha(folder/'collection-recipe.json'),frames=500,
            physical_gpu=1,source_hashes={str(p):sha(p) for p in paths},chosen=[p['chosen'] for p in pairs])
        immutable(folder/'recovery-recipe.json',signature)
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(71)
        began=time.monotonic();pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        for name in ('language_model','rvq_depth_decoder'):
            pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');pipe.language_model.eval().requires_grad_(False);pipe.rvq_depth_decoder.eval().requires_grad_(False)
        load_seconds=time.monotonic()-began
        try:
            for pair in pairs:
                observation=pair['chosen_observation'];path=folder/'tokens'/f"{pair['chosen']}.pt"
                target_signature=dict(recovery_sha256=digest(signature),trajectory_sha256=observation['trajectory_sha256'],
                    audio_sha256=observation['audio_sha256'],source_arm=observation['arm'])
                if path.exists():
                    if (not read(path.with_suffix('.json'))['passed'] or read(path.with_suffix('.json'))['sha256']!=sha(path) or
                        torch.load(path,map_location='cpu',weights_only=True)['signature']!=target_signature):
                        raise IntegrityError('Matched target cache changed')
                    continue
                for label in ('audio','trajectory'):
                    if sha(observation[label])!=observation[label+'_sha256']:raise IntegrityError('Chosen capture corrupted')
                components=list(manifest['style_components'][pair['family']]);arm=observation['arm']
                if pair['chosen_is_parent']:
                    if arm['checkpoint_sha256']!=collection['parent_sha256'] or sha(arm['checkpoint'])!=arm['checkpoint_sha256']:
                        raise IntegrityError('Chosen source parent changed')
                    components.append(component(arm['checkpoint'],arm['coefficient']))
                elif arm!={'name':'off'}:raise IntegrityError('Unexpected chosen source policy')
                generator._merge_sliders(pipe,'cuda:0',components)
                started=time.monotonic();trajectory=torch.load(observation['trajectory'],map_location='cpu',weights_only=True)
                recovered=recover(pipe,trajectory,500)
                save_tensor(path,dict(signature=target_signature,**recovered))
                immutable(path.with_suffix('.json'),dict(passed=True,sha256=sha(path),source=pair['chosen'],codebooks=8,
                    frames=500,verified_feedback_frames=501,seconds=time.monotonic()-started,new_audio=0))
                print('RECOVERED '+pair['chosen']+'; 501 exact feedback frames under recorded source policy',flush=True)
            # The unchanged residual objective already passed both actual-host and tiny-model gradient audits.
            audit=ROOT/'analysis/reward_game_v1_20260908/training/residual-imitation-v1/live-loss-audit.json'
            previous=read(audit)
            if not previous['passed']:raise IntegrityError('Residual loss gradient audit failed')
            immutable(folder/'live-loss-audit.json',dict(passed=True,reused_audit=str(audit),reused_audit_sha256=sha(audit),
                residual_policy_sha256=sha(Path(__file__).with_name('residual_policy.py')),
                interpretation='Same frozen objective and differentiable merger; target replay independently exact for each newly selected source policy.',
                new_audit_updates=0))
            write(folder/'prepare-status.json',dict(state='targets_prepared',verified_targets=8,actual_updates=0,
                new_audio=0,load_seconds=load_seconds,elapsed_seconds=time.monotonic()-began))
        finally:
            generator._merge_sliders(pipe,'cuda:0',[])
            exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in generator._merge_state('cuda:0').pristine.items())
            write(folder/'prepare-off-restoration.json',dict(exact=exact,physical_gpu=1))
            if not exact:raise IntegrityError('Code recovery base restoration failed')


if __name__=='__main__':
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True)
    a=p.parse_args();prepare(a.home,a.folder)
