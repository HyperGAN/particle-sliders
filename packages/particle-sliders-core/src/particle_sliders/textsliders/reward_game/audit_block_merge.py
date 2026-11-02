"""Actual native equivalence of disjoint parent updates and one block adapter."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import time

from .core import read,write,immutable,sha,digest,locked,IntegrityError,Store
from .setup import verify
from .resources import gpu_lease
from .block_artifact import checkpoint,component,STRUCTURE
from .acoustic_artifact import component as attention_component,checkpoint as attention_checkpoint
from .ff_artifact import component as ff_component,checkpoint as ff_checkpoint


def audit(home,folder,candidate_path):
    import torch
    from diffusers import ModularPipeline
    from safetensors.torch import load_file,save_file
    from ..reward_sliders.specs import MODEL
    from app import generator
    from app.lora_runtime import LoRANetwork
    home=Path(home).resolve();folder=Path(folder).resolve();verify(home)
    candidate=checkpoint(candidate_path,1.);recipe_path=Path(candidate['path']).parent/'recipe.json';recipe=read(recipe_path)
    if read(Path(candidate['path']).with_suffix('.json'))['recipe_sha256']!=digest(recipe):raise IntegrityError('Composition recipe differs from sidecar')
    for name,validator in (('attention',attention_checkpoint),('feed_forward',ff_checkpoint)):
        parent=recipe['parents'][name]
        if validator(parent['path'],1.)!=parent:raise IntegrityError('Frozen composition parent changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise IntegrityError('Native block audit owns physical GPU 1')
    for path,h in recipe['sources'].items():
        if sha(path)!=h:raise IntegrityError('Composition source changed')
    parent_components=[attention_component(recipe['parents']['attention']['path'],1.),ff_component(recipe['parents']['feed_forward']['path'],1.)]
    capture_file=home/'audit/acoustic-tail-v1/acoustic-tail.pt'
    if sha(capture_file)!=read(home/'audit/acoustic-tail-v1/capture-result.json')['capture_sha256']:
        raise IntegrityError('Audited probe capture changed')
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('block_artifact.py','block_compose.py','acoustic_artifact.py','ff_artifact.py','resources.py')]
    protocol=dict(version=1,candidate=candidate,parents=recipe['parents'],composition_recipe_sha256=sha(recipe_path),
        capture_sha256=sha(capture_file),physical_gpu=1,
        source_hashes={str(p.resolve()):sha(p) for p in sources},
        required=dict(native_parent_pair_equals_single_composite=True,all_216_projection_weights_exact=True,
            both_captured_condition_branches_exact=True,zero_up_equals_off=True,off_restored=True),
        budget=dict(new_audio_generated=0,optimizer_updates=0),
        interpretation='Numerical host/format compatibility; the separate game still requires its own Off PCM audit and actual candidate audio')
    immutable(folder/'protocol.json',protocol)
    for path,h in protocol['source_hashes'].items():
        target=folder/'sources'/h/Path(path).name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        if sha(target)!=h:raise IntegrityError('Archived native audit source differs')
    if (folder/'result.json').exists():return read(folder/'result.json')
    if (folder/'attempt.json').exists():raise IntegrityError('Interrupted native block audit retained; explicit amendment required')
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    with locked(folder/'audit.lock'),gpu_lease(home,1):
        immutable(folder/'attempt.json',dict(started_unix=time.time(),protocol_sha256=digest(protocol)))
        began=time.monotonic();torch.set_num_threads(4);torch.cuda.set_device(0)
        pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        pipe.load_components(names='transformer',pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');tf=pipe.transformer;tf.eval().requires_grad_(False)
        network=LoRANetwork(tf,rank=8,alpha=8.,multiplier=1.,target_replace=['MiniMaxMusic3TransformerBlock'],
            prefix='lora_unet',delimiter='-',train_method='full',attach=False)
        if len(network.hosts)!=216:raise IntegrityError('Unexpected full-block native topology')
        pristine={name:host.weight.detach().cpu().clone() for name,host in network.hosts.items()}
        capture=torch.load(capture_file,map_location='cpu',weights_only=True)
        probes=capture['chunks'][0]['steps'][capture['steps']-1]['branches']
        def outputs():
            with torch.no_grad():
                return [tf(hidden_states=p['latent'].to('cuda:0'),timestep=p['timestep'].to('cuda:0'),
                    encoder_hidden_states=p['condition'].to('cuda:0'),return_dict=False)[0].detach().cpu() for _,p in sorted(probes.items())]
        def unchanged():return all(torch.equal(host.weight.detach().cpu(),pristine[name]) for name,host in network.hosts.items())
        off=outputs()
        try:
            zero={name:value.clone() for name,value in load_file(candidate['path']).items()}
            for name,value in zero.items():
                if name.endswith('.lora_up.weight'):value.zero_()
            zero_path=folder/'zero-up-control.safetensors';save_file(zero,str(zero_path));del zero
            zero_component=dict(STRUCTURE,weights=str(zero_path),mtime=zero_path.stat().st_mtime,multiplier=1.)
            generator._merge_sliders(pipe,'cuda:0',[zero_component])
            if not unchanged() or not all(torch.equal(a,b) for a,b in zip(off,outputs())):
                raise IntegrityError('Zero-up block adapter differs from Off')
            generator._merge_sliders(pipe,'cuda:0',[])
            generator._merge_sliders(pipe,'cuda:0',parent_components)
            expected={name:host.weight.detach().cpu().clone() for name,host in network.hosts.items()}
            parent_outputs=outputs()
            generator._merge_sliders(pipe,'cuda:0',[])
            generator._merge_sliders(pipe,'cuda:0',[component(candidate['path'],1.)])
            rows={name:torch.equal(host.weight.detach().cpu(),expected[name]) for name,host in network.hosts.items()}
            branch_equal=[torch.equal(a,b) for a,b in zip(parent_outputs,outputs())]
            if not all(rows.values()) or not all(branch_equal):raise IntegrityError('Single block adapter differs from native parent pair')
            generator._merge_sliders(pipe,'cuda:0',[])
            exact_off=unchanged() and all(torch.equal(a,b) for a,b in zip(off,outputs()))
            if not exact_off:raise IntegrityError('Off changed after native block composition')
            result=dict(passed=True,protocol_sha256=digest(protocol),candidate=candidate,
                parent_pair_matches_composite=True,projection_equal=rows,projections_verified=len(rows),branch_equal=branch_equal,
                zero_up_matches_off=True,off_restored=True,seconds=time.monotonic()-began,
                new_audio_generated=0,optimizer_updates=0)
            immutable(folder/'result.json',result);Store(home).event('acoustic_block_native_audit',passed=True,candidate_sha256=candidate['weights_sha256'],new_audio_generated=0)
            return result
        finally:
            generator._merge_sliders(pipe,'cuda:0',[])
            exact=unchanged();write(folder/'off-restoration.json',dict(exact=exact,physical_gpu=1))
            if not exact:raise IntegrityError('Native block audit failed to restore the base')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
    r=audit(a.home,a.folder,a.candidate)
    print(__import__('json').dumps({k:v for k,v in r.items() if k!='projection_equal'},indent=2),flush=True)
