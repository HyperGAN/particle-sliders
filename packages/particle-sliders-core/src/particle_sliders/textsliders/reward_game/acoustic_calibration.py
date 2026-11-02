"""One explicit global gain folded into an ordinary acoustic adapter export."""
import argparse
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError
from .acoustic_artifact import checkpoint


def fold(values, gain):
    import math
    if not math.isfinite(gain) or not 0 <= gain <= 1.5:
        raise IntegrityError('Acoustic calibration gain must lie in [0, 1.5]')
    return {k: v.clone() * gain if k.endswith('lora_up.weight') else v.clone()
            for k, v in values.items()}


def construct(parent, folder, gain=1.5):
    import torch
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(4)
    if gain != 1.5:
        raise IntegrityError('This pilot declares one gain, 1.5; zero/unit are algebra audits only')
    parent=Path(parent).resolve(); folder=Path(folder).resolve()
    identity=checkpoint(parent, 1.); source=load_file(str(parent))
    state=parent.parent/'state-step8.pt'
    if not read(parent.parent/'off-restoration-8.json')['exact']:
        raise IntegrityError('Parent Off restoration did not pass')
    recipe=dict(method='single-global-acoustic-gain-calibration-v1', parent=identity,
        gain=gain, multiplier=1., inherited_gradient_updates=8, local_optimizer_updates=0,
        parent_training_recipe=str(parent.parent/'recipe.json'),
        parent_training_recipe_sha256=sha(parent.parent/'recipe.json'),
        parent_full_state=str(state), parent_full_state_sha256=sha(state),
        selection='Exposed development calibration after 16/16 meaningful wins and +0.095898 mean CE failed the unchanged +0.10 gate. One fixed 1.5 gain; no per-case selection or grid.',
        limitation='A stronger ordinary adapter may regress. Full development and new independent confirmation remain required.',
        source_hashes={str(Path(__file__).resolve()):sha(__file__)})
    immutable(folder/'recipe.json', recipe)
    zero, unit, changed=fold(source,0.),fold(source,1.),fold(source,gain)
    up=[k for k in source if k.endswith('lora_up.weight')]
    assert len(up)==144 and len(source)==432
    assert all(torch.equal(unit[k],v) for k,v in source.items())
    assert all(torch.count_nonzero(zero[k])==0 for k in up)
    assert all(torch.equal(changed[k],source[k]*gain if k in up else source[k]) for k in source)
    path=folder/'reward-ce-acoustic-gain150.safetensors'
    if path.exists():
        existing=load_file(str(path))
        if set(existing)!=set(changed) or not all(torch.equal(existing[k],v) for k,v in changed.items()):
            raise IntegrityError('Existing calibrated weights changed')
    else:
        save_file(changed,str(path))
    metadata=read(parent.with_suffix('.json'))
    metadata.update(weights_sha256=sha(path), calibration=dict(gain=gain, parent_sha256=identity['weights_sha256'],
        recipe_sha256=digest(recipe), local_optimizer_updates=0, inherited_gradient_updates=8))
    immutable(path.with_suffix('.json'),metadata)
    output=checkpoint(path,1.)
    audit=dict(passed=True,parent=identity,candidate=output,zero_up_exact=True,unit_all_tensors_exact=True,
        folded_gain=gain,folded_up_exact=True,unchanged_down_and_alpha=True,local_optimizer_updates=0,
        inherited_gradient_updates=8,ordinary_host_off_restoration_inherited=True,new_audio_generated=0)
    immutable(folder/'audit.json',audit)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--folder',required=True)
    a=p.parse_args();print(__import__('json').dumps(construct(a.parent,a.folder),indent=2))
