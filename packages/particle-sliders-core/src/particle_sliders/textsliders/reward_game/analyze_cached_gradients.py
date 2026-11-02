"""Descriptive CPU analysis of the retained failed cache audit; no new pass rule."""
import argparse
import math
from pathlib import Path
from .core import read, immutable, sha


def analyze(home):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    home = Path(home).resolve(); folder = home/'audit/attention-cache-v1'
    paths = {name: folder/f'{name}-gradients.safetensors' for name in ('legacy', 'cached')}
    legacy, cached = (load_file(str(paths[name])) for name in ('legacy', 'cached'))
    if legacy.keys() != cached.keys():
        raise ValueError('Gradient supports differ')
    sums = dict(a2=0., b2=0., diff2=0., dot=0., a1=0., sign_flip_a1=0.)
    count = flips = exact = 0; max_absolute = 0.; per_tensor = []
    for name in legacy:
        a, b = legacy[name].double(), cached[name].double()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Malformed retained gradient')
        difference = a-b; changed_sign = a.sign() != b.sign()
        a2, b2, diff2 = (float(x.square().sum()) for x in (a, b, difference))
        dot = float((a*b).sum())
        for key, value in dict(a2=a2, b2=b2, diff2=diff2, dot=dot,
                a1=float(a.abs().sum()), sign_flip_a1=float(a[changed_sign].abs().sum())).items():
            sums[key] += value
        count += a.numel(); flips += int(changed_sign.sum()); exact += int(torch.equal(a,b))
        maximum = float(difference.abs().max()); max_absolute = max(max_absolute, maximum)
        per_tensor.append(dict(name=name, relative_l2=math.sqrt(diff2/a2) if a2 else None,
                               max_absolute=maximum, legacy_norm=math.sqrt(a2)))
    result = dict(input_sha256={name:sha(path) for name,path in paths.items()},
        protocol_file_sha256=sha(folder/'protocol.json'), source_sha256=sha(__file__),
        tensors=len(legacy), exact_tensors=exact, coefficients=count,
        legacy_norm=math.sqrt(sums['a2']), cached_norm=math.sqrt(sums['b2']),
        global_relative_l2=math.sqrt(sums['diff2']/sums['a2']),
        global_cosine=sums['dot']/math.sqrt(sums['a2']*sums['b2']),
        max_absolute=max_absolute, sign_disagreements=flips,
        sign_disagreement_fraction=flips/count,
        legacy_absolute_gradient_fraction_on_sign_disagreements=sums['sign_flip_a1']/sums['a1'],
        largest_relative_tensor=max(per_tensor,key=lambda r:r['relative_l2'] or 0),
        interpretation='Descriptive aggregation of an already failed exact-gradient audit. No baseline repeat or cause isolation, no new acceptance tolerance, no cached implementation authorization. Current recomputed ordinary training remains unchanged.',
        new_audio_generated=0, new_optimizer_updates=0, per_tensor=per_tensor)
    immutable(home/'audit/cache-gradient-aggregate-diagnostic-v1.json',result)
    return {key:value for key,value in result.items() if key!='per_tensor'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--home',required=True)
    args=parser.parse_args();print(__import__('json').dumps(analyze(args.home),indent=2))
