"""Measure deployed LoRA update geometry on CPU, without materializing dense deltas.

Run with the minimax-music3 interpreter. No model loads or studio mutations.
The metric is parameter-space geometry, not perceived strength or model quality.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

WORK = Path(__file__).resolve().parent
APP_ROOT = WORK.parents[2]
sys.path.insert(0, str(APP_ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def gram(up, down):
    """G_ij = <B_i A_i, B_j A_j>_F, including any scale baked into B.

    Factors may have different ranks. Multiplication stays in their joint
    rank space. Float64 accumulation avoids cancellation when summing modules.
    """
    ranks = [x.shape[1] for x in up]
    bounds = [0]
    for rank in ranks:
        bounds.append(bounds[-1] + rank)
    b = torch.cat(up, dim=1).double()
    a = torch.cat(down, dim=0).double()
    cross = (b.T @ b) * (a @ a.T)
    return torch.stack([
        torch.stack([cross[bounds[i]:bounds[i+1], bounds[j]:bounds[j+1]].sum()
                     for j in range(len(ranks))])
        for i in range(len(ranks))
    ])


def summary(g, ids):
    norms = g.diag().clamp_min(0).sqrt()
    cosine = g / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    pairs = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            # Equal mix at the SAME studio energy, versus the mean solo norm.
            # This includes unequal adapter norms and their mutual overlap.
            retention = ((g[i, i] + g[j, j] + 2*g[i, j]).clamp_min(0).sqrt()
                         / (norms[i] + norms[j]).clamp_min(1e-30))
            pairs.append(dict(a=ids[i], b=ids[j], cosine=cosine[i, j].item(),
                              equal_mix_norm_retention=retention.item()))
    pairs.sort(key=lambda p: p['cosine'])
    equal = torch.ones(len(ids), dtype=torch.float64) / len(ids)
    return dict(norms=dict(zip(ids, norms.tolist())), cosine=cosine.tolist(),
                norm_ratio_max_min=(norms.max()/norms.min()).item(),
                pairs=pairs,
                equal_all_mix_norm_retention=((equal @ g @ equal).clamp_min(0).sqrt()
                                             / norms.mean()).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=WORK/'geometry.json')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from app import sliders
    registry_hash = sha(sliders.REGISTRY_PATH)
    ids = [s['id'] for s in sliders.catalog()['sliders']]
    components = []
    for slider_id in ids:
        resolved = sliders.resolve([dict(id=slider_id, scale=1)], budget=None)
        if len(resolved) != 1 or resolved[0]['kind'] != 'language_model':
            raise ValueError('This audit requires one LM component per slider')
        components.append(resolved[0])
    records = [dict(id=i, weights=c['weights'], sha256=sha(c['weights']),
                    sidecar_sha256=sha(sliders._sidecar_path(Path(c['weights']))),
                    multiplier_at_absolute_one=c['multiplier'])
               for i, c in zip(ids, components)]
    with ExitStack() as stack:
        files = [stack.enter_context(safe_open(c['weights'], framework='pt', device='cpu'))
                 for c in components]
        keys = set(files[0].keys())
        if any(set(f.keys()) != keys for f in files):
            raise ValueError('Adapters target different module sets')
        modules = sorted(k[:-len('.alpha')] for k in keys if k.endswith('.alpha'))
        total = torch.zeros(len(ids), len(ids), dtype=torch.float64)
        per_module = {}
        for index, module in enumerate(modules):
            up, down = [], []
            for f, c in zip(files, components):
                a = f.get_tensor(module+'.lora_down.weight').double()
                b = f.get_tensor(module+'.lora_up.weight').double()
                alpha = float(f.get_tensor(module+'.alpha')) or a.shape[0]
                if a.ndim != 2 or b.ndim != 2:
                    raise ValueError('Only linear adapters are supported')
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError('Nonfinite adapter weights')
                if alpha != c['alpha'] or a.shape[0] != c['rank']:
                    raise ValueError('Sidecar and tensor scaling disagree')
                up.append(b * (alpha/a.shape[0]) * c['multiplier'])
                down.append(a)
            g = gram(up, down)
            total += g
            per_module[module] = g.tolist()
            if (index+1) % 24 == 0:
                print(f'{index+1}/{len(modules)} modules', flush=True)
    if sha(sliders.REGISTRY_PATH) != registry_hash:
        raise ValueError('Registry changed during audit')
    for rec in records:
        if sha(rec['weights']) != rec['sha256']:
            raise ValueError('Weights changed during audit')
        if sha(sliders._sidecar_path(Path(rec['weights']))) != rec['sidecar_sha256']:
            raise ValueError('Sidecar changed during audit')
    result = dict(schema=1, registry_sha256=registry_hash, script_sha256=sha(__file__),
                  interpretation='Exact fp64 parameter-update Gram matrices; no activation or audio quality claim.',
                  ids=ids, checkpoints=records, modules=len(modules), gram=total.tolist(),
                  summary=summary(total, ids), per_module_gram=per_module)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['summary'].items() if k not in ('cosine','pairs')}, indent=2))
    print('Lowest overlap:', result['summary']['pairs'][0])
    print('Highest overlap:', result['summary']['pairs'][-1])


if __name__ == '__main__':
    main()
