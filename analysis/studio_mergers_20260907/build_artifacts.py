"""Stream exact dense merger outputs to disk, one attention projection at a time."""
from contextlib import ExitStack
import json
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from common import WORK, OLD, PAIRS, sha, write
from mergers import ties, knots_ties, match_norm
from app import sliders


def main():
    torch.set_num_threads(4)
    old = json.loads((OLD/'screen.json').read_text())
    checks = {c['weights']: c['sha256'] for c in old['checkpoints']}
    for path, expected in checks.items():
        assert sha(path) == expected, path
    for pair in PAIRS:
        key = '+'.join(pair)
        comps = sliders.resolve([dict(id=i, scale=.5) for i in pair], host_energy={'language_model':2.8})
        out = WORK/'artifacts'/key
        out.mkdir(parents=True, exist_ok=True)
        manifests = {}
        for method in ('ties', 'knots_ties'):
            p = out/method
            p.mkdir(exist_ok=True)
            assert not (p/'manifest.json').exists(), 'Refuse to replace frozen artifacts'
            manifests[method] = dict(schema=1, method=method, density=.3,
                trimming='per_projection', strength='per_projection_Frobenius_matched_to_linear_energy_2.8',
                source_components=comps, source_sha256={c['weights']:sha(c['weights']) for c in comps},
                algorithm_sha256=sha(WORK/'mergers.py'), builder_sha256=sha(__file__), modules=[])
        with ExitStack() as stack:
            files = [stack.enter_context(safe_open(c['weights'],framework='pt',device='cpu')) for c in comps]
            keys = set(files[0].keys())
            assert all(set(f.keys()) == keys for f in files)
            modules = sorted(k[:-6] for k in keys if k.endswith('.alpha'))
            assert len(modules) == 144
            for index, module in enumerate(modules):
                started = time.time()
                ups, downs, deltas = [], [], []
                for f, c in zip(files, comps):
                    a = f.get_tensor(module+'.lora_down.weight').float()
                    b = f.get_tensor(module+'.lora_up.weight').float()
                    alpha = float(f.get_tensor(module+'.alpha')) or a.shape[0]
                    assert alpha == c['alpha'] and a.shape[0] == c['rank']
                    scale = c['multiplier']*(alpha/a.shape[0])
                    # Keep the ordinary baseline's FP32 contraction order.
                    deltas.append((b @ a)*scale)
                    ups.append(b*scale)
                    downs.append(a)
                linear = sum(deltas)
                for method in manifests:
                    if method == 'ties':
                        delta, rank = ties(deltas, .3), None
                    else:
                        delta, rank = knots_ties(ups, downs, .3)
                    delta, metrics = match_norm(delta, linear)
                    assert abs(metrics['matched_norm']/metrics['reference_norm']-1) < 2e-5
                    path = out/method/f'{index:03d}.safetensors'
                    save_file({'delta':delta.contiguous()},str(path))
                    manifests[method]['modules'].append(dict(name=module, path=str(path),sha256=sha(path),
                        shape=list(delta.shape), joint_rank=rank, **metrics))
                    del delta
                del ups, downs, deltas, linear
                if (index+1)%12 == 0:
                    print(f'{key}: {index+1}/144 projections; last {time.time()-started:.2f}s',flush=True)
        for method, manifest in manifests.items():
            write(out/method/'manifest.json', manifest)
        print(f'COMPLETE {key}',flush=True)
    print('ALL ARTIFACTS COMPLETE',flush=True)


if __name__ == '__main__':
    main()
