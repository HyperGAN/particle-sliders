"""Fresh-process replay qualification of tiny native image bindings; CPU only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))


def worker(args):
    import torch
    from test_pg09_images import FAMILIES, game_fixture
    families = list(FAMILIES) + ['iris'] if args.suite == 'images' else ['qwen_image21']
    previous = torch.load(args.input, weights_only=False) if args.input else {}
    result = {}
    for family in families:
        for preset in ('bispf', 'bispf_optimized'):
            key = family + '/' + preset
            if family == 'qwen_image21':
                from test_pg09_qwen_image import fixture
                game = fixture(preset)[-1]
            else:
                game = game_fixture(family, preset)[-1]
            history = []
            if previous:
                game.load_state_dict(previous[key]['state'])
                history = previous[key]['metrics']
            for _ in range(args.steps):
                history.append(game.step())
            state = deepcopy(game.state_dict())
            result[key] = dict(state=state, metrics=history, output=game.evaluate(0, seed=91))
    assert not torch.cuda.is_initialized()
    torch.save(result, args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['images','qwen'], default='images')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU qualification requires CUDA_VISIBLE_DEVICES empty')
    args.output.mkdir(parents=True, exist_ok=True)
    def run(name, steps, prior=None):
        target = args.output / (name+'.pt')
        command = [sys.executable, __file__, '--worker', '--suite', args.suite,
                   '--output', str(target), '--steps', str(steps)]
        if prior is not None:
            command += ['--input', str(prior)]
        subprocess.run(command, check=True, timeout=120)
        return target
    full = run('continuous',5)
    split = run('split',2)
    resumed = run('resumed',3,split)
    import torch
    from test_pg09_images import equal
    expected, actual = (torch.load(path, weights_only=False) for path in (full,resumed))
    equal(expected,actual)
    receipt = dict(core_commit='ec234e184676c46060ecbf6122df0e7b726f45a7',
                   cuda_initialized=False, protocol='5 updates == 2 + fresh-process 3 (includes lazy fourth-step penalty)',
                   cases={key:dict(exact_state=True, exact_losses=True, exact_evaluation=True,
                         signature=value['state']['config']) for key,value in actual.items()})
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'cases'}))
    print('Exact replay:',', '.join(receipt['cases']))


if __name__ == '__main__':
    main()
