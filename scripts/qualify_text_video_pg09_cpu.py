"""Bounded, offline, two-process continuation gates for the shared game."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROGRAM = r'''
import json, sys, torch
from pathlib import Path
from test_pg09_text_video import make, equal
root, family, preset, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
adapter, rows, game = make(family, preset)
if mode == 'produce':
    game.step(); game.step()
    torch.save(game.state_dict(), root / 'checkpoint.pt')
    metrics = [game.step(), game.step()]
    torch.save(dict(metrics=metrics, state=game.state_dict(), sample=game.evaluate(0, seed=912)),
               root / 'expected.pt')
else:
    game.load_state_dict(torch.load(root / 'checkpoint.pt', weights_only=False))
    before = game.state_dict()
    game.evaluate(0, seed=912)
    equal(before, game.state_dict())
    metrics = [game.step(), game.step()]
    expected = torch.load(root / 'expected.pt', weights_only=False)
    assert metrics == expected['metrics']
    equal(game.state_dict(), expected['state'])
    assert torch.equal(game.evaluate(0, seed=912), expected['sample'])
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(family=family, preset=preset, completed_steps=4,
                         exact_losses=True, exact_full_state=True, exact_sample=True,
                         evaluation_preserves_training_rng=True, cuda_initialized=False,
                         response_shape=list(rows[0].target.shape),
                         generator_arm=game.state_dict()['config']['generator_arm'],
                         implementation_sources=game.state_dict()['config']['implementation_sources'],
                         pg_sources_sha256=game.state_dict()['config']['pg_sources_sha256'],
                         conditions_sha256=game.state_dict()['config']['conditions_sha256'])))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', required=True, type=Path,
                        help='verified core checkout, containing packages/particle-sliders-core')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    core_src = args.core.resolve() / 'packages/particle-sliders-core/src'
    core_module = core_src / 'particle_sliders/pg09.py'
    if not core_module.is_file():
        parser.error('selected checkout does not contain the shared ConditionalPG09 module')
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(map(str, (core_src, root, root / 'tests'))))
    cases = []
    core_digest = hashlib.sha256(core_module.read_bytes()).hexdigest()
    for family in ('tiny', 'ltx'):
        for preset in ('bispf', 'bispf_optimized'):
            folder = args.output / f'{family}-{preset}'
            folder.mkdir(exist_ok=True)
            for mode in ('produce', 'restore'):
                result = subprocess.run([sys.executable, '-c', PROGRAM, str(folder), family, preset, mode],
                                        env=env, text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                if hashlib.sha256(core_module.read_bytes()).hexdigest() != core_digest:
                    raise RuntimeError('selected core changed during qualification; use an immutable checkout')
                if mode == 'restore':
                    cases.append(json.loads(result.stdout))
    receipt = dict(qualification='CPU software continuation only; no pretrained quality or GPU admission',
                   core_sha256=core_digest,
                   adapter_sha256=hashlib.sha256((root / 'conceptmod/textsliders/pg09_text_video.py').read_bytes()).hexdigest(),
                   cases=cases)
    (args.output / 'fresh-process-resume.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
