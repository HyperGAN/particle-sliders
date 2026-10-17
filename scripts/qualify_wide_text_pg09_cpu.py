"""Historical LTX connector fixture; Tiny/Bonsai routes are retired here.

Uses the existing reviewed overlay via PYTHONPATH, never installs or patches.
No pretrained assets are opened. ConditionalPG09 owns every optimizer update.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROGRAM = r'''
import json, pickle, random, sys, torch
import numpy as np
from pathlib import Path
from test_pg09_wide_text import make_wide
from test_pg09_text_video import equal
root, family, preset, profile, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
if family == 'ltx_connector':
    from test_pg09_ltx_embed import make_connector
    adapter, rows, game = make_connector(preset)
else:
    adapter, rows, game = make_wide(family, preset, profile=profile, width=64 if mode.startswith('cross') else 1024)
if mode in ('produce', 'cross-produce'):
    game.step(); game.step()
    torch.save(game.state_dict(), root / 'checkpoint.pt')
    if mode == 'produce':
        metrics = [game.step(), game.step()]
        torch.save(dict(metrics=metrics, state=game.state_dict(), sample=game.evaluate(0, seed=912)), root / 'expected.pt')
elif mode == 'restore':
    game.load_state_dict(torch.load(root / 'checkpoint.pt', weights_only=True))
    before = game.state_dict()
    game.evaluate(0, seed=912)
    equal(before, game.state_dict())
    metrics = [game.step(), game.step()]
    expected = torch.load(root / 'expected.pt', weights_only=True)
    assert metrics == expected['metrics']
    equal(game.state_dict(), expected['state'])
    assert torch.equal(game.evaluate(0, seed=912), expected['sample'])
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(family=family, preset=preset, profile=profile, completed_steps=4,
        response_shape=list(rows[0].target.shape), exact_losses=True, exact_full_state=True,
        exact_sample=True, evaluation_preserves_training_rng=True, cuda_initialized=False,
        implementation_sources=game.state_dict()['config']['implementation_sources'],
        source_profile=game.state_dict()['config']['pg_source_profile'],
        penalty_input_domain=game.state_dict()['config']['penalty_input_domain'],
        resolved_penalty_domain=game.state_dict()['config']['resolved_penalty_domain'])))
else:
    before, torch_rng = game.state_dict(), torch.get_rng_state().clone()
    other_rng = pickle.dumps((random.getstate(), np.random.get_state()))
    try:
        game.load_state_dict(torch.load(root / 'checkpoint.pt', weights_only=True))
    except ValueError as exc:
        assert 'mismatch' in str(exc)
    else:
        raise AssertionError('cross-profile checkpoint was accepted')
    equal(before, game.state_dict())
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert other_rng == pickle.dumps((random.getstate(), np.random.get_state()))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(profile=profile, cross_profile_rejected=True,
                         live_state_unchanged=True, global_rng_unchanged=True, cuda_initialized=False)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-ltx', action='store_true',
                        help='historical flag; explicitly select --family ltx_connector')
    parser.add_argument('--family', choices=('tiny', 'bonsai', 'ltx_connector'), default='tiny',
                        help='LTX connector only here; Tiny/Bonsai use installed-package acceptance')
    args = parser.parse_args()
    if args.family != 'ltx_connector':
        parser.error('legacy Tiny/Bonsai source-checkout qualification is retired; use '
                     'the pinned installed-package acceptance; select --family '
                     'ltx_connector for the separately owned LTX harness')
    root = Path(__file__).resolve().parents[1]
    core_src = args.core.resolve() / 'packages/particle-sliders-core/src'
    module = core_src / 'particle_sliders/pg09.py'
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, cross = [], []

    def run(folder, family, preset, profile, mode):
        parts = [core_src, root, root / 'tests']
        if profile == 'alz-continuous-v1':
            parts.insert(0, args.overlay.resolve())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=os.pathsep.join(map(str, parts)))
        result = subprocess.run([sys.executable, '-c', PROGRAM, str(folder), family, preset, profile, mode],
                                env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        if hashlib.sha256(module.read_bytes()).hexdigest() != digest:
            raise RuntimeError('selected core changed during qualification')
        return json.loads(result.stdout) if result.stdout else None

    families = ['ltx_connector']
    for family in families:
        for preset in ('bispf', 'bispf_optimized'):
            folder = args.output / f'{family}-{preset}'
            folder.mkdir(exist_ok=True)
            run(folder, family, preset, 'alz-continuous-v1', 'produce')
            cases.append(run(folder, family, preset, 'alz-continuous-v1', 'restore'))
    receipt = dict(qualification='synthetic native-width CPU software gates only; no pretrained model admission',
                   core_sha256=digest, cases=cases, cross_profile_cases=cross,
                   installed_package_modified=False, pretrained_assets_loaded=False)
    (args.output / 'wide-fresh-process.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), cross_profile_cases=len(cross), all_passed=True,
                          pretrained_assets_loaded=False, installed_package_modified=False)))


if __name__ == '__main__':
    main()
