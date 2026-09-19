"""Serial execution record for the first adaptive MMD challenge."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = '/home/mikkel/anaconda3/envs/minimax-music3/bin/python'
GAME = str(HERE/'game.py')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1',
           HF_HOME='/ml2/music/.cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')


def run(command, log):
    start = time.monotonic()
    record = dict(command=command, log=str(log), started_utc=datetime.now(timezone.utc).isoformat())
    print(json.dumps(record), flush=True)
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
    record.update(seconds=time.monotonic()-start, returncode=result.returncode)
    with (HERE/'execution.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)
    result.check_returncode()


def main():
    # The first control job was already started interactively. Wait for its actual
    # process exit before launching another model, then require its registration.
    if len(sys.argv) > 1:
        path = Path('/proc')/sys.argv[1]/'cmdline'
        while path.exists():
            try:
                command = path.read_bytes()
            except FileNotFoundError:
                break
            if b'game.py' not in command or b'gan_original_600' not in command:
                break
            time.sleep(5)
    state = json.loads((HERE/'state.json').read_text())
    if 'gan_original_600' not in state['entries']:
        raise RuntimeError('The initial baseline did not register; inspect its logs')
    seed = ROOT/'models/conditional-mmd-cfg-research-20260905'
    seed_weights = seed/'conditional-mmd-cfg-research-20260905_step720.safetensors'
    references = [
        ('mmd_seed_720', 'mmd', seed_weights),
        ('gan_fm_cap_1950', 'reference', ROOT/'models/gan-v2/fm_capped-to1950-20260905/fm_capped-to1950-20260905_step1950.safetensors'),
        ('gan_bounded_1200', 'reference', ROOT/'models/gan-v2/baseline-to1200-20260905/baseline-to1200-20260905_step1200.safetensors'),
    ]
    for name, family, weights in references:
        run([PYTHON, GAME, 'evaluate', '--id', name, '--family', family, '--weights', str(weights)], HERE/f'{name}-execution.log')
    run([PYTHON, GAME, 'begin', '--hypothesis',
         'The MMD 720 plateau is partly caused by an oversized Adam proposal and a coarse six-fraction search.',
         '--change', 'Restore MMD 720 weights, Adam moments and RNG; preserve all fixed paired RBF targets, calibration and CFG branches; adapt proposal fractions with a normalized gradient fallback.',
         '--parent', 'mmd_seed_720', '--budget', '150'], HERE/'begin-round0001.log')
    state = json.loads((HERE/'state.json').read_text())
    name = state['active_round']
    if name != 'round-0001':
        raise RuntimeError('Unexpected round; this execution plan is specifically for round 0001')
    folder = HERE/'rounds'/name
    candidate = folder/'adaptive-candidate'
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='Candidate sets a new MMD personal best or overall lead on the locked board',
        candidate_id='round0001_adaptive',
        prompts=str(ROOT/'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml'),
        prompts_sha256=json.loads((HERE/'protocol.json').read_text())['prompts_sha256'],
        rows=[2, 3], seeds=[101, 303], duration=30., gpu=1,
        reference=state['champions']['overall'],
        reference_weights=state['entries'][state['champions']['overall']]['weights'],
        judge='Unchanged render-heuristic-v2; scores remain outside the board',
        diagnostics=['per-clip components and ASR errors', 'same-condition cross-seed melody and rhythm differences with quality and lyric checks'],
        limitation='Two seeds per prompt can detect obvious sameness but cannot establish full distributional mode coverage')
    confirmation = folder/'confirmation'; confirmation.mkdir()
    (confirmation/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    run([PYTHON, '-m', 'pytest', '-q', 'tests/test_mmd_game.py', 'tests/test_mmd_adaptive.py',
         'tests/test_distribution_objective.py', 'tests/test_conditional_energy.py'], folder/'validation.txt')
    run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.train_adaptive',
         '--source-state', str(seed/'state.pt'), '--source-weights', str(seed_weights),
         '--out', str(candidate), '--updates', '150'], folder/'training.log')
    completion = json.loads((candidate/'completion.json').read_text())
    run([PYTHON, GAME, 'evaluate', '--id', 'round0001_adaptive', '--family', 'mmd',
         '--round', name, '--weights', completion['final_weights']], folder/'evaluation.log')
    print('Candidate evaluated. Inspect results, run triggered confirmation, and write the round notes before finish.', flush=True)


if __name__ == '__main__':
    main()
