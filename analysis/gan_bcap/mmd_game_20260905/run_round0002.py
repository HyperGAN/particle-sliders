"""Serial, logged precision experiment and fixed-judge evaluation for round two."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .run_round0001 import run, ROOT, HERE, PYTHON, GAME


def main():
    folder = HERE/'rounds/round-0002'
    state = json.loads((HERE/'state.json').read_text())
    if state['active_round'] != 'round-0002':
        raise ValueError('Expected active round two')
    seed = ROOT/'models/conditional-mmd-cfg-research-20260905'
    candidate = folder/'float32-candidate'
    confirmation = folder/'confirmation'
    confirmation.mkdir(exist_ok=True)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='New MMD personal best or overall lead', candidate_id='round0002_float32',
        prompts=str(ROOT/'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml'),
        prompts_sha256=json.loads((HERE/'protocol.json').read_text())['prompts_sha256'],
        rows=[2, 3], seeds=[101, 303], duration=30., gpu=1,
        reference='gan_fm_cap_1950', reference_weights=state['entries']['gan_fm_cap_1950']['weights'],
        judge='Unchanged render-heuristic-v2, outside the board',
        diagnostics=['per-clip scores and lyric failures', 'cross-seed chroma, rhythm and content spread with quality checks'],
        substantial_development_gain=0.15,
        substantial_gain_definition='At least +0.15 over the previous overall leader, plus positive extra-fixture mean gain; still subject to listening',
        limitation='Two samples per condition cannot establish full mode coverage')
    plan_path = confirmation/'plan.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    tests = ['test_mmd_precision.py', 'test_mmd_adaptive.py', 'test_mmd_game.py',
             'test_distribution_objective.py', 'test_conditional_energy.py']
    validation = folder/'validation-source'
    validation.mkdir(exist_ok=True)
    for filename in tests:
        shutil.copyfile(ROOT/'tests'/filename, validation/filename)
    run([PYTHON, '-m', 'pytest', '-q', *['tests/'+filename for filename in tests]], folder/'validation.txt')
    run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.train_float32',
         '--source-state', str(seed/'state.pt'),
         '--source-weights', str(seed/'conditional-mmd-cfg-research-20260905_step720.safetensors'),
         '--prepared', str(HERE/'rounds/round-0001/adaptive-candidate/prepared.pt'),
         '--out', str(candidate), '--updates', '150'], folder/'training.log')
    completion = json.loads((candidate/'completion.json').read_text())
    run([PYTHON, GAME, 'evaluate', '--id', 'round0002_float32', '--family', 'mmd',
         '--round', 'round-0002', '--weights', completion['final_weights']], folder/'evaluation.log')
    state = json.loads((HERE/'state.json').read_text())
    score = state['entries']['round0002_float32']['score']
    card = json.loads((folder/'round.json').read_text())
    if score > card['initial_best_scores']['mmd']+1e-9:
        run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.confirm_round',
             '--round', 'round-0002'], confirmation/'driver.log')
    else:
        (confirmation/'outcome.json').write_text(json.dumps(dict(status='not_triggered',
            reason='Candidate did not improve the MMD personal best'), indent=2)+'\n')
    # Extend coverage by one prioritized historical contender on the same judge.
    run([PYTHON, GAME, 'evaluate', '--id', 'gan_v2_baseline', '--family', 'reference',
         '--weights', str(ROOT/'models/gan-v2/baseline-660-20260905/baseline-660-20260905_step660.safetensors')],
        folder/'opponent-evaluation.log')
    print('Experiment and scoring complete; inspect evidence and finish the round.', flush=True)


if __name__ == '__main__':
    main()
