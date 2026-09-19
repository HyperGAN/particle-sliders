"""Register and run the frozen active-history coverage challenge after round two."""
from datetime import datetime, timezone
import json
import shutil

from .run_round0001 import run, ROOT, HERE, PYTHON, GAME


def main():
    state = json.loads((HERE/'state.json').read_text())
    if state['active_round'] or len(state['completed_rounds']) != 2:
        raise ValueError('Complete round two before starting this experiment')
    run([PYTHON, GAME, 'begin', '--parent', 'mmd_seed_720', '--budget', '150',
         '--hypothesis', 'Matching only neutral histories undercovers histories encountered with the slider active, contributing to the float32 candidate audio regression.',
         '--change', 'From retained MMD 720, freeze one additional parent-generated +1 history per training condition using seeds 17 through 20; train on an equal mixture of original and added histories with the round-two float32 MMD recipe.'], HERE/'begin-round0003.log')
    folder = HERE/'rounds/round-0003'
    seed = ROOT/'models/conditional-mmd-cfg-research-20260905'
    candidate = folder/'coverage-candidate'
    state = json.loads((HERE/'state.json').read_text())
    reference = state['champions']['overall']
    extras = [name for name in ('gan_v2_baseline', 'gan_fm_cap_1950') if name != reference]
    confirmation = folder/'confirmation'
    confirmation.mkdir()
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='New MMD personal best or overall lead', candidate_id='round0003_coverage',
        prompts=str(ROOT/'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml'),
        prompts_sha256=json.loads((HERE/'protocol.json').read_text())['prompts_sha256'],
        rows=[2, 3], seeds=[101, 303], duration=30., gpu=1,
        reference=reference, reference_weights=state['entries'][reference]['weights'],
        extra_references=extras, extra_reference_weights={name:state['entries'][name]['weights'] for name in extras},
        judge='Unchanged render-heuristic-v2; separate from development leaderboard',
        substantial_development_gain=0.15,
        substantial_gain_definition='At least +0.15 over the round starting leader plus positive extra-fixture mean gain, subject to individual lyric/quality regressions and listening',
        diagnostics=['per-example concept, enjoyment, production, lyrics and artifact failures',
                     'within-condition cross-seed chroma, rhythm and content spread with quality checks'],
        limitation='Two seeds per condition cannot establish full mode coverage')
    (confirmation/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    tests = ['test_mmd_confirmation_cache.py', 'test_mmd_coverage.py', 'test_mmd_precision.py', 'test_mmd_adaptive.py',
             'test_mmd_game.py', 'test_distribution_objective.py', 'test_conditional_energy.py']
    validation = folder/'validation-source'
    validation.mkdir()
    for filename in tests:
        shutil.copyfile(ROOT/'tests'/filename, validation/filename)
    run([PYTHON, '-m', 'pytest', '-q', *['tests/'+filename for filename in tests]], folder/'validation.txt')
    run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.train_coverage',
         '--source-state', str(seed/'state.pt'),
         '--source-weights', str(seed/'conditional-mmd-cfg-research-20260905_step720.safetensors'),
         '--prepared', str(HERE/'rounds/round-0001/adaptive-candidate/prepared.pt'),
         '--reference-float32-prepared', str(HERE/'rounds/round-0002/float32-candidate/prepared.pt'),
         '--out', str(candidate), '--updates', '150'], folder/'training.log')
    completion = json.loads((candidate/'completion.json').read_text())
    run([PYTHON, GAME, 'evaluate', '--id', 'round0003_coverage', '--family', 'mmd',
         '--round', 'round-0003', '--weights', completion['final_weights']], folder/'evaluation.log')
    state = json.loads((HERE/'state.json').read_text())
    card = json.loads((folder/'round.json').read_text())
    entry = state['entries']['round0003_coverage']
    if entry['eligible'] and entry['score'] > card['initial_best_scores']['mmd']+1e-9:
        run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.confirm_cached',
             '--round', 'round-0003', '--reuse', str(HERE/'rounds/round-0002/confirmation-baseline')], confirmation/'driver.log')
    else:
        (confirmation/'outcome.json').write_text(json.dumps(dict(status='not_triggered',
            reason='No new eligible MMD personal best'), indent=2)+'\n')
    print('Coverage candidate scored. Inspect evidence, finish this round, and continue toward the user objective.', flush=True)


if __name__ == '__main__':
    main()
