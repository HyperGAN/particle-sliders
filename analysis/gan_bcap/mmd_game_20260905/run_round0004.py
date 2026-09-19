"""One preset strength calibration, followed by the unchanged audio challenge."""
from datetime import datetime, timezone
import json
import shutil

from .run_round0001 import run, ROOT, HERE, PYTHON, GAME


def main():
    state = json.loads((HERE/'state.json').read_text())
    if state['active_round'] or len(state['completed_rounds']) != 3:
        raise ValueError('Complete round three before this experiment')
    parent = state['champions']['overall']
    if parent != 'gan_v2_baseline':
        raise ValueError('Reconsider this declared baseline660 branch if the leader changed')
    run([PYTHON, GAME, 'begin', '--parent', parent, '--family', 'calibration', '--budget', '0',
         '--hypothesis', 'The strongest measured GAN may be below its best useful adapter strength; closer positive-caption hidden matching regressed the measured concept effect.',
         '--change', 'Construct one preset1.25 strength calibration by multiplying all alpha buffers and sidecar alpha, retaining every trained matrix; zero optimizer updates and unchanged renderer scale+1.'], HERE/'begin-round0004.log')
    folder = HERE/'rounds/round-0004'
    candidate = folder/'strength125-candidate'
    confirmation = folder/'confirmation'
    confirmation.mkdir()
    reference = state['entries'][parent]
    extra = state['entries']['gan_fm_cap_1950']
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='New eligible overall development lead', candidate_id='round0004_strength125',
        prompts=str(ROOT/'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml'),
        prompts_sha256=json.loads((HERE/'protocol.json').read_text())['prompts_sha256'],
        rows=[4, 5], seeds=[515, 727], duration=30., gpu=1,
        reference=parent, reference_weights=reference['weights'],
        extra_references=[extra['id']], extra_reference_weights={extra['id']:extra['weights']},
        judge='Unchanged render-heuristic-v2, outside development aggregation',
        substantial_development_gain=.15,
        substantial_gain_definition='At least +0.15 over the starting overall leader plus positive added-fixture mean gain, with individual lyric and quality regressions inspected',
        diagnostics=['per-clip concept, enjoyment, production, lyrics and artifact failures',
                     'within-condition cross-seed chroma, rhythm and content spread alongside quality'],
        limitation='Two seeds per condition cannot establish mode coverage; human listening remains unmeasured',
        fresh_scope='Rows4/5 and seeds515/727 have not been used by preceding game comparisons')
    (confirmation/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    tests = ['test_mmd_game.py', 'test_mmd_families.py', 'test_mmd_strength.py', 'test_mmd_confirmation_cache.py']
    validation = folder/'validation-source'
    validation.mkdir()
    for name in tests:
        shutil.copyfile(ROOT/'tests'/name, validation/name)
    provenance = folder/'bookkeeping-provenance'
    provenance.mkdir()
    for name in ('game.py', 'family_bookkeeping.py', 'run_round0004.py', 'confirm_cached.py', 'confirmation_cache.py'):
        shutil.copyfile(HERE/name, provenance/name)
    from .game import sha, load_game
    protocol, _ = load_game(HERE)
    (provenance/'manifest.json').write_text(json.dumps(dict(
        authorization='The user explicitly authorized GANs/other methods and continued experiments for top score.',
        change='Method labels and zero-update construction rounds; MMD champion remains restricted to MMD entries.',
        sources={str((HERE/name).resolve()):sha(provenance/name) for name in ('game.py', 'family_bookkeeping.py', 'run_round0004.py', 'confirm_cached.py', 'confirmation_cache.py')},
        frozen_judge_sources=protocol['sources'], frozen_protocol_sha256=sha(HERE/'protocol.json')), indent=2)+'\n')
    run([PYTHON, '-m', 'pytest', '-q', *['tests/'+name for name in tests]], folder/'validation.txt')
    run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.calibrate_strength',
         '--source-weights', reference['weights'], '--source-state', str(__import__('pathlib').Path(reference['weights']).parent/'state.pt'),
         '--factor', '1.25', '--out', str(candidate)], folder/'construction.log')
    completion = json.loads((candidate/'completion.json').read_text())
    run([PYTHON, GAME, 'evaluate', '--id', 'round0004_strength125', '--family', 'calibration',
         '--label', 'GAN660 strength1.25', '--round', 'round-0004', '--weights', completion['final_weights']], folder/'evaluation.log')
    state = json.loads((HERE/'state.json').read_text())
    entry = state['entries']['round0004_strength125']
    if entry['eligible'] and entry['score'] > reference['score']+1e-9:
        run([PYTHON, '-m', 'analysis.gan_bcap.mmd_game_20260905.confirm_cached',
             '--round', 'round-0004'], confirmation/'driver.log')
    else:
        (confirmation/'outcome.json').write_text(json.dumps(dict(status='not_triggered',
            reason='No new eligible overall development lead'), indent=2)+'\n')
    print('Strength candidate scored. Inspect, finish this round, and continue if the improvement is insufficient.', flush=True)


if __name__ == '__main__':
    main()
