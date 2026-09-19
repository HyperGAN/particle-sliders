"""Register the fixed eight-step attempt after its actual full-flow audits."""
import argparse
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError


def prepare(home):
    from .acoustic_controller import register
    home = Path(home).resolve()
    folder = home/'training/acoustic-full-flow-legacy-v1'
    audit = home/'audit/acoustic-full-flow-legacy-v1'
    previous = home/'audit/acoustic-tail-v1'
    game_home = home/'acoustic-v1'
    intent_path = home/'recipes/full-within-chunk-flow-legacy-intent-v1.json'
    intent = read(intent_path)
    for path, expected in intent['sources'].items():
        if sha(path) != expected:
            raise IntegrityError('Declared full-flow source changed')
    result = read(audit/'gradient-result.json')
    if not all(result[k] for k in ('passed', 'zero_waveform_exact',
            'nonzero_complete_waveform_matches_ordinary_merge', 'off_waveform_exact',
            'base_and_vocoder_frozen')):
        raise IntegrityError('Actual full-flow gradient/native/Off audit must pass')
    if not read(audit/'gradient-off-restoration.json')['exact']:
        raise IntegrityError('Actual full-flow audit failed cleanup')
    cp = read(previous/'gradient-protocol.json'); cr = read(previous/'gradient-result.json')
    if not cr['passed'] or cp['target_replace'] != ['MiniMaxMusic3Attention']:
        raise IntegrityError('Existing ordinary attention gradient audit must pass')
    collection = read(folder/'collection-recipe.json')
    captures = read(folder/'captures.json')
    if collection['retained_flow_steps'] != 30 or len(captures) != 8:
        raise IntegrityError('Need the complete eight-case thirty-step collection')
    if any(r['status'] != 'complete' or r['recipe_sha256'] != digest(collection)
           for r in captures) or sum(r['new_clips'] for r in captures) != 8:
        raise IntegrityError('Full-flow collection is incomplete or its accounting differs')
    for row in captures:
        for key in ('audio', 'capture'):
            if sha(row[key]) != row[key+'_sha256']:
                raise IntegrityError('Full-flow capture changed')
    recipe = read(home/'recipes/acoustic-ce-8.json')
    recipe.update(
        hypothesis=intent['hypothesis'],
        failure_mechanism='Two-step acoustic derivatives improved development but failed fresh mean gains; test earlier within-chunk flow credit.',
        parent_checkpoint=None, changed_variables=intent['changed_variables'],
        checkpoint=str(folder/'reward-ce-full-flow_step8.safetensors'),
        training_state=str(folder/'state-step8.pt'), training_status=str(folder/'status.json'),
        expected_updates=8, initial_updates=0,
        interpretation='All thirty within-chunk flow steps; fixed conditioning and detached reference overlap. Not full generator backpropagation.',
        next_after_failure='Retain the result and select a materially different method; no broad strength or duration grid.')
    recipe['sources'].update(intent['sources'])
    recipe['sources'][str(Path(__file__).resolve())] = sha(__file__)
    recipe['budget'].update(optimizer_updates=8, new_training_clips=0,
        new_duplicate_engineering_clips=8, new_clips=16, max_seconds=14400)
    recipe['argv'] = [recipe['argv'][0], '-u', '-m',
        'conceptmod.textsliders.reward_game.train_full_flow_legacy_ce', '--home', str(home),
        '--game-home', str(game_home), '--folder', str(folder),
        '--full-audit-folder', str(audit), '--previous-audit-folder', str(previous),
        '--total', '8', '--lr', '0.001']
    recipe['preprocessing'].update(
        acoustic_collection=str(folder/'collection-recipe.json'),
        collection_recipe_sha256=sha(folder/'collection-recipe.json'),
        captures_sha256=sha(folder/'captures.json'),
        capture_sha256={r['id']:r['capture_sha256'] for r in captures},
        full_flow_intent_sha256=sha(intent_path),
        full_flow_gradient_audit_sha256=sha(audit/'gradient-result.json'),
        previous_attention_audit_sha256=sha(previous/'gradient-result.json'),
        independent_training_clips_added=0, duplicate_engineering_clips_added=8)
    path = home/'recipes/acoustic-full-flow-legacy-8.json'
    immutable(path, recipe)
    return dict(register(game_home, path), game_home=str(game_home), recipe=str(path),
                research_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--home', required=True)
    args = parser.parse_args()
    print(__import__('json').dumps(prepare(args.home), indent=2))
