"""Register joint robust block refinement after exact parent capture replay."""
import argparse
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError


def prepare(home):
    from .robust_block_controller import register
    from .block_artifact import checkpoint, STRUCTURE
    home = Path(home).resolve(); package = Path(__file__).parent
    intent_path = home/'recipes/robust-block-intent-v1.json'; intent = read(intent_path)
    collection = home/'training/acoustic-block-parent-v2'; folder = home/'training/acoustic-block-robust-v1'
    for path, expected in intent['source_hashes'].items():
        if sha(path) != expected:
            raise IntegrityError('Frozen robust block intent source changed')
    parent = checkpoint(intent['parent']['path'], 1.)
    if parent != intent['parent']:
        raise IntegrityError('Declared joint block parent changed')
    cr = read(collection/'collection-recipe.json'); rows = read(collection/'captures.json')
    if read(collection/'collection-status.json')['state'] != 'complete' or len(rows) != 8 or cr['parent'] != parent:
        raise IntegrityError('All eight matching parent captures are required')
    if not read(collection/'collection-off-restoration.json')['exact']:
        raise IntegrityError('Parent collection Off restoration failed')
    for row in rows:
        if row['status'] != 'complete' or not row['exact_waveform_replay'] or row['recipe_sha256'] != digest(cr):
            raise IntegrityError('Unverified parent training waveform')
        if any(sha(row[k]) != row[k+'_sha256'] for k in ('audio', 'capture')):
            raise IntegrityError('Parent training bytes changed')
    sources = dict(cr['source_hashes'])
    for name in ('train_robust_block.py', 'prepare_robust_block_attempt.py', 'robust_block_controller.py',
                 'robust_block_search.py', 'robust_block_artifacts.py', 'audit_robust_block_export.py',
                 'acoustic_artifacts.py', 'block_evaluate.py', 'block_identity.py', 'block_worker.py',
                 'block_artifact.py', 'block_renderer.py'):
        path = package/name; sources[str(path.resolve())] = sha(path)
    test = package.parents[2]/'tests/test_reward_game_robust_block_training.py'
    sources[str(test.resolve())] = sha(test)
    for name in ('collection-recipe.json', 'captures.json', 'collection-status.json', 'collection-off-restoration.json'):
        folder.mkdir(parents=True, exist_ok=True); link = folder/name
        if not link.exists():
            link.symlink_to(collection/name)
        if link.resolve() != collection/name:
            raise IntegrityError('Wrong joint block capture metadata')
    training = dict(method='joint-regression-sensitive-full-batch-block-v1', parent=parent, structure=STRUCTURE,
        seed=9131, optimizer='AdamW', learning_rate=.0005, betas=[.9, .999], weight_decay=0., gradient_norm=1.,
        checkpoint_steps=[1, 2], total_updates=2, physical_gpu=1, gradient_scope=intent['method']['gradient_scope'],
        objective=intent['method']['objective'], full_generation_gradient=False, latent_fidelity_weight=10.,
        hard_stops=intent['method']['hard_stops'], sampling=intent['method']['batch'],
        collection_recipe_sha256=sha(collection/'collection-recipe.json'), captures_sha256=sha(collection/'captures.json'),
        intent_sha256=sha(intent_path), source_hashes=sources, fresh_optimizer=True,
        new_audio_generated_by_training=0, reused_parent_training_clips=8)
    immutable(folder/'recipe.json', training)
    recipe = read(home/'recipes/acoustic-latent-transport-80.json')
    recipe.update(hypothesis=intent['hypothesis'], failure_mechanism=intent['failure_mechanism'],
        parent_checkpoint=parent['path'], changed_variables=intent['method'],
        checkpoint=str(folder/'reward-ce-robust-block_step2.safetensors'), training_state=str(folder/'state-step2.pt'),
        training_status=str(folder/'status.json'), expected_updates=2, initial_updates=0,
        interpretation=intent['method']['gradient_scope'], next_after_failure='Retain result and select a materially different method; no broad strength/duration grid.')
    recipe['sources'].update(read(home/'recipes/acoustic-block-compose-v1.json')['sources'])
    recipe['sources'].update(sources)
    recipe['budget'] = dict(optimizer_updates=2, full_batch_waveform_gradient_evaluations=16,
        new_training_clips=0, prior_parent_training_clips=8, new_duplicate_engineering_clips=0,
        new_clips=16, max_seconds=14400)
    recipe['argv'] = [recipe['argv'][0], '-u', '-m', 'conceptmod.textsliders.reward_game.train_robust_block',
                      '--home', str(home), '--folder', str(folder)]
    recipe['preprocessing'] = dict(collection_recipe_sha256=sha(collection/'collection-recipe.json'),
        captures_sha256=sha(collection/'captures.json'), capture_sha256={r['id']: r['capture_sha256'] for r in rows},
        parent_checkpoint=parent, intent_sha256=sha(intent_path), parent_collection_folder=str(collection))
    path = home/'recipes/robust-block-2.json'; immutable(path, recipe)
    return dict(register(home/'block-v1', path), recipe=str(path), research_complete=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--home', required=True); a = p.parse_args()
    print(__import__('json').dumps(prepare(a.home), indent=2))
