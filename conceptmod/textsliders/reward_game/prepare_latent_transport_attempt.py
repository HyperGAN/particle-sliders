"""Register the frozen latent-target distillation pilot after teacher feasibility."""
import argparse
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError


def prepare(home):
    from .acoustic_controller import register
    home = Path(home).resolve(); package = Path(__file__).parent
    teacher = home/'training/acoustic-latent-teacher-v1'
    folder = home/'training/acoustic-transport-v1'
    collection = home/'training/acoustic-full-flow-legacy-v1'
    intent_path = home/'recipes/latent-transport-intent-v1.json'; intent = read(intent_path)
    for path, expected in intent['source_hashes'].items():
        if sha(path) != expected:
            raise IntegrityError('Declared latent transport source changed')
    result = read(teacher/'result.json')
    if not result['passed'] or result['actual_target_ascent_steps'] != 32 or not read(teacher/'off-restoration.json')['exact']:
        raise IntegrityError('Teacher feasibility and cleanup must pass')
    targets = read(teacher/'targets.json')
    if len(targets) != 8 or any(sha(r['target']) != r['target_sha256'] for r in targets):
        raise IntegrityError('Teacher target identity changed')
    for row in targets:
        for trial in row['history']:
            if sha(trial['audio']) != trial['audio_sha256'] or sha(trial['target_state']) != trial['target_state_sha256']:
                raise IntegrityError('A retained teacher trial changed')
    for name in ('collection-recipe.json', 'captures.json', 'collection-status.json', 'collection-off-restoration.json'):
        folder.mkdir(parents=True, exist_ok=True); link = folder/name
        if not link.exists():
            link.symlink_to(collection/name)
        if link.resolve() != collection/name:
            raise IntegrityError('Wrong transport capture metadata')
    immutable(folder/'reused-collection.json', dict(source=str(collection), exact_captures_reused=8,
              new_audio_generated=0, teacher_folder=str(teacher), teacher_audio_counted_in_teacher_folder=32))
    sources = dict(intent['source_hashes'])
    for path in (Path(__file__), package/'train_latent_transport.py', package/'merged_forward.py',
                 package/'acoustic_artifact.py', package/'audit_latent_transport_export.py',
                 package.parents[2]/'tests/test_reward_game_transport_training.py'):
        sources[str(path.resolve())] = sha(path)
    training = dict(method='bounded-ce-latent-transport-attention-v1', structure=read(home/'training/acoustic-full-flow-recompute-v1/recipe.json')['structure'],
        seed=9131, learning_rate=.0003, optimizer='AdamW', betas=[.9, .999], weight_decay=0., gradient_norm=1.,
        checkpoint_steps=[1, 40, 80], total_updates=80, physical_gpu=1,
        initial='Fresh Kaiming down factors and zero up factors; no previous candidate state',
        objective=intent['changed_variables']['objective'], target_transport=intent['changed_variables']['distillation'],
        sampling='case_index=update%8; chunk_index=(update%40)//8; epoch=update//40; flow_step=(5*chunk_index+10*epoch+3*case_index)%30',
        full_generation_gradient=False, teacher_result_sha256=sha(teacher/'result.json'), teacher_targets_sha256=sha(teacher/'targets.json'),
        teacher_recipe_sha256=sha(teacher/'teacher-recipe.json'), collection_recipe_sha256=sha(collection/'collection-recipe.json'),
        intent_sha256=sha(intent_path), source_hashes=sources)
    immutable(folder/'recipe.json', training)
    recipe = read(home/'recipes/acoustic-full-flow-recompute-8.json')
    recipe.update(hypothesis=intent['hypothesis'], failure_mechanism='Full-flow CE training exceeded its latent drift limit after two updates; distill bounded targets using supervised path transport.',
        parent_checkpoint=None, changed_variables=intent['changed_variables'],
        checkpoint=str(folder/'reward-ce-latent-transport_step80.safetensors'), training_state=str(folder/'state-step80.pt'),
        training_status=str(folder/'status.json'), expected_updates=80, initial_updates=0,
        interpretation='Supervised velocity transport toward bounded CE latent targets; fixed conditions and overlap; no full generator gradient.',
        next_after_failure='Retain all outcomes and select a materially different hypothesis; no broad strength or duration grid.')
    recipe['sources'].update(sources)
    recipe['budget'].update(optimizer_updates=80, new_training_clips=0, new_duplicate_engineering_clips=0,
                            reused_full_flow_captures=8, new_clips=16, max_seconds=7200,
                            prior_teacher_audio=32, new_audio_generated_by_training=0)
    recipe['argv'] = [recipe['argv'][0], '-u', '-m', 'conceptmod.textsliders.reward_game.train_latent_transport',
        '--home', str(home), '--game-home', str(home/'acoustic-v1'), '--folder', str(folder), '--teacher-folder', str(teacher)]
    recipe['preprocessing'] = dict(collection_recipe_sha256=sha(collection/'collection-recipe.json'),
        captures_sha256=sha(collection/'captures.json'), teacher_recipe_sha256=sha(teacher/'teacher-recipe.json'),
        teacher_result_sha256=sha(teacher/'result.json'), teacher_targets_sha256=sha(teacher/'targets.json'),
        target_sha256={r['id']: r['target_sha256'] for r in targets}, intent_sha256=sha(intent_path))
    path = home/'recipes/acoustic-latent-transport-80.json'; immutable(path, recipe)
    return dict(register(home/'acoustic-v1', path), recipe=str(path), research_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--home', required=True); args = parser.parse_args()
    print(__import__('json').dumps(prepare(args.home), indent=2))
