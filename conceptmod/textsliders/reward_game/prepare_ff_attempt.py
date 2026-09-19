"""Freeze the feed-forward pilot only after its actual-host audit passes."""
import argparse
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError


def prepare(home):
    from .ff_setup import init
    from .ff_controller import register
    home=Path(home).resolve();audit=home/'audit/acoustic-ff-v1';game_home=home/'ff-v1'
    result=read(audit/'gradient-result.json')
    if not result['passed'] or not result['frozen_attention_unchanged']:
        raise IntegrityError('Actual feed-forward gradient and frozen-attention audit must pass')
    intent=read(home/'recipes/acoustic-ff-intent-v1.json')
    for path,expected in intent['sources'].items():
        if sha(path)!=expected:raise IntegrityError('Feed-forward audit intent source changed')
    init(game_home,home,audit)
    recipe=read(home/'recipes/acoustic-ce-8.json')
    folder=home/'training/acoustic-ff-ce-v1'
    recipe.update(hypothesis=intent['hypothesis'],failure_mechanism='The attention-only acoustic pilot failed fresh magnitude/original-control gates, and adding the signed LM policy failed fresh Off consistency. Test nonlinear acoustic feed-forward feature updates with attention deltas fixed at zero.',
        parent_checkpoint=None,changed_variables=intent['changed_variables'],
        checkpoint=str(folder/'reward-ce-acoustic-ff_step8.safetensors'),
        training_state=str(folder/'state-step8.pt'),training_status=str(folder/'status.json'),
        expected_updates=8,initial_updates=0,
        interpretation='Fresh zero feed-forward factors, ordinary rank-8 block export with 144 exactly zero attention deltas. Same eight Off captures and truncated CE/latent-anchor objective. Development remains exposed and uses unchanged gates.',
        next_after_failure='Record outcome and review feed-forward versus attention support. A failed pilot does not complete research or authorize a broad strength/duration grid.')
    names=('ff_artifact.py','ff_network.py','ff_identity.py','ff_renderer.py','ff_worker.py','ff_evaluate.py',
        'ff_setup.py','ff_controller.py','ff_artifacts.py','ff_search.py','ff.py','ff_sequence.py',
        'audit_ff_gradient.py','train_ff_ce.py','prepare_ff_attempt.py')
    for name in names:
        path=Path(__file__).with_name(name).resolve();recipe['sources'][str(path)]=sha(path)
    recipe['budget'].update(optimizer_updates=8,new_training_clips=0,new_clips=16,new_own_off_engineering_clips=1,max_seconds=7200)
    recipe['argv']=[recipe['argv'][0],'-u','-m','conceptmod.textsliders.reward_game.train_ff_ce',
        '--home',str(home),'--game-home',str(game_home),'--folder',str(folder),'--total','8','--lr','0.001']
    recipe['preprocessing'].update(acoustic_collection=str(folder/'collection-recipe.json'),
        collection_recipe_sha256=sha(folder/'collection-recipe.json'),captures_sha256=sha(folder/'captures.json'),
        feed_forward_audit=str(audit/'gradient-result.json'),feed_forward_audit_sha256=sha(audit/'gradient-result.json'),
        capture_reuse_sha256=sha(folder/'capture-reuse.json'),feed_forward_intent_sha256=digest(intent))
    path=home/'recipes/acoustic-ff-8.json';immutable(path,recipe)
    return dict(register(game_home,path),game_home=str(game_home),recipe=str(path),
                next_command='ff_sequence --home '+str(game_home),research_complete=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--home',required=True)
    args=parser.parse_args();print(__import__('json').dumps(prepare(args.home),indent=2),flush=True)
