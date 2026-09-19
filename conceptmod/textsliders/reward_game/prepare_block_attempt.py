"""Register one fixed composition only after actual native merge equivalence."""
import argparse
from pathlib import Path
from .core import read,immutable,sha,digest,IntegrityError


def prepare(home):
    from .block_setup import init
    from .block_controller import register
    from .block_artifact import checkpoint
    home=Path(home).resolve();audit=home/'audit/acoustic-block-merge-v1';game_home=home/'block-v1'
    folder=home/'training/acoustic-block-compose-v2';composition=read(folder/'composition.json')
    candidate=checkpoint(composition['candidate']['path'],1.)
    if candidate!=composition['candidate'] or read(audit/'result.json')['candidate']!=candidate:
        raise IntegrityError('Audited composition changed')
    init(game_home,home,audit)
    fixed=read(folder/'recipe.json')
    sources=dict(read(home/'recipes/acoustic-gain150.json')['sources'])
    names=('block_artifact.py','block_compose.py','block_identity.py','block_renderer.py','block_worker.py',
        'block_evaluate.py','block_setup.py','block_controller.py','block_artifacts.py','block_search.py',
        'block.py','block_sequence.py','prepare_block_attempt.py','audit_block_merge.py','ff_artifact.py','listen.py')
    for name in names:
        path=Path(__file__).with_name(name).resolve();sources[str(path)]=sha(path)
    paths=dict(attention_training_recipe=home/'training/acoustic-ce-v1/recipe.json',
        attention_training_state=home/'training/acoustic-ce-v1/state-step8.pt',
        attention_calibration_recipe=home/'training/acoustic-gain150-v1/recipe.json',
        attention_calibration_audit=home/'training/acoustic-gain150-v1/audit.json',
        shared_capture_recipe=home/'training/acoustic-ce-v1/collection-recipe.json',
        shared_capture_manifest=home/'training/acoustic-ce-v1/captures.json',
        feed_forward_training_recipe=home/'training/acoustic-ff-ce-v1/recipe.json',
        feed_forward_training_state=home/'training/acoustic-ff-ce-v1/state-step8.pt',
        feed_forward_export_audit=home/'audit/acoustic-ff-export-v1.json')
    recipe=dict(hypothesis=fixed['hypothesis'],failure_mechanism=fixed['failure_mechanism'],method='checkpoint',
        parent_checkpoint=None,parents=fixed['parents'],
        changed_variables=dict(combined_support='144 attention and 72 feed-forward projections',
            factor_rule='Exact source-factor copies on disjoint projections',runtime_multiplier=1.,new_optimizer_updates=0),
        budget=dict(optimizer_updates=0,new_training_clips=0,new_clips=16,new_own_off_engineering_clips=1),
        sources=sources,checkpoint=candidate['path'],multiplier=1.,
        parent_evidence={name:dict(path=str(path),sha256=sha(path)) for name,path in paths.items()},
        preprocessing=dict(composition_recipe_sha256=sha(folder/'recipe.json'),composition_sha256=sha(folder/'composition.json'),
            native_audit_sha256=sha(audit/'result.json'),field_name_amendment_sha256=sha(home/'amendments/block-field-names-v1/amendment.json')),
        after_failure='Retain the outcome and select a materially different intervention; the complete within-chunk flow method is prepared but unselected.',
        after_success='Freeze fresh confirmation, independent replication, preservation and per-host composition before success.')
    path=home/'recipes/acoustic-block-compose-v1.json';immutable(path,recipe)
    return dict(register(game_home,path),game_home=str(game_home),recipe=str(path),benchmark_sha256=read(game_home/'game.json')['benchmark_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);a=p.parse_args()
    print(__import__('json').dumps(prepare(a.home),indent=2))
