"""Declare a separate acoustic adapter game after actual host audits pass."""
from pathlib import Path
from copy import deepcopy
from .core import read,write,immutable,digest,sha,IntegrityError,Store
from .setup import verify
from .block_artifact import STRUCTURE


def init(home,parent,audit_folder):
    home=Path(home).resolve();parent=Path(parent).resolve();audit_folder=Path(audit_folder).resolve()
    game,manifest=verify(parent);old_config=read(parent/'game.json')
    required=['result.json','off-restoration.json']
    result=read(audit_folder/'result.json');protocol=read(audit_folder/'protocol.json')
    if not result['passed'] or result['protocol_sha256']!=digest(protocol) or result['projections_verified']!=216:
        raise IntegrityError('Actual native block composition audit must pass')
    if not result['parent_pair_matches_composite'] or not result['zero_up_matches_off'] or not result['off_restored']:
        raise IntegrityError('Native block audit is incomplete')
    if not read(audit_folder/'off-restoration.json')['exact']:raise IntegrityError('Native block audit failed Off restoration')
    from .block_artifact import checkpoint
    if checkpoint(result['candidate']['path'],1.)!=result['candidate']:
        raise IntegrityError('Audited block candidate changed')
    for file,h in protocol['source_hashes'].items():
        if sha(file)!=h:raise IntegrityError('Native block audit sources changed')
    # Validate imported references explicitly, without generating replacement controls.
    for case in game['cases']:
        for control in case['controls'].values():
            for label in ('audio','observation'):
                if sha(control[label])!=control[label+'_sha256']:raise IntegrityError('Frozen comparison control changed')
    import sys
    from .core import ROOT
    sys.path.insert(0,str(ROOT.parent));from app.sliders import catalog
    limits={kind:catalog()['energy'][kind]['max'] for kind in ('language_model','transformer')}
    amended=deepcopy(game);amended.update(game_id=game['game_id']+'-block-v1',parent_benchmark_sha256=digest(game),
        candidate_structure=STRUCTURE,host_energy_by_kind=limits,
        experiment='Separate full acoustic block adapter support; fixed composition of disjoint attention and feed-forward factors. Same development cases/order/thresholds/controls and frozen scoring. Host budgets are separate.')
    spec=home/'spec/benchmark.json';immutable(spec,amended)
    references=deepcopy(read(Path(old_config['spec']).parent/'reference-scorecards.json'))
    references.update(parent_benchmark_sha256=digest(game),benchmark_sha256=digest(amended),
        provenance='Same unchanged comparison recordings and CE values; only supported candidate host/identity changed')
    immutable(spec.parent/'reference-scorecards.json',references)
    config=dict(spec=str(spec),benchmark_sha256=digest(amended),game_id=amended['game_id'],verified_stats=old_config['verified_stats'],parent_home=str(parent))
    immutable(home/'game.json',config)
    paths=[Path(__file__).with_name(n) for n in ('block_setup.py','block_worker.py','block_evaluate.py','block_identity.py','block_renderer.py','block_artifact.py')]
    immutable(home/'renderer.json',dict(version='block-v1',sources={str(p.resolve()):sha(p) for p in paths},
        contract='Inherited SearchRenderer.generate unchanged; new typed checkpoint and per-host component/identity checks. Own Off PCM audit remains required before candidate rendering.'))
    immutable(home/'audit/host-audits.json',dict(parent_home=str(parent),audit_folder=str(audit_folder),
        files={str(audit_folder/f):sha(audit_folder/f) for f in required},verified_reference_audio_count=48,new_audio_generated=0))
    Store(home).event('acoustic_game_initialized',parent_home=str(parent),benchmark_sha256=digest(amended),new_audio_generated=0)
    verify(home);return config
