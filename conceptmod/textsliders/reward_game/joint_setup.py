"""Separate fixed-pair game using the original cases, controls and thresholds."""
from copy import deepcopy
from pathlib import Path
from .core import read,immutable,sha,digest,Store,IntegrityError
from .setup import verify
from .joint_artifact import checkpoint,FORMAT


def init(home,parent,acoustic_home,bundle):
    home=Path(home).resolve();parent=Path(parent).resolve();acoustic_home=Path(acoustic_home).resolve()
    game,manifest=verify(parent);acoustic_game,_=verify(acoustic_home);pair=checkpoint(bundle,1.)
    for previous in (parent,acoustic_home):
        if not read(previous/'audit/off-pcm-v1/result.json')['passed']:raise IntegrityError('Previous ordinary host Off audit missing')
    for case in game['cases']:
        for control in case['controls'].values():
            for kind in ('audio','observation'):
                if sha(control[kind])!=control[kind+'_sha256']:raise IntegrityError('Joint imported control changed')
    amended=deepcopy(game);amended.update(game_id=game['game_id']+'-joint-v1',parent_benchmark_sha256=digest(game),
        candidate_structure=dict(format=FORMAT,hosts=['language_model','transformer'],rank_per_host=8,alpha_per_host=8),
        host_energy_by_kind=acoustic_game['host_energy_by_kind'],audit_candidate=pair,
        experiment='Fixed pair of ordinary host LoRAs. Native component loading and merging; same exposed development cases, physical GPUs, style strengths, controls and gates.')
    spec=home/'spec/benchmark.json';immutable(spec,amended)
    original_config=read(parent/'game.json');refs=deepcopy(read(Path(original_config['spec']).parent/'reference-scorecards.json'))
    refs.update(benchmark_sha256=digest(amended),parent_benchmark_sha256=digest(game))
    immutable(spec.parent/'reference-scorecards.json',refs)
    config=dict(spec=str(spec),benchmark_sha256=digest(amended),game_id=amended['game_id'],verified_stats=original_config['verified_stats'],parent_home=str(parent))
    immutable(home/'game.json',config)
    names=('joint_setup.py','joint_worker.py','joint_evaluate.py','joint_identity.py','joint_renderer.py','joint_artifact.py','joint_merge_audit.py','acoustic_artifact.py')
    paths=[Path(__file__).with_name(n) for n in names]
    immutable(home/'renderer.json',dict(version='joint-v1',sources={str(p.resolve()):sha(p) for p in paths},
        contract='Inherited ordinary generate loop; pair flattened into native typed components. Own Off PCM and actual joint merge audit required before candidate audio.'))
    Store(home).event('joint_game_initialized',pair=pair,new_audio_generated=0)
    verify(home);return config
