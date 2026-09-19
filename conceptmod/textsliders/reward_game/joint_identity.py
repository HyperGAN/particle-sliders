"""Content identity of the two fixed ordinary host components."""
from pathlib import Path
from .core import STRUCTURE,candidate_identity,sha,IntegrityError


def generation_identity(game,manifest,case,candidate):
    styles=[];energy=dict(language_model=0.,transformer=0.)
    for comp in case['style_components']:
        styles.append(dict(structure={k:comp[k] for k in STRUCTURE if k!='unit_scale'},weights_sha256=manifest['style_hashes'][comp['weights']],multiplier=comp['multiplier']))
        energy[comp['kind']]+=abs(comp['multiplier']*comp['alpha']/comp['rank'])
    for part in candidate.get('components',[]):
        s=part['structure'];energy[s['kind']]+=abs(part['multiplier']*s['alpha']/s['rank'])
    for kind,value in energy.items():
        if value>game['host_energy_by_kind'][kind]+1e-8:raise IntegrityError('Joint experiment exceeds '+kind+' energy')
    return dict(schema='ordinary-music3-joint-render-v1',candidate=candidate_identity(candidate),
        case={k:case[k] for k in ('id','family','seed','physical_gpu')},styles=styles,base_hashes=manifest['model_hashes'],
        renderer_sources=manifest['source_hashes'],bridge_sha256=sha(Path(__file__).with_name('joint_worker.py')),
        joint_identity_sha256=sha(__file__),host_energy_by_kind=game['host_energy_by_kind'],resolved_energy=energy,
        sampler=game['sampler'],duration_seconds=game['duration_seconds'])
