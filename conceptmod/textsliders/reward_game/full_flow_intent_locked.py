"""Verify cached diagnostic models before the unchanged preservation procedure."""
import argparse
import os
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError, Store
from .full_flow_confirmation_intent import run as measure


def run(home,name):
    from slider_selection.features import CLAP
    from scripts.lm_score import WHISPER_MODEL
    home=Path(home).resolve();folder=home/'confirmation'/name
    from .full_flow_confirmation import verify_batch
    verify_batch(home,folder)
    if os.environ.get('HF_HUB_OFFLINE')!='1':
        raise IntegrityError('Locked preservation requires offline model resolution')
    lock=read(folder/'intent/model-lock.json')
    if not lock['passed'] or not lock['actual_processors_and_models_loaded']:
        raise IntegrityError('Actual diagnostic model preflight missing')
    expected={CLAP[0]:CLAP[1],WHISPER_MODEL:None}
    for model in lock['models']:
        if model['model'] not in expected:
            raise IntegrityError('Undeclared diagnostic model')
        cache=Path(model['path']).parent.parent
        revision=expected[model['model']] or (cache/'refs/main').read_text().strip()
        if revision!=model['resolved_revision']:
            raise IntegrityError('Cached diagnostic model revision changed')
        for path,record in model['files'].items():
            if sha(path)!=record['sha256']:
                raise IntegrityError('Cached diagnostic model bytes changed')
    if {m['model'] for m in lock['models']}!=set(expected):
        raise IntegrityError('Incomplete diagnostic model lock')
    source_root=Path(__file__).resolve().parents[3]
    sources=[Path(__file__),source_root/'slider_selection/features.py',source_root/'scripts/lm_score.py']
    declaration=dict(model_lock_sha256=sha(folder/'intent/model-lock.json'),
        sources={str(p.resolve()):sha(p) for p in sources},device='cpu',diagnostic_calculation_unchanged=True)
    immutable(folder/'intent/locked-invocation.json',declaration)
    Store(home).event('confirmation_intent_model_lock_verified',batch=name,declaration_sha256=digest(declaration))
    return measure(home,name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--name',required=True)
    a=p.parse_args();print(__import__('json').dumps(run(a.home,a.name),indent=2))
