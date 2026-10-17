#!/usr/bin/env python3
"""Verify an already installed video wheel from outside every source checkout.

Requires an isolated Python environment with the frozen particle-sliders wheel
and particlegan==0.9.1 installed. Copies the explicit reviewed PG profile to the
receipt directory, validates every source against the installed core manifest,
and uses it only for these isolated child processes. Never modifies an install.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', required=True)
    p.add_argument('--profile-root', required=True, help='Directory containing the reviewed particlegan package')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    python = str(Path(args.python).absolute())
    out = Path(args.output).absolute()
    out.mkdir(parents=True, exist_ok=True)
    profile = out / 'isolated-pg091-profile'
    shutil.copytree(Path(args.profile_root) / 'particlegan', profile / 'particlegan', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {str(x.relative_to(profile / 'particlegan')): hashlib.sha256(x.read_bytes()).hexdigest()
                     for x in sorted((profile / 'particlegan').rglob('*.py'))}
    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    prefix = 'import sys; sys.path.insert(0, ' + repr(str(profile)) + '); '
    with tempfile.TemporaryDirectory(prefix='ltx-sourcefree-') as directory:
        cwd = Path(directory)
        def run(label, code):
            result = subprocess.run([python, '-I', '-c', prefix + code], cwd=cwd, env=env,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            (out / (label + '.log')).write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(label + ' failed: ' + result.stdout)
            return result.stdout
        origin = json.loads(run('origins', '''
import json
from pathlib import Path
from importlib.metadata import version
import torch, particle_sliders, particlegan
from particle_sliders import ltx25_cli, ltx25_backend, ltx25_uni, pg09_ltx_embed
from particle_sliders.pg09 import validate_pg09_domain
assert version('particlegan') == '0.9.1'
validate_pg09_domain((1,8208), response_domain='continuous_vector', pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
modules = (particle_sliders, ltx25_cli, ltx25_backend, ltx25_uni, pg09_ltx_embed)
assert all('site-packages' in str(Path(m.__file__).resolve()) for m in modules)
assert not any(name.startswith('conceptmod') for name in sys.modules)
assert not torch.cuda.is_initialized()
print(json.dumps({'origins': {m.__name__: m.__file__ for m in modules}, 'pg_version': version('particlegan'), 'pg_origin': particlegan.__file__, 'core_version': version('particle-sliders-core')}))
'''))
        common = ['--dummy', '--steps', '4', '--rank', '2', '--alpha', '2', '--te_last_n', '1', '--seed', '0',
                  '--pg_source_profile', 'alz-continuous-v1', '--penalty_input_domain', 'continuous']
        receipts = []
        for preset in ('bispf', 'bispf_optimized'):
            root = cwd / preset
            full, split, infer = (root / name for name in ('full', 'split', 'infer'))
            for label, extra in [('full', ['--save_dir', str(full)]),
                                 ('split', ['--save_dir', str(split), '--stop_after', '2']),
                                 ('resume', ['--save_dir', str(split), '--resume', str(split / 'checkpoint.pt')])]:
                run(preset + '-' + label, 'from particle_sliders.ltx25_cli import main; main(' + repr(common + ['--preset', preset] + extra) + ')')
            run(preset + '-infer', 'from particle_sliders.ltx25_cli import main; main(' + repr([
                '--dummy', '--steps', '0', '--load_export', str(full / 'adapter-export.pt'), '--save_dir', str(infer)]) + ')')
            result = json.loads(run(preset + '-compare', '''
import json
from pathlib import Path
import torch
from particle_sliders.ltx25_cli import main
full, split, infer = map(Path, ''' + repr([str(full), str(split), str(infer)]) + ''')
def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b
load=lambda p:torch.load(p,weights_only=True,map_location='cpu')
equal(load(full/'checkpoint.pt'), load(split/'checkpoint.pt'))
equal(load(full/'adapter-export.pt'), load(split/'adapter-export.pt'))
equal(load(full/'response-probes.pt'), load(infer/'response-probes.pt'))
assert not list(full.glob('*.mp4'))
# Corrupted source signatures must fail before creating the destination.
packet=load(full/'adapter-export.pt')
packet['game']['config']['pg_version']='0.0.0'
tampered=full/'tampered.pt'; torch.save(packet,tampered)
target=full/'must-not-exist'
try: main(['--dummy','--steps','0','--load_export',str(tampered),'--save_dir',str(target)])
except ValueError: pass
else: raise AssertionError('tampered source accepted')
assert not target.exists()
assert not torch.cuda.is_initialized()
print(json.dumps({'exact_full_vs_2plus2':True,'export_probe_reload_exact':True,'source_rejected_before_output':True,'cuda_initialized':False}))
'''))
            receipts.append(dict(preset=preset, **result))
        receipt = dict(schema='ltx25-sourcefree-wheel-v1', python=python, **origin,
                       pg_profile_sources=source_hashes, presets=receipts,
                       production_runtime_qualified=False, audiovisual_render_qualified=False)
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
