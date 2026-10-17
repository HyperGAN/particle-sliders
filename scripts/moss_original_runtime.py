"""Attest the original cache-producer runtime before invoking the shared GAN worker."""
import argparse
import json
import os
from pathlib import Path
import sys

from conceptmod.textsliders.pg09_moss_worker import BoundRecipe, atomic_json, sha_file

EXECUTABLE = '/mnt/nvme/sliders/moss-image-sliders/.venv/bin/python'
SITE = '/mnt/nvme/sliders/moss-image-sliders/.venv/lib/python3.12/site-packages'
EXPECTED = {'torch': '2.9.1+cu128', 'diffusers': '0.38.0', 'transformers': '5.3.0'}
PROFILE = dict(name='moss-cache-producer-v1', executable=EXECUTABLE, versions=EXPECTED,
               pg_source_directory='/mnt/nvme/sliders/qualification/pg09-audit/wheel')


def verify_profile(recipe, *, executable, modules):
    if recipe.data.get('runtime_profile') != PROFILE:
        raise ValueError('recipe must bind exact original cache-producer runtime')
    if executable != EXECUTABLE:
        raise ValueError('actual interpreter differs from original runtime')
    pins = recipe.data['source_pins']
    source = str(Path(__file__).resolve())
    if pins.get(source) != sha_file(source):
        raise ValueError('original-runtime bootstrap source is not pinned')
    result = {}
    for name, expected in EXPECTED.items():
        module = modules[name]
        actual = str(Path(module.__file__).resolve())
        if str(module.__version__) != expected or actual != f'{SITE}/{name}/__init__.py':
            raise ValueError('actual native module version/origin differs: ' + name)
        result[name] = dict(version=str(module.__version__), file=actual, sha256=sha_file(actual))
    return result



def initial_gradient_parity(run, *, emit, indices=(0,39), relative_limit=1e-6):
    """Zero-adapter forward must retain teacher parity with and without autograd."""
    import torch
    records=[]
    for index in indices:
        cached=run.rows[index]
        with torch.no_grad():
            no_grad=run.backend.generated(cached).detach()
        with torch.enable_grad():
            enabled=run.backend.generated(cached)
        if not enabled.requires_grad:
            raise ValueError('initial generated response is detached from adapters')
        enabled=enabled.detach()
        base=cached.neutral.to(enabled)
        edit=(cached.target-cached.neutral).to(enabled)
        denominator=edit.square().mean().clamp_min(1e-20)
        if not all(t.isfinite().all() for t in (no_grad,enabled,base)):
            raise FloatingPointError('initial generated response nonfinite')
        record=dict(index=index, grad_no_grad_exact=torch.equal(enabled,no_grad),
            grad_cache_exact=torch.equal(enabled,base),
            no_grad_cache_exact=torch.equal(no_grad,base),
            grad_relative_mse=float((enabled-base).square().mean()/denominator),
            no_grad_relative_mse=float((no_grad-base).square().mean()/denominator),
            threshold_relative_mse=relative_limit)
        records.append(record)
        emit('initial_gradient_parity', initial_gradient_parity=records)
        if max(record['grad_relative_mse'],record['no_grad_relative_mse'])>relative_limit:
            raise ValueError('initial grad-enabled zero-adapter response differs from frozen cache')
    return records

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('probe','train','evaluate'), required=True)
    parser.add_argument('--lease-fd', type=int, action='append', required=True)
    args = parser.parse_args(argv)
    recipe = BoundRecipe.read(args.recipe)
    recipe.verify_sources()
    recipe.verify_output(args.output)
    from conceptmod.textsliders.pg09_moss_worker import inherited_leases
    inherited_leases(args.lease_fd, recipe.data['lease_paths'])
    import torch
    import diffusers
    import transformers
    modules = dict(torch=torch, diffusers=diffusers, transformers=transformers)
    attestation = verify_profile(recipe, executable=sys.executable, modules=modules)
    from particle_sliders.pg09 import validate_pg09_domain
    pg = validate_pg09_domain((1,128,32,32), response_domain='continuous_field')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before original-runtime attestation')
    record = dict(schema='moss-original-runtime-attestation-v1', recipe_sha256=recipe.identity,
        modules=attestation, executable=sys.executable, pg=pg,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    output = args.output / 'runtime-attestation.json'
    if output.exists() and json.loads(output.read_text()) != record:
        raise ValueError('output runtime attestation belongs to different sources/profile')
    atomic_json(output,record)
    # All objectives, optimizer/controller ownership, parity and phase gates remain shared.
    from conceptmod.textsliders.pg09_moss_worker import execute
    execute(recipe,args.output,args.mode,args.lease_fd,initial_qualifier=initial_gradient_parity)


if __name__ == '__main__':
    main()
