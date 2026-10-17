"""Stdlib-only source/card audit. Never imports Torch or native model runtimes."""
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys

CANONICAL = Path('/ml2/codex-storage/future-workspaces/2026-10-16/task-6/particle-sliders-pg09-core')
DRAFTS = CANONICAL / 'evidence/remaining-canonical-readiness-v1/admission-drafts'


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','diffusers','transformers','particlegan',
                                       'supra','moss_sliders','safetensors'}:
            raise AssertionError('forbidden heavy/native import: ' + fullname)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    sys.meta_path.insert(0, BlockNative())
    sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'scripts')]
    from conceptmod.textsliders import pg09_remaining_worker as worker
    from conceptmod.textsliders import pg09_remaining_admission as admission
    from conceptmod.textsliders import pg09_remaining_bind as binder
    imports = {m.__name__:str(Path(m.__file__).resolve()) for m in (worker, admission, binder)}
    plan = json.loads((DRAFTS / 'sequential-plan.json').read_text())
    assert plan['status'] == 'DRAFT_NOT_ADMITTED' and plan['automatic_dispatch'] is False
    assert Path(plan['source_root']) == CANONICAL
    expected = [('FF01','final-form','probe'),('FF02','final-form','resume-check'),
                ('FF03','final-form','train'),('FF04','final-form','evaluate'),
                ('S01','supra-contexts','contexts'),('S02','supra-final-boss','probe'),
                ('S03','supra-final-boss','resume-check'),('S04','supra-final-boss','train'),
                ('S05','supra-final-boss','evaluate')]
    assert [(x['phase'],x['case'],x['mode']) for x in plan['commands']] == expected
    contracts = {}
    source_checks = 0
    for case in ('final-form','supra-contexts','supra-final-boss'):
        path = DRAFTS / (case + '.draft.json')
        recipe = admission.RemainingRecipe.read(path)
        recipe.verify_sources()
        assert len(recipe.data['source_pins']) == 11
        source_checks += len(recipe.data['source_pins'])
        for module in (worker, admission, binder):
            recipe.verify_origin(module)
        assert recipe.data['admission'] == 'DRAFT_NOT_ADMITTED'
        assert recipe.data['owner_priority_gate']['sha256'] is None
        assert not Path(recipe.data['output_directory']).exists()
        assert recipe.phase_reason('probe', {}).startswith('default denied')
        for name,path in recipe.data['package_origins'].items():
            assert sha(path) == recipe.data['package_source_sha256'][name]
        for source in recipe.data['source_pins']:
            assert '/backend-adapters-pg09-20261017/' not in source
        assert recipe.data['source_pins'][str(CANONICAL/'conceptmod/textsliders/pg09_remaining_worker.py')] == 'bbbea675b751158dbe804f8d02e9fd2a38d6b9d85707c78c800901009525e36e'
        contracts[case] = recipe
    for command in plan['commands']:
        recipe = contracts[command['case']]
        args,env = command['argv'],command['environment']
        assert args[:3] == [recipe.data['python'],'-m','conceptmod.textsliders.pg09_remaining_worker']
        assert len(args) == 13 and args[3] == '--recipe' and args[5] == '--output' and args[7] == '--mode'
        assert Path(args[4]) == DRAFTS / (command['case'] + '.draft.json')
        assert args[6] == recipe.data['output_directory'] and args[8] == command['mode']
        assert args[9:] == ['--lease-fd','{owner_admission_fd}','--lease-fd','{owner_gpu_fd}']
        assert env['CUDA_VISIBLE_DEVICES'] == str(recipe.data['physical_gpu'])
        assert env['HF_HOME'] == recipe.data['hf_home']
        assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
        assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        paths = env['PYTHONPATH'].split(':')
        assert paths[:3] == [str(CANONICAL),str(CANONICAL/'packages/particle-sliders-core/src'),str(CANONICAL/'scripts')]
        assert not any('site-packages' in p or 'backend-adapters-pg09-20261017' in p for p in paths)
        spec = importlib.machinery.PathFinder.find_spec('conceptmod.textsliders.pg09_remaining_campaigns', [str(CANONICAL/'conceptmod/textsliders')])
        assert Path(spec.origin) == CANONICAL/'conceptmod/textsliders/pg09_remaining_campaigns.py'
        core_spec = importlib.machinery.PathFinder.find_spec('particle_sliders.pg09', [str(CANONICAL/'packages/particle-sliders-core/src/particle_sliders')])
        assert recipe.data['source_pins'][str(Path(core_spec.origin).resolve())] == sha(core_spec.origin)
        assert command['executed'] is False and command['root_guardian_required'] is True
    a,b = contracts['supra-contexts'].data,contracts['supra-final-boss'].data
    assert a['cohort_root'] == b['cohort_root'] and a['cohort_id'] == b['cohort_id']
    assert a['caps']['wall_seconds'] == b['caps']['wall_seconds'] == 21600
    assert a['contexts'] == b['contexts']
    assert all(a['contexts'][k] is None for k in ('sha256','receipt_sha256','preparation_phase_sha256','case_budget_sha256'))
    assert a['source_pins'] == b['source_pins']
    assert not any(x in sys.modules for x in ('torch','numpy','diffusers','transformers','particlegan','supra','moss_sliders'))
    return dict(status='PASS_STDLIB_ONLY_DRAFT_REVIEW_NOT_ADMISSION',drafts=3,phase_vectors=9,
                source_pins_verified=source_checks,actual_stdlib_import_origins=imports,
                heavy_core_backend_resolution='ModuleSpec paths/hash only; no heavy module execution',
                input_payloads_opened=0,heavy_native_imports=0,gpu_or_dispatch=False,
                draft_sha256={p.name:sha(p) for p in DRAFTS.glob('*.json')},
                remaining=['Root guardian/priority receipt/leases/admission',
                           '19 Supra contexts and binding','Actual native import attestation/parity/resources',
                           'Fresh-process GPU disk resume','Adequate full budgets and matched quality'])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
