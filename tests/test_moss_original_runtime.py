from types import SimpleNamespace

import pytest

from conceptmod.textsliders.pg09_moss_worker import BoundRecipe, sha_file
import moss_original_runtime as runtime


def test_exact_native_profile_origin_gate():
    pins = {runtime.__file__:sha_file(runtime.__file__)}
    recipe = SimpleNamespace(data=dict(runtime_profile=runtime.PROFILE,source_pins=pins))
    modules = {k:SimpleNamespace(__version__=v,__file__=f'{runtime.SITE}/{k}/__init__.py')
               for k,v in runtime.EXPECTED.items()}
    actual = runtime.verify_profile(recipe,executable=runtime.EXECUTABLE,modules=modules)
    assert actual['torch']['version'] == '2.9.1+cu128'
    with pytest.raises(ValueError,match='interpreter'):
        runtime.verify_profile(recipe,executable='/tmp/python',modules=modules)
    for name in modules:
        wrong = dict(modules)
        wrong[name] = SimpleNamespace(__version__=modules[name].__version__,__file__='/tmp/shadow.py')
        with pytest.raises(ValueError,match='origin'):
            runtime.verify_profile(recipe,executable=runtime.EXECUTABLE,modules=wrong)
    recipe.data['runtime_profile'] = dict(runtime.PROFILE,name='other')
    with pytest.raises(ValueError,match='cache-producer'):
        runtime.verify_profile(recipe,executable=runtime.EXECUTABLE,modules=modules)


def test_original_recipes_preserve_fixed_gan_budget_and_shared_worker():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for case in ('bad-intent','afterimage'):
        recipe=BoundRecipe.read(root/f'qualification/moss-reruns/{case}-original-runtime.json')
        recipe.verify_sources()
        assert recipe.data['runtime_profile']==runtime.PROFILE
        assert recipe.data['total_steps']==3840
        assert recipe.data['output_directory'].endswith('/'+case)
        assert 'original-runtime' in recipe.data['output_directory']


def test_grad_enabled_initial_parity_rejects_dispatch_change():
    import torch
    from types import SimpleNamespace
    base=torch.ones(1,4,2,2)
    cached=SimpleNamespace(neutral=base,target=base+.01)
    leaf=torch.nn.Parameter(torch.zeros_like(base))
    class Backend:
        offset=0.
        def generated(self,row):
            return row.neutral+leaf+(self.offset if torch.is_grad_enabled() else 0.)
    backend=Backend()
    run=SimpleNamespace(rows=[cached],backend=backend)
    assert runtime.initial_gradient_parity(run,emit=lambda *a,**k:None,indices=(0,))[0]['grad_cache_exact']
    backend.offset=.001
    with pytest.raises(ValueError,match='grad-enabled'):
        runtime.initial_gradient_parity(run,emit=lambda *a,**k:None,indices=(0,))
    assert leaf.grad is None
