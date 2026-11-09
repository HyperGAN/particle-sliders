"""Run from both a checkout and an installed wheel, with no repository on sys.path."""
import os
import subprocess
import sys
import textwrap

import pytest


def run_python(tmp_path, source):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-c', textwrap.dedent(source)],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('first', ['conceptmod', 'particle_sliders'])
@pytest.mark.parametrize('suffix', ['textsliders.lora', 'textsliders.particle_bridge_gan',
                                     'textsliders.gan_v2.critic', 'notrigger'])
def test_single_module_identity_in_both_import_orders(tmp_path, first, suffix):
    second = 'particle_sliders' if first == 'conceptmod' else 'conceptmod'
    run_python(tmp_path, f'''
        import importlib, sys
        a = importlib.import_module('{first}.{suffix}')
        b = importlib.import_module('{second}.{suffix}')
        assert a is b
        assert a.__name__ == 'particle_sliders.{suffix}'
        assert a.__spec__.name == a.__name__
        a._namespace_test_sentinel = object()
        assert b._namespace_test_sentinel is a._namespace_test_sentinel
        if '{suffix}' != 'notrigger':
            assert importlib.reload(a) is b
        assert sys.modules['{first}.{suffix}'] is sys.modules['{second}.{suffix}']
    ''')


def test_glimmer_subclass_and_legacy_pickle_resolution(tmp_path):
    run_python(tmp_path, '''
        import io, pickle, torch
        from conceptmod.textsliders.lora import LoRAModule as Old
        from particle_sliders.textsliders.lora import LoRAModule as New
        from particle_sliders.textsliders import particle_bridge_gan as game
        from conceptmod.textsliders import particle_bridge_gan as old_game
        assert Old is New
        class Adapter(New):
            pass
        adapter = Adapter('test', torch.nn.Linear(3, 2), lora_dim=2)
        assert isinstance(adapter, Old)
        assert game.update is old_game.update
        # Protocol-0 globals are the qualified names in pre-migration pickles.
        assert pickle.loads(b'cconceptmod.textsliders.lora\\nLoRAModule\\n.') is New
        assert pickle.loads(b'cconceptmod.textsliders.particle_bridge_gan\\nRoutedMLP\\n.') is game.RoutedMLP
        # Emulate a full-object checkpoint saved by the pre-migration class.
        original = New.__module__
        try:
            New.__module__ = 'conceptmod.textsliders.lora'
            stream = io.BytesIO()
            torch.save(New('test', torch.nn.Linear(3, 2), lora_dim=2), stream)
        finally:
            New.__module__ = original
        stream.seek(0)
        restored = torch.load(stream, weights_only=False, map_location='cpu')
        assert type(restored) is New
        # Tensor-only checkpoints keep their names and values across imports.
        target = Old('test', torch.nn.Linear(3, 2), lora_dim=2)
        target.load_state_dict(restored.state_dict(), strict=True)
        assert target.state_dict().keys() == restored.state_dict().keys()
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    ''')


def test_packaged_prompts_and_license(tmp_path):
    run_python(tmp_path, '''
        from importlib.resources import files
        assert 'Rohit Gandikota' in files('particle_sliders').joinpath('LICENSE').read_text()
        a = files('particle_sliders.textsliders').joinpath('data/prompts-yue2-metal-arm-b.yaml')
        b = files('conceptmod.textsliders').joinpath('data/prompts-yue2-metal-arm-b.yaml')
        assert a.read_bytes() == b.read_bytes()
        assert a.read_bytes()
    ''')


def test_backend_imports_do_not_require_shared_core_or_native_runtimes(tmp_path):
    run_python(tmp_path, '''
        import importlib.abc, sys
        assert 'torch' not in sys.modules
        assert not any(name in sys.modules for name in
                       ('particlegan', 'diffusers', 'app', 'hypergan', 'hndl'))
        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'particlegan', 'diffusers', 'app', 'hypergan', 'hndl'}:
                    raise AssertionError('unexpected optional import: ' + fullname)
        sys.meta_path.insert(0, BlockOptional())
        from particle_sliders.textsliders.lora import LoRAModule
        from conceptmod.textsliders.particle_bridge_gan import update
        assert callable(update)
        assert 'particle_sliders.pg09' not in sys.modules
    ''')


def test_legacy_module_command(tmp_path):
    # Weight-free CLI with no runtime/model dependency.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    env.pop('PYTHONPATH', None)
    results = [subprocess.run([sys.executable, '-m', name, '--help'], cwd=tmp_path,
                             env=env, text=True, capture_output=True, check=True)
               for name in ('conceptmod.textsliders.distribution_probe',
                            'particle_sliders.textsliders.distribution_probe')]
    assert 'usage:' in results[0].stdout
    assert results[0].stdout.replace('conceptmod', 'particle_sliders') == results[1].stdout


def test_all_lazy_root_exports_survive_star_import(tmp_path):
    run_python(tmp_path, '''
        import particle_sliders
        assert len(particle_sliders.__all__) == 28
        namespace = {}
        exec('from particle_sliders import *', namespace)
        assert set(particle_sliders.__all__) <= namespace.keys()
        for name in particle_sliders.__all__:
            assert namespace[name] is getattr(particle_sliders, name)
    ''')


def test_concurrent_legacy_import_never_mutates_canonical_spec(tmp_path):
    run_python(tmp_path, '''
        import importlib, threading
        from particle_sliders import _legacy_namespace as aliases
        canonical = importlib.import_module('particle_sliders.textsliders.lora')
        ready, finish = threading.Event(), threading.Event()
        original = aliases._AliasLoader.exec_module
        def paused(loader, module):
            if module.__name__.endswith('.lora'):
                ready.set()
                assert finish.wait(10)
            return original(loader, module)
        aliases._AliasLoader.exec_module = paused
        results, errors = [], []
        def load_legacy():
            try:
                results.append(importlib.import_module('conceptmod.textsliders.lora'))
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=load_legacy)
        worker.start()
        try:
            assert ready.wait(10)
            assert canonical.__spec__.name == 'particle_sliders.textsliders.lora'
            assert importlib.import_module(canonical.__name__) is canonical
        finally:
            finish.set()
            worker.join(10)
        assert not worker.is_alive() and not errors
        assert results == [canonical]
    ''')
