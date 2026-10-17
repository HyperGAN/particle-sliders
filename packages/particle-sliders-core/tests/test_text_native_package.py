"""Native text extraction from reviewed fe54d05daa2a3dbc91e7c6697ba8386ff74a8c30.

Frozen Tiny backend SHA256: 392344c0c2e966db72a46bd44fd4363c05914f6ab134414133e87bb983523186.
Frozen Tiny adapter source SHA256: 99096f5cbfeb9c29ac0fe6daa84d4892c286165055ef3b912437659bc0bb1ae7.
No historical training game is part of this installable native surface.
"""
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import zipfile

import pytest
import torch

from particle_sliders import pg09_text_video as bridge
from particle_sliders.bonsai_native import BonsaiParticleHead
from particle_sliders.reference import RoutedMLP
from particle_sliders.tiny_llm_backend import TinyLLMBackend
from particle_sliders.tiny_native import TinyParticleSlider, _attention_targets


def test_repository_text_backends_reexport_exact_canonical_objects():
    if importlib.util.find_spec('conceptmod') is None:
        pytest.skip('repository compatibility aliases are absent from the distributable')
    for name in ('tiny_llm_backend', 'pg09_text_video'):
        assert importlib.import_module(f'conceptmod.textsliders.{name}') is importlib.import_module(
            f'particle_sliders.{name}')
    tiny = importlib.import_module('conceptmod.textsliders.tiny_llm_particle')
    bonsai = importlib.import_module('conceptmod.textsliders.bonsai_gguf_particle')
    assert tiny.TinyParticleSlider is TinyParticleSlider
    assert tiny._attention_targets is _attention_targets
    assert bonsai.BonsaiParticleHead is BonsaiParticleHead
    # Archival entrypoints stay outside the package and retain their behavior.
    assert tiny.update.__module__ == 'conceptmod.textsliders.tiny_llm_particle'
    assert bonsai.update.__module__ == 'conceptmod.textsliders.bonsai_gguf_particle'


@pytest.mark.parametrize('name', ['tiny_llm_backend', 'tiny_native', 'bonsai_native', 'pg09_text_video'])
def test_native_text_surface_has_no_repository_or_historical_game_imports(name):
    module = importlib.import_module(f'particle_sliders.{name}')
    assert not any(hasattr(module, attr) for attr in ('RECIPE', 'build_game', 'update', 'shared'))
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(('conceptmod', 'analysis')) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith(('conceptmod', 'analysis'))


def test_tiny_native_router_and_particles_follow_host_device_without_cuda():
    host = TinyLLMBackend(dummy=True).model.to(device='meta', dtype=torch.bfloat16)
    slider = TinyParticleSlider(host)
    assert all(p.device.type == 'meta' and p.dtype == torch.bfloat16 for p in slider.parameters())
    slider.float()
    assert all(p.device.type == 'meta' and p.dtype == torch.float32 for p in slider.parameters())
    assert all(type(adapter.bridge) is RoutedMLP for adapter in slider.adapters.values())


@pytest.mark.parametrize('host_dtype', [torch.float32, torch.bfloat16])
def test_tiny_native_fp32_export_reload_preserves_host_and_signed_outputs(tmp_path, host_dtype):
    pytest.importorskip('safetensors')
    torch.manual_seed(91)
    original = TinyLLMBackend(dummy=True)
    original.model.to(dtype=host_dtype)
    slider = TinyParticleSlider(original.model).float()
    with torch.no_grad():
        for parameter in slider.parameters():
            parameter.copy_(torch.rand_like(parameter) * .02)
    path = tmp_path / 'tiny.safetensors'
    slider.save(path, {})
    torch.manual_seed(91)
    fresh = TinyLLMBackend(dummy=True)
    fresh.model.to(dtype=host_dtype)
    base = deepcopy(fresh.model.state_dict())
    restored, _ = TinyParticleSlider.load(fresh.model, path, adapter_dtype=torch.float32)
    assert all(value.dtype == torch.float32 and torch.equal(value, restored.state_dict()[key])
               for key, value in slider.state_dict().items())
    assert all(p.dtype == host_dtype for p in fresh.model.parameters())
    assert all(torch.equal(value, fresh.model.state_dict()[key]) for key, value in base.items())
    ids = original.encode('a quiet garden').ids
    assert ids == fresh.encode('a quiet garden').ids
    for scale in (-1., 0., .5, 1.):
        with slider.scaled(scale), restored.scaled(scale), torch.no_grad():
            assert torch.equal(original.hidden(ids), fresh.hidden(ids))


@pytest.mark.parametrize('corruption', ['shape', 'integer', 'bf16', 'nonfinite', 'targets'])
def test_tiny_native_loader_rejects_before_host_hooks_or_rng_change(tmp_path, corruption):
    pytest.importorskip('safetensors')
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    slider = TinyParticleSlider(TinyLLMBackend(dummy=True).model)
    path = tmp_path / 'valid.safetensors'
    slider.save(path, {})
    state = load_file(str(path))
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
    if corruption == 'shape':
        state['particles'] = state['particles'][:-1]
    elif corruption == 'integer':
        state['particles'] = state['particles'].long()
    elif corruption == 'bf16':
        state['particles'] = state['particles'].bfloat16()
    elif corruption == 'nonfinite':
        state['particles'][0, 0] = float('nan')
    else:
        record = json.loads(metadata['conceptmod'])
        record['targets'] = ['unrelated']
        metadata['conceptmod'] = json.dumps(record)
    corrupt = tmp_path / 'corrupt.safetensors'
    save_file(state, str(corrupt), metadata=metadata)
    backend = TinyLLMBackend(dummy=True)
    before = deepcopy(backend.model.state_dict())
    forwards = {name: module.forward for name, module in _attention_targets(backend.model).items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        TinyParticleSlider.load(backend.model, corrupt, adapter_dtype=torch.float32)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(module.forward == forwards[name] for name, module in _attention_targets(backend.model).items())
    assert all(torch.equal(value, backend.model.state_dict()[key]) for key, value in before.items())
    TinyParticleSlider.load(backend.model, path, adapter_dtype=torch.float32)


def test_bonsai_native_head_uses_shared_router_and_preserves_strict_binding(monkeypatch):
    head = BonsaiParticleHead()
    assert type(head.bridge) is RoutedMLP
    values = torch.randn(1, 5120)
    assert head(values) is values
    with torch.no_grad():
        head.up.weight.fill_(.01)
    with head.scaled(1.):
        assert not torch.equal(head(values), values)
    # This tests class ownership only; native-width source admission is a
    # separate reviewed runtime gate and is not bypassed by package imports.
    monkeypatch.setattr(bridge, 'require_continuous_vector', lambda *args, **kwargs: {})
    adapter = bridge.BonsaiReadoutResponseAdapter(object(), head, model_identity={'fixture': True})
    assert adapter.adapter is head and adapter.particles is head.particles
    with pytest.raises(bridge.ResponseCompatibilityError, match='existing native Bonsai head'):
        bridge.BonsaiReadoutResponseAdapter(object(), torch.nn.Linear(1, 1), model_identity={'fixture': True})


def test_built_wheel_tiny_runs_shared_game_without_repository_or_native_runtime(tmp_path):
    import particlegan

    source = tmp_path / 'source'
    shutil.copytree(Path(__file__).resolve().parents[1], source,
        ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache'))
    dist = tmp_path / 'dist'
    dist.mkdir()
    build = subprocess.run([sys.executable, '-c',
        'from setuptools import build_meta; import sys; build_meta.build_wheel(sys.argv[1])', str(dist)],
        cwd=source, capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stdout + build.stderr
    wheel, = dist.glob('*.whl')
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    dependency = Path(particlegan.__file__).resolve().parent.parent
    script = textwrap.dedent('''
        import importlib.abc
        from pathlib import Path
        import sys
        sys.path[:0] = sys.argv[1:3]
        import torch
        class Guard(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'conceptmod', 'analysis', 'transformers', 'diffusers',
                        'safetensors', 'huggingface_hub', 'llama_cpp'}:
                    raise AssertionError('unexpected optional/historical import: ' + fullname)
        sys.meta_path.insert(0, Guard())
        from particle_sliders import ConditionalPG09
        from particle_sliders.tiny_llm_backend import TinyLLMBackend
        from particle_sliders.tiny_native import TinyParticleSlider
        from particle_sliders.bonsai_native import BonsaiParticleHead
        from particle_sliders.pg09_text_video import TinyQwenResponseAdapter, TextResponseRow
        from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic
        assert Path(sys.modules[TinyLLMBackend.__module__].__file__).is_relative_to(sys.argv[1])
        torch.set_num_threads(1)
        torch.manual_seed(51)
        backend = TinyLLMBackend(dummy=True)
        base = {key: value.clone() for key, value in backend.model.state_dict().items()}
        slider = TinyParticleSlider(backend.model).float()
        adapter = TinyQwenResponseAdapter(backend, slider)
        rows = [adapter.prepare(TextResponseRow((1, 2), (3, 4))),
                adapter.prepare(TextResponseRow((5, 6), (7, 8)))]
        conditions = torch.cat([row.neutral for row in rows])
        critic = ConditionalReadoutCritic(32, 32)
        game = ConditionalPG09(adapter, rows, critic, conditions=conditions,
            particles=slider.particles, response_domain='continuous_vector', total_steps=2, batch_size=2)
        metrics = game.step()
        assert abs(metrics['g_loss'] - metrics['g_adv'] - metrics['prior_regularization']) < 1e-6
        assert all(torch.equal(value, backend.model.state_dict()[key]) for key, value in base.items())
        assert game.prior.z is slider.particles
        sources = game.state_dict()['config']['implementation_sources']
        assert 'particle_sliders.pg09_text_video' in sources
        assert 'particle_sliders.tiny_native' in sources
        assert 'particle_sliders.tiny_llm_backend' in sources
        assert not any(name.startswith('conceptmod') for name in sys.modules)
        assert not torch.cuda.is_initialized()
    ''')
    result = subprocess.run([sys.executable, '-I', '-c', script, str(installed), str(dependency)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
