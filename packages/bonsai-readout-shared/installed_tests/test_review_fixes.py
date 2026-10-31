"""Opt-in installed CPU regressions for Claude's bounded source review."""
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

if os.environ.get('BONSAI_INSTALLED_TESTS') != '1':
    pytest.skip('opt-in installed Bonsai fixture: set BONSAI_INSTALLED_TESTS=1', allow_module_level=True)

import torch
from bonsai_readout_shared import binding as product
from particle_sliders.lifecycle import SessionConfig


def _subprocess(code, tmp_path, pythonpath=None):
    env = dict(os.environ)
    if pythonpath is not None:
        env['PYTHONPATH'] = str(pythonpath)
    return subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                          text=True, capture_output=True)


def test_real_path_shadow_rejected_before_core_execution(tmp_path):
    shadow = tmp_path / 'shadow'
    package = shadow / 'particle_sliders'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text("raise RuntimeError('SHADOW_EXECUTED')\n")
    result = _subprocess('import bonsai_readout_shared.binding', tmp_path, shadow)
    assert result.returncode != 0
    assert 'import origin differs' in result.stderr
    assert 'SHADOW_EXECUTED' not in result.stderr


def test_real_executable_stale_bytecode_rejected_before_core_import(tmp_path):
    from importlib.util import cache_from_source
    import importlib._bootstrap_external as bootstrap
    source = Path(distribution('particle-sliders-core').locate_file('particle_sliders'))
    private = tmp_path / 'private-install'
    destination = private / 'particle_sliders'
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__'))
    metadata = private / 'particle_sliders_core-0.12.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text(distribution('particle-sliders-core').read_text('METADATA'))
    module = destination / 'pg094.py'
    cached = Path(cache_from_source(str(module)))
    cached.parent.mkdir(exist_ok=True)
    code = compile("raise RuntimeError('STALE_BYTECODE_EXECUTED')", str(module), 'exec')
    cached.write_bytes(bootstrap._code_to_timestamp_pyc(code, int(module.stat().st_mtime), module.stat().st_size))
    # Control proves this is a usable poisoned cache, not a merely altered file.
    control = _subprocess('import particle_sliders.pg094', tmp_path, private)
    assert control.returncode != 0 and 'STALE_BYTECODE_EXECUTED' in control.stderr
    guarded = _subprocess('import bonsai_readout_shared.binding', tmp_path, private)
    assert guarded.returncode != 0 and 'bytecode cache differs from source' in guarded.stderr
    assert 'STALE_BYTECODE_EXECUTED' not in guarded.stderr


def test_namespace_does_not_replace_sibling_conceptmod_preview(tmp_path):
    sibling = tmp_path / 'sibling'
    package = sibling / 'conceptmod/textsliders'
    package.mkdir(parents=True)
    (sibling / 'conceptmod/__init__.py').write_text('SIBLING = True\n')
    (package / '__init__.py').write_text('PREVIEW = "sibling-owned"\n')
    result = _subprocess('''
import conceptmod.textsliders as sibling
before = sibling.__file__
import bonsai_readout_shared.binding
assert sibling.__file__ == before and sibling.PREVIEW == 'sibling-owned'
''', tmp_path, sibling)
    assert result.returncode == 0, result.stderr
    files = [str(path) for path in distribution('bonsai-readout-shared').files]
    assert not any(path.startswith('conceptmod/') for path in files)


@pytest.mark.parametrize('fail', [False, True])
def test_direct_factory_restores_caller_cpu_rng_on_success_and_failure(tmp_path, monkeypatch, fail):
    path = tmp_path / 'rows.json'
    path.write_text(json.dumps([dict(positive='bright room', neutral='room'),
                                dict(positive='sunny day', neutral='day')]))
    options = dict(rows_file=str(path), steps=2, batch=2, seed=71)
    before = torch.get_rng_state().clone()
    if fail:
        def refuse(*args):
            raise RuntimeError('construction failed')
        monkeypatch.setattr(product, 'HNDLReadoutCritic', refuse)
        with pytest.raises(RuntimeError, match='construction failed'):
            product.prepare_fixture(options)
    else:
        binding = product.prepare_fixture(options)
        binding.close()
    assert torch.equal(before, torch.get_rng_state())


def test_same_size_model_content_change_rejected_without_native_admission(tmp_path):
    path = tmp_path / 'tiny-content-fixture.bin'
    path.write_bytes(b'original content')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert product.file_content_identity(path, expected)['content_sha256'] == expected
    path.write_bytes(b'changed! content')
    with pytest.raises(ValueError, match='content SHA256 differs'):
        product.file_content_identity(path, expected)


def test_native_identity_is_still_gated_before_reading_file(tmp_path):
    with pytest.raises(ValueError, match='width >=170'):
        product._native_model_identity(dict(weights=str(tmp_path / 'absent.gguf'), model_sha256='0' * 64))


def test_unselected_installed_lane_skips_without_distribution_or_reference_assumptions(tmp_path):
    source = Path(__file__)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-c', '/dev/null', str(source)],
        cwd=tmp_path, env={k: v for k, v in os.environ.items() if k not in ('BONSAI_INSTALLED_TESTS', 'BONSAI_REFERENCE_ROOT')},
        text=True, capture_output=True)
    assert result.returncode == 5 and '1 skipped' in result.stdout


def test_symlinkless_build_materializes_native_loader_and_self_contained_sdist(tmp_path):
    reference = os.environ.get('BONSAI_REFERENCE_ROOT')
    if not reference:
        pytest.skip('optional packaging check needs BONSAI_REFERENCE_ROOT')
    reference = Path(reference)
    package = tmp_path / 'plain-checkout/packages/bonsai-readout-shared'
    shutil.copytree(reference / 'packages/bonsai-readout-shared', package,
        ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__', '_native.py'))
    # All selected checkout files are regular files; no symlink dereference is
    # available to make a build pass. Supply only the unchanged historical input.
    assert not any(path.is_symlink() for path in package.rglob('*'))
    native = package.parents[1] / 'conceptmod/textsliders/bonsai_gguf_backend.py'
    native.parent.mkdir(parents=True)
    shutil.copyfile(reference / 'conceptmod/textsliders/bonsai_gguf_backend.py', native)
    target = package / 'src/bonsai_readout_shared/_native.py'
    assert not target.exists()
    result = subprocess.run([sys.executable, '-c', 'import _bonsai_build; _bonsai_build.build_sdist("dist")'],
        cwd=package, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == native.read_bytes()
    archive = next((package / 'dist').glob('*.tar.gz'))
    extracted = tmp_path / 'sdist-only'
    with tarfile.open(archive) as handle:
        assert all('..' not in Path(item.name).parts and not item.name.startswith('/') and not item.issym()
                   for item in handle.getmembers())
        handle.extractall(extracted, filter='data')
    root = next(extracted.iterdir())
    wheel_dir = tmp_path / 'wheels'
    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
        '--wheel-dir', str(wheel_dir), str(root)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(next(wheel_dir.glob('*.whl'))) as wheel:
        assert wheel.read('bonsai_readout_shared/_native.py') == native.read_bytes()
        assert not any(name.startswith('conceptmod/') for name in wheel.namelist())
