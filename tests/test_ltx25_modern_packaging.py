"""Archive/source admission, namespace and reproducibility regressions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


def load_builder():
    root = Path(__file__).resolve().parents[1]
    source = root / 'backends' / 'ltx25-modern' / 'build_backend.py'
    spec = importlib.util.spec_from_file_location('ltx_profile_build', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return root, builder


def test_isolated_reproducible_builds_and_self_contained_namespace(tmp_path, monkeypatch):
    root, builder = load_builder()
    cwd = Path.cwd()
    def forbidden(*args):
        pytest.fail('build changed the parent process cwd')
    monkeypatch.setattr(os, 'chdir', forbidden)
    paths = [tmp_path / name for name in ('wheel-a','wheel-b','sdist-a','sdist-b','rebuilt')]
    for path in paths:
        path.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        names = list(pool.map(lambda output: builder.build_wheel(str(output)), paths[:2]))
    direct = (paths[0] / names[0]).read_bytes()
    assert direct == (paths[1] / names[1]).read_bytes()
    sdist_a = builder.build_sdist(str(paths[2]))
    sdist_b = builder.build_sdist(str(paths[3]))
    assert (paths[2] / sdist_a).read_bytes() == (paths[3] / sdist_b).read_bytes()
    assert Path.cwd() == cwd
    with tarfile.open(paths[2] / sdist_a) as archive:
        assert all(not Path(x).is_absolute() and '..' not in Path(x).parts for x in archive.getnames())
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    str(paths[2] / sdist_a), '-w', str(paths[4])], check=True, capture_output=True)
    rebuilt = next(paths[4].glob('*.whl'))
    assert direct == rebuilt.read_bytes()
    with zipfile.ZipFile(rebuilt) as wheel:
        assert not any(name.startswith('conceptmod/') for name in wheel.namelist())
        for installed, original in builder.MODULES.items():
            source = (root / 'conceptmod' / 'textsliders' / original).read_bytes()
            if original == 'ltx25_backend.py':
                for module in ('ltx25_uni','slider_targets'):
                    old = ('from conceptmod.textsliders.' + module + ' import').encode()
                    assert source.count(old) == 1
                    source = source.replace(old, ('from .' + module + ' import').encode(), 1)
            assert wheel.read(builder.PACKAGE + '/' + installed) == source
        assert not any('train_lora_' in name or 'h3_' in name or 'bonsai_' in name for name in wheel.namelist())


def test_build_refuses_ambiguous_replacement_before_emitting_artifact(tmp_path, monkeypatch):
    root, builder = load_builder()
    profile = tmp_path / 'backends' / 'ltx25-modern'
    shutil.copytree(root / 'backends' / 'ltx25-modern', profile, ignore=shutil.ignore_patterns('__pycache__'))
    source = tmp_path / 'conceptmod' / 'textsliders'
    source.mkdir(parents=True)
    for original in builder.MODULES.values():
        shutil.copyfile(root / 'conceptmod' / 'textsliders' / original, source / original)
    project = profile / 'pyproject.toml'
    project.write_text(project.read_text().replace('build-backend = "build_backend"', 'build-backend = "unexpected"'))
    monkeypatch.setattr(builder, 'ROOT', profile)
    with pytest.raises(ValueError, match='exactly once'):
        builder.build_wheel(str(tmp_path / 'unpublished'))
    assert not (tmp_path / 'unpublished').exists()
