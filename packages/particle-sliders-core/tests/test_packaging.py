"""Build the distributable, then exercise its imports without the source tree."""
from email import message_from_bytes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


@pytest.fixture(scope="module")
def built_package(tmp_path_factory):
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("distribution smoke test requires the setuptools build backend")
    root = tmp_path_factory.mktemp("core-distribution")
    source = root / "source"
    shutil.copytree(
        Path(__file__).resolve().parents[1], source,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"),
    )
    # Build from an extracted sdist, so an omitted runtime file cannot be
    # accidentally supplied by the checkout. No installer or network is used.
    script = """
from pathlib import Path
import os
import tarfile
from setuptools import build_meta

root = Path.cwd().parent
dist = root / 'dist'
dist.mkdir()
archive = dist / build_meta.build_sdist(str(dist))
with tarfile.open(archive) as stream:
    stream.extractall(root / 'sdist')
os.chdir(next((root / 'sdist').iterdir()))
build_meta.build_wheel(str(dist))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=source,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return root


def test_sdist_retains_release_installation_contract(built_package):
    archive, = (built_package / "dist").glob("*.tar.gz")
    with tarfile.open(archive) as stream:
        paths = {Path(member.name).relative_to(Path(member.name).parts[0]) for member in stream.getmembers()}
    assert Path("CONDITIONAL-PG09-CONTRACT.md") in paths
    assert Path("requirements-pg09.txt") in paths


def test_wheel_metadata_pins_released_particle_gan(built_package):
    wheel, = (built_package / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as stream:
        metadata_path, = (name for name in stream.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = message_from_bytes(stream.read(metadata_path))
    assert metadata["Name"] == "particle-sliders-core"
    assert metadata["Requires-Python"] == ">=3.10"
    requirements = metadata.get_all("Requires-Dist")
    assert "particlegan==0.9.1" in requirements
    assert "torch<3,>=2.6" in requirements


def test_wheel_imports_alias_versions_and_verified_release_resources(built_package):
    import particlegan

    wheel, = (built_package / "dist").glob("*.whl")
    installed = built_package / "installed"
    with zipfile.ZipFile(wheel) as stream:
        stream.extractall(installed)
    environment = os.environ.copy()
    # Do not inherit pytest's source paths. The only package search overrides
    # are our built wheel and the already qualified upstream dependency.
    environment["PYTHONPATH"] = os.pathsep.join((
        str(installed), str(Path(particlegan.__file__).resolve().parent.parent),
    ))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
from importlib.metadata import version
from importlib.resources import files
import importlib
import importlib.abc
import json
from pathlib import Path
import sys
import warnings

class NoRepositoryOrNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'conceptmod', 'diffusers', 'peft', 'transformers', 'safetensors',
            'qwen_image_sliders', 'lumen_studio', 'moss_sliders', 'supra',
        }:
            raise AssertionError(f'core import depends on optional repository/native package: {fullname}')
sys.meta_path.insert(0, NoRepositoryOrNativeImports())

import particle_sliders
from particle_sliders import pg09
for name in ('pg09_backends', 'pg09_audio', 'pg09_images', 'pg09_readout_critic',
             'pg09_moss_campaign', 'pg09_remaining_campaigns', 'supra_model',
             'native_lora', 'music3_native', 'tiny_llm_backend', 'tiny_native',
             'bonsai_native', 'pg09_text_video'):
    backend = importlib.import_module(f'particle_sliders.{name}')
    assert Path(backend.__file__).resolve().is_relative_to(Path.cwd() / 'installed')
assert 'conceptmod' not in sys.modules
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import concept_slider_core
root = Path.cwd() / 'installed'
assert Path(particle_sliders.__file__).resolve().is_relative_to(root)
assert Path(concept_slider_core.__file__).resolve().is_relative_to(root)
assert any(item.category is DeprecationWarning for item in caught)
assert concept_slider_core.__all__ == particle_sliders.__all__
assert all(getattr(concept_slider_core, name) is getattr(particle_sliders, name)
           for name in particle_sliders.__all__)
assert particle_sliders.__version__ == concept_slider_core.__version__ == version('particle-sliders-core')
for name in ('provenance.json', 'locked_provenance.json', 'bispf_provenance.json',
             'pg09_release_sources.json', 'pg09_continuous_profile.json'):
    json.loads(files('particle_sliders').joinpath(name).read_text())
assert files('particle_sliders').joinpath('LICENSE').read_text()
print(json.dumps({'version': particle_sliders.__version__,
                  'pg_version': version('particlegan'),
                  'verified_sources': pg09._released_sources()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=built_package, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["pg_version"] == "0.9.1"
    assert len(receipt["verified_sources"]) == 64
