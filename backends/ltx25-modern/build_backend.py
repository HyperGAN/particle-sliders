"""Stage only this profile's unchanged native helpers for portable wheel/sdist.

No generated copies are checked into the repository. An sdist is self-contained
and uses ordinary setuptools; it does not depend on parent checkout paths.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from setuptools import build_meta

ROOT = Path(__file__).resolve().parent
MODULES = ('__init__.py', 'ltx25_gan.py', 'ltx25_backend.py', 'ltx25_uni.py', 'slider_targets.py')


@contextmanager
def staged():
    previous = Path.cwd()
    with TemporaryDirectory(prefix='ltx25-modern-build-') as directory:
        target = Path(directory)
        source = ROOT.parents[1] / 'conceptmod'
        destination = target / 'src' / 'conceptmod'
        (destination / 'textsliders').mkdir(parents=True)
        shutil.copyfile(source / '__init__.py', destination / '__init__.py')
        for name in MODULES:
            shutil.copyfile(source / 'textsliders' / name, destination / 'textsliders' / name)
        for name in ('LICENSE', 'README.md'):
            shutil.copyfile(ROOT / name, target / name)
        project = (ROOT / 'pyproject.toml').read_text().replace(
            'build-backend = "build_backend"\nbackend-path = ["."]',
            'build-backend = "setuptools.build_meta"')
        (target / 'pyproject.toml').write_text(project)
        try:
            os.chdir(target)
            yield
        finally:
            os.chdir(previous)


def get_requires_for_build_wheel(config_settings=None):
    return []


get_requires_for_build_sdist = get_requires_for_build_wheel


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    output = str(Path(metadata_directory).resolve())
    with staged():
        return build_meta.prepare_metadata_for_build_wheel(output, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    output = str(Path(wheel_directory).resolve())
    metadata = str(Path(metadata_directory).resolve()) if metadata_directory else None
    with staged():
        return build_meta.build_wheel(output, config_settings, metadata)


def build_sdist(sdist_directory, config_settings=None):
    output = str(Path(sdist_directory).resolve())
    with staged():
        return build_meta.build_sdist(output, config_settings)
