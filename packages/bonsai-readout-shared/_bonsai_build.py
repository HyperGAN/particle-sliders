"""Materialize the pinned legacy loader at build time without source symlinks."""
import hashlib
from pathlib import Path

from setuptools import build_meta

NATIVE_SHA256 = 'bc83fa01340cc11efcb6ba4d7aaaddc05ce3a01a6ce1b086f933205358bfd84a'


def _materialize():
    root = Path(__file__).resolve().parent
    source = root.parents[1] / 'conceptmod/textsliders/bonsai_gguf_backend.py'
    target = root / 'src/bonsai_readout_shared/_native.py'
    data = source.read_bytes() if source.is_file() else target.read_bytes()
    if hashlib.sha256(data).hexdigest() != NATIVE_SHA256:
        raise ValueError('Bonsai packaged native loader differs from its pinned historical source')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def get_requires_for_build_wheel(config_settings=None):
    _materialize()
    return build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _materialize()
    return build_meta.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _materialize()
    return build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _materialize()
    return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _materialize()
    return build_meta.build_sdist(sdist_directory, config_settings)
