"""Build the shared distribution without product namespace startup hooks."""
from setuptools import build_meta as _setuptools
from setuptools.build_meta import (
    build_sdist,
    build_wheel,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    # A plain src/ .pth exposes excluded product modules. Strict PEP660 uses
    # only build_py's reviewed outputs, matching the regular wheel boundary.
    config_settings = dict(config_settings or {})
    if config_settings.get('editable_mode', 'strict') != 'strict':
        raise ValueError('Shared core editable installs require editable_mode=strict')
    config_settings['editable_mode'] = 'strict'
    return _setuptools.build_editable(
        wheel_directory, config_settings, metadata_directory)
