"""Setuptools builds plus one distribution-owned, RECORD-tracked startup hook.

Wheel-root .pth placement is necessary: setuptools data-files install below the
environment prefix, where Python's site module does not process them. Apply the
same operation to regular and PEP 660 editable wheels; no conceptmod files ship.
"""
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile
import zipfile

from setuptools import build_meta as _setuptools
from setuptools.build_meta import (
    build_sdist,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)

_HOOK = "particle_sliders_legacy_imports.pth"


def _with_startup_hook(directory, filename):
    wheel = Path(directory) / filename
    hook = Path(__file__).with_name(_HOOK).read_bytes()
    with zipfile.ZipFile(wheel) as source:
        records = [name for name in source.namelist() if name.endswith(".dist-info/RECORD")]
        if len(records) != 1 or _HOOK in source.namelist():
            raise ValueError("Expected one unsigned setuptools wheel without a startup hook")
        record = records[0]
        rows = list(csv.reader(io.StringIO(source.read(record).decode())))
        digest = base64.urlsafe_b64encode(hashlib.sha256(hook).digest()).rstrip(b"=").decode()
        rows.append([_HOOK, "sha256=" + digest, str(len(hook))])
        output = io.StringIO(newline="")
        csv.writer(output).writerows(rows)
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".whl")
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as target:
                for info in source.infolist():
                    target.writestr(info, output.getvalue().encode() if info.filename == record
                                    else source.read(info.filename))
                target.writestr(_HOOK, hook)
            os.replace(temporary, wheel)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return filename


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _with_startup_hook(wheel_directory, _setuptools.build_wheel(
        wheel_directory, config_settings, metadata_directory))


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _with_startup_hook(wheel_directory, _setuptools.build_editable(
        wheel_directory, config_settings, metadata_directory))
