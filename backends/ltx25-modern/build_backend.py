"""Reproducible isolated-namespace builds; the caller's cwd never changes."""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PACKAGE = 'particle_sliders_ltx25_dummy_preview'
MODULES = {'binding.py': 'ltx25_gan.py', 'ltx25_backend.py': 'ltx25_backend.py',
           'ltx25_uni.py': 'ltx25_uni.py', 'slider_targets.py': 'slider_targets.py'}
EPOCH = 315532800


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('build replacement must match exactly once: ' + old)
    return text.replace(old, new, 1)


@contextmanager
def staged():
    with TemporaryDirectory(prefix='ltx25-dummy-build-') as directory:
        target = Path(directory)
        source = ROOT.parents[1] / 'conceptmod' / 'textsliders'
        destination = target / 'src' / PACKAGE
        destination.mkdir(parents=True)
        (destination / '__init__.py').write_text('"""LTX dummy CPU preview only; no pretrained/native model admission."""\n')
        for installed, original in MODULES.items():
            text = (source / original).read_text()
            if original == 'ltx25_backend.py':
                for module in ('ltx25_uni', 'slider_targets'):
                    text = replace_once(text, 'from conceptmod.textsliders.' + module + ' import',
                                        'from .' + module + ' import')
            (destination / installed).write_text(text)
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(destination.glob('*.py'))}
        core = json.loads((ROOT / 'core012-runtime.json').read_text())
        if core['commit'] != '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231':
            raise ValueError('unreviewed core source profile')
        admission = replace_once((ROOT / 'admission.py').read_text(), 'PRODUCT_SOURCES = {}',
                                 'PRODUCT_SOURCES = ' + repr(sources))
        admission = replace_once(admission, 'CORE_FILES = {}', 'CORE_FILES = ' + repr(core['files']))
        (destination / '_admission.py').write_text(admission)
        for name in ('LICENSE', 'README.md'):
            shutil.copyfile(ROOT / name, target / name)
        project = replace_once((ROOT / 'pyproject.toml').read_text(),
            'build-backend = "build_backend"\nbackend-path = ["."]',
            'build-backend = "setuptools.build_meta"')
        (target / 'pyproject.toml').write_text(project)
        (target / 'setup.py').write_text(
            'import os\nfrom setuptools import setup\n'
            'os.environ["SOURCE_DATE_EPOCH"] = "315532800"\nsetup()\n')
        for path in sorted(target.rglob('*'), reverse=True):
            os.utime(path, (EPOCH, EPOCH))
        yield target


def get_requires_for_build_wheel(config_settings=None):
    return []


get_requires_for_build_sdist = get_requires_for_build_wheel


def _run(method, arguments):
    with staged() as target:
        code = ('import json,sys; from setuptools import build_meta; '
                'value=getattr(build_meta,sys.argv[1])(*json.loads(sys.argv[2])); '
                'print("__LTX_BUILD_RESULT__"+json.dumps(value))')
        environment = dict(os.environ, SOURCE_DATE_EPOCH=str(EPOCH))
        result = subprocess.run([sys.executable, '-c', code, method, json.dumps(arguments)],
            cwd=target, env=environment, check=True, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        marker = '__LTX_BUILD_RESULT__'
        matches = [line[len(marker):] for line in lines if line.startswith(marker)]
        if len(matches) != 1:
            raise ValueError('build subprocess must return exactly one receipt')
        return json.loads(matches[0])


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _run('prepare_metadata_for_build_wheel', [str(Path(metadata_directory).resolve()), config_settings])


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    metadata = str(Path(metadata_directory).resolve()) if metadata_directory else None
    return _run('build_wheel', [str(Path(wheel_directory).resolve()), config_settings, metadata])


def build_sdist(sdist_directory, config_settings=None):
    output = Path(sdist_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = _run('build_sdist', [str(output), config_settings])
    # Normalize tar ownership/timestamps and gzip header, which setuptools does
    # not consistently bind to SOURCE_DATE_EPOCH across supported versions.
    archive_bytes = io.BytesIO()
    with tarfile.open(output / name, 'r:gz') as original, tarfile.open(fileobj=archive_bytes, mode='w') as archive:
        for member in sorted(original.getmembers(), key=lambda value: value.name):
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('unsafe source archive path')
            member.uid = member.gid = 0
            member.uname = member.gname = ''
            member.mtime = EPOCH
            member.pax_headers = {}
            archive.addfile(member, original.extractfile(member) if member.isfile() else None)
    with (output / name).open('wb') as destination:
        with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=EPOCH) as compressed:
            compressed.write(archive_bytes.getvalue())
    return name
