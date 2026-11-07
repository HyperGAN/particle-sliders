"""Real pip install/uninstall isolation, including a separate conceptmod owner.

The fixture owns the same top-level file as the separate Conceptmod project,
exports a native backend, and deliberately provides no legacy textsliders tree.
No network, model download, or modification of the invoking environment.
"""
import base64
import csv
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile

import pytest


def run(args, cwd, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result


@pytest.fixture(scope="module")
def wheels(tmp_path_factory):
    folder = tmp_path_factory.mktemp("namespace-wheels")
    configured = os.environ.get("PARTICLE_SLIDERS_TEST_WHEEL")
    if configured:
        core = Path(configured).resolve()
    else:
        project = Path(__file__).resolve().parents[1]
        assert (project / "pyproject.toml").is_file(), "Set PARTICLE_SLIDERS_TEST_WHEEL outside a checkout"
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--no-cache-dir", "--no-index", "--wheel-dir", str(folder), str(project)], folder)
        core, = folder.glob("particle_sliders_core-*.whl")
    with zipfile.ZipFile(core) as wheel:
        assert not any(name.startswith("conceptmod/") for name in wheel.namelist())
        hook = "particle_sliders_legacy_imports.pth"
        record, = [name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")]
        rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(wheel.read(record).decode()))}
        content = wheel.read(hook)
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        assert rows[hook] == ["sha256=" + digest, str(len(content))]
    fixture = folder / "conceptmod-0.0.0-py3-none-any.whl"
    files = {
        "conceptmod/__init__.py": b'PROJECT = "separate-conceptmod"\n',
        "conceptmod/backends/__init__.py": b'NATIVE_API = "native-backend"\n',
        "conceptmod-0.0.0.dist-info/METADATA": b'Metadata-Version: 2.1\nName: conceptmod\nVersion: 0.0.0\n',
        "conceptmod-0.0.0.dist-info/WHEEL": b'Wheel-Version: 1.0\nGenerator: namespace-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    rows = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, "sha256=" + digest, str(len(data))])
    record = "conceptmod-0.0.0.dist-info/RECORD"
    rows.append([record, "", ""])
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    files[record] = output.getvalue().encode()
    with zipfile.ZipFile(fixture, "w") as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
    return core, fixture


def isolated_python(folder):
    environment = folder / "env"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Reuse CPU libraries by symlinking only dependency files, excluding all
    # Sliders/Conceptmod packages, metadata, and startup hooks. This prevents an
    # older core in the caller's dependency environment masking uninstall bugs.
    dependencies = folder / "dependencies"
    dependencies.mkdir()
    roots = dict.fromkeys(Path(importlib.util.find_spec(name).origin).parent.parent
                         for name in ("torch", "safetensors", "particlegan"))
    for root in roots:
        for entry in root.iterdir():
            if entry.name.endswith(".pth") or entry.name.startswith(
                    ("particle_sliders", "concept_slider_core", "conceptmod", "__editable__")):
                continue
            target = dependencies / entry.name
            if target.is_symlink():
                target.unlink()
            target.symlink_to(entry, target_is_directory=entry.is_dir())
    site = run([str(python), "-c", "import site; print(site.getsitepackages()[0])"], folder).stdout.strip()
    Path(site, "dependencies.pth").write_text(str(dependencies) + "\n")
    return str(python)


@pytest.mark.parametrize("core_first", [False, True])
def test_coinstall_order_and_both_uninstalls(tmp_path, wheels, core_first):
    core, conceptmod = wheels
    python = isolated_python(tmp_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    def pip(*args):
        return run([python, "-m", "pip", *args], tmp_path, env)
    def check(source):
        result = run([python, "-c", source], tmp_path, env)
        assert "Error processing line" not in result.stderr
    for wheel in ((core, conceptmod) if core_first else (conceptmod, core)):
        pip("install", "--no-deps", "--no-index", "--no-cache-dir", str(wheel))
    check('''
import importlib, importlib.metadata
import conceptmod
assert conceptmod.PROJECT == 'separate-conceptmod'
from conceptmod.backends import NATIVE_API
assert NATIVE_API == 'native-backend'
from conceptmod.textsliders.lora import LoRAModule as Old
from particle_sliders.textsliders.lora import LoRAModule as New
assert Old is New and New.__module__ == 'particle_sliders.textsliders.lora'
assert importlib.import_module('conceptmod.textsliders.lora') is importlib.import_module('particle_sliders.textsliders.lora')
core = importlib.metadata.distribution('particle-sliders-core')
other = importlib.metadata.distribution('conceptmod')
assert not set(map(str, core.files)) & set(map(str, other.files))
''')
    # Site-disabled interpreters do not process startup hooks. Explicit
    # activation is supported and does not require importing any model runtime.
    site = run([python, "-c", "import site; print(site.getsitepackages()[0])"], tmp_path, env).stdout.strip()
    run([python, "-S", "-c", f'''
import importlib.util, sys
sys.path.insert(0, {site!r})
assert importlib.util.find_spec('conceptmod.textsliders') is None
from particle_sliders._legacy_namespace import install
install()
assert importlib.util.find_spec('conceptmod.textsliders.lora') is not None
assert 'torch' not in sys.modules
'''], tmp_path, env)
    pip("uninstall", "-y", "conceptmod")
    check('''
from conceptmod.textsliders.lora import LoRAModule as Old
from particle_sliders.textsliders.lora import LoRAModule as New
assert Old is New
import conceptmod
assert not hasattr(conceptmod, 'PROJECT')
''')
    pip("install", "--no-deps", "--no-index", "--no-cache-dir", str(conceptmod))
    pip("uninstall", "-y", "particle-sliders-core")
    check('''
import importlib.util, pathlib, site
import conceptmod
assert conceptmod.PROJECT == 'separate-conceptmod'
from conceptmod.backends import NATIVE_API
assert NATIVE_API == 'native-backend'
assert importlib.util.find_spec('particle_sliders') is None
assert importlib.util.find_spec('conceptmod.textsliders') is None
assert not pathlib.Path(site.getsitepackages()[0], 'particle_sliders_legacy_imports.pth').exists()
''')


def test_editable_wheel_owns_and_removes_startup_hook(tmp_path):
    project = Path(os.environ.get("PARTICLE_SLIDERS_TEST_PROJECT", Path(__file__).resolve().parents[1]))
    assert (project / "pyproject.toml").is_file(), "Set PARTICLE_SLIDERS_TEST_PROJECT outside a checkout"
    python = isolated_python(tmp_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    run([python, "-m", "pip", "install", "--no-deps", "--no-index", "--no-cache-dir",
         "--no-build-isolation", "--editable", str(project)], tmp_path, env)
    run([python, "-c", '''
import importlib.metadata, pathlib, site, sys
from conceptmod.textsliders.lora import LoRAModule as Old
from particle_sliders.textsliders.lora import LoRAModule as New
assert Old is New
record = importlib.metadata.distribution('particle-sliders-core').files
assert 'particle_sliders_legacy_imports.pth' in set(map(str, record))
assert pathlib.Path(site.getsitepackages()[0], 'particle_sliders_legacy_imports.pth').is_file()
assert not any(name in sys.modules for name in ('particlegan', 'diffusers', 'hypergan', 'hndl'))
'''], tmp_path, env)
    run([python, "-m", "pip", "uninstall", "-y", "particle-sliders-core"], tmp_path, env)
    result = run([python, "-c", '''
import importlib.util, pathlib, site
assert importlib.util.find_spec('conceptmod') is None
assert importlib.util.find_spec('particle_sliders') is None
assert not pathlib.Path(site.getsitepackages()[0], 'particle_sliders_legacy_imports.pth').exists()
'''], tmp_path, env)
    assert "Error processing line" not in result.stderr
