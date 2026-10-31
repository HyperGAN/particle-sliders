"""Regression: a source archive must retain the selected native helper files."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


def test_self_contained_sdist_rebuild_preserves_native_source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / 'backends' / 'ltx25-modern' / 'build_backend.py'
    spec = importlib.util.spec_from_file_location('ltx_profile_build', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    name = builder.build_sdist(str(tmp_path))
    with tarfile.open(tmp_path / name) as archive:
        assert all(not Path(x).is_absolute() and '..' not in Path(x).parts for x in archive.getnames())
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    str(tmp_path / name), '-w', str(tmp_path)], check=True, capture_output=True)
    with zipfile.ZipFile(next(tmp_path.glob('*.whl'))) as wheel:
        for filename in builder.MODULES:
            target = 'conceptmod/textsliders/' + filename
            assert wheel.read(target) == (root / target).read_bytes()
        assert not any('train_lora_' in name or 'h3_' in name or 'bonsai_' in name for name in wheel.namelist())
