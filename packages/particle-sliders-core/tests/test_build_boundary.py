"""A wheel rebuild cannot adopt stale product outputs or symlink aliases."""
from pathlib import Path
import shutil
import sys
import zipfile

from test_namespace_installation import run


def test_rebuild_removes_stale_product_files_and_aliases(tmp_path):
    project = Path(__file__).resolve().parents[1]
    copy = tmp_path / 'core'
    shutil.copytree(project, copy, ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
    stale = copy / 'build/lib'
    (stale / 'particle_sliders/textsliders').mkdir(parents=True)
    (stale / 'particle_sliders/native_lora.py').write_text('# old allowed output\n')
    (stale / 'particle_sliders/music3_native.py').symlink_to('native_lora.py')
    (stale / 'particle_sliders/textsliders/leak.py').write_text('raise RuntimeError("stale product")\n')
    (stale / 'particle_sliders_legacy_imports.pth').write_text('import conceptmod\n')
    wheels = tmp_path / 'wheels'
    run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-index',
         '--no-build-isolation', '--wheel-dir', str(wheels), str(copy)], tmp_path)
    wheel, = wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert 'particle_sliders/native_lora.py' in names
        assert 'particle_sliders/music3_native.py' not in names
        assert not any(name.startswith('particle_sliders/textsliders/') for name in names)
        assert 'particle_sliders_legacy_imports.pth' not in names
