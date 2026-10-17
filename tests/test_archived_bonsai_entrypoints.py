"""Historical diagnostic CLIs reject before source, artifact or native access."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', [
    'diagnose_bonsai_pg09_readonly.py',
    'ablate_bonsai_pg09_readonly.py',
])
@pytest.mark.parametrize('existing_output', [False, True])
def test_archived_diagnostic_rejects_before_source_artifact_or_native_access(
        tmp_path, script, existing_output):
    # Valid CLI arguments exercise the retirement guard, while any former body
    # immediately trips a path-access or native-import sentinel. No real cohort
    # checkout, checkpoint, readout cache or runtime package is needed.
    trap = tmp_path / 'trap'
    trap.mkdir()
    (trap / 'sitecustomize.py').write_text('''
import importlib.abc
from pathlib import Path
import sys

def forbidden(*args, **kwargs):
    raise AssertionError('forbidden archive path access')

for name in ('resolve', 'read_text', 'read_bytes', 'write_text', 'write_bytes',
             'open', 'stat', 'exists', 'is_file', 'is_dir', 'mkdir', 'iterdir',
             'glob', 'rglob'):
    setattr(Path, name, forbidden)

class NoNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'yaml', 'conceptmod', 'particle_sliders'}:
            raise AssertionError('forbidden archive native import: ' + fullname)

sys.meta_path.insert(0, NoNativeImports())
''')
    output = tmp_path / 'report.json'
    original = b'preserve existing historical evidence\n'
    if existing_output:
        output.write_bytes(original)
    env = dict(os.environ, PYTHONPATH=str(trap), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / script),
         '--code', str(tmp_path / 'missing-code'),
         '--run', str(tmp_path / 'missing-run'), '--output', str(output)],
        env=env, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 2
    assert 'archived source-bound Bonsai diagnostic retired here' in result.stderr
    assert 'original immutable cohort source for historical reproduction' in result.stderr
    assert 'not new package qualification' in result.stderr
    assert 'forbidden archive' not in result.stderr
    assert 'Traceback' not in result.stderr
    assert result.stdout == ''
    if existing_output:
        assert output.read_bytes() == original
    else:
        assert not output.exists()
    assert not (tmp_path / 'missing-code').exists()
    assert not (tmp_path / 'missing-run').exists()
