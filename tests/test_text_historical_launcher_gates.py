"""Historical public routes must reject before source, asset or process access."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('script,extra', [
    ('qualify_text_video_pg09_cpu.py', []),
    ('qualify_text_video_pg09_cpu.py', ['--family', 'tiny']),
    ('qualify_wide_text_pg09_cpu.py', ['--overlay', '/absent-overlay']),
    ('qualify_wide_text_pg09_cpu.py', ['--overlay', '/absent-overlay', '--family', 'tiny']),
    ('qualify_wide_text_pg09_cpu.py', ['--overlay', '/absent-overlay', '--family', 'bonsai']),
    ('verify_bonsai_pg09_cached.py', ['--overlay', '/absent-overlay', '--mode', 'produce']),
    ('verify_bonsai_pg09_cached.py', ['--overlay', '/absent-overlay', '--mode', 'restore',
                                      '--plan', '/absent-plan.json']),
])
def test_retired_text_routes_reject_before_source_assets_and_output(tmp_path, script, extra):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'must-not-exist'
    result = subprocess.run(
        [sys.executable, '-I', str(root / 'scripts' / script), '--core', '/absent-core',
         '--output', str(output), *extra],
        cwd=tmp_path, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'retired' in result.stderr
    assert 'installed-package' in result.stderr or 'installed package' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not output.exists()
