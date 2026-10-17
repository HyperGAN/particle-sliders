from pathlib import Path
import subprocess
import sys

import torch

from test_pg09_audio import equal


def test_bounded_worker_exact_fresh_process_resume(tmp_path):
    driver = Path(__file__).resolve().parents[1] / 'scripts/qualify_audio_pg09_live.py'
    full, first, resumed = [tmp_path / name for name in ('full', 'first', 'resumed')]
    base = [sys.executable, str(driver), '--family', 'music3', '--fixture',
            '--device', 'cpu', '--total-steps', '4']
    for extra in (['--until', '3', '--out', str(full)],
                  ['--until', '1', '--out', str(first)],
                  ['--until', '3', '--out', str(resumed), '--resume', str(first / 'checkpoint.pt')]):
        result = subprocess.run([*base, *extra], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    equal(torch.load(full / 'qualification.pt', weights_only=True),
          torch.load(resumed / 'qualification.pt', weights_only=True))


def test_native_worker_without_admission_stops_before_model_load(tmp_path):
    driver = Path(__file__).resolve().parents[1] / 'scripts/qualify_audio_pg09_live.py'
    output = tmp_path / 'not-admitted'
    result = subprocess.run([sys.executable, str(driver), '--family', 'music3',
        '--device', 'cuda:0', '--until', '3', '--out', str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'explicit owner admission' in result.stderr
    assert not output.exists()
