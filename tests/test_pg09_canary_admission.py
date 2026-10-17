"""Pure CPU checks for the bounded new-objective canary's admission thresholds."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pg09_canary_guard', Path(__file__).parents[1] / 'scripts/guarded_pg09_canary.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize('values', [
    (82, 20000, 100, 225, 20000, 1000),
    (40, 6143, 100, 225, 20000, 1000),
    (40, 20000, 259, 225, 20000, 1000),
    (40, 20000, 100, 225, 4095, 1000),
    (40, 20000, 100, 225, 20000, 99),
    (float('nan'), 20000, 100, 225, 20000, 1000),
    (40, 20000, 100, 0, 20000, 1000),
])
def test_guard_rejects_resource_limit(values):
    assert guard.resource_reason(*values)


def test_guard_accepts_healthy_snapshot():
    assert guard.resource_reason(44, 46000, 21, 225, 50000, 1900) is None
