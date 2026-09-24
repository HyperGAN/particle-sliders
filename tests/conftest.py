import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_global_rng():
    """Isolate global RNG state per test so order cannot change results.

    Library fits already run under ``analysis.slider2d.rng.isolated_rng``;
    this fixture covers the remaining direct ``torch.randn`` / ``random``
    draws in test bodies: each test starts from a fixed seed and whatever
    it consumes is restored afterwards.
    """
    torch_state = torch.get_rng_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(0)
        random.seed(0)
        yield
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(py_state)


# Some tests exercise the shared music workstation: the parent workspace's
# ``app`` package, its pinned Music 3 Diffusers build, and campaign files
# generated there and never committed. Off the workstation (no ``app``) those
# tests skip; on it they run and fail as usual.
ON_WORKSTATION = importlib.util.find_spec('app') is not None
WORKSTATION_MODULES = {'app', 'diffusers'}
WORKSTATION_FILES = {
    ROOT / 'analysis/reward_slider_game_20260908/benchmark.json',
}


def _workstation_only(exc):
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        top = exc.name.split('.')[0]
        if top in WORKSTATION_MODULES and importlib.util.find_spec(top) is None:
            return f'needs {top!r} from the music workstation'
    if isinstance(exc, FileNotFoundError) and exc.filename:
        path = Path(exc.filename)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.resolve() in WORKSTATION_FILES:
            return f'needs local campaign file {path.relative_to(ROOT)}'
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if ON_WORKSTATION or call.excinfo is None or call.when == 'teardown':
        return
    reason = _workstation_only(call.excinfo.value)
    if reason:
        report.outcome = 'skipped'
        report.longrepr = (item.location[0], item.location[1] or 0, f'Skipped: {reason}')
