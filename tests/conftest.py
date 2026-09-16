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
