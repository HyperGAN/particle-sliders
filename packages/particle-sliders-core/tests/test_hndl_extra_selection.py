"""The two HNDL profiles have different supported environments, not interchangeable pins."""
from importlib.metadata import requires, version

import pytest
from torch import nn


def test_legacy_and_modern_extras_keep_exact_separate_dependencies():
    declared = requires('particle-sliders-core')
    assert any('hndl==0.8.2' in item and 'extra == "hndl"' in item for item in declared)
    assert any('hndl==0.10.0' in item and 'extra == "hypergan-hndl"' in item for item in declared)
    assert any('5f368a3770cf5bdae0448995c9f76e29af8049f7' in item
               and 'extra == "hypergan-hndl"' in item for item in declared)


def test_legacy_readout_loudly_refuses_modern_hndl_environment():
    pytest.importorskip('hndl')
    if version('hndl') != '0.10.0':
        pytest.skip('this refusal exercises the separate modern HNDL environment')
    from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic
    with pytest.raises(RuntimeError, match='requires hndl==0.8.2'):
        HNDLConditionalReadoutCritic(8, 2)
