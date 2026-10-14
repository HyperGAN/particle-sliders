"""Opt-in hold04 stamp: default stays put, hold04 differs only in noise_hold_ratio."""
import hashlib
import json
import pytest

import particle_sliders
from particle_sliders import formulation_stamp, winning_formulation
from particle_sliders.formulation import (
    CURRENT_FORMULATION_ID,
    GMIX_1600_HOLD04_FORMULATION_ID,
    _thaw,
)

# sha256 of the default stamp card at a119ca1, before the opt-in stamp existed.
DEFAULT_CARD_SHA256 = "69f228c108d05ec4b9f7b33d56923e40070dafdb672137a66c2ce62ff2fdb187"


def _card_sha(stamp):
    card = {k: _thaw(stamp.spec[k]) for k in sorted(stamp.spec)}
    return hashlib.sha256(json.dumps(card, sort_keys=True, default=list).encode()).hexdigest()


def test_default_stamp_unchanged():
    stamp = winning_formulation()
    assert stamp.formulation_id == CURRENT_FORMULATION_ID == "particle-gmix-1600-v2"
    assert stamp.spec["noise_hold_ratio"] == 1.3
    assert _card_sha(stamp) == DEFAULT_CARD_SHA256
    assert formulation_stamp() is stamp
    assert formulation_stamp(CURRENT_FORMULATION_ID) is stamp


def test_hold04_differs_only_in_noise_hold_ratio():
    default = winning_formulation().as_dict()
    hold = formulation_stamp(GMIX_1600_HOLD04_FORMULATION_ID)
    assert hold.formulation_id == GMIX_1600_HOLD04_FORMULATION_ID
    diff = {k for k in set(default) | set(hold.as_dict()) if default.get(k) != hold.as_dict().get(k)}
    assert diff == {"noise_hold_ratio"}
    assert hold.spec["noise_hold_ratio"] == 0.4
    assert hold.spec["noise_decay_steps"] == default["noise_decay_steps"]


def test_hold04_noise_curve_matches_until_hold():
    default = winning_formulation()
    hold = formulation_stamp(GMIX_1600_HOLD04_FORMULATION_ID)
    assert hold.noise_std_at(0, 1.0) == default.noise_std_at(0, 1.0)
    assert hold.noise_std_at(1600, 1.0) == pytest.approx(0.4)
    assert default.noise_std_at(1600, 1.0) == pytest.approx(1.3)


def test_require_rejects_cross_stamp_card():
    hold = formulation_stamp(GMIX_1600_HOLD04_FORMULATION_ID)
    with pytest.raises(ValueError):
        winning_formulation().require(hold.as_dict())
    hold.require(hold.as_dict())


def test_unknown_stamp_raises():
    with pytest.raises(KeyError):
        formulation_stamp("nope")
    assert "formulation_stamp" in particle_sliders.__all__
