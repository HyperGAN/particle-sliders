"""Strength scheduling uses native intervals and keeps CFG branches scoped."""

from contextlib import contextmanager
from fractions import Fraction
from math import inf, nan

import pytest
import torch

from particle_sliders.backends import FlowSchedule, FrozenResponseAdapter, ResponseRow
from particle_sliders.strength_schedule import interval_strengths, match_binary32_interval_dose


@pytest.mark.parametrize("strength", [-2.0, 0.0, 1.0])
@pytest.mark.parametrize("profile", ["constant", "early", "late"])
def test_nonuniform_intervals_preserve_signed_nominal_dose(strength, profile):
    widths = (0.4, 0.2, 0.4)
    factors = {"constant": (1.0, 1.0, 1.0), "early": (1.5, 1.0, 0.5),
               "late": (0.5, 1.0, 1.5)}
    actual = interval_strengths(strength, [-value for value in widths], profile=profile)
    assert actual == pytest.approx([strength * value for value in factors[profile]])
    assert sum(width * abs(value) for width, value in zip(widths, actual)) == pytest.approx(abs(strength))


@pytest.mark.parametrize("strength, deltas, profile", [
    (1.0, [], "early"), (1.0, [0.0, 1.0], "early"), (1.0, [inf], "early"),
    (nan, [1.0], "early"), (1.0, [1.0], "unknown"), (1e308, [1e308] * 2, "late"),
    (1.7e308, [0.25, 0.75], "early"),
])
def test_invalid_requests_fail_before_sampling(strength, deltas, profile):
    with pytest.raises(ValueError):
        interval_strengths(strength, deltas, profile=profile)


def test_one_interval_uses_nominal_strength():
    assert interval_strengths(1.0, [5e-324], profile="early") == (1.0,)


def test_dose_match_uses_signed_coefficients_and_native_interval_widths():
    result = match_binary32_interval_dose([-2.0, 0.0, 8.0], [1.0, 0.25, 0.0, 0.0])
    assert result.constant_strength == -1.5
    assert result.adaptive_dose == result.constant_dose == Fraction(-3, 2)


@pytest.mark.parametrize("sign", [-1, 1])
def test_dose_match_rounds_exact_binary32_midpoint_to_even(sign):
    next_value = 1.0 + 2**-23
    result = match_binary32_interval_dose([sign, sign * next_value], [2.0, 1.0, 0.0])
    assert result.constant_strength == sign
    assert result.adaptive_dose - result.constant_dose == Fraction(sign, 2**23)


def test_dose_match_avoids_double_rounding_near_binary32_midpoint():
    next_value = 1.0 + 2**-23
    result = match_binary32_interval_dose([next_value, 1.0, next_value], [-2**-60, 0, 1, 2])
    assert result.constant_strength == next_value
    assert result.adaptive_dose < result.constant_dose


@pytest.mark.parametrize("strengths, coordinates", [
    ([], [1.0]), ([1.0], [1.0]), ([1.0], [0.0, 0.0]),
    ([1.0], [0.0, inf]), ([nan], [0.0, 1.0]), ([0.1], [0.0, 1.0]),
    ([1e300], [0.0, 1.0]),
])
def test_dose_match_rejects_invalid_native_clock_or_coefficients(strengths, coordinates):
    with pytest.raises(ValueError):
        match_binary32_interval_dose(strengths, coordinates)


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 7.0

    @contextmanager
    def scaled(self, scale):
        previous = self.scale
        self.scale = scale
        try:
            yield
        finally:
            self.scale = previous


class TestResponse(FrozenResponseAdapter):
    __test__ = False

    def __init__(self, network):
        super().__init__(network, decoder=lambda latent: latent)
        self.calls = []

    def _velocity(self, condition, latent, timestep):
        self.calls.append((condition, self.network.scale))
        return latent + self.network.scale


def test_render_scopes_each_strength_to_the_conditional_forward():
    network = Adapter()
    response = TestResponse(network)
    row = ResponseRow(neutral="conditional", positive="teacher", negative="unconditional",
                      latent=torch.zeros(1, 1), schedule=FlowSchedule((1.0, 0.5), (-0.5, -0.5)),
                      guidance=2.0)
    response.render(row, scale=1.0, strength_profile="early")
    assert response.calls == [("conditional", 1.5), ("unconditional", 0.0),
                              ("conditional", 0.5), ("unconditional", 0.0)]
    assert network.scale == 7.0

    response.calls.clear()
    response.render(row)
    assert response.calls == [("conditional", 1.0), ("unconditional", 0.0)] * 2

    response.calls.clear()
    response.render(row, scale=0.0, strength_profile="late")
    assert response.calls == [("conditional", 0.0), ("unconditional", 0.0)] * 2
    assert network.scale == 7.0
