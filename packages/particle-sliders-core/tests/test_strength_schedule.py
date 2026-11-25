"""Strength scheduling uses native intervals and keeps CFG branches scoped."""

from contextlib import contextmanager
from fractions import Fraction
from math import inf, nan

import pytest
import torch

from particle_sliders.backends import FlowSchedule, FrozenResponseAdapter, ResponseRow
from particle_sliders.strength_schedule import (
    interval_dose,
    interval_strengths,
    match_binary32_interval_dose,
    match_binary64_interval_dose,
)


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


@pytest.mark.parametrize("coefficient_format", ["binary32", "binary64"])
def test_interval_dose_resolves_two_native_clocks_independently(coefficient_format):
    gate = (-1.0, 0.0, 0.0, 4.0)
    fixed = (0.0, -1.0, -1.0, 0.0)
    matching_grids = ([1, 0.5, 0.25, 0, 0], [0, 0.5, 0.625, 1, 1])
    checked = [*matching_grids, [1, 0.75, 0.5, 0, 0]]
    differences = [interval_dose(gate, grid, coefficient_format=coefficient_format)
                   - interval_dose(fixed, grid, coefficient_format=coefficient_format)
                   for grid in checked]
    assert differences == [0, 0, Fraction(1, 2)]


def test_interval_dose_requires_resolved_scalar_format():
    value = 1 + 2 ** -24
    assert interval_dose([value], [0, 1], coefficient_format="binary64") == 1 + Fraction(1, 2 ** 24)
    with pytest.raises(ValueError, match="resolved binary32"):
        interval_dose([value], [0, 1], coefficient_format="binary32")
    with pytest.raises(ValueError, match="format"):
        interval_dose([1], [0, 1], coefficient_format="float16")


@pytest.mark.parametrize("match", [match_binary32_interval_dose, match_binary64_interval_dose])
def test_dose_match_uses_signed_coefficients_and_native_interval_widths(match):
    result = match([-2.0, 0.0, 8.0], [1.0, 0.25, 0.0, 0.0])
    assert result.constant_strength == -1.5
    assert result.adaptive_dose == result.constant_dose == Fraction(-3, 2)


@pytest.mark.parametrize("match", [match_binary32_interval_dose, match_binary64_interval_dose])
def test_dose_match_uses_the_control_arms_own_observed_clock(match):
    result = match([1, 0, 0, 0, 0], range(6), constant_coordinates=[9, 9, 7, 5])
    assert result.constant_strength == 0.25
    assert result.adaptive_dose == result.constant_dose == 1


@pytest.mark.parametrize("match, residual", [
    (match_binary32_interval_dose, Fraction(1, 2**25)),
    (match_binary64_interval_dose, Fraction(-1, 2**54)),
])
def test_dose_match_does_not_label_a_nondyadic_native_control_exact(match, residual):
    result = match([1, 0, 0, 0, 0], range(6), constant_coordinates=range(4))
    assert result.adaptive_dose == 1
    assert result.constant_dose - result.adaptive_dose == residual
    assert result.constant_dose == 3 * Fraction(result.constant_strength)


@pytest.mark.parametrize("match, maximum", [
    (match_binary32_interval_dose, float.fromhex("0x1.fffffep127")),
    (match_binary64_interval_dose, float.fromhex("0x1.fffffffffffffp1023")),
])
def test_dose_match_returns_nearest_finite_coefficient_on_shorter_clock(match, maximum):
    result = match([-maximum], [0, 2], constant_coordinates=[0, 1])
    assert result.constant_strength == -maximum
    assert result.constant_dose == Fraction(-maximum)
    assert result.adaptive_dose == 2 * result.constant_dose


@pytest.mark.parametrize("match", [match_binary32_interval_dose, match_binary64_interval_dose])
@pytest.mark.parametrize("constant_coordinates", [[], [1], [1, 1], [0, inf]])
def test_dose_match_rejects_invalid_control_clocks(match, constant_coordinates):
    with pytest.raises(ValueError, match="constant control"):
        match([1], [0, 1], constant_coordinates=constant_coordinates)


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


def test_equal_passes_can_match_binary64_even_when_binary32_rounds():
    next_binary32 = 1.0 + 2**-23
    result = match_binary64_interval_dose([1.0, next_binary32], [0.0, 1.0, 2.0])
    assert result.constant_strength == 1.0 + 2**-24
    assert result.adaptive_dose == result.constant_dose == 2 + Fraction(1, 2**23)


@pytest.mark.parametrize("sign", [-1, 1])
def test_dose_match_rounds_exact_binary64_midpoint_to_even(sign):
    next_binary64 = 1.0 + 2**-52
    result = match_binary64_interval_dose([sign, sign * next_binary64], [2.0, 1.0, 0.0])
    assert result.constant_strength == sign
    assert result.adaptive_dose - result.constant_dose == Fraction(sign, 2**52)


def test_dose_match_uses_exact_native_widths_to_resolve_a_near_midpoint():
    next_binary64 = 1.0 + 2**-52
    result = match_binary64_interval_dose([next_binary64, 1.0, next_binary64], [-2**-60, 0, 1, 2])
    assert result.constant_strength == next_binary64
    assert result.adaptive_dose < result.constant_dose


@pytest.mark.parametrize("match", [match_binary32_interval_dose, match_binary64_interval_dose])
@pytest.mark.parametrize("strengths, coordinates", [
    ([], [1.0]), ([1.0], [1.0]), ([1.0], [0.0, 0.0]),
    ([1.0], [0.0, inf]), ([nan], [0.0, 1.0]),
])
def test_dose_match_rejects_invalid_native_clock_or_coefficients(match, strengths, coordinates):
    with pytest.raises(ValueError):
        match(strengths, coordinates)


@pytest.mark.parametrize("strength", [0.1, 1e300])
def test_binary32_dose_match_rejects_unresolved_native_coefficients(strength):
    with pytest.raises(ValueError):
        match_binary32_interval_dose([strength], [0.0, 1.0])


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
