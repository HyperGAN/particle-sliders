"""Resolve adapter strengths over a product's native trajectory intervals."""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from struct import Struct


_FLOAT32 = Struct(">f")
_UINT32 = Struct(">I")


@dataclass(frozen=True)
class IntervalDoseMatch:
    """Same-direction control and exact coefficient doses on the native clock."""

    constant_strength: float
    adaptive_dose: Fraction
    constant_dose: Fraction


def _finite_intervals(strengths, coordinates):
    strengths = tuple(map(float, strengths))
    coordinates = tuple(map(float, coordinates))
    if not strengths or len(coordinates) != len(strengths) + 1:
        raise ValueError("dose matching requires one coefficient per native interval")
    if any(not isfinite(x) for x in (*strengths, *coordinates)):
        raise ValueError("dose matching requires finite coefficients and coordinates")
    return strengths, coordinates


def _exact_interval_dose(strengths, coordinates):
    edges = tuple(map(Fraction, coordinates))
    widths = tuple(abs(end - start) for start, end in zip(edges, edges[1:]))
    total_width = sum(widths)
    if not total_width:
        raise ValueError("dose matching requires a nonzero native trajectory width")
    dose = sum(width * Fraction(s) for width, s in zip(widths, strengths, strict=True))
    return dose, total_width


def _require_binary32_coefficients(strengths):
    try:
        exact_binary32 = all(_FLOAT32.unpack(_FLOAT32.pack(s))[0] == s for s in strengths)
    except OverflowError:
        exact_binary32 = False
    if not exact_binary32:
        raise ValueError("dose matching requires resolved binary32 coefficients")


def interval_dose(strengths, coordinates, *, coefficient_format):
    """Return the exact signed coefficient dose on one native clock.

    Use coefficients already resolved at the product's scalar application
    boundary and endpoints in execution order. Choose `binary32` to reject
    coefficients that would round in that format, or `binary64` for Python
    floats. Zero-width intervals contribute zero. For coupled clocks,
    compare the two schedules independently on each clock.
    """
    if coefficient_format not in ("binary32", "binary64"):
        raise ValueError("coefficient format must be binary32 or binary64")
    strengths, coordinates = _finite_intervals(strengths, coordinates)
    if coefficient_format == "binary32":
        _require_binary32_coefficients(strengths)
    return _exact_interval_dose(strengths, coordinates)[0]


def match_binary32_interval_dose(strengths, coordinates):
    """Match a fixed-direction edit with the nearest binary32 constant.

    Provide one resolved binary32 coefficient per interval and its native
    endpoint coordinates in execution order. Zero-width intervals carry no
    dose. Return exact signed coefficient-space doses, with ties rounded to
    even for the constant. Measure any subsequent tensor-rounding effects in
    the native product separately.
    """
    strengths, coordinates = _finite_intervals(strengths, coordinates)
    _require_binary32_coefficients(strengths)

    dose, total_width = _exact_interval_dose(strengths, coordinates)
    ideal = dose / total_width
    rounded = _UINT32.unpack(_FLOAT32.pack(float(ideal)))[0]

    # A binary64 intermediate can land exactly on a binary32 midpoint even
    # when the exact mean is just to one side. Compare the adjacent values in
    # rational space. The low significand bit selects even at an exact tie.
    candidates = [(b, _FLOAT32.unpack(_UINT32.pack(b))[0])
                  for b in range(max(0, rounded - 1), min(0xFFFFFFFF, rounded + 1) + 1)]
    _, constant = min(((b, value) for b, value in candidates if isfinite(value)),
                      key=lambda item: (abs(Fraction(item[1]) - ideal), item[0] & 1))
    return IntervalDoseMatch(constant, dose, total_width * Fraction(constant))


def match_binary64_interval_dose(strengths, coordinates):
    """Match a fixed-direction edit with the nearest binary64 constant.

    Return exact signed doses for the adaptive coefficients and the constant
    on the supplied native clock. Compare the two doses when a caller requires
    equality at the coefficient boundary.
    """
    strengths, coordinates = _finite_intervals(strengths, coordinates)
    dose, total_width = _exact_interval_dose(strengths, coordinates)
    constant = float(dose / total_width)
    return IntervalDoseMatch(constant, dose, total_width * Fraction(constant))


def interval_strengths(strength, deltas, *, profile="constant"):
    """Return one signed adapter strength per integration interval.

    Supply intervals in execution order. Early and late emphasize opposite
    halves of the total traveled distance with nominal dose equal to constant.
    Products own interval selection and permitted effective strengths.
    """
    if profile not in ("constant", "early", "late"):
        raise ValueError("strength profile must be constant, early, or late")
    strength = float(strength)
    if not isfinite(strength):
        raise ValueError("strength must be finite")
    widths = tuple(abs(float(delta)) for delta in deltas)
    if not widths or any(not isfinite(width) or width <= 0 for width in widths):
        raise ValueError("strength profiles require finite, nonzero intervals")
    total = sum(widths)
    if not isfinite(total):
        raise ValueError("total interval width must be finite")
    if profile == "constant" or strength == 0 or len(widths) == 1:
        return (strength,) * len(widths)

    remaining = total / 2
    resolved = []
    for width in widths:
        fraction = min(width, remaining) / width
        early = 0.5 + fraction
        gain = early if profile == "early" else 2.0 - early
        resolved.append(strength * gain)
        remaining = max(0.0, remaining - width)
    if any(not isfinite(value) for value in resolved):
        raise ValueError("effective strength must be finite")
    return tuple(resolved)
