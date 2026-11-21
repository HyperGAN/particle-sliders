"""Resolve adapter strengths over a product's native trajectory intervals."""

from math import isfinite


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
