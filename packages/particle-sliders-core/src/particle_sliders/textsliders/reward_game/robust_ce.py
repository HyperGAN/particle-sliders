"""Regression-sensitive direct CE loss and its scalar waveform pullback."""
import math


def value_and_weight(ce, off, margin=.1, temperature=.1):
    if not all(math.isfinite(x) for x in (ce, off)):
        raise ValueError('Nonfinite CE comparison')
    z = (margin-(ce-off))/temperature
    value = temperature*(max(z, 0.)+math.log1p(math.exp(-abs(z))))
    weight = 1./(1.+math.exp(-z)) if z >= 0 else math.exp(z)/(1.+math.exp(z))
    return value, weight
