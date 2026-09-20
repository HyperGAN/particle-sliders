"""Studio strength controls; the pinned training/runtime contract stays intact."""
import math

from .contracts import normalized_strengths

MAX_ENERGY = 10.


def inference_strengths(mix, energy):
    if not math.isfinite(energy) or not 0 <= energy <= MAX_ENERGY:
        raise ValueError(f"Energy must be finite and in [0, {MAX_ENERGY:g}]")
    # Preserve the established arithmetic exactly throughout its original range.
    if energy <= 1:
        return normalized_strengths(mix, energy)
    return {name: value * energy for name, value in normalized_strengths(mix, 1.).items()}


def set_inference_mix(mixer, mix, energy):
    strengths = inference_strengths(mix, energy)
    # Retain the mixer's check that every active branch has a loaded checkpoint.
    mixer.set_mix(mix, min(energy, 1.))
    mixer.strengths = strengths
    return strengths
