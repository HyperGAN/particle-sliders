"""Fit a routed bottleneck by a homogeneous linear map; retain its up matrix."""
import math

import torch


@torch.no_grad()
def fit_routed_down(x, routed_targets, *, ridge_fraction=0.01):
    """Return (down, ridge) with down shaped [rank, input_width].

    ``x`` is [calibration_rows, input_width] and ``routed_targets`` is
    [calibration_rows, rank], measured before the teacher's up projection.
    The consumer retains that up projection unchanged. This is the Anima v1
    dual ridge solve, including its operation order and scale floor. It uses
    O(calibration_rows**2) memory; collecting suitable activations and choosing
    a calibration budget are responsibilities of the model integration.
    """
    if x.ndim != 2 or routed_targets.ndim != 2:
        raise ValueError("Expected two matrices")
    if x.shape[0] != routed_targets.shape[0] or min(*x.shape, routed_targets.shape[1]) < 1:
        raise ValueError("Expected nonempty matrices with matching calibration rows")
    if x.device != routed_targets.device or x.dtype != routed_targets.dtype:
        raise ValueError("Inputs must share dtype and device")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("Fit in float32 or float64")
    if not math.isfinite(ridge_fraction) or ridge_fraction <= 0:
        raise ValueError("ridge_fraction must be finite and positive")
    if not torch.isfinite(x).all() or not torch.isfinite(routed_targets).all():
        raise ValueError("Calibration values must be finite")
    gram = x @ x.T
    ridge = ridge_fraction * gram.diag().mean().clamp_min(1e-8)
    gram.diagonal().add_(ridge)
    down = torch.linalg.solve(gram, routed_targets).T @ x
    if not torch.isfinite(down).all():
        raise FloatingPointError("Non-finite fitted projection")
    return down, ridge
