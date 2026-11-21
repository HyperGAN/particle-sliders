"""Prompt Steering Replacement (PSR) intervention and activation-matching loss."""

from collections.abc import Sequence

import torch
from torch import nn


class PromptSteering(nn.Module):
    """One trainable PSR layer from Steer Like the LLM, Eqs. (7)-(8).

    Fit the affine ReLU probe and additive direction jointly on aligned pairs.
    One instance at the chosen layer gives S-PSR. A-PSR uses one instance at
    each layer and forwards each edited residual through the frozen backbone.
    """

    def __init__(self, hidden_size, *, device=None, dtype=torch.float32):
        super().__init__()
        self.probe = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
        self.direction = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)

    def coefficients(self, residual):
        """The position-specific λ(h) before multiplying by attribute strength."""
        return self.probe(residual.to(dtype=self.probe.weight.dtype)).squeeze(-1).relu()

    def forward(self, residual, strength, *, mask=None):
        """Return edited residual and masked λ, using scalar or per-example strength."""
        coefficients = self.coefficients(residual)
        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("PSR mask must be a Boolean tensor")
            valid = torch.broadcast_to(mask.to(residual.device), coefficients.shape)
            coefficients = coefficients.masked_fill(~valid, 0)
        scale = torch.as_tensor(strength, dtype=coefficients.dtype, device=residual.device)
        if scale.ndim:
            if residual.ndim != 3 or scale.shape != (residual.shape[0],):
                raise ValueError("per-example strength requires a [batch, position, hidden] residual")
            scale = scale[:, None]
        weighted = scale * coefficients
        edited = residual.to(dtype=self.probe.weight.dtype) + weighted[..., None] * self.direction.weight[0]
        return torch.where(weighted[..., None] == 0, residual, edited.to(residual.dtype)), coefficients


def psr_mse_loss(edited: Sequence[torch.Tensor], target: Sequence[torch.Tensor], mask):
    """Sum aligned per-layer MSE, averaging positions within each example first.

    Both streams must already be gathered to the same positions and padded to
    [batch, position, hidden]. Supply every measured layer, including downstream
    layers for S-PSR. Targets are detached prompt-steered reference activations.
    """
    if (not edited or len(edited) != len(target) or mask.ndim != 2 or mask.dtype != torch.bool
            or not mask.shape[0] or not mask.any(dim=1).all()):
        raise ValueError("provide paired layers and at least one valid aligned position per example")
    total = edited[0].new_zeros((), dtype=torch.float32)
    for prediction, reference in zip(edited, target):
        if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[:2] != mask.shape:
            raise ValueError("paired activations must match the [batch, position, hidden] alignment")
        valid = mask.to(device=prediction.device)
        dtype = torch.promote_types(prediction.dtype, reference.dtype)
        dtype = torch.promote_types(dtype, torch.float32)
        difference = prediction.to(dtype=dtype) - reference.detach().to(device=prediction.device, dtype=dtype)
        difference = difference.masked_fill(~valid[..., None], 0)
        per_position = difference.square().mean(dim=-1)
        loss = (per_position.sum(dim=1) / valid.sum(dim=1)).mean()
        total = total + loss.to(total.device)
    return total


def psr_location_penalty(coefficients):
    """Mean of max(0, 1 − sum_position λ) over examples (Section 3.5)."""
    if coefficients.ndim != 2 or not all(coefficients.shape):
        raise ValueError("expected masked PSR coefficients shaped [batch, position]")
    return (1 - coefficients.sum(dim=-1)).relu().mean()
