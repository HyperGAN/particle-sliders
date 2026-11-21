"""Check hidden-basis changes through the shipped routed-distillation fit."""
import torch

from particle_sliders import fit_routed_down


def test_routed_distillation_preserves_heldout_readout_across_aligned_rotation():
    generator = torch.Generator().manual_seed(245)
    fit_rows = torch.randn(24, 3, generator=generator, dtype=torch.float64)
    heldout_rows = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    teacher_down = fit_rows.new_tensor([[1., .5, .25], [-.75, .5, .125]])
    targets = fit_rows @ teacher_down.T
    heldout_targets = heldout_rows @ teacher_down.T
    rotation = fit_rows.new_tensor([
        [3 ** .5 / 2, -.5, 0.], [.5, 3 ** .5 / 2, 0.], [0., 0., 1.],
    ])

    down, ridge = fit_routed_down(fit_rows, targets)
    aligned_down, aligned_ridge = fit_routed_down(fit_rows @ rotation, targets)
    torch.testing.assert_close(aligned_ridge, ridge, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(aligned_down, down @ rotation, rtol=1e-10, atol=1e-12)

    rotated_heldout = heldout_rows @ rotation
    aligned = rotated_heldout @ aligned_down.T
    torch.testing.assert_close(aligned, heldout_rows @ down.T, rtol=1e-10, atol=1e-12)
    aligned_mse = (aligned - heldout_targets).square().mean()
    assert aligned_mse < 1e-5
    # An unchanged matrix in the new basis is a real intervention mismatch even
    # though the coordinates have the same shape and the rotation is modest.
    stale_mse = (rotated_heldout @ down.T - heldout_targets).square().mean()
    assert stale_mse > .1
