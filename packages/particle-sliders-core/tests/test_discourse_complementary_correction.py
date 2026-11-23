"""Complementary correction through the shipped endpoint teacher (arXiv:2610.41661)."""
import torch

from particle_sliders import SliderRecipe, teacher_poles


def test_teacher_removes_overlapping_nuisance_without_moving_the_pole_midpoint():
    midpoint = torch.tensor([[10., -3., 4.]])
    neutral = torch.tensor([[9., 1., 3.]])
    direction = torch.tensor([[1., 0., 0.]])
    # The nuisance shares a component with the intended slider. Removing the
    # raw nuisance vector would also subtract part of the intended edit.
    nuisance = torch.tensor([[2., 1., 0.]])
    edit = torch.tensor([[3., .75, .5]])

    plus, minus, mode = teacher_poles(
        SliderRecipe(), midpoint + edit, midpoint - edit, neutral,
        leak_dir=nuisance, slider_dir=direction,
    )

    assert mode == 'guard_admitted'
    expected_edit = torch.tensor([[3., 0., .5]])
    torch.testing.assert_close(plus, midpoint + expected_edit, rtol=0, atol=0)
    torch.testing.assert_close(minus, midpoint - expected_edit, rtol=0, atol=0)


def test_teacher_keeps_original_poles_when_correction_would_erase_the_contrast():
    midpoint = torch.tensor([[10., -3., 4.]])
    edit = torch.tensor([[.25, 3., 0.]])
    original_plus, original_minus = midpoint + edit, midpoint - edit

    plus, minus, mode = teacher_poles(
        SliderRecipe(), original_plus, original_minus, torch.zeros_like(midpoint),
        leak_dir=torch.tensor([[2., 1., 0.]]), slider_dir=torch.tensor([[1., 0., 0.]]),
    )

    assert mode == 'guard_fallback_raw_poles'
    assert plus is original_plus
    assert minus is original_minus
