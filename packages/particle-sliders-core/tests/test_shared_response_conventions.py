"""Products choose response conventions explicitly, independent of model names."""
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from particle_sliders.pg09_backends import FrozenResponseAdapter, FlowSchedule, ResponseRow


class Network(torch.nn.Module):
    names = ()

    @contextmanager
    def scaled(self, value):
        yield


class Response(FrozenResponseAdapter):
    # A name is metadata, never an implicit convention selector.
    family = 'future-product'

    def _velocity(self, condition, latent, timestep):
        return torch.full_like(latent, condition)


def row(**changes):
    return replace(ResponseRow(neutral=2., positive=5., negative=1.,
                               latent=torch.zeros(1, 2), timestep=torch.tensor(.5)), **changes)


def test_new_product_increasing_trajectory_and_direction_admission():
    increasing = Response(Network(), trajectory_direction='increasing')
    trajectory = row(timestep=None, schedule=FlowSchedule.increasing_linear(2))
    prepared = increasing.prepare(trajectory)
    torch.testing.assert_close(prepared.target, torch.full((1, 2), 5.))
    torch.testing.assert_close(increasing.generated(prepared), torch.full((1, 2), 2.))
    with pytest.raises(ValueError, match='negative Euler'):
        Response(Network()).prepare(trajectory)
    assert increasing.metadata()['trajectory_direction'] == 'increasing'


def test_teacher_convention_is_explicit_and_model_name_has_no_effect():
    positive = Response(Network())
    positive.family = 'moss'
    torch.testing.assert_close(positive.prepare(row()).target, torch.full((1, 2), 5.))
    with pytest.raises(ValueError, match='unipolar'):
        positive.prepare(row(scale=-1., strength=.5))
    signed = Response(Network(), teacher_convention='signed_difference')
    signed.family = 'supra-any-name'
    torch.testing.assert_close(signed.prepare(row(scale=-1., strength=.5)).target,
                               torch.zeros(1, 2))
    assert signed.metadata()['teacher_convention'] == 'signed_difference'
    with pytest.raises(ValueError, match='raw velocity'):
        signed.prepare(row(guidance=2.))


@pytest.mark.parametrize('options', [dict(trajectory_direction='guess'),
                                    dict(teacher_convention='moss')])
def test_unknown_convention_refused_at_binding(options):
    with pytest.raises(ValueError):
        Response(Network(), **options)
