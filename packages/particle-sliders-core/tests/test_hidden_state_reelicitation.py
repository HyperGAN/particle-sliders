"""Preserve native prefix-conditioned states under a matched AR continuation."""
import torch

from particle_sliders.audio import AudioRow, audio_conditions
from test_cut_points import NativeLastTokenModel, NativeResponse, NativeSlider


class CausalPrefixModel(NativeLastTokenModel):
    """CPU fixture with a known, causal dependence on the full token prefix."""

    def forward(self, ids):
        return super().forward(ids).cumsum(dim=1)


def test_same_continuation_keeps_distinct_native_prefix_conditioning():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        model = CausalPrefixModel()
        slider = NativeSlider(model)
    response = NativeResponse(model, slider, model_identity={'fixture': 'causal-prefix'})
    neutral_tokens = [1, 2, 3]
    positive_tokens = [4, 2, 3]
    with torch.no_grad():
        neutral = model(torch.tensor([neutral_tokens]))[:, -1]
        target = model(torch.tensor([positive_tokens]))[:, -1]
    assert not torch.allclose(neutral, target)

    prepared = response.prepare(AudioRow(neutral_tokens, positive_tokens, history_tokens=2))
    neutral_tokens[0] = positive_tokens[0]
    torch.testing.assert_close(prepared.target, target, rtol=0, atol=0)
    torch.testing.assert_close(prepared.neutral, neutral, rtol=0, atol=0)
    torch.testing.assert_close(response.sample(prepared, scale=0.), neutral, rtol=0, atol=0)
    condition = audio_conditions([prepared])
    expected = torch.cat((neutral, torch.tensor([[3., 3., 2., 1.]])), dim=1)
    torch.testing.assert_close(condition, expected, rtol=0, atol=0)
