"""Cut-point conformance on native-shaped CPU fixtures, never model qualification."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.fibo_native import FiboEditResponseAdapter, FiboParticleSlider
from particle_sliders.native_lora import lora_target_plan
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter, AudioRow
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.routed_projection import TinyParticleProjection


@pytest.fixture(autouse=True)
def bounded_cpu():
    before = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(20261031)
        yield
    torch.set_num_threads(before)


def projection_fixture():
    host = nn.Linear(6, 5).eval().requires_grad_(False)
    original = host.forward
    particles = nn.Parameter(torch.randn(128, 4))
    projection = TinyParticleProjection('attention.to_q', host, particles, rank=3, alpha=6.)
    # Zero-up initialization intentionally blocks upstream branch gradients.
    # Warm the up matrix directly; this is a gradient probe, not training.
    with torch.no_grad():
        projection.lora_up.weight.normal_(std=.1)
    return host, original, particles, projection


@pytest.mark.parametrize('shape', [(6,), (2, 6), (1, 3, 6), (1, 2, 3, 6)])
def test_projection_preserves_leading_native_axes_and_one_delta_scale(shape):
    host, original, _, projection = projection_fixture()
    x = torch.randn(shape)
    projection.multiplier = 1.
    base, full = original(x), host(x)
    projection.multiplier = .5
    half = host(x)
    assert half.shape == full.shape == (*shape[:-1], 5)
    torch.testing.assert_close(half - base, .5 * (full - base), rtol=1e-5, atol=1e-7)
    # The bridge operates independently at each leading coordinate. This does
    # not admit an arbitrary shaped response into the GAN's native-domain gate.
    torch.testing.assert_close(half, host(x.reshape(-1, 6)).reshape(half.shape))


def test_projection_zero_bypasses_branch_even_with_nonzero_adapter(monkeypatch):
    host, original, _, projection = projection_fixture()
    x = torch.randn(1, 4, 6)

    def forbidden(*args):
        raise AssertionError('zero must not execute the routed branch')

    monkeypatch.setattr(projection.bridge, 'forward', forbidden)
    projection.multiplier = 0.
    assert torch.equal(host(x), original(x))


def test_projection_cannot_broadcast_a_wrong_feature_width():
    host, _, _, projection = projection_fixture()
    projection.multiplier = .5
    with pytest.raises(RuntimeError, match='shapes cannot be multiplied'):
        host(torch.randn(1, 2, 7))


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(6, 6)
        self.to_v = nn.Linear(6, 6)


class NativeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionBlock()


def test_target_plan_is_read_only_and_deduplicates_overlapping_class_selection():
    model = NativeBackbone()
    before = deepcopy(model.state_dict())
    forwards = [module.forward for module in model.modules()]
    rng = torch.get_rng_state().clone()
    plan = lora_target_plan(model, ['NativeBackbone', 'AttentionBlock'],
                            prefix='lora_unet', delimiter='-')
    assert [name for name, _ in plan] == ['lora_unet-attention-to_q', 'lora_unet-attention-to_v']
    assert len({id(module) for _, module in plan}) == 2
    assert forwards == [module.forward for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


class NativeLastTokenModel(nn.Module):
    """Readout is after final norm; injection is an earlier Linear output."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)
        self.embed = nn.Embedding(16, 32)
        self.projection = nn.Linear(32, 32)
        self.final_norm = nn.LayerNorm(32)
        self.eval().requires_grad_(False)

    def forward(self, ids):
        return self.final_norm(torch.tanh(self.projection(self.embed(ids))))


class NativeSlider(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.projection = TinyParticleProjection('projection', model.projection,
                                                self.particles, rank=8, alpha=8.)

    @contextmanager
    def scaled(self, scale):
        previous = self.projection.multiplier
        self.projection.multiplier = scale
        try:
            yield
        finally:
            self.projection.multiplier = previous


class NativeResponse(ARLastHiddenResponseAdapter):
    def _hidden(self, ids):
        return self.model(torch.tensor([ids]))[:, -1]


def response_fixture():
    model = NativeLastTokenModel()
    slider = NativeSlider(model)
    response = NativeResponse(model, slider, model_identity={'fixture': 'cut-point-native-last-token'})
    return model, slider, response


def test_readout_and_injection_need_not_be_the_same_tensor():
    model, slider, response = response_fixture()
    prepared = response.prepare(AudioRow((1, 2), (1, 3)))
    assert prepared.target.shape == prepared.neutral.shape == (1, 32)
    assert not prepared.target.requires_grad and not prepared.neutral.requires_grad
    with slider.scaled(0.), torch.no_grad():
        assert torch.equal(prepared.target, model(torch.tensor([[1, 3]]))[:, -1])
        assert torch.equal(prepared.neutral, model(torch.tensor([[1, 2]]))[:, -1])
    assert torch.equal(response.sample(prepared, scale=0.), prepared.neutral)


def test_native_readout_keeps_branch_gradients_and_frozen_host_values():
    model, slider, response = response_fixture()
    with torch.no_grad():
        slider.projection.lora_up.weight.normal_(std=.1)
    before = deepcopy(model.state_dict())
    prepared = response.prepare(AudioRow((1, 2), (1, 3)))
    generated = response.generated(prepared)
    (generated * torch.arange(32)).sum().backward()
    # Every adapter leaf, including the single shared cloud, participates after
    # warm-up. No optimizer or GAN objective is introduced by this probe.
    assert len({id(p) for p in response.network.parameters()}) == len(list(slider.parameters()))
    assert response.network.particles is slider.particles
    for parameter in slider.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize('row', [
    AudioRow((1, 2), (1, 3), history_tokens=1),
    AudioRow((1, 2), (1, 3), history_tokens=2),
])
def test_ar_history_must_match_and_leave_a_prompt_prefix(row):
    _, _, response = response_fixture()
    with pytest.raises(ValueError, match='history'):
        response.prepare(row)


class FiboAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = [nn.Linear(4, 4) for _ in range(3)]
        self.to_out = nn.ModuleList([nn.Linear(4, 4)])


class FiboNativeShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_layers=1, n_refiner_layers=0,
            injection_layer_ids=[], cap_feat_dim=4, in_channels=4, patch_size=2)
        block = nn.Module()
        block.attention = FiboAttention()
        self.layers = nn.ModuleList([block])
        self.noise_refiner = nn.ModuleList()

    def forward(self, *, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask, context_latents, return_dict):
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(1, encoder_hidden_states.shape[2], dtype=torch.bool)
        context = (encoder_hidden_states * mask[:, None, :, None]).sum((1, 2)) / mask.sum()
        return (hidden_states + context[:, :, None, None],)


def fibo_shape_fixture():
    class Pipeline:
        def __init__(self):
            self.transformer = FiboNativeShape()
            self.scheduler = SimpleNamespace(fixture=True)

        def __call__(self, **kwargs):
            raise RuntimeError('native capture fixture failure')

    pipeline = Pipeline()
    slider = FiboParticleSlider(pipeline.transformer)
    response = FiboEditResponseAdapter(pipeline, slider,
        model_identity={'qualification': 'CPU fixture only', 'fixture': 'cut-point-mask'})
    condition = dict(embeds=torch.randn(1, 1, 3, 4),
                     mask=torch.tensor([[True, True, False]]), context_0=torch.zeros(1, 4, 2, 2))
    return pipeline, response, condition


@pytest.mark.parametrize('mask', [torch.ones(1, 3), torch.ones(3, dtype=torch.bool),
                                  torch.zeros(1, 3, dtype=torch.bool)])
def test_fibo_padding_mask_rejects_dtype_shape_and_all_padding(mask):
    _, response, condition = fibo_shape_fixture()
    condition['mask'] = mask
    with pytest.raises(ValueError, match='boolean padding mask'):
        response.prepare(ResponseRow(condition, condition, condition,
            torch.zeros(1, 4, 2, 2), timestep=torch.tensor(500.)))


def test_fibo_valid_token_mask_is_not_an_adapter_or_loss_gate():
    _, response, condition = fibo_shape_fixture()
    altered = deepcopy(condition)
    altered['embeds'][:, :, -1] = 1e6  # masked padding cannot change readout
    x, time = torch.zeros(1, 4, 2, 2), torch.tensor(500.)
    first = response.velocity(condition, x, time, scale=0.)
    assert torch.equal(first, response.velocity(altered, x, time, scale=0.))
    altered['mask'] = None  # None explicitly means no padding mask
    assert not torch.equal(first, response.velocity(altered, x, time, scale=0.))


def test_fibo_capture_failure_removes_only_owned_temporary_hooks():
    pipeline, response, _ = fibo_shape_fixture()
    foreign = pipeline.transformer.register_forward_pre_hook(lambda module, args: None)
    hooks = dict(pipeline.transformer._forward_pre_hooks)
    scheduler = pipeline.scheduler
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='native capture fixture failure'):
                response.capture_pair('neutral', 'positive', [object()])
            assert dict(pipeline.transformer._forward_pre_hooks) == hooks
            assert pipeline.scheduler is scheduler
    finally:
        foreign.remove()
