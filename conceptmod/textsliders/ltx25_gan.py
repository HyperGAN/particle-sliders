"""Opt-in experimental LTX video-field GAN; historical UNI remains independent.

The shared package owns the game/session/checkpoints. This module owns native
packing and an identity-bound HyperGAN/HNDL critic, never GAN mathematics.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from particle_sliders.fibo_hndl import verified_builder
from particle_sliders.pg094 import ConditionalPG094, PG_SOURCE_PROFILE
from particle_sliders.lifecycle import Binding, SessionConfig
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg09 import _hash
from conceptmod.textsliders.ltx25_backend import (
    LTX25Backend, DEFAULT_MODEL, embed_lora_hosts, ltx_pack_feature_dim,
)
from conceptmod.textsliders.ltx25_uni import (
    RECIPE_EMBED, resolve_concept_token_ids, require_concept_tokens, unused_hold_mask,
)

COHORT = 'ltx25-video-field-gan-hg-hndl-v1-preview'
BACKEND = 'ltx25-modern-video-gan'
OBJECTIVE = 'frozen-plus-video-velocity;student-neutral;frozen-neutral-audio'
NATIVE_SOURCES = {
    'ltx25_backend.py': '863fdf0aa352507f7cf139c0f70e0df53949452b2714a7cecef86c8b755088d2',
    'ltx25_uni.py': '77bb72d84d5af0e220241639e910ebeabbbe95ef582ac8a05c68475f48b0e61d',
    'slider_targets.py': '5db3a2cdc9635a0f15f6bc59c80f00700ef04d35241650b935b373f610a85412',
}


def native_sources():
    actual = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in NATIVE_SOURCES}
    if actual != NATIVE_SOURCES:
        raise ValueError('native helper source cohort differs; preserve the original environment')
    return actual

# Spatial HNDL graph runs framewise INSIDE the critic. The game and penalty
# always receive the genuine [B,C,T,H,W] field, including the temporal axis.
CRITIC_GRAPH = '''field = conv(response, $width, kernel_size=1, name="field0")
field = silu(field)
field = conv(field, $width, kernel_size=3, padding=1, name="field2")
context = linear(condition, $double_width, name="condition")
gain, bias = split(context, $width)
ones = constant(gain, $width, value=1.0)
gain = add(gain, ones)
gain = reshape(gain, $width, 1, 1)
bias = reshape(bias, $width, 1, 1)
field = broadcast_mul(field, gain)
field = broadcast_add(field, bias)
field = silu(field)
field = mean(field, 3)
field = mean(field, 2)
score = linear(field, $width, name="score0")
score = silu(score)
linear(score, 1, name="score2")'''


class VideoFieldCritic(nn.Module):
    score_bound = 0

    def __init__(self, channels, condition_width, geometry, width=8):
        super().__init__()
        builder, identity = verified_builder()
        self.shape = (channels, *geometry)
        self.graph = builder(CRITIC_GRAPH,
            input_shape={'response': ['B', channels, *geometry[1:]], 'condition': ['B', condition_width]},
            output_shape=['B', 1], parameters={'width': width, 'double_width': 2 * width})
        self.identity = dict(identity, product_graph_sha256=hashlib.sha256(CRITIC_GRAPH.encode()).hexdigest(),
                             response_shape=list(self.shape), condition_width=condition_width, width=width,
                             temporal_readout='framewise-spatial-score-mean;no-temporal-convolution')

    def forward(self, response, condition):
        if tuple(response.shape[1:]) != self.shape:
            raise ValueError('native video field geometry differs')
        b, c, t, h, w = response.shape
        frames = response.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        context = condition[:, None, :].expand(-1, t, -1).reshape(b * t, -1)
        return self.graph(response=frames, condition=context).reshape(b, t).mean(1)

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)


class NativeLeaves(nn.Module):
    """Own the original adapter leaves, restoring native scale after every call."""
    def __init__(self, native):
        super().__init__()
        self.adapter = native.network

    @contextmanager
    def scaled(self, scale):
        old = self.adapter.lora_scale
        multipliers = [leaf.multiplier for leaf in self.adapter.loras]
        try:
            self.adapter.set_lora_slider(float(scale))
            yield
        finally:
            self.adapter.set_lora_slider(old)
            for leaf, value in zip(self.adapter.loras, multipliers):
                leaf.multiplier = value


@dataclass(frozen=True)
class VideoRow:
    neutral: str
    positive: str
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    scale: float = 1.


@dataclass(frozen=True)
class PreparedVideo:
    row: VideoRow
    target: torch.Tensor
    neutral: torch.Tensor
    audio_condition: torch.Tensor
    audio_mask: torch.Tensor


class VideoResponse:
    def __init__(self, native, geometry, model_identity):
        self.backend = native
        self.geometry = tuple(geometry)
        self.model_identity = deepcopy(model_identity)
        self.network = NativeLeaves(native)
        # Rollback is product-owned; no host module registrations are changed.
        self._hosts = embed_lora_hosts(native.encoder, native.connectors, last_n=native.te_last_n)

    def metadata(self):
        return dict(cohort=COHORT, objective=OBJECTIVE, geometry=self.geometry,
                    model=self.model_identity, audio='frozen-neutral-conditioning-and-latents',
                    targets=self.backend.lora_module_names(), recipe_host=RECIPE_EMBED)

    def _pack(self, prompt, row, scale, *, teacher=False):
        native = self.backend
        with self.network.scaled(scale), native.network:
            text = native.encode_text(prompt, frozen=(scale == 0))
            hold = {}
            if teacher:
                neutral = native.encode_text(row.neutral, frozen=True)
                concept = resolve_concept_token_ids(native.tokenizer, row.positive, row.neutral, '')
                require_concept_tokens(text.token_ids, concept)
                hold = dict(hold_neu=neutral, hold_mask=unused_hold_mask(
                    text.token_ids, [], concept, hold_mode='non_concept'))
            t, h, w = self.geometry
            return native.pack_t2v(text, video_latents=row.video_latents,
                audio_latents=row.audio_latents, num_frames=(t - 1) * 8 + 1,
                height=h * 32, width=w * 32, **hold)

    def _field(self, packed):
        # Native LTX packing order is [B,T*H*W,C] with unit patch sizes.
        # The exact T/H/W and full feature width are validated before execution.
        with self.network.scaled(0.):
            sample = self.backend.forward_velocity(packed, scale=0.).sample
        t, h, w = self.geometry
        channels = ltx_pack_feature_dim(self.backend.transformer, kind='video')
        if tuple(sample.shape) != (1, t * h * w, channels):
            raise ValueError('LTX output does not match declared native latent packing')
        return sample.reshape(1, t, h, w, channels).permute(0, 4, 1, 2, 3).contiguous().float()

    def prepare(self, row):
        with torch.no_grad():
            neutral = self._pack(row.neutral, row, 0.)
            positive = self._pack(row.positive, row, 0., teacher=True)
            # No audio concept objective: teacher and student share frozen neu audio.
            positive = replace(positive, audio_encoder_hidden_states=neutral.audio_encoder_hidden_states,
                               audio_encoder_attention_mask=neutral.audio_encoder_attention_mask)
            target, base = self._field(positive), self._field(neutral)
        return PreparedVideo(row, target.detach(), base.detach(),
                             neutral.audio_encoder_hidden_states.detach(), neutral.audio_encoder_attention_mask.detach())

    def generated(self, prepared):
        with self.network.scaled(prepared.row.scale):
            packed = self._pack(prepared.row.neutral, prepared.row, prepared.row.scale)
            packed = replace(packed, audio_encoder_hidden_states=prepared.audio_condition,
                             audio_encoder_attention_mask=prepared.audio_mask)
            return self._field(packed)

    def close(self):
        for (_, host), leaf in zip(self._hosts, self.backend.network.loras):
            host.forward = leaf.org_forward


def validate(options):
    allowed = {'geometry', 'dummy', 'model_id', 'transformer_subfolder',
               'steps', 'batch', 'seed', 'ema_decay'}
    if set(options) - allowed:
        raise ValueError('unknown preview options: ' + str(sorted(set(options) - allowed)))
    geometry = options.get('geometry', [2, 1, 1])
    if len(geometry) != 3 or any(type(x) is not int or x < 1 for x in geometry):
        raise ValueError('positive native T/H/W required')
    if options.get('dummy') is not True:
        raise ValueError('pretrained/GPU qualification pending; only explicit CPU fixture is admitted')
    if options.get('transformer_subfolder', 'transformer') != 'transformer':
        raise ValueError('preview is distilled only; SFT requires a distinct qualified cohort')
    if options.get('model_id', DEFAULT_MODEL) != DEFAULT_MODEL:
        raise ValueError('model identity differs')
    native_sources()
    verified_builder()  # Fail before attaching any native hooks.


def capabilities(options):
    validate(options)
    return ResponseCapabilities('adapter_gan', 'video', 'ltx-video-latent-velocity',
        'continuous_field', (1, 128, *options.get('geometry', [2, 1, 1])), '0.9.4', PG_SOURCE_PROFILE,
        qualification='Tiny CPU fixture only; experimental velocity objective; pretrained/GPU unqualified')


def prepare(options, **staged):
    validate(options)
    rows = staged.get('rows')
    if rows is None or len(rows) < 2:
        raise ValueError('provide at least two native VideoRows')
    geometry = tuple(options.get('geometry', [2, 1, 1]))
    for row in rows:
        validate_row(row, geometry)
    adapter = make_native(options)
    try:
        prepared = [adapter.prepare(row) for row in rows]
        conditions = torch.tensor([[i, .725] for i in range(len(rows))], dtype=torch.float32)
        with torch.random.fork_rng():
            torch.manual_seed(options.get('seed', 7) + 1)
            critic = VideoFieldCritic(128, 2, geometry)
        game = ConditionalPG094(adapter, prepared, critic,
            conditions=conditions, total_steps=options.get('steps', 4), batch_size=options.get('batch', 2),
            seed=options.get('seed', 7))
        native_config = {key: options[key] for key in ('geometry', 'dummy', 'seed') if key in options}
        specification = inference_backend.specification(adapter, model_identity=adapter.model_identity,
            protocol=COHORT, configuration=native_config)
        return Binding(game, adapter.close, inference=specification)
    except BaseException:
        adapter.close()
        raise


class LTXResponseBackend(ResponseBackend):
    def open(self, config, **request):
        if config.cohort != COHORT:
            raise ValueError('new GAN cohort required; historical UNI is not migrated')
        return super().open(config, **request)


backend = LTXResponseBackend(BACKEND, ResponseCapabilities('adapter_gan', 'video',
    'ltx-video-latent-velocity', 'continuous_field', (1, 128, 2, 1, 1), '0.9.4', PG_SOURCE_PROFILE),
    prepare, validate, configure_capabilities=capabilities)


def config(**options):
    return SessionConfig(BACKEND, COHORT, dummy=True, **options)


def validate_row(row, geometry):
    if (not isinstance(row, VideoRow) or not isinstance(row.neutral, str) or not row.neutral
            or not isinstance(row.positive, str) or not row.positive
            or tuple(row.video_latents.shape) != (1, geometry[0]*geometry[1]*geometry[2], 128)
            or tuple(row.audio_latents.shape) != (1, 2, 8)
            or any(x.device.type != 'cpu' or x.requires_grad or not torch.isfinite(x).all()
                   for x in (row.video_latents, row.audio_latents))):
        raise ValueError('finite frozen native CPU latent geometry required')


def make_native(options):
    validate(options)
    with torch.random.fork_rng():
        torch.manual_seed(options.get('seed', 7))
        native = LTX25Backend(dummy=True, recipe=RECIPE_EMBED, lora_rank=4, lora_alpha=4)
    frozen = {prefix + name: value for prefix, module in (
        ('encoder.', native.encoder), ('connectors.', native.connectors), ('transformer.', native.transformer))
        for name, value in module.named_parameters() if not value.requires_grad}
    return VideoResponse(native, tuple(options.get('geometry', [2, 1, 1])),
        {'model_id': DEFAULT_MODEL, 'transformer_subfolder': 'transformer',
         'qualification': 'dummy-only', 'seed': options.get('seed', 7),
         'native_sources': native_sources(), 'frozen_sha256': _hash(frozen)})


def validate_inference(options, spec):
    if options:
        raise ValueError('native preview takes its exact configuration from the artifact')
    validate(spec.configuration)


def prepare_inference(options, spec):
    validate_inference(options, spec)
    adapter = make_native(spec.configuration)

    def response(*, neutral, video_latents, audio_latents, strength=1., seed=0):
        row = VideoRow(neutral, neutral, video_latents, audio_latents, strength)
        validate_row(row, adapter.geometry)
        with torch.no_grad(), adapter.network.scaled(strength):
            packed = adapter._pack(neutral, row, strength)
            # Audio conditioning remains frozen at the neutral caption.
            frozen = adapter._pack(neutral, row, 0.)
            packed = replace(packed, audio_encoder_hidden_states=frozen.audio_encoder_hidden_states,
                             audio_encoder_attention_mask=frozen.audio_encoder_attention_mask)
            return adapter._field(packed).detach().clone()

    return InferenceBinding(adapter.network,
        native_identity(adapter, model_identity=adapter.model_identity, protocol=COHORT),
        adapter.close, response=response)


inference_backend = InferenceBackend(BACKEND, backend.capabilities, prepare_inference,
    validate_inference, configure_capabilities=capabilities)
