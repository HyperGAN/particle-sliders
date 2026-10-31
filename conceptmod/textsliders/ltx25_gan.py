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
from particle_sliders.lifecycle import Binding, SessionConfig, UnsupportedOperation
from particle_sliders.response import ResponseBackend, ResponseCapabilities, ResponseSession
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg09 import _hash
from .ltx25_backend import (
    LTX25Backend, DEFAULT_MODEL, embed_lora_hosts, ltx_pack_feature_dim,
    DummyTokenizer, DummyEncoder, DummyLTX2Connectors, DummyLTX2Transformer,
    _FrozenStub, _DummyPipe, ModuleLoRANetwork, _masked_token_pool,
)
from .ltx25_uni import (
    RECIPE_EMBED, resolve_concept_token_ids, require_concept_tokens, unused_hold_mask,
)

COHORT = 'ltx25-dummy-video-field-gan-hg-hndl-v2-preview'
BACKEND = 'ltx25-dummy-video-gan-preview'
DUMMY_MODEL = 'dummy-ltx25-api-fixture-v2'
OBJECTIVE = 'frozen-plus-video-velocity;student-neutral;frozen-neutral-audio'
DUMMY_QUALIFICATION = 'DUMMY PREVIEW ONLY; no native/pretrained/GPU/media qualification'
NATIVE_SOURCES = {
    'ltx25_backend.py': '863fdf0aa352507f7cf139c0f70e0df53949452b2714a7cecef86c8b755088d2',
    'ltx25_uni.py': '77bb72d84d5af0e220241639e910ebeabbbe95ef582ac8a05c68475f48b0e61d',
    'slider_targets.py': '5db3a2cdc9635a0f15f6bc59c80f00700ef04d35241650b935b373f610a85412',
}


try:
    from ._admission import admit
except ImportError:
    def admit():
        raise ValueError('install the isolated dummy preview wheel; source checkout is not admitted')


def native_sources():
    actual = {}
    for name in NATIVE_SOURCES:
        source = Path(__file__).with_name(name).read_bytes()
        if name == 'ltx25_backend.py' and __package__ == 'particle_sliders_ltx25_dummy_preview':
            for module in ('ltx25_uni', 'slider_targets'):
                old = ('from .' + module + ' import').encode()
                if source.count(old) != 1:
                    raise ValueError('isolated native import count differs')
                source = source.replace(old, ('from conceptmod.textsliders.' + module + ' import').encode(), 1)
        actual[name] = hashlib.sha256(source).hexdigest()
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


class NativeLedger:
    """Capture before hooks; restore by object identity, independently of adapter lists."""
    def __init__(self, native, roots, hosts):
        self.native, self.roots, self.hosts = native, tuple(roots), tuple(hosts)
        if not self.hosts or len({id(host) for _, host in self.hosts}) != len(self.hosts):
            raise ValueError('unique native host ownership required')
        self.forwards = {host: host.forward for _, host in self.hosts}
        self.modules = {module: (module.training, dict(module._modules), dict(module._parameters),
                                 dict(module._buffers), set(module._non_persistent_buffers_set))
                        for root in self.roots for module in root.modules()}
        self.parameters = {p: (p.requires_grad, p.grad) for root in self.roots for p in root.parameters()}
        self.network_before = native.network
        self.closed = False

    def freeze(self):
        for root in self.roots:
            root.requires_grad_(False)
            root.eval()

    def check(self, network):
        if self.closed:
            raise ValueError('dummy native binding is closed')
        if len(network.loras) != len(self.hosts):
            raise ValueError('adapter host/leaf ownership count differs')
        expected = set()
        for index, (_, host) in enumerate(self.hosts):
            leaf = network.loras[index]
            if leaf.org_forward != self.forwards[host] or host.forward != leaf.forward:
                raise ValueError('adapter host identity differs')
            expected.update(leaf.parameters())
        owned = tuple(network.parameters())
        if not expected or set(owned) != expected or len(owned) != len(expected):
            raise ValueError('adapter parameter ownership differs')
        if any(not p.requires_grad or p.dtype != torch.float32 for p in owned):
            raise ValueError('adapter leaves must be trainable float32')
        current = {p for root in self.roots for p in root.parameters()}
        if current != set(self.parameters):
            raise ValueError('native base parameter registration differs')
        if any(p.requires_grad for p in current):
            raise ValueError('unowned trainable native base parameter')
        if any(module.training for root in self.roots for module in root.modules()):
            raise ValueError('frozen native base must stay in evaluation mode')

    def close(self):
        if self.closed:
            return
        for host, forward in self.forwards.items():
            host.forward = forward
        for module, (mode, modules, parameters, buffers, nonpersistent) in self.modules.items():
            module._modules = dict(modules)
            module._parameters = dict(parameters)
            module._buffers = dict(buffers)
            module._non_persistent_buffers_set = set(nonpersistent)
            module.training = mode
        for parameter, (requires_grad, gradient) in self.parameters.items():
            parameter.requires_grad_(requires_grad)
            parameter.grad = gradient
        self.native.network = self.network_before
        self.closed = True


class DummyPreviewBackend(LTX25Backend):
    """Reuse the exact dummy architecture, capturing state before LoRA attachment."""
    def _init_dummy(self):
        self.tokenizer, self.encoder = DummyTokenizer(), DummyEncoder()
        self.connectors, self.transformer = DummyLTX2Connectors(), DummyLTX2Transformer()
        self.vae, self.audio_vae, self.vocoder = (_FrozenStub(name) for name in ('vae', 'audio_vae', 'vocoder'))
        roots = (self.encoder, self.connectors, self.transformer, self.vae, self.audio_vae, self.vocoder)
        hosts = embed_lora_hosts(self.encoder, self.connectors, last_n=self.te_last_n)
        self.ledger = NativeLedger(self, roots, hosts)
        try:
            self.ledger.freeze()
            self.network = ModuleLoRANetwork(hosts, rank=self.lora_rank, alpha=self.lora_alpha,
                                            up_init_std=self.lora_up_init_std)
            self.ledger.check(self.network)
            self.pipe = _DummyPipe(tokenizer=self.tokenizer, text_encoder=self.encoder,
                connectors=self.connectors, transformer=self.transformer, vae=self.vae,
                audio_vae=self.audio_vae, vocoder=self.vocoder)
        except BaseException:
            self.ledger.close()
            raise


class NativeLeaves(nn.Module):
    """Own the original adapter leaves, restoring native scale after every call."""
    def __init__(self, native):
        super().__init__()
        self.adapter = native.network

    @contextmanager
    def scaled(self, scale):
        old = self.adapter.lora_scale
        multipliers = {leaf: leaf.multiplier for leaf in self.adapter.loras}
        try:
            self.adapter.set_lora_slider(float(scale))
            yield
        finally:
            self.adapter.lora_scale = old
            for leaf, value in multipliers.items():
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
    condition: torch.Tensor


class VideoResponse:
    def __init__(self, native, geometry, model_identity):
        self.backend = native
        self.geometry = tuple(geometry)
        self.model_identity = deepcopy(model_identity)
        self.network = NativeLeaves(native)
        # Rollback is product-owned; no host module registrations are changed.
        self.ledger = native.ledger
        self.ledger.check(native.network)

    def metadata(self):
        return dict(cohort=COHORT, objective=OBJECTIVE, geometry=self.geometry,
                    model=self.model_identity, audio='frozen-neutral-conditioning-and-latents',
                    targets=self.backend.lora_module_names(), recipe_host=RECIPE_EMBED)

    def _pack(self, prompt, row, scale, *, teacher=False):
        native = self.backend
        self.ledger.check(native.network)
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
        feature = _masked_token_pool(neutral.encoder_hidden_states, neutral.encoder_attention_mask, 1).squeeze(1)
        sigma = neutral.timestep.float().mean().reshape(1, 1) / 1000.
        condition = torch.cat((feature.float(), sigma), dim=1).detach()
        return PreparedVideo(row, target.detach(), base.detach(),
                             neutral.audio_encoder_hidden_states.detach(), neutral.audio_encoder_attention_mask.detach(),
                             condition)

    def generated(self, prepared):
        with self.network.scaled(prepared.row.scale):
            packed = self._pack(prepared.row.neutral, prepared.row, prepared.row.scale)
            packed = replace(packed, audio_encoder_hidden_states=prepared.audio_condition,
                             audio_encoder_attention_mask=prepared.audio_mask)
            return self._field(packed)

    def close(self):
        self.ledger.close()


def validate(options):
    admit()
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
    if options.get('model_id', DUMMY_MODEL) != DUMMY_MODEL:
        raise ValueError('model identity differs')
    native_sources()
    verified_builder()  # Fail before attaching any native hooks.


def capabilities(options):
    validate(options)
    return ResponseCapabilities('adapter_gan', 'video', 'dummy-ltx-video-latent-velocity',
        'continuous_field', (1, 128, *options.get('geometry', [2, 1, 1])), '0.9.4', PG_SOURCE_PROFILE,
        qualification=DUMMY_QUALIFICATION)


@dataclass
class DummyBinding(Binding):
    dummy_specification: object = None


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
        conditions = torch.cat([item.condition for item in prepared]).detach()
        with torch.random.fork_rng():
            torch.manual_seed(options.get('seed', 7) + 1)
            critic = VideoFieldCritic(128, conditions.shape[1], geometry)
        game = ConditionalPG094(adapter, prepared, critic,
            conditions=conditions, total_steps=options.get('steps', 4), batch_size=options.get('batch', 2),
            seed=options.get('seed', 7))
        native_config = {key: options[key] for key in ('geometry', 'dummy', 'seed') if key in options}
        specification = inference_backend.specification(adapter, model_identity=adapter.model_identity,
            protocol=COHORT, configuration=native_config)
        return DummyBinding(game, adapter.close, dummy_specification=specification)
    except BaseException:
        adapter.close()
        raise


class DummyResponseSession(ResponseSession):
    def export_inference(self, *args, **kwargs):
        raise UnsupportedOperation('dummy preview cannot export a native model artifact; use export_dummy_inference')

    def export_dummy_inference(self, *args, **kwargs):
        if 'specification' in kwargs:
            raise ValueError('dummy export uses only its exact dummy specification')
        return super().export_inference(*args, specification=self.binding.dummy_specification, **kwargs)


class LTXResponseBackend(ResponseBackend):
    def open(self, config, **request):
        if config.cohort != COHORT:
            raise ValueError('new GAN cohort required; historical UNI is not migrated')
        return super().open(config, **request)

    def _session(self, config, capabilities, binding):
        return DummyResponseSession(config, capabilities, binding)


backend = LTXResponseBackend(BACKEND, ResponseCapabilities('adapter_gan', 'video',
    'dummy-ltx-video-latent-velocity', 'continuous_field', (1, 128, 2, 1, 1), '0.9.4', PG_SOURCE_PROFILE,
    qualification=DUMMY_QUALIFICATION),
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
    native = None
    try:
        with torch.random.fork_rng():
            torch.manual_seed(options.get('seed', 7))
            native = DummyPreviewBackend(dummy=True, model_id=DUMMY_MODEL,
                recipe=RECIPE_EMBED, lora_rank=4, lora_alpha=4)
        native.ledger.check(native.network)
        frozen = {str(index) + '.' + name: value for index, root in enumerate(native.ledger.roots)
                  for name, value in root.named_parameters()}
        return VideoResponse(native, tuple(options.get('geometry', [2, 1, 1])),
            {'model_id': DUMMY_MODEL, 'reference_model': DEFAULT_MODEL, 'transformer_subfolder': 'transformer',
             'qualification': 'dummy-preview-only', 'seed': options.get('seed', 7),
             'runtime': admit(), 'native_sources': native_sources(), 'frozen_sha256': _hash(frozen)})
    except BaseException:
        if native is not None:
            native.ledger.close()
        raise


def validate_inference(options, spec):
    if options:
        raise ValueError('native preview takes its exact configuration from the artifact')
    validate(spec.configuration)
    if (spec.identity['protocol'] != COHORT or spec.identity['model'].get('qualification') != 'dummy-preview-only'
            or spec.identity['model'].get('model_id') != DUMMY_MODEL):
        raise ValueError('only explicitly identified dummy-preview artifacts are admitted')
    if (spec.identity['model'].get('runtime') != admit()
            or spec.identity['model'].get('native_sources') != native_sources()):
        raise ValueError('dummy artifact runtime/source identity differs before hooks')


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

    try:
        return InferenceBinding(adapter.network,
            native_identity(adapter, model_identity=adapter.model_identity, protocol=COHORT),
            adapter.close, response=response)
    except BaseException:
        adapter.close()
        raise


def inference_capabilities(options):
    return replace(capabilities(options), operations=('response',))


inference_backend = InferenceBackend(BACKEND, replace(backend.capabilities, operations=('response',)),
    prepare_inference, validate_inference, configure_capabilities=inference_capabilities)
