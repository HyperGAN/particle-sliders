"""Thin text/video response bridges; ConditionalPG09 owns the whole game.

Tiny reuses the 7039652 native routed adapter, without its PairedPG09 loop.
LTX admits only its explicit velocity arm, with genuine video coordinates.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path

import torch

from .pg09_backends import (
    AdapterParameters, FrozenResponseAdapter, PreparedResponse,
    bind_native_gradient_checkpointing,
)


class ResponseCompatibilityError(ValueError):
    pass


def require_continuous_vector(width, family, *, pg_source_profile='release', penalty_input_domain='auto'):
    # Source/profile validation belongs to the reviewed core; no local manifest
    # copy, implicit profile selection, response reshape or installed patch.
    if (pg_source_profile, penalty_input_domain) == ('release', 'auto') and not 0 < width < 170:
        raise ResponseCompatibilityError(
            f'{family}: released PG09 cannot admit flat continuous width {width}; '
            'no reshape, projection, categorical or MSE fallback is provided')
    from particle_sliders.pg09 import validate_pg09_domain
    validate_pg09_domain((1, width), response_domain='continuous_vector',
                         pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
    return dict(response_domain='continuous_vector', pg_source_profile=pg_source_profile,
                penalty_input_domain=penalty_input_domain)


@dataclass(frozen=True)
class TextResponseRow:
    neutral_ids: tuple[int, ...]
    positive_ids: tuple[int, ...]
    scale: float = 1.0


class TinyQwenResponseAdapter:
    """Existing TinyParticleSlider and actual bank, native last-token vector.

    The standard live 1024-wide Qwen3 readout requires the exact reviewed
    explicit continuous profile. Release-auto behavior remains blocked.
    """
    def __init__(self, backend, slider, *, model_identity=None,
                 pg_source_profile='release', penalty_input_domain='auto'):
        config = backend.model.config
        width = int(config['hidden_size'] if isinstance(config, dict) else config.hidden_size)
        self.domain_options = require_continuous_vector(
            width, 'TinyQwen', pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
        if not backend.dummy and not model_identity:
            raise ResponseCompatibilityError('verified staged-model identity required')
        from .tiny_llm_particle import _attention_targets
        targets = _attention_targets(backend.model)
        if (list(targets) != slider.target_names or any(
                getattr(m.forward, '__self__', None) is not slider.adapters[
                    'lora_tiny-' + name.replace('.', '-')]
                for name, m in targets.items())):
            raise ResponseCompatibilityError('slider must be attached to this exact backend')
        if any(p.requires_grad for p in backend.model.parameters()):
            raise ResponseCompatibilityError('TinyQwen base must be frozen')
        self.backend, self.slider = backend, slider
        self.network = AdapterParameters(slider.named_parameters(), slider.scaled)
        self.particles = slider.particles
        self.model_identity = deepcopy(model_identity or {'qualification': 'CPU fixture only'})
        bind_native_gradient_checkpointing(backend.model, self.network)

    def prepare(self, row):
        row = deepcopy(row)
        if (row.scale != 1 or not row.neutral_ids or not row.positive_ids or
                row.neutral_ids == row.positive_ids):
            raise ValueError('distinct neutral/positive token ids and unipolar +1 required')
        with torch.no_grad(), self.network.scaled(0):
            neutral = self.backend.hidden(row.neutral_ids)[:, -1].float().detach().clone()
            target = self.backend.hidden(row.positive_ids)[:, -1].float().detach().clone()
        return PreparedResponse(row, target, neutral)

    def generated(self, prepared):
        with self.network.scaled(prepared.row.scale):
            value = self.backend.hidden(prepared.row.neutral_ids)[:, -1].float()
        if value.shape != prepared.target.shape:
            raise ValueError('TinyQwen response shape changed')
        return value

    @torch.no_grad()
    def render(self, row, *, scale=1.):
        # This protocol is a hidden-state readout, not autoregressive text generation.
        with self.network.scaled(scale):
            return self.backend.hidden(row.neutral_ids)[:, -1].float().detach().clone()

    def metadata(self):
        return dict(protocol='tiny-qwen-shared-pg090-response-v1',
                    model_identity=self.model_identity, source_port='7039652',
                    response='native last-caption-token hidden vector',
                    student_conditioning='neutral token ids', target_convention='frozen positive token ids',
                    time_coordinates='none; fixed text readout', scale_convention='unipolar +1',
                    adapter_names=list(self.network.names), adapter_dtype='float32',
                    particles='actual TinyParticleSlider dense-routed bank',
                    domain_options=self.domain_options, quality_qualified=False)


@dataclass(frozen=True)
class BonsaiResponseRow:
    neutral: str
    positive: str
    scale: float = 1.0


class BonsaiReadoutResponseAdapter:
    """Existing native routed residual head on immutable real GGUF readouts.

    The backend may be the existing checksummed FrozenReadouts cache with no
    server. No gradient through GGUF or autoregressive generation is claimed.
    Released-auto defaults reject width5120 before reading any cached payload.
    """
    def __init__(self, backend, head, *, model_identity,
                 pg_source_profile='release', penalty_input_domain='auto'):
        self.domain_options = require_continuous_vector(
            5120, 'Bonsai', pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
        from .bonsai_gguf_particle import BonsaiParticleHead
        if not isinstance(head, BonsaiParticleHead) or not model_identity:
            raise ResponseCompatibilityError('existing native Bonsai head and verified model/readout identity required')
        self.backend, self.adapter = backend, head
        self.network = AdapterParameters(head.named_parameters(), head.scaled)
        self.particles = head.particles
        self.model_identity = deepcopy(model_identity)

    def prepare(self, row):
        row = deepcopy(row)
        if row.scale != 1 or not row.neutral or not row.positive or row.neutral == row.positive:
            raise ValueError('distinct neutral/positive captions and unipolar +1 required')
        with torch.no_grad():
            neutral = self.backend.hidden(self.backend.encode(row.neutral)).detach().clone()
            target = self.backend.hidden(self.backend.encode(row.positive)).detach().clone()
        for response in (neutral, target):
            if (response.shape != (1, 5120) or response.dtype != torch.float32
                    or not torch.isfinite(response).all()):
                raise ResponseCompatibilityError('Bonsai requires finite native [1,5120] FP32 frozen readouts')
        return PreparedResponse(row, target, neutral)

    def generated(self, prepared):
        with self.network.scaled(prepared.row.scale):
            return self.adapter(prepared.neutral).float()

    @torch.no_grad()
    def render(self, row, *, scale=1.):
        value = self.backend.hidden(self.backend.encode(row.neutral))
        with self.network.scaled(scale):
            return self.adapter(value).float().detach().clone()

    def metadata(self):
        return dict(protocol='bonsai-readout-shared-pg090-response-v1',
                    model_identity=self.model_identity, response='native [1,5120] frozen last-token readout',
                    adapter_placement='native torch-side routed residual head; no GGUF autograd',
                    student_conditioning='neutral readout', target_convention='frozen positive readout',
                    time_coordinates='none; fixed text readout', scale_convention='unipolar +1',
                    adapter_names=list(self.network.names), adapter_dtype='float32',
                    particles='actual BonsaiParticleHead dense-routed bank', domain_options=self.domain_options,
                    quality_qualified=False)


@contextmanager
def _native_lora_scale(native, value):
    """Native forward resets scales on exit; restore every prior value exactly."""
    layers = list(getattr(native, 'loras', ()))
    previous = [(m, m.multiplier) for m in layers]
    peft = [(m, dict(m.scaling)) for m in getattr(native, 'transformer', torch.nn.Module()).modules()
            if hasattr(m, 'lora_A') and hasattr(m, 'scaling')]
    previous_scale = native.lora_scale
    try:
        native.set_lora_slider(value)
        yield
    finally:
        native.set_lora_slider(previous_scale)
        for module, scale in previous:
            module.multiplier = scale
        for module, scaling in peft:
            module.scaling = scaling


class LTX25VideoResponseAdapter(FrozenResponseAdapter):
    """One-step native video velocity, frozen explicit audio context, no render.

    Conditions are (caption, fixed packed audio latent). Input/output are actual
    video volumes. Live use requires the staged pipeline's own pack/unpack
    helpers, patch=1 config and verified identity. CPU fixtures use the same
    patch=1 coordinate permutation; no generic vector is disguised as a field.
    """
    family = 'ltx25_video_velocity'
    time_coordinates = 'per-token LTX timestep = 1000*sigma, video and audio matched'

    def __init__(self, backend, *, model_identity=None):
        if backend.embed_recipe:
            raise ResponseCompatibilityError(
                'LTX default connector-embedding sequence is not an admitted native field; '
                'select the explicit velocity arm, with no automatic fallback')
        self.backend = backend
        self._live_codec_identity = None
        if not backend.dummy:
            config = backend.transformer.config
            if getattr(config, 'patch_size', None) != 1 or getattr(config, 'patch_size_t', None) != 1:
                raise ResponseCompatibilityError('only verified patch=1 native LTX geometry is supported')
            if not model_identity or any(not callable(getattr(backend.pipe, k, None))
                                         for k in ('_pack_latents', '_unpack_latents')):
                raise ResponseCompatibilityError('verified staged identity and native pipeline codec required')
            source = inspect.getsourcefile(type(backend.pipe))
            self._live_codec_identity = dict(module=type(backend.pipe).__module__,
                                            sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
        requested = {id(p) for p in backend.trainable_parameters()}
        for module in (backend.transformer, backend.encoder, backend.connectors):
            module.eval()
            if any(p.requires_grad and id(p) not in requested for p in module.parameters()):
                raise ResponseCompatibilityError('LTX base, encoder and connectors must be frozen')
        # Existing LoRA leaves are registered on the native network, via hooks.
        if hasattr(backend.network, 'named_parameters'):
            named = list(backend.network.named_parameters())
        else:
            named = [(n, p) for n, p in backend.transformer.named_parameters() if id(p) in requested]
        if set(id(p) for _, p in named) != requested:
            raise ResponseCompatibilityError('native LTX adapter ownership differs')
        network = AdapterParameters(named,
                                    lambda value: _native_lora_scale(backend.network, value))
        super().__init__(network, model_identity=model_identity)
        bind_native_gradient_checkpointing(backend.transformer, network)

    def prepare(self, row):
        if row.schedule is not None or row.guidance != 1:
            raise ResponseCompatibilityError(
                'LTX bridge admits raw one-step velocity only; coupled AV render/CFG is unqualified')
        if row.scale != 1 or row.strength != 1:
            raise ResponseCompatibilityError('LTX raw velocity uses the explicit unipolar +1 target')
        if row.latent.ndim != 5 or row.latent.shape[1] != self.backend.transformer.proj_in.in_features:
            raise ResponseCompatibilityError('LTX input must be a native [1,C,T,H,W] video volume')
        for context in (row.neutral, row.positive, row.negative):
            if (not isinstance(context, tuple) or len(context) != 2 or not isinstance(context[0], str)
                    or not isinstance(context[1], torch.Tensor) or context[1].ndim != 3
                    or context[1].shape[0] != 1 or not torch.isfinite(context[1]).all()):
                raise ResponseCompatibilityError('caption and fixed packed audio latent required')
            if not torch.equal(context[1], row.neutral[1]):
                raise ResponseCompatibilityError('all captions must share identical audio context')
        return super().prepare(row)

    def _velocity(self, condition, latent, timestep):
        caption, audio = condition
        b, c, t, h, w = latent.shape
        if self.backend.dummy:
            packed_video = latent.permute(0, 2, 3, 4, 1).reshape(b, t*h*w, c)
        else:
            packed_video = self.backend.pipe._pack_latents(latent, 1, 1)
        encoded = self.backend.encode_text(caption, frozen=True)
        packed = self.backend.pack_t2v(encoded, video_latents=packed_video,
                                      audio_latents=audio, num_frames=8*(t-1)+1,
                                      height=32*h, width=32*w)
        time = timestep.to(device=packed.hidden_states.device, dtype=torch.float32).reshape(1, 1)
        packed.timestep = time.expand(b, packed_video.shape[1])
        packed.audio_timestep = time.expand(b, audio.shape[1])
        output = self.backend.forward_velocity(packed, scale=self.network.scale).sample
        if output.shape != packed_video.shape:
            raise ResponseCompatibilityError('native video velocity packing changed')
        if self.backend.dummy:
            return output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return self.backend.pipe._unpack_latents(output, t, h, w, 1, 1)

    def metadata(self):
        return dict(super().metadata(), response='native [B,C,T,H,W] one-step video velocity',
                    audio='fixed packed context; audio response excluded', patch_size=1, patch_size_t=1,
                    native_codec=self._live_codec_identity,
                    render_qualified=False, teacher_quality_qualified=False)
