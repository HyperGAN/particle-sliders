"""Native AR last-state bindings for the single shared ConditionalPG09 game.

No loaders, losses, optimizers, token generation or audio decoding run here.
Released PG0.9 admits these responses only below width170. Wider native AR
states require the shared core's exact reviewed explicit-continuous source
profile. No projection, truncation, field reshape or objective fallback.
"""
from dataclasses import dataclass, replace

import torch
from torch import nn

from .pg09_backends import (
    AdapterParameters, checkpoint_with_adapter_scale,
)


@dataclass(frozen=True)
class AudioRow:
    neutral_ids: tuple[int, ...]
    positive_ids: tuple[int, ...]
    history_tokens: int = 0
    scale: float = 1.0


@dataclass(frozen=True)
class PreparedAudioResponse:
    row: AudioRow
    target: torch.Tensor
    neutral: torch.Tensor


class AudioResponseCritic(nn.Module):
    """Stable import identity for shared-game checkpoint and inference reload."""
    def __init__(self, width, condition_width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + condition_width, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, response, condition):
        return self.net(torch.cat((response, condition), dim=1)).squeeze(-1)


def require_audio_width(width, *, pg_source_profile='release', penalty_input_domain='auto'):
    from particle_sliders.pg09 import validate_pg09_domain
    if type(width) is not int or width < 1:
        raise ValueError('positive native AR hidden width required')
    return validate_pg09_domain((1, width), response_domain='continuous_vector',
        pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)


class ARLastHiddenResponseAdapter:
    """Wrap an already attached native slider; retain its actual routed bank."""
    family = 'abstract_audio_ar'

    def __init__(self, model, slider, *, model_identity, checkpoint_modules=(),
                 pg_source_profile='release', penalty_input_domain='auto'):
        self.domain_gate = require_audio_width(model.config.hidden_size,
            pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
        if not model_identity:
            raise ValueError('verified model identity or explicit CPU fixture identity required')
        if any(getattr(m, 'gradient_checkpointing', False) for m in model.modules()):
            raise ValueError('disable existing checkpointing before binding scale-capturing modules')
        self.model, self.slider = model, slider
        from copy import deepcopy
        self.model_identity = deepcopy(model_identity)
        self.model.eval()
        named = [(name, p) for name, p in slider.named_parameters() if p.requires_grad]
        self.network = AdapterParameters(named, slider.scaled)
        owned = {id(p) for p in self.network.parameters()}
        if any(p.requires_grad and id(p) not in owned for p in model.parameters()):
            raise ValueError('unowned trainable frozen-model parameter')
        registered = {id(m) for m in model.modules()}
        modules = tuple(checkpoint_modules)
        if len({id(m) for m in modules}) != len(modules):
            raise ValueError('duplicate checkpoint modules')
        if any(id(m) not in registered or getattr(m, '_pg09_scale_checkpoint', False) for m in modules):
            raise ValueError('checkpoint modules must be unwrapped native host modules')
        for module in modules:
            checkpoint_with_adapter_scale(module, self.network)

    @property
    def particles(self):
        return self.network.particles

    def _hidden(self, ids):
        raise NotImplementedError

    def _response(self, ids, *, scale):
        with self.network.scaled(scale):
            value = self._hidden(ids).float()
        if value.shape != (1, int(self.model.config.hidden_size)) or not torch.isfinite(value).all():
            raise ValueError('native last-hidden response has invalid shape or values')
        return value

    def prepare(self, row):
        # Copy token bytes, never let a caller change prepared conditioning.
        row = replace(row, neutral_ids=tuple(row.neutral_ids), positive_ids=tuple(row.positive_ids))
        if row.scale != 1.0:
            raise ValueError('audio teacher is unipolar positive at scale +1')
        for ids in (row.neutral_ids, row.positive_ids):
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError('nonempty nonnegative integer token ids required')
        if row.neutral_ids == row.positive_ids:
            raise ValueError('distinct neutral and positive context required')
        if type(row.history_tokens) is not int or not 0 <= row.history_tokens < min(
                len(row.neutral_ids), len(row.positive_ids)):
            raise ValueError('history length must leave a prompt prefix')
        if row.history_tokens and row.neutral_ids[-row.history_tokens:] != row.positive_ids[-row.history_tokens:]:
            raise ValueError('teacher and student must share the same continuation history')
        with torch.no_grad():
            neutral = self._response(row.neutral_ids, scale=0.).detach().clone()
            target = self._response(row.positive_ids, scale=0.).detach().clone()
        return PreparedAudioResponse(row, target, neutral)

    def generated(self, prepared):
        return self._response(prepared.row.neutral_ids, scale=prepared.row.scale)

    @torch.no_grad()
    def sample(self, prepared, *, scale=1.):
        """A hidden-state probe, not an AR token sampler or an audio render."""
        if not 0 <= scale <= 1:
            raise ValueError('unipolar probe scale must be in [0,1]')
        # Keep even stochastic native forwards independent of the training RNG.
        devices = sorted({p.device.index for p in self.network.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            return self._response(prepared.row.neutral_ids, scale=scale).detach().clone()

    def metadata(self):
        return dict(protocol='audio-ar-last-hidden-pg090-v1', family=self.family,
                    model_identity=self.model_identity, adapter_names=list(self.network.names),
                    response_domain='continuous_vector', response_width=int(self.model.config.hidden_size),
                    pg_domain_gate=dict(self.domain_gate),
                    response_coordinates='native final AR token hidden state; no projection',
                    target_convention='frozen positive last hidden; shared continuation history',
                    student_conditioning='neutral', time_coordinates='no diffusion time; AR history length',
                    trained_scales=[1.0], cfg_adapter_branch='not applicable to hidden-state probe',
                    checkpoint_scale_capture='shared per-forward captured context',
                    audio_render_qualified=False, quality_qualified=False)


class YuE2ARResponseAdapter(ARLastHiddenResponseAdapter):
    family = 'yue2_native_ar'

    def __init__(self, backend, slider, **kwargs):
        self.backend = backend
        super().__init__(backend.model, slider, **kwargs)

    def _hidden(self, ids):
        # Native product selects the genuine final AR state, never logits/NAR.
        return self.backend.hidden(list(ids), checkpointing=False)[:, -1]


class Music3ARResponseAdapter(ARLastHiddenResponseAdapter):
    family = 'music3_native_qwen3_ar'

    def _hidden(self, ids):
        device = next(self.model.parameters()).device
        tokens = torch.tensor([ids], device=device, dtype=torch.long)
        # Unpadded token rows: the final index is the native last-real-token read.
        return self.model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                use_cache=False).last_hidden_state[:, -1]


def audio_conditions(prepared_rows):
    """Frozen context coordinates plus AR position and unipolar adapter sign.

    The full frozen neutral last state supplies model-derived context features;
    exact neutral/positive token bytes and history are additionally bound by
    ConditionalPG09's row signature. No row-number embedding is used.
    """
    return torch.cat([torch.cat((p.neutral.detach().clone(), p.neutral.new_tensor([[
        len(p.row.neutral_ids), len(p.row.positive_ids), p.row.history_tokens, p.row.scale,
    ]])) , dim=1) for p in prepared_rows])
