"""LTX's default text/connector protocol, an explicit continuous GAN arm.

Retains native frozen-positive preconnector nonconcept hold as target
conditioning. No hold loss, MSE, reconstruction term or optimizer lives here.
The genuine token/feature sequence is vectorized without pooling/truncation.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path

import torch

from .pg09_backends import AdapterParameters, PreparedResponse, bind_native_gradient_checkpointing
from .pg09_text_video import ResponseCompatibilityError, require_continuous_vector, _native_lora_scale
from .ltx25_uni import resolve_concept_token_ids, require_concept_tokens, unused_hold_mask


@dataclass(frozen=True)
class LTXEmbedRow:
    neutral: str
    positive: str
    concept_words: str = ''
    scale: float = 1.


@dataclass(frozen=True)
class PreparedLTXEmbedRow:
    caption: LTXEmbedRow
    native_shape: tuple[int, ...]
    target_mask: torch.Tensor
    student_mask: torch.Tensor
    common_mask: torch.Tensor
    hold_mask: torch.Tensor
    positive_token_ids: tuple[int, ...]
    neutral_token_ids: tuple[int, ...]


class LTX25ConnectorResponseAdapter:
    """Default TE/video-connector placement; video readout response only.

    Fixed common-valid mask zeros inactive coordinates while retaining the
    complete declared shape/order. Same-T mask intersection matches the native
    target support; empty or changed masks/shapes fail, with no pooling fallback.
    Rendering a complete audiovisual trajectory remains a separate gate.
    """
    def __init__(self, backend, *, native_shape, model_identity=None,
                 pg_source_profile='release', penalty_input_domain='auto'):
        shape = tuple(native_shape)
        if len(shape) != 3 or shape[0] != 1 or any(type(v) is not int or v < 1 for v in shape):
            raise ResponseCompatibilityError('declare genuine native [1,tokens,features] connector shape')
        self.domain_options = require_continuous_vector(shape[1] * shape[2], 'LTX connector sequence',
                              pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
        if not backend.embed_recipe:
            raise ResponseCompatibilityError('connector adapter requires the native default text/embed placement')
        if not backend.dummy and not model_identity:
            raise ResponseCompatibilityError('verified staged text/connector model identity required')
        self.backend, self.native_shape = backend, shape
        # Core's public provenance walker follows ``model``. Keep the actual TE
        # and connectors reachable for source/config/architecture binding while
        # backend.network and optimizer/export ownership remain adapter-only.
        self.model = torch.nn.ModuleDict({'encoder': backend.encoder, 'connectors': backend.connectors})
        self.model_identity = deepcopy(model_identity or {'qualification': 'CPU synthetic connector fixture'})
        roots = [(name, getattr(backend, name, None)) for name in ('encoder', 'connectors', 'transformer')]
        requested = {id(p) for p in backend.trainable_parameters()}
        for name, root in roots:
            if root is None:
                continue
            root.eval()
            if any(p.requires_grad and id(p) not in requested for p in root.parameters()):
                raise ResponseCompatibilityError(f'unowned trainable LTX base parameter in {name}')
        if hasattr(backend.network, 'named_parameters'):
            named = list(backend.network.named_parameters())
        else:
            named = [(prefix + '.' + name, p) for prefix, root in roots if root is not None
                     for name, p in root.named_parameters() if id(p) in requested]
        if set(id(p) for _, p in named) != requested:
            raise ResponseCompatibilityError('LTX TE/connector adapter ownership differs')
        self.network = AdapterParameters(named, lambda scale: _native_lora_scale(backend.network, scale))
        for name, root in roots[:2]:
            if root is not None:
                bind_native_gradient_checkpointing(root, self.network)

    def _mask(self, value):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != self.native_shape[:2]
                or value.requires_grad or not torch.isfinite(value).all()
                or not torch.all((value == 0) | (value == 1))):
            raise ResponseCompatibilityError('native connector mask must be fixed binary [1,tokens]')
        return value.detach().bool().clone()

    def _shape(self, value):
        if tuple(value.shape) != self.native_shape or not torch.isfinite(value).all():
            raise ResponseCompatibilityError('native connector shape changed; pooling/truncation is forbidden')
        return value.float()

    def _vector(self, value, mask):
        return (self._shape(value) * mask.to(value.device).unsqueeze(-1)).flatten(1)

    def prepare(self, row):
        row = deepcopy(row)
        if row.scale != 1 or not row.neutral or not row.positive or row.neutral == row.positive:
            raise ValueError('distinct neutral/positive captions and unipolar +1 required')
        with torch.no_grad(), self.network.scaled(0):
            positive = self.backend.encode_text(row.positive, frozen=True)
            neutral = self.backend.encode_text(row.neutral, frozen=True)
            concept = resolve_concept_token_ids(self.backend.tokenizer, row.positive,
                                                row.neutral, row.concept_words)
            require_concept_tokens(positive.token_ids, concept)
            hold = unused_hold_mask(positive.token_ids, (), concept, hold_mode='non_concept')
            target, target_mask = self.backend.encode_post_connector_video(
                positive, hold_neu=neutral, hold_mask=hold, scale=None)
            baseline, student_mask = self.backend.encode_post_connector_video(neutral, scale=None)
            target_mask, student_mask = self._mask(target_mask), self._mask(student_mask)
            common = target_mask & student_mask
            if not common.any():
                raise ResponseCompatibilityError('native target/student masks have no common valid coordinates')
            prepared = PreparedLTXEmbedRow(row, self.native_shape, target_mask, student_mask, common,
                                          hold.detach().clone(), tuple(positive.token_ids), tuple(neutral.token_ids))
            target, baseline = self._vector(target, common).detach(), self._vector(baseline, common).detach()
        return PreparedResponse(prepared, target, baseline)

    def conditioning(self, prepared, *, scale=1.):
        row = prepared.row
        if tuple(row.native_shape) != self.native_shape:
            raise ResponseCompatibilityError('prepared connector geometry differs')
        with self.network.scaled(scale):
            encoded = self.backend.encode_text(row.caption.neutral, frozen=False)
            if tuple(encoded.token_ids) != row.neutral_token_ids:
                raise ResponseCompatibilityError('neutral tokenizer identity changed')
            value, mask = self.backend.encode_post_connector_video(encoded, scale=None)
            if not torch.equal(self._mask(mask), row.student_mask):
                raise ResponseCompatibilityError('student connector mask changed')
            return self._shape(value), mask

    def generated(self, prepared):
        value, _ = self.conditioning(prepared, scale=prepared.row.caption.scale)
        return self._vector(value, prepared.row.common_mask)

    @torch.no_grad()
    def render(self, prepared, *, scale=1.):
        value, _ = self.conditioning(prepared, scale=scale)
        return self._vector(value, prepared.row.common_mask).detach().clone()

    def metadata(self):
        return dict(protocol='ltx25-connector-continuous-gan-v1', model_identity=self.model_identity,
                    native_shape=list(self.native_shape), response='vectorized native post-connector video sequence',
                    order='token-major, feature-minor; complete coordinates retained',
                    fixed_mask='frozen target_mask AND neutral_mask; inactive coordinates zero, no truncation',
                    teacher='frozen positive caption with native preconnector nonconcept hold to frozen neutral',
                    student='neutral caption through live adapted TE and video connectors',
                    hold_objective='none; hold defines frozen teacher conditioning only',
                    hold_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(resolve_concept_token_ids)).read_bytes()).hexdigest(),
                    adapter_names=list(self.network.names), adapter_dtype='float32',
                    domain_options=self.domain_options, particles=None, generator_arm='adapter_gan',
                    time_coordinates='none; text/connector readout', scale_convention='unipolar +1',
                    audiovisual_render_qualified=False, quality_qualified=False)
