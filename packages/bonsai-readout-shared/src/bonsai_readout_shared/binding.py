"""Separate Bonsai readout cohort using the installed core and actual HNDL.

Historical particle_bridge_gan is deliberately not imported. The new game is
the shared public BISPF preset, not a reproduction of its archived Rp/noise/VIC
game. Native width5120 is refused before model/server/graph construction.
The separately named width16 fixture tests software only, never Bonsai quality.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from ._integrity import CORE_PYTHON_SHA256, CORE_ASSETS_SHA256, require_source_environment

# Check disk sources/caches and ordinary path shadows before importing core
# entrypoints. This is cooperative integrity checking, not loaded-code proof.
require_source_environment()

import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLParticleProjection, verified_builder
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding
from particle_sliders.pg094 import ConditionalPG094, PG_SOURCE_PROFILE, PG_VERSION, validate_pg09_domain
from particle_sliders.response import ResponseBackend, ResponseCapabilities

CORE_COMMIT = '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
# Informational archive reference only; rebuilt wheels may have different ZIP
# timestamps. Admission checks exact source/assets, not this archived wheel.
CORE_REFERENCE_WHEEL_SHA256 = '9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad'
PROFILE = 'bonsai-readout-shared-pg094-hypergan-hndl-v1'
PROTOCOL = PROFILE + '-native-response'
FIXTURE_WIDTH = 16
NATIVE_WIDTH = 5120
CRITIC_SOURCE = '''joined = concat(response, condition, axis=1)
hidden = linear(joined, 16, name="hidden")
hidden = silu(hidden)
linear(hidden, 1, name="score")
'''


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def file_content_identity(path, expected_sha256):
    """Stream a local file and reject changed bytes; never downloads a model."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('native model changed while hashing')
    if digest.hexdigest() != expected_sha256:
        raise ValueError('native model content SHA256 differs')
    return dict(content_sha256=digest.hexdigest(), file_bytes=after.st_size)


def _native_model_identity(options):
    # The native gate remains first: this cannot read weights under core0.12.
    validate_pg09_domain((1, NATIVE_WIDTH), response_domain='continuous_vector')
    from . import _native as original
    original.resolve_weights(options['weights'])
    original.check_weights_size(options['weights'])
    content = file_content_identity(options['weights'], options['model_sha256'])
    original.probe_gguf(options['weights'])
    return dict(kind='frozen-prism-gguf-last-token-readout', repo=original.REPO,
        filename=original.FILENAME, **content, fork_release=original.FORK_RELEASE_PIN,
        pooling=original.EMBED_POOLING, width=NATIVE_WIDTH,
        native_source_sha256=_sha(Path(original.__file__).read_bytes()),
        server_identity_qualification='requires independent server-to-file attestation; native remains blocked')


def verify_runtime():
    """Check installed files/caches/origins, then the shared builder gate.

    No claim is made about arbitrary preloaded/monkeypatched in-memory code.
    """
    scope = require_source_environment()
    builder, identity = verified_builder()
    return builder, dict(identity, verification_scope=scope['scope'])


class _ReadoutIdentity(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_features = self.out_features = width

    def forward(self, value):
        return value


class SharedReadoutHead(nn.Module):
    """One bank and the shared exact rank8/alpha8 routed HNDL projection.

    This is a residual head on frozen server vectors, never in-attention LoRA.
    Width is architectural here; game admission remains a separate hard gate.
    """
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width not in (FIXTURE_WIDTH, NATIVE_WIDTH):
            raise ValueError('only the declared Bonsai/CPU fixture head shapes are supported')
        builder, identity = verify_runtime()
        self.width = width
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.projection = HNDLParticleProjection('bonsai_readout_residual',
            _ReadoutIdentity(width), self.particles, 8, 8,
            builder=builder, identity=identity, attach=False)
        self._identity = dict(profile=PROFILE, core_commit=CORE_COMMIT,
            core_python_sha256=CORE_PYTHON_SHA256, core_assets_sha256=CORE_ASSETS_SHA256, builder=identity,
            native_input_shape=['B', width], native_output_shape=['B', width],
            bank_shape=[128, 4], placement='frozen_readout_residual', rank=8, alpha=8)

    @property
    def scale(self):
        return self.projection.multiplier

    @contextmanager
    def scaled(self, strength):
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('shared Bonsai strength must be in [0,1]')
        previous = self.scale
        self.projection.multiplier = strength
        try:
            yield self
        finally:
            self.projection.multiplier = previous

    def forward(self, value):
        if value.ndim != 2 or value.shape[-1] != self.width:
            raise ValueError('native pooled readout shape must remain [B,width]')
        return self.projection(value)

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)


class HNDLReadoutCritic(nn.Module):
    """Thin real HG-built Linear16/SiLU/Linear1 native vector graph.

    Matches the shared ConditionalReadoutCritic, not the legacy local critic.
    No loss, penalty, optimizer or training equation is implemented here.
    """
    score_bound = 0

    def __init__(self, width, condition_width):
        super().__init__()
        builder, identity = verify_runtime()
        self.graph = builder(CRITIC_SOURCE,
            input_shape={'response': ['B', width], 'condition': ['B', condition_width]},
            output_shape=['B', 1])
        self._identity = dict(profile=PROFILE, builder=identity,
            graph_sha256=_sha(CRITIC_SOURCE.encode()),
            input_shapes={'response': ['B', width], 'condition': ['B', condition_width]},
            output_shape=['B', 1], matched_architecture='ConditionalReadoutCritic(hidden=16)',
            plan=repr(self.graph))

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)


class FixtureReadout:
    """Explicit non-model width16 stand-in; cannot select native width/model."""
    width = FIXTURE_WIDTH
    identity = dict(kind='synthetic-sha256-seeded-readout-v1', width=FIXTURE_WIDTH,
                    qualification='CPU fixture only; no pretrained Bonsai or server')

    @torch.no_grad()
    def hidden(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63)
        return torch.randn(1, self.width, generator=torch.Generator().manual_seed(seed))


class PrismReadout:
    """The existing frozen text-only backend, behind native domain admission.

    This does not launch/download a server or add a gradient path through GGUF.
    The current pinned core refuses this constructor before any server request.
    """
    width = NATIVE_WIDTH

    def __init__(self, options):
        validate_pg09_domain((1, self.width), response_domain='continuous_vector')
        from . import _native as original
        self.identity = _native_model_identity(options)
        self.backend = original.BonsaiGGUFBackend(device='cpu', weights=options['weights'],
            server_url=options['server_url'])

    @torch.no_grad()
    def hidden(self, text):
        return self.backend.hidden(self.backend.encode(text)).detach()


@dataclass(frozen=True)
class ReadoutRow:
    positive: str
    neutral: str
    scale: float = 1.


@dataclass(frozen=True)
class PreparedReadout:
    row: ReadoutRow
    target: torch.Tensor
    neutral: torch.Tensor


class BonsaiReadoutAdapter:
    """Frozen readout capture and head ownership; shared core owns the game."""
    def __init__(self, runtime):
        validate_pg09_domain((1, runtime.width), response_domain='continuous_vector')
        self.runtime = runtime
        self.network = SharedReadoutHead(runtime.width)

    def prepare(self, rows):
        prepared = []
        with torch.no_grad():
            for row in rows:
                target = self.runtime.hidden(row.positive).detach().clone()
                neutral = self.runtime.hidden(row.neutral).detach().clone()
                for value in (target, neutral):
                    if value.shape != (1, self.runtime.width) or not torch.isfinite(value).all():
                        raise ValueError('frozen readout must be finite and preserve native shape')
                prepared.append(PreparedReadout(row, target, neutral))
        return prepared

    def generated(self, prepared):
        # Historical Bonsai adapts the positive-input readout. Preserve this
        # placement/convention, while declaring the new public BISPF objective.
        with self.network.scaled(prepared.row.scale):
            return self.network(prepared.target.detach())

    def response(self, *, text, strength=1., seed=0):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('nonempty readout text required')
        with torch.no_grad():
            value = self.runtime.hidden(text).detach()
        with self.network.scaled(strength):
            return self.network(value)

    def metadata(self):
        return dict(protocol=PROTOCOL, model_identity=self.runtime.identity,
            network_profile=PROFILE, response_domain='continuous_vector',
            response_width=self.runtime.width, placement='frozen_readout_residual',
            student_conditioning='positive_input', target='frozen_positive_readout',
            objective='shared_public_bispf_optimized; separate from legacy particle_bridge_gan',
            native_model_differentiable=False, token_generation_qualified=False)

    def close(self):
        pass  # No host hooks, server ownership, or persistent native mutation.


def _caps(width, fixture):
    return ResponseCapabilities('routed_particle_gan', 'synthetic_vector' if fixture else 'text_readout',
        'synthetic_pooled_readout' if fixture else 'frozen_gguf_last_token_embedding',
        'continuous_vector', (1, width), PG_VERSION, PG_SOURCE_PROFILE,
        qualification=('CPU fixture only; no pretrained/native qualification' if fixture else
                       'BLOCKED: width5120 needs reviewed explicit continuous-domain support'))


def _validate_options(options, fixture, inference=False):
    common = {'network_profile', 'pg_version', 'pg_source_profile', 'penalty_input_domain', 'base_precision'}
    allowed = common | (set() if inference else {'rows_file', 'steps', 'batch', 'seed', 'ema_decay'})
    if not fixture:
        allowed |= {'weights', 'server_url', 'model_sha256'}
    if set(options) - allowed:
        raise ValueError('unknown Bonsai shared options: ' + str(sorted(set(options) - allowed)))
    for key, expected in dict(network_profile=PROFILE, pg_version=PG_VERSION,
            pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain='auto', base_precision='float32').items():
        if options.get(key, expected) != expected:
            raise ValueError('Bonsai profile/source/precision differs: ' + key)
    if not fixture:
        from ._native import resolve_weights
        if not options.get('weights') or not options.get('server_url'):
            raise ValueError('native Bonsai needs local pinned weights and an existing fork server')
        resolve_weights(options['weights'])
        digest = options.get('model_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('native Bonsai requires an explicit expected model_sha256')
    validate_pg09_domain((1, FIXTURE_WIDTH if fixture else NATIVE_WIDTH), response_domain='continuous_vector')
    verify_runtime()


def _read_rows(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError('at least two positive/neutral JSON rows required')
    rows = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {'positive', 'neutral'}:
            raise ValueError('row requires only positive and neutral strings')
        if any(not isinstance(item[k], str) or not item[k].strip() for k in item) or item['positive'] == item['neutral']:
            raise ValueError('distinct nonempty positive/neutral text required')
        rows.append(ReadoutRow(**item))
    return rows


def _native_runtime(options):
    # Remains unreachable under the pinned core. Do not replace the guard with
    # truncation, projection, categorical fallback or a synthetic native claim.
    validate_pg09_domain((1, NATIVE_WIDTH), response_domain='continuous_vector')
    return PrismReadout(options)


def _prepare(options, fixture):
    # Also safe when the factory is called directly, outside the core's fork.
    # Seed CPU only; do not mutate queued CUDA RNG state in this CPU profile.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(options.get('seed', 7))
        return _prepare_seeded(options, fixture)


def _prepare_seeded(options, fixture):
    _validate_options(options, fixture)
    rows = _read_rows(options['rows_file'])
    native = BonsaiReadoutAdapter(FixtureReadout() if fixture else _native_runtime(options))
    prepared = native.prepare(rows)
    conditions = torch.cat([p.neutral for p in prepared])
    critic = HNDLReadoutCritic(native.runtime.width, conditions.shape[1])
    game = ConditionalPG094(native, prepared, critic, conditions=conditions,
        particles=native.network.particles, response_domain='continuous_vector',
        preset='bispf_optimized', total_steps=options.get('steps', 4),
        batch_size=options.get('batch', 2), seed=options.get('seed', 7))
    backend = FIXTURE_INFERENCE if fixture else INFERENCE
    specification = backend.specification(native, model_identity=native.runtime.identity,
        protocol=PROTOCOL, configuration={'network_profile': PROFILE})
    return Binding(game, native.close, inference=specification)


def validate_training(options):
    _validate_options(options, False)
    _read_rows(options['rows_file'])


def validate_fixture(options):
    _validate_options(options, True)
    _read_rows(options['rows_file'])


def prepare_training(options):
    return _prepare(options, False)


def prepare_fixture(options):
    return _prepare(options, True)


def validate_inference(options, spec):
    _validate_inference(options, spec, False)


def validate_fixture_inference(options, spec):
    _validate_inference(options, spec, True)


def _validate_inference(options, spec, fixture):
    if spec.configuration != {'network_profile': PROFILE}:
        raise ValueError('inference network profile differs')
    _validate_options(options, fixture, inference=True)
    expected = FixtureReadout.identity if fixture else _native_model_identity(options)
    if spec.identity['model'] != expected:
        raise ValueError('inference frozen model identity differs')


def prepare_inference(options, spec):
    return _prepare_inference(options, spec, False)


def prepare_fixture_inference(options, spec):
    return _prepare_inference(options, spec, True)


def _prepare_inference(options, spec, fixture):
    _validate_inference(options, spec, fixture)
    native = BonsaiReadoutAdapter(FixtureReadout() if fixture else _native_runtime(options))
    return InferenceBinding(native.network,
        native_identity(native, model_identity=native.runtime.identity, protocol=PROTOCOL),
        native.close, response=native.response)


TRAINING = ResponseBackend('bonsai-gguf-shared', _caps(NATIVE_WIDTH, False), prepare_training, validate_training)
FIXTURE_TRAINING = ResponseBackend('bonsai-readout-fixture', _caps(FIXTURE_WIDTH, True), prepare_fixture, validate_fixture)
INFERENCE = InferenceBackend('bonsai-gguf-shared', _caps(NATIVE_WIDTH, False), prepare_inference, validate_inference)
FIXTURE_INFERENCE = InferenceBackend('bonsai-readout-fixture', _caps(FIXTURE_WIDTH, True), prepare_fixture_inference, validate_fixture_inference)
