"""Modality-neutral numerical responses over the common slider lifecycle.

V1 provides train/resume/evaluate/export and native-only response inference.
It does not supply a token, image or waveform renderer. Native products own
loaders, response geometry, scale semantics and frozen/native observation state.
"""
from dataclasses import dataclass, field, asdict
import hashlib
from importlib.metadata import entry_points
from pathlib import Path

from .lifecycle import (
    Backend, BackendUnavailable, Capabilities, SliderSession, UnsupportedOperation,
    _open_registered,
)
from .pg09 import PG_VERSION, validate_pg09_domain

API = 'particle-sliders-response-session-v1'
TRAINING, INFERENCE = API + '-training', API + '-inference'
CAPABILITIES = 'particle-sliders-response-capabilities-v1'


@dataclass(frozen=True)
class ResponseCapabilities:
    generator_arm: str
    modality: str
    response_kind: str
    response_domain: str
    response_shape: tuple
    pg_version: str
    pg_source_profile: str
    penalty_input_domain: str = 'auto'
    strength_min: float | None = 0.
    strength_max: float | None = 1.
    base_precisions: tuple = ('float32',)
    operations: tuple = ('train', 'resume', 'evaluate', 'export', 'response')
    unavailable_reasons: tuple = (('render', 'native media rendering is unavailable; this backend exposes numerical responses'),)
    qualification: str = 'CPU conformance only; pretrained/native quality unqualified'
    schema: str = field(default=CAPABILITIES, init=False)

    def __post_init__(self):
        if self.generator_arm not in ('routed_particle_gan', 'adapter_gan'):
            raise ValueError('declare the actual routed-particle or adapter-only arm')
        if any(not isinstance(v, str) or not v for v in (
                self.modality, self.response_kind, self.pg_version, self.pg_source_profile)):
            raise ValueError('explicit modality, response kind, PG version and source profile are required')
        shape = tuple(self.response_shape)
        if (not shape or shape[0] != 1 or any(type(v) is not int or v < 1 for v in shape)
                or self.response_domain == 'continuous_vector' and len(shape) != 2
                or self.response_domain == 'continuous_field' and len(shape) not in (4, 5)
                or self.response_domain not in ('continuous_vector', 'continuous_field')):
            raise ValueError('declare actual native continuous response geometry without reshaping')
        if set(self.operations) - {'train', 'resume', 'evaluate', 'export', 'response'}:
            raise ValueError('response v1 cannot advertise media rendering or unknown operations')
        for key in ('response_shape', 'base_precisions', 'operations'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(self, 'unavailable_reasons', tuple(tuple(pair) for pair in self.unavailable_reasons))

    @property
    def render(self):
        return False

    @property
    def render_outputs(self):
        return ()

    def refusal(self, operation):
        return dict(self.unavailable_reasons).get(operation, operation + ' is unavailable for this response backend')

    def check(self, operation='train', *, strength=1., references=False, precision=None, output_type=None):
        if references:
            raise UnsupportedOperation('reference conditioning is unavailable in response-session v1')
        # One common strength/operation/precision checker; no image fields in our descriptor.
        Capabilities(self.generator_arm, False, strength_min=self.strength_min,
            strength_max=self.strength_max, base_precisions=self.base_precisions,
            operations=self.operations, unavailable_reasons=self.unavailable_reasons).check(
                operation, strength=strength, precision=precision, output_type=output_type)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop('schema', None) != CAPABILITIES:
            raise ValueError('unsupported response capability version')
        result = cls(**value)
        if set(asdict(result)) != set(value) | {'schema'}:
            raise ValueError('response capability descriptor fields differ')
        return result


@dataclass(frozen=True)
class ResponseBackend(Backend):
    """A complete product factory returning the common Binding; no local loop."""
    capabilities: ResponseCapabilities

    def _check_pg(self, options, capabilities):
        if not isinstance(capabilities, ResponseCapabilities):
            raise TypeError('response registration requires ResponseCapabilities')
        if capabilities.pg_version != PG_VERSION or options.get('pg_version', PG_VERSION) != PG_VERSION:
            raise BackendUnavailable('response requires its exact core/PG version ' + PG_VERSION)
        profile = options.get('pg_source_profile', capabilities.pg_source_profile)
        domain = options.get('penalty_input_domain', capabilities.penalty_input_domain)
        if (profile, domain) != (capabilities.pg_source_profile, capabilities.penalty_input_domain):
            raise BackendUnavailable('configured PG profile/domain differs from response capabilities')
        # Checks actual installed hashes and native dimensions before prepare/weights.
        validate_pg09_domain(capabilities.response_shape, response_domain=capabilities.response_domain,
                             pg_source_profile=profile, penalty_input_domain=domain)

    def _session(self, config, capabilities, binding):
        return ResponseSession(config, capabilities, binding)


class ResponseSession(SliderSession):
    api, training_schema, inference_schema = API, TRAINING, INFERENCE

    def __init__(self, config, capabilities, binding):
        super().__init__(config, capabilities, binding)
        signature = self.game._signature
        if (self.game.response_domain != capabilities.response_domain
                or tuple(self.game.response_shape) != capabilities.response_shape
                or signature['pg_version'] != capabilities.pg_version
                or signature['pg_source_profile'] != capabilities.pg_source_profile
                or signature['penalty_input_domain'] != capabilities.penalty_input_domain):
            raise ValueError('actual game response geometry/profile contradicts declared capabilities')
        self._manifest['response_contract_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def evaluate(self, index=0, *, seed=0, strength=1.):
        """Return the declared numerical native response; never media or tokens."""
        return super().evaluate(index, seed=seed, strength=strength)


def response_backend(name):
    """Resolve the separate response registry, without an image fallback."""
    matches = list(entry_points(group='particle_sliders.response_backends', name=name))
    if len(matches) != 1:
        raise BackendUnavailable(f'expected one installed {name!r} response backend, found {len(matches)}')
    try:
        result = matches[0].load()
    except ImportError as exc:
        raise BackendUnavailable(name + ': install its isolated native backend environment') from exc
    if not isinstance(result, ResponseBackend) or result.name != name:
        raise BackendUnavailable('response registration must expose a matching ResponseBackend')
    return result


def open_response(config, *, resume=None, artifact=None, operation='train', **request):
    """Shared preparation, step/EMA, exact resume, observation and cleanup."""
    return _open_registered(config, resume=resume, artifact=artifact, operation=operation,
        request=request, resolve=response_backend, training_schema=TRAINING, inference_schema=INFERENCE)
