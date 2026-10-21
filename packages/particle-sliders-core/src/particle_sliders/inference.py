"""Data-independent native inference, separate from PG training/resume state.

Products register a native-only factory. No training factory, prepared rows,
critic, optimizer or ConditionalPG09 instance is constructed by this module.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import entry_points
import inspect
import json
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .lifecycle import (
    BackendUnavailable, Capabilities, CheckpointError, LifecycleError,
    UnsupportedOperation, atomic_save, _rng_devices, _network_observation, capabilities_from_dict,
)
from .pg09 import _architectures, _hash, _identity, _validate_module

INFERENCE_V2 = 'particle-sliders-native-inference-v2'
IDENTITY_V1 = 'particle-sliders-native-identity-v1'


def _json(value):
    return json.loads(json.dumps(_identity(value), sort_keys=True, allow_nan=False))


def _source(owner):
    source = inspect.getsourcefile(owner if inspect.isfunction(owner) else type(owner))
    if source is None:
        raise ValueError('native inference implementations require inspectable source')
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def native_identity(native, *, model_identity, protocol):
    """Describe actual native objects without training data or a critic.

    model_identity must come from the product's verified base loader, not the
    training backend's cache-augmented metadata. Mode/gradient flags are mutable
    observation state, so they do not change the inference architecture.
    """
    if not isinstance(model_identity, dict) or not model_identity or not isinstance(protocol, str) or not protocol:
        raise ValueError('verified native model identity and versioned protocol are required')
    forbidden = {'rows', 'teacher_cache', 'frozen_cache', 'optimizer', 'critic', 'training_data'}
    def check(value):
        if isinstance(value, dict):
            if forbidden & value.keys():
                raise ValueError('native model identity must not contain training state/data')
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
    check(model_identity)
    if not isinstance(getattr(native, 'network', None), nn.Module):
        raise TypeError('native inference requires the actual adapter network')
    architectures = _architectures(native, None)
    for architecture in architectures.values():
        architecture['tree'] = [(name, cls, extra, {key: value for key, value in flags.items()
                                                    if key != 'gradient_checkpointing'})
                                for name, cls, extra, mode, flags in architecture['tree']]
        architecture['parameters'] = {key: (shape, dtype)
                                     for key, (shape, dtype, grad) in architecture['parameters'].items()}
    objects, seen, sources = [native, native.network], set(), {}
    for _ in range(4):
        for owner in tuple(objects):
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            sources[type(owner).__module__] = _source(owner)
            for name in ('backend', 'runtime', 'model', 'transformer', 'slider', 'adapter'):
                child = getattr(owner, name, None)
                if child is not None and not isinstance(child, (str, bytes, int, float, bool)):
                    objects.append(child)
            if isinstance(owner, nn.Module):
                objects.extend(owner.modules())
    return _json(dict(schema=IDENTITY_V1, model=model_identity, protocol=protocol,
        architectures=architectures, sources=sources,
        network_schema={key: (list(value.shape), str(value.dtype))
                        for key, value in native.network.state_dict().items()}))


@dataclass(frozen=True, init=False)
class InferenceSpec:
    """Immutable native-only export descriptor, produced by InferenceBackend."""
    _json: str

    def __init__(self, value):
        expected = {'backend', 'configuration', 'capabilities', 'response', 'identity', 'factory_sources'}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError('invalid inference specification fields')
        if not isinstance(value['backend'], str) or not value['backend']:
            raise ValueError('native inference backend name is required')
        if not isinstance(value['configuration'], dict) or type(value['response']) is not bool:
            raise ValueError('native configuration and response capability are required')
        if not isinstance(value['identity'], dict) or value['identity'].get('schema') != IDENTITY_V1:
            raise ValueError('unsupported native identity version')
        if set(value['factory_sources']) != {'prepare', 'validate', 'preflight'}:
            raise ValueError('native factory source identities are required')
        if set(value['identity']) != {'schema', 'model', 'protocol', 'architectures', 'sources', 'network_schema'}:
            raise ValueError('invalid native identity fields')
        for digest in (*value['identity']['sources'].values(), *value['factory_sources'].values()):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('invalid native source digest')
        capabilities_from_dict(value['capabilities'])
        object.__setattr__(self, '_json', json.dumps(_json(value), sort_keys=True, allow_nan=False))

    def as_dict(self):
        return json.loads(self._json)

    @property
    def configuration(self):
        return self.as_dict()['configuration']

    @property
    def identity(self):
        return self.as_dict()['identity']


@dataclass
class InferenceBinding:
    network: nn.Module
    identity: dict
    close: Callable
    render: Callable | None = None
    response: Callable | None = None
    observation: Callable | None = None


@dataclass(frozen=True)
class InferenceBackend:
    """Product-owned native factory: prepare(local_options, spec) -> binding."""
    name: str
    capabilities: Capabilities
    prepare: Callable
    validate: Callable
    response: bool = True
    configure_capabilities: Callable | None = None
    preflight: Callable | None = None

    def describe(self, configuration):
        return self.configure_capabilities(configuration) if self.configure_capabilities else self.capabilities

    def _sources(self):
        return dict(prepare=_source(self.prepare), validate=_source(self.validate),
                    preflight=_source(self.preflight) if self.preflight else None)

    def specification(self, native, *, model_identity, protocol, configuration=None):
        configuration = _json(configuration or {})
        return InferenceSpec(dict(backend=self.name, configuration=configuration,
            capabilities=asdict(self.describe(configuration)), response=self.response,
            identity=native_identity(native, model_identity=model_identity, protocol=protocol),
            factory_sources=self._sources()))


def inference_backend(name):
    matches = list(entry_points(group='particle_sliders.inference_backends', name=name))
    if len(matches) != 1:
        raise BackendUnavailable(f'expected one native-only {name!r} inference backend, found {len(matches)}; '
                                 'the training factory is never used as a fallback')
    try:
        result = matches[0].load()
    except ImportError as exc:
        raise BackendUnavailable(name + ': install its isolated native environment') from exc
    if not isinstance(result, InferenceBackend) or result.name != name:
        raise BackendUnavailable('native registration must expose a matching InferenceBackend')
    return result


def export_inference(session, path=None, *, selection='raw', specification=None):
    """Export selected adapter values without embedding game/config/data state."""
    session._open()
    session.capabilities.check('export')
    if session.inference_only:
        raise UnsupportedOperation('re-export native inference through its original artifact receipt')
    spec = specification if specification is not None else session.binding.inference
    if not isinstance(spec, InferenceSpec):
        raise UnsupportedOperation('the product must provide a native-only InferenceSpec before v2 export')
    if spec.as_dict()['backend'] != session.config.backend:
        raise ValueError('native inference specification backend differs')
    if selection not in ('raw', 'ema') or selection == 'ema' and session.ema.values is None:
        raise ValueError('select raw or a configured EMA')
    # Also refuses failed/incomplete PG updates, while leaving training identity intact.
    state = session.game.export_state_dict()
    values = session.ema.values if selection == 'ema' else state['network']
    _validate_module(session.game.network, values)
    schema = _json({key: (list(value.shape), str(value.dtype)) for key, value in values.items()})
    if spec.identity['network_schema'] != schema:
        raise ValueError('inference specification network schema differs from the trained adapter')
    # A spec is tied to the actual adapter/native architecture, not an arbitrary map.
    identity = native_identity(session.game.backend, model_identity=spec.identity['model'],
                               protocol=spec.identity['protocol'])
    if identity != spec.identity:
        raise ValueError('inference specification differs from the actual trained native backend')
    packet = dict(schema=INFERENCE_V2, specification=spec.as_dict(),
        origin=dict(cohort=session.config.cohort, training_manifest_sha256=_hash(session.manifest),
                    completed_steps=session.game.completed_steps, generator_arm=session.capabilities.generator_arm,
                    pg_version=state['config']['pg_version']),
        selection=selection, network={key: value.detach().cpu().clone() for key, value in values.items()})
    packet['sha256'] = _hash(packet)
    if path is not None:
        atomic_save(path, packet)
    return packet


def _validate_packet(packet):
    expected = {'schema', 'specification', 'origin', 'selection', 'network', 'sha256'}
    try:
        if not isinstance(packet, dict) or set(packet) != expected or packet['schema'] != INFERENCE_V2:
            raise CheckpointError('expected native inference v2; legacy v1 artifacts require their original loader')
        if packet['selection'] not in ('raw', 'ema'):
            raise CheckpointError('unknown raw/EMA inference selection')
        origin = packet['origin']
        if (not isinstance(origin, dict) or set(origin) != {'cohort', 'training_manifest_sha256', 'completed_steps',
                                                         'generator_arm', 'pg_version'}
                or not isinstance(origin['cohort'], str) or not origin['cohort']
                or type(origin['completed_steps']) is not int or origin['completed_steps'] < 0
                or origin['generator_arm'] not in ('routed_particle_gan', 'adapter_gan')):
            raise CheckpointError('invalid inference origin receipt')
        spec = InferenceSpec(packet['specification'])
        if packet['sha256'] != _hash({key: value for key, value in packet.items() if key != 'sha256'}):
            raise CheckpointError('native inference packet checksum differs')
        return spec
    except CheckpointError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise CheckpointError('invalid native inference packet') from exc


def read_inference(path):
    try:
        packet = torch.load(path, map_location='cpu', weights_only=True)
        _validate_packet(packet)
        return packet
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError('invalid native inference file') from exc


def open_inference(artifact, *, options=None, backend=None, operation='render', strength=1.,
                   references=False, output_type=None, native_request=None):
    """Open an export using only native local options; never reconstruct training."""
    packet = read_inference(artifact) if isinstance(artifact, (str, Path)) else deepcopy(artifact)
    spec = _validate_packet(packet)
    descriptor = spec.as_dict()
    native_backend = backend or inference_backend(descriptor['backend'])
    if not isinstance(native_backend, InferenceBackend) or native_backend.name != descriptor['backend']:
        raise BackendUnavailable('native inference registration differs')
    if operation not in ('render', 'response'):
        raise UnsupportedOperation('native inference accepts render or response; training/resume remain separate')
    if operation == 'response' and not descriptor['response']:
        raise UnsupportedOperation('native response evaluation is unavailable')
    options = _json(options or {})
    capabilities = native_backend.describe(spec.configuration)
    output_type = output_type or next(iter(capabilities.render_outputs), None)
    capabilities.check(operation, strength=strength, references=references,
                       precision=options.get('base_precision'), output_type=output_type)
    if (_json(asdict(capabilities)) != descriptor['capabilities'] or native_backend.response != descriptor['response']
            or native_backend._sources() != descriptor['factory_sources']):
        raise CheckpointError('native factory source, protocol or capabilities differ')
    if capabilities.generator_arm != packet['origin']['generator_arm']:
        raise CheckpointError('inference generator arm differs from origin')
    try:
        native_backend.validate(options, spec)
        if native_backend.preflight is not None:
            native_backend.preflight(deepcopy(options), dict(operation=operation, strength=strength,
                references=references, output_type=output_type, native=deepcopy(native_request or {})), spec)
        devices = _rng_devices(options)
        with torch.random.fork_rng(devices=devices):
            binding = native_backend.prepare(options, spec)
    except ImportError as exc:
        raise BackendUnavailable(native_backend.name + ': missing isolated native dependency') from exc
    try:
        if not isinstance(binding, InferenceBinding) or not isinstance(binding.network, nn.Module):
            raise TypeError('native factory must return InferenceBinding with its actual adapter network')
        if (capabilities.render != (binding.render is not None)
                or native_backend.response != (binding.response is not None)):
            raise ValueError('native callbacks contradict declared capabilities')
        if _json(binding.identity) != spec.identity:
            raise CheckpointError('native model, architecture, source or protocol differs')
        _validate_module(binding.network, packet['network'])  # Entire state before mutation.
        binding.network.load_state_dict(packet['network'], strict=True)
        def preflight(operation, strength, references, output_type, native):
            if native_backend.preflight is not None:
                native_backend.preflight(deepcopy(options), dict(operation=operation, strength=strength,
                    references=references, output_type=output_type, native=deepcopy(native)), spec)
        return InferenceSession(binding, capabilities, packet, output_type, preflight, rng_devices=devices)
    except BaseException:
        if callable(getattr(binding, 'close', None)):
            binding.close()
        raise


class InferenceSession:
    """Native base/adapter only; no game, rows, critic or optimizer attributes."""
    def __init__(self, binding, capabilities, packet, output_type, preflight, *, rng_devices=()):
        self.binding, self.capabilities = binding, capabilities
        self._receipt = {key: deepcopy(value) for key, value in packet.items() if key != 'network'}
        self.closed = False
        self._output_type = output_type
        self._preflight = preflight
        self._rng_devices = tuple(rng_devices)

    @property
    def selection(self):
        return self._receipt['selection']

    @property
    def manifest(self):
        return deepcopy(self._receipt)

    @contextmanager
    def _observation(self, seed, request=None):
        if self.closed:
            raise LifecycleError('inference session is closed')
        network = self.binding.network
        devices = sorted(set(self._rng_devices) | set(_rng_devices(network, request)))
        with _network_observation(network, seed, devices=devices, observation=self.binding.observation):
            yield

    def render(self, *, strength=1., seed=0, images=None, output_type=None, **native):
        output_type = output_type or self._output_type
        self.capabilities.check('render', strength=strength, references=bool(images),
                                output_type=output_type)
        self._preflight('render', strength, bool(images), output_type, dict(native, images=images))
        with self._observation(seed, dict(native, images=images)):
            return self.binding.render(strength=strength, seed=seed, images=images, output_type=output_type, **native)

    def response(self, *, strength=1., seed=0, **native):
        self.capabilities.check('response', strength=strength)
        if self.binding.response is None:
            raise UnsupportedOperation('native response callback is unavailable')
        self._preflight('response', strength, False, None, native)
        with self._observation(seed, native):
            return self.binding.response(strength=strength, seed=seed, **native)

    def train(self, **kwargs):
        raise UnsupportedOperation('native inference artifact cannot resume training')

    step = checkpoint = restore = train

    def close(self):
        if not self.closed:
            self.closed = True
            self.binding.close()

    def __enter__(self):
        if self.closed:
            raise LifecycleError('inference session is closed')
        return self

    def __exit__(self, *exc):
        self.close()
