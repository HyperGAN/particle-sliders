"""Public image-session contract. Native factories own geometry; PG owns the game.

Importing this module never imports a product, downloads weights or selects a
GPU. See IMAGE-LIFECYCLE.md for packet/cohort and backend isolation contracts.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import entry_points
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import torch

from .pg09 import ConditionalPG09, _hash, _validate_module

API = 'particle-sliders-image-session-v1'
TRAINING = API + '-training'
INFERENCE = API + '-inference'


class LifecycleError(RuntimeError):
    """Base error for the public lifecycle boundary."""


class UnsupportedOperation(LifecycleError):
    """The declared backend cannot execute the requested operation."""


class BackendUnavailable(LifecycleError):
    """Install/select the backend in its own compatible dependency environment."""


class CheckpointError(LifecycleError):
    """Invalid packet or incompatible cohort; no implicit migration is supported."""


@dataclass(frozen=True, init=False)
class SessionConfig:
    backend: str
    cohort: str
    options_json: str

    def __init__(self, backend, cohort, **options):
        if not isinstance(backend, str) or not backend or not isinstance(cohort, str) or not cohort:
            raise ValueError('backend and a nonempty new cohort name are required')
        encoded = json.dumps(options, sort_keys=True, allow_nan=False)
        object.__setattr__(self, 'backend', backend)
        object.__setattr__(self, 'cohort', cohort)
        object.__setattr__(self, 'options_json', encoded)

    @property
    def options(self):
        return json.loads(self.options_json)

    def as_dict(self):
        return dict(backend=self.backend, cohort=self.cohort, options=self.options)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'backend', 'cohort', 'options'}:
            raise ValueError('config requires backend, cohort and options')
        return cls(value['backend'], value['cohort'], **value['options'])


@dataclass(frozen=True)
class Capabilities:
    generator_arm: str
    render: bool
    strength_min: float | None = 0.
    strength_max: float | None = 1.
    reference_images: bool = False
    references_required: bool = False
    base_precisions: tuple = ('float32', 'bfloat16')
    render_outputs: tuple = ('pil', 'latent')
    qualification: str = 'CPU conformance only; pretrained/GPU quality unqualified'
    operations: tuple = ('train', 'resume', 'evaluate', 'export', 'response', 'render')
    unavailable_reasons: tuple = ()

    def refusal(self, operation):
        return dict(self.unavailable_reasons).get(operation, operation + ' is unavailable for this backend')

    def check(self, operation='train', *, strength=1., references=False, precision=None, output_type='pil'):
        if operation not in ('train', 'resume', 'evaluate', 'export', 'response', 'render'):
            raise UnsupportedOperation('unknown operation: ' + operation)
        if operation not in self.operations or operation == 'render' and not self.render:
            raise UnsupportedOperation(self.refusal(operation))
        if (not math.isfinite(strength) or self.strength_min is not None and strength < self.strength_min
                or self.strength_max is not None and strength > self.strength_max):
            raise UnsupportedOperation(f'strength must be in [{self.strength_min}, {self.strength_max}]')
        if references and not self.reference_images:
            raise UnsupportedOperation('reference-image conditioning is unsupported')
        if operation == 'render' and self.references_required and not references:
            raise UnsupportedOperation('native edit rendering requires reference images')
        if operation == 'render' and output_type not in self.render_outputs:
            raise UnsupportedOperation('unsupported native render output: ' + output_type)
        if not self.base_precisions:
            raise UnsupportedOperation(self.refusal(operation))
        if precision is None:
            precision = self.base_precisions[0]
        if precision not in self.base_precisions:
            raise UnsupportedOperation('unsupported base precision: ' + precision)


@dataclass
class Binding:
    """A factory transfers ownership of one prepared game and cleanup callback."""
    game: ConditionalPG09
    close: Callable
    render: Callable | None = None
    inference: object | None = None
    before_update: Callable | None = None
    observation: Callable | None = None


@dataclass(frozen=True)
class Backend:
    name: str
    capabilities: Capabilities
    prepare: Callable
    validate: Callable
    configure_capabilities: Callable | None = None
    preflight: Callable | None = None

    def describe(self, config):
        return self.configure_capabilities(config.options) if self.configure_capabilities else self.capabilities

    def open(self, config, *, operation='train', strength=1., references=False, output_type=None, native_request=None, **staged):
        if config.backend != self.name:
            raise ValueError('backend configuration differs')
        capabilities = self.describe(config)
        output_type = output_type or next(iter(capabilities.render_outputs), None)
        request = dict(operation=operation, strength=strength, references=references, output_type=output_type,
                       native=deepcopy(native_request or {}))
        capabilities.check(operation, strength=strength, references=references,
                           precision=config.options.get('base_precision'), output_type=output_type)
        options = config.options
        for key, default, low in (('steps', 4, 1), ('batch', 2, 2), ('seed', 7, 0)):
            if type(options.get(key, default)) is not int or options.get(key, default) < low:
                raise ValueError(key + ' has an invalid lifecycle budget')
        if options.get('batch', 2) > 512 or options.get('seed', 7) >= 2**63 - 100:
            raise ValueError('batch or seed exceeds the public PG budget')
        decay = options.get('ema_decay')
        if decay is not None and (not math.isfinite(decay) or not 0 <= decay < 1):
            raise ValueError('EMA decay must be in [0,1)')
        try:
            self.validate(options)  # No weight/device allocation in these hooks.
            if self.preflight is not None:
                self.preflight(deepcopy(options), deepcopy(request), staged)
        except ImportError as exc:
            raise BackendUnavailable(self.name + ': missing backend dependency; use its isolated environment') from exc
        from .pg09 import _released_sources, PG_VERSION
        from importlib.metadata import version
        if version('particlegan') != PG_VERSION:
            raise BackendUnavailable('install the exact released ParticleGAN ' + PG_VERSION)
        _released_sources()
        devices = _rng_devices(config.options, staged)
        try:
            with torch.random.fork_rng(devices=devices):
                binding = self.prepare(config.options, **staged)
        except ImportError as exc:
            raise BackendUnavailable(self.name + ': missing backend dependency; use its isolated environment') from exc
        try:
            session = SliderSession(config, capabilities, binding)
            import inspect
            session._manifest['factory_sha256'] = hashlib.sha256(Path(inspect.getsourcefile(self.prepare)).read_bytes()).hexdigest()
            return session
        except BaseException:
            if callable(getattr(binding, 'close', None)):
                binding.close()
            raise


def _rng_devices(*values):
    """Discover explicit CUDA devices, including nested native arguments/staging."""
    devices, seen = set(), set()
    def visit(value, device_key=False):
        if not isinstance(value, (str, int, float, bool, type(None), torch.device)):
            if id(value) in seen:
                return
            seen.add(id(value))
        if isinstance(value, torch.device) or device_key and isinstance(value, str):
            device = torch.device(value)
            if device.type == 'cuda':
                devices.add(device.index if device.index is not None else torch.cuda.current_device())
        elif isinstance(value, torch.Tensor):
            if value.is_cuda:
                devices.add(value.device.index)
        elif isinstance(value, torch.nn.Module):
            for tensor in (*value.parameters(), *value.buffers()):
                visit(tensor)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, key in ('device', 'torch_device', 'execution_device'))
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif hasattr(value, 'device'):
            visit(value.device, True)
    for value in values:
        visit(value)
    return sorted(devices)


def _seed_observation(seed, devices):
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    for device in devices:
        torch.cuda.set_rng_state(torch.Generator(device=f'cuda:{device}').manual_seed(seed).get_state(), device)


@contextmanager
def _network_observation(network, seed, *, devices, observation=None):
    """Preserve every registered adapter buffer, including nonpersistent ones.

    Buffer references/registrations are retained independently of state_dict,
    which intentionally excludes nonpersistent buffers. Native state outside
    this network remains the product observation context's responsibility.
    """
    state = deepcopy(network.state_dict())
    modes = [(module, module.training) for module in network.modules()]
    buffers = [(module, dict(module._buffers), set(module._non_persistent_buffers_set))
               for module in network.modules()]
    values = {id(buffer): (buffer, buffer.detach().clone())
              for _, registered, _ in buffers for buffer in registered.values() if buffer is not None}
    def restore_buffers():
        with torch.no_grad():
            for buffer, saved in values.values():
                if (buffer.shape, buffer.dtype, buffer.device) == (saved.shape, saved.dtype, saved.device):
                    buffer.copy_(saved)
                else:
                    buffer.data = saved.clone()
            for module, registered, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(registered)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(nonpersistent)
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            _seed_observation(seed, devices)
            network.eval()
            with observation() if observation else nullcontext():
                yield
    finally:
        try:
            restore_buffers()
            network.load_state_dict(state, strict=True)
        finally:
            try:
                restore_buffers()
            finally:
                for module, training in modes:
                    module.training = training


def native_cleanup(root):
    """Capture borrowed module modes/gradient flags/checkpoint settings before binding.

    Factories call this before changing a supplied model, and compose the returned
    callback with removal of their owned hooks. It is safe to call repeatedly.
    """
    checkpoint = checkpoint_cleanup(root)
    modes = [(module, module.training) for module in root.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in root.parameters()]
    def restore():
        checkpoint()
        for module, mode in modes:
            module.training = mode
        for parameter, grad in gradients:
            parameter.requires_grad_(grad)
    return restore


@contextmanager
def new_directory(path):
    """Own a new cache directory; remove all partial output if capture fails."""
    import shutil
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    except BaseException:
        shutil.rmtree(path)
        raise


def backend(name):
    """Resolve exactly one installed backend; native imports stay in its factory."""
    matches = list(entry_points(group='particle_sliders.image_backends', name=name))
    if len(matches) != 1:
        raise BackendUnavailable(f'expected one installed {name!r} image backend, found {len(matches)}')
    try:
        result = matches[0].load()
    except ImportError as exc:
        raise BackendUnavailable(f'{name}: incompatible or missing backend dependency; use its isolated environment') from exc
    if not isinstance(result, Backend) or result.name != name:
        raise BackendUnavailable('backend registration must expose a matching Backend contract')
    return result


def open_session(config, *, resume=None, artifact=None, operation='train', **request):
    """Load/preflight a packet before constructing the native backend."""
    if resume is not None and artifact is not None:
        raise ValueError('choose a training checkpoint or inference artifact')
    packet = read_packet(resume or artifact) if resume is not None or artifact is not None else None
    if packet is not None:
        expected = TRAINING if resume is not None else INFERENCE
        if packet['schema'] != expected or packet['manifest']['config'] != config.as_dict():
            raise CheckpointError('packet kind/config/cohort differs; use its original pin or explicit migration')
    session = backend(config.backend).open(config, operation=operation, **request)
    try:
        if resume is not None:
            session.restore(packet)
        elif artifact is not None:
            session.load_inference(packet)
        return session
    except BaseException:
        session.close()
        raise


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_save(path, value):
    """Write a weights-only-loadable packet via same-filesystem atomic replace."""
    _atomic_write(path, lambda stream: torch.save(value, stream))


def atomic_json(path, value):
    """Atomically write a finite JSON manifest."""
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _atomic_write(path, lambda stream: stream.write(data))


def read_packet(path):
    try:
        packet = torch.load(path, map_location='cpu', weights_only=True)
        _check_packet(packet)
        return packet
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError('invalid session packet; legacy formats require their original entrypoint') from exc


def _check_packet(packet):
    if (not isinstance(packet, dict) or packet.get('schema') not in (TRAINING, INFERENCE)
            or set(packet) != {'schema', 'manifest', 'payload', 'sha256'}):
        raise CheckpointError('unsupported session packet version; no automatic migration')
    try:
        digest = _hash({key: value for key, value in packet.items() if key != 'sha256'})
    except (ValueError, TypeError) as exc:
        raise CheckpointError('invalid session packet values') from exc
    if packet['sha256'] != digest:
        raise CheckpointError('session packet checksum differs')


class EMA:
    """EMA of the actual network, including particles exactly once via ownership."""
    def __init__(self, network, decay=None):
        if decay is not None and (not math.isfinite(decay) or not 0 <= decay < 1):
            raise ValueError('EMA decay must be in [0,1), or None')
        self.decay = decay
        self.values = deepcopy(network.state_dict()) if decay is not None else None

    @torch.no_grad()
    def update(self, network):
        if self.values is not None:
            for key, value in network.state_dict().items():
                if value.is_floating_point():
                    self.values[key].lerp_(value, 1 - self.decay)
                else:
                    self.values[key].copy_(value)

    def validate(self, network, state):
        if not isinstance(state, dict) or set(state) != {'decay', 'values'} or state['decay'] != self.decay:
            raise CheckpointError('EMA configuration differs')
        if self.decay is None:
            if state['values'] is not None:
                raise CheckpointError('unexpected EMA values')
        else:
            _validate_module(network, state['values'])

    def state_dict(self):
        return deepcopy(dict(decay=self.decay, values=self.values))

    def load_state_dict(self, network, state):
        self.validate(network, state)
        self.values = (None if self.decay is None else
                       {key: value.to(network.state_dict()[key]).clone() for key, value in state['values'].items()})


def advance(game, *, ema=None):
    """One PG update with scale captured through backward; preserve native math."""
    with game.network.scaled(1.):
        metrics = game.step()
    if ema is not None:
        ema.update(game.network)
    return metrics


def updates(game, stop_after, *, ema=None):
    """Compatibility iterator for product writers retaining historical envelopes."""
    if type(stop_after) is not int or not game.completed_steps < stop_after <= game.recipe.total_steps:
        raise ValueError('stop_after must exceed completed steps within the fixed budget')
    while game.completed_steps < stop_after:
        yield advance(game, ema=ema)


def checkpoint_cleanup(root):
    """Capture caller-owned checkpoint hooks before a native adapter binds them."""
    keys = ('gradient_checkpointing', '_gradient_checkpointing_func')
    saved = [(module, {key: getattr(module, key) for key in keys if hasattr(module, key)})
             for module in root.modules()]
    def restore():
        for module, previous in saved:
            for key in keys:
                if key in previous:
                    setattr(module, key, previous[key])
                elif hasattr(module, key):
                    delattr(module, key)
    return restore


class SliderSession:
    def __init__(self, config, capabilities, binding):
        if not isinstance(binding.game, ConditionalPG09):
            raise TypeError('image sessions require ConditionalPG09')
        self.config, self.capabilities, self.binding = config, capabilities, binding
        self.game = binding.game
        actual_arm = 'routed_particle_gan' if self.game.prior is not None else 'adapter_gan'
        if capabilities.generator_arm != actual_arm or capabilities.render != (binding.render is not None):
            raise ValueError('backend implementation contradicts its declared capabilities')
        self.ema = EMA(self.game.network, config.options.get('ema_decay'))
        self.closed = False
        self.inference_only = False
        self.last_metrics = None
        self._manifest = dict(api=API, config=config.as_dict(), capabilities=asdict(capabilities),
            lifecycle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            game_signature=deepcopy(self.game._signature))

    @property
    def manifest(self):
        """Snapshot for external run observers; caller edits cannot rebind a cohort."""
        return deepcopy(self._manifest)

    def _open(self):
        if self.closed:
            raise LifecycleError('session is closed')

    def _training(self):
        self._open()
        if self.inference_only:
            raise UnsupportedOperation('inference-loaded sessions cannot train, resume or checkpoint')

    def step(self):
        self._training()
        self.capabilities.check('train')
        if self.binding.before_update is not None:
            state = self.game.state_dict()
            restore_native = native_cleanup(self.game.network)
            try:
                with torch.random.fork_rng(devices=_rng_devices(self.game.network)):
                    allowed = self.binding.before_update(dict(completed_steps=self.game.completed_steps,
                                                              manifest=self.manifest))
                if allowed is False:
                    raise UnsupportedOperation('native update admission rejected')
            except BaseException:
                self.game.load_state_dict(state)
                restore_native()
                raise
        try:
            stats = advance(self.game, ema=self.ema)
            self.last_metrics = deepcopy(stats)
            return stats
        except BaseException:
            self.close()
            raise

    def train(self, *, stop_after=None, on_step=None):
        self._training()
        self.capabilities.check('train')
        stop = self.game.recipe.total_steps if stop_after is None else stop_after
        if type(stop) is not int or not self.game.completed_steps < stop <= self.game.recipe.total_steps:
            raise ValueError('stop_after must exceed completed steps within the fixed budget')
        while self.game.completed_steps < stop:
            metrics = self.step()
            if on_step is not None:
                try:
                    on_step(deepcopy(metrics))
                except BaseException:
                    self.close()
                    raise
        return deepcopy(self.last_metrics)

    def _packet(self, schema, payload):
        result = dict(schema=schema, manifest=deepcopy(self.manifest), payload=payload)
        result['sha256'] = _hash(result)
        return result

    def checkpoint(self, path=None):
        self._training()
        result = self._packet(TRAINING, dict(game=self.game.state_dict(), ema=self.ema.state_dict(),
                                             last_metrics=deepcopy(self.last_metrics)))
        if path is not None:
            atomic_save(path, result)
        return result

    def _validate(self, packet, schema):
        self._open()
        _check_packet(packet)
        if packet['schema'] != schema or packet['manifest'] != self.manifest:
            raise CheckpointError('packet kind, source, capabilities or cohort differs; no implicit rebinding')
        return packet['payload']

    def restore(self, packet):
        self._training()
        self.capabilities.check('resume')
        payload = self._validate(packet, TRAINING)
        try:
            if not isinstance(payload, dict) or set(payload) != {'game', 'ema', 'last_metrics'}:
                raise CheckpointError('invalid training packet fields')
            metrics = payload['last_metrics']
            if ((payload['game']['completed_steps'] == 0 and metrics is not None) or
                    (payload['game']['completed_steps'] > 0 and
                     (not isinstance(metrics, dict) or metrics.get('step') != payload['game']['completed_steps']))):
                raise CheckpointError('metrics step differs from checkpoint')
            _hash(metrics)  # Reject nonfinite or nonserializable metrics before mutation.
            self.ema.validate(self.game.network, payload['ema'])
            # PG validates source, network, optimizer, RNG and coverage BEFORE mutation.
            self.game.load_state_dict(payload['game'])
            self.ema.load_state_dict(self.game.network, payload['ema'])
        except (ValueError, KeyError, TypeError) as exc:
            raise CheckpointError('invalid training/EMA state') from exc
        self.last_metrics = deepcopy(payload['last_metrics'])
        self.inference_only = False

    def export(self, path=None, *, selection='raw'):
        self._open()
        self.capabilities.check('export')
        if selection not in ('raw', 'ema') or selection == 'ema' and self.ema.values is None:
            raise ValueError('select raw or a configured EMA')
        state = self.game.export_state_dict()
        if selection == 'ema':
            state['network'] = deepcopy(self.ema.values)
        result = self._packet(INFERENCE, dict(adapter=state, selection=selection))
        if path is not None:
            atomic_save(path, result)
        return result

    def export_inference(self, path=None, *, selection='raw', specification=None):
        """Native-only v2 artifact; requires an explicit product inference spec."""
        from .inference import export_inference
        return export_inference(self, path, selection=selection, specification=specification)

    def load_inference(self, packet):
        payload = self._validate(packet, INFERENCE)
        if not isinstance(payload, dict) or set(payload) != {'selection', 'adapter'}:
            raise CheckpointError('invalid inference packet fields')
        if payload['selection'] not in ('raw', 'ema'):
            raise CheckpointError('unknown inference selection')
        try:
            self.game.load_export_state_dict(payload['adapter'])
        except (ValueError, KeyError, TypeError) as exc:
            raise CheckpointError('invalid inference adapter') from exc
        self.inference_only = True

    @contextmanager
    def _observation(self, seed, request=None):
        self._open()
        network = self.game.network
        devices = _rng_devices(network, self.game.device, self.config.options, request)
        with _network_observation(network, seed, devices=devices, observation=self.binding.observation):
            yield

    def evaluate(self, index=0, *, seed=0, strength=1.):
        """Return a native velocity response, never an image or a generic solver."""
        self.capabilities.check('evaluate', strength=strength)
        self.capabilities.check('response', strength=strength)
        if type(index) is not int or not 0 <= index < len(self.game.rows):
            raise ValueError('response row index out of bounds')
        from dataclasses import replace
        prepared = self.game.rows[index]
        prepared = replace(prepared, row=replace(prepared.row, scale=strength))
        with self._observation(seed):
            return self.game.backend.generated(prepared).detach().clone()

    def render(self, *, strength=1., seed=0, images=None, **native):
        self.capabilities.check('render', strength=strength, references=bool(images),
                                precision=self.capabilities.base_precisions[0],
                                output_type=native.get('output_type', next(iter(self.capabilities.render_outputs), None)))
        native.setdefault('output_type', next(iter(self.capabilities.render_outputs), None))
        with self._observation(seed, dict(native, images=images)):
            return self.binding.render(strength=strength, seed=seed, images=images, **native)

    def close(self):
        if not self.closed:
            self.closed = True
            self.binding.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
