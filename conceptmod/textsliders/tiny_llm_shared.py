"""Separate TinyQwen shared PG/HG/HNDL cohort; no historical objective migration.

Only bounded CPU fixtures are executable with the reviewed release-auto core.
Native Qwen3-0.6B hidden width1024 is refused before model loading or hooks.
All losses, penalties, optimizers, continuation and inference belong to core.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLParticleProjection, verified_builder
from particle_sliders.inference import InferenceBackend, InferenceBinding
from particle_sliders.lifecycle import Binding, SessionConfig, native_cleanup
from particle_sliders.pg094 import ConditionalPG094, PG_SOURCE_PROFILE, PG_VERSION, engine_options
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter, AudioRow, audio_conditions
from particle_sliders.response import ResponseBackend, ResponseCapabilities

from .tiny_llm_backend import ATTN_CLASS_NAMES, LORA_LINEAR_NAMES, TinyLLMBackend

NAME = 'tinyqwen-shared-preview'
COHORT = 'tinyqwen-pg094-hypergan-hndl-v1-cpu-fixture'
PROTOCOL = 'tinyqwen-neutral-to-positive-last-hidden-v1'
CRITIC_GRAPH = '''joined = concat(response, condition, axis=1)
features = linear(joined, 16, name="hidden")
features = silu(features)
linear(features, 1, name="score")
'''
ROOT = Path(__file__).parent


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime():
    """Admit exactly the installed reviewed wheel closure and real HG builder."""
    import particle_sliders
    if version('particle-sliders-core') != '0.12.0':
        raise ValueError('TinyQwen preview requires its separate exact core0.12 environment')
    lock = json.loads((ROOT / 'tiny_llm_shared.lock.json').read_text())
    core_root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(core_root)): _digest(p) for p in core_root.rglob('*') if p.suffix in ('.py', '.json', '.hndl')}
    if actual != lock['core_sources']:
        raise ValueError('TinyQwen preview core source closure differs; use original cohort wheel')
    builder, identity = verified_builder()
    return builder, dict(identity, tinyqwen_lock_sha256=_digest(ROOT / 'tiny_llm_shared.lock.json'),
                         tinyqwen_critic_graph_sha256=hashlib.sha256(
        CRITIC_GRAPH.encode()).hexdigest(), tinyqwen_binding_sha256=_digest(__file__))


class TinyQwenHNDLCritic(nn.Module):
    """Product-bound Linear16/SiLU/Linear1 via the verified HyperGAN builder."""
    score_bound = 0

    def __init__(self, width, condition_width):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (width, condition_width)):
            raise ValueError('positive native response and condition widths required')
        builder, identity = verify_runtime()
        self.graph = builder(CRITIC_GRAPH,
            input_shape={'response': ['B', width], 'condition': ['B', condition_width]},
            output_shape=['B', 1])
        self._identity = dict(identity, response_shape=[1, width],
                              condition_shape=[1, condition_width], plan=CRITIC_GRAPH)

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)


def _targets(model):
    targets = {}
    seen = set()
    for name, module in model.named_modules():
        if type(module).__name__ not in ATTN_CLASS_NAMES:
            continue
        for leaf in LORA_LINEAR_NAMES:
            child = getattr(module, leaf, None)
            if not isinstance(child, nn.Linear) or id(child) in seen:
                raise ValueError('expected unique native Qwen q/k/v/o Linear targets')
            if isinstance(getattr(child.forward, '__self__', None), HNDLParticleProjection) or (
                    getattr(child.forward, '__func__', None) is not nn.Linear.forward):
                raise ValueError('TinyQwen target already wrapped; refusing mixed cohorts')
            seen.add(id(child))
            targets[f'{name}.{leaf}' if name else leaf] = child
    if not targets:
        raise ValueError('expected native Qwen attention q/k/v/o targets')
    return targets


class TinyQwenHNDLSlider(nn.Module):
    """One real shared bank and shared HNDL projections; transactional attachment."""
    def __init__(self, model, *, attach=True):
        super().__init__()
        from particle_sliders.pg09_audio import require_audio_width
        cfg = model.config
        require_audio_width(cfg['hidden_size'] if isinstance(cfg, dict) else cfg.hidden_size)
        targets = _targets(model)
        builder, identity = verify_runtime()  # Verify before changing any native state.
        self.target_names = tuple(targets)
        self.particles = nn.Parameter(torch.randn(128, 4, device=next(model.parameters()).device, dtype=torch.float32))
        self.adapters = nn.ModuleDict()
        self._hosts = []
        self._restore = native_cleanup(model)
        self._closed = False
        # Construct ALL real graphs before hooking; a failed builder changes no host.
        for index, (name, module) in enumerate(targets.items()):
            adapter = HNDLParticleProjection(name, module, self.particles, 8, 8,
                builder=builder, identity=identity, attach=False)
            adapter.to(device=self.particles.device, dtype=torch.float32)
            self.adapters[str(index)] = adapter
        if not attach:
            return  # Real graph construction/identity preflight, with no native hooks.
        try:
            model.eval().requires_grad_(False)
            for module, adapter in zip(targets.values(), self.adapters.values()):
                self._hosts.append((module, module.forward, adapter))
                module.forward = adapter.forward
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, strength):
        if self._closed:
            raise RuntimeError('TinyQwen slider is closed')
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError('finite strength required')
        previous = [a.multiplier for a in self.adapters.values()]
        try:
            for adapter in self.adapters.values():
                adapter.multiplier = strength
            yield self
        finally:
            for adapter, value in zip(self.adapters.values(), previous):
                adapter.multiplier = value

    def close(self):
        if self._closed:
            return
        # Detect an ownership conflict before changing any host.
        if any(module.forward != adapter.forward for module, _, adapter in self._hosts):
            raise RuntimeError('TinyQwen forward ownership changed; restore owner before close')
        for module, original, _ in reversed(self._hosts):
            module.forward = original
        self._restore()
        self._closed = True


class _Host(nn.Module):
    """Expose fixture config through the same AR config protocol as native Qwen."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(**model.config) if isinstance(model.config, dict) else model.config


class TinyQwenResponse(ARLastHiddenResponseAdapter):
    family = 'tinyqwen_native_last_hidden_shared_v1'

    def __init__(self, backend, slider, *, model_identity):
        self.backend = backend
        super().__init__(_Host(backend.model), slider, model_identity=model_identity)

    def _hidden(self, ids):
        return self.backend.hidden(list(ids))[:, -1]


def capabilities(options):
    return ResponseCapabilities('routed_particle_gan', 'text', 'last_caption_token_hidden',
        'continuous_vector', (1, 32 if options.get('fixture', False) else 1024),
        PG_VERSION, PG_SOURCE_PROFILE,
        qualification='Bounded random CPU fixture only; native Qwen width1024 blocked; no token generation')


def validate(options):
    allowed = {'fixture', 'steps', 'batch', 'seed', 'ema_decay', 'pg_version',
               'pg_source_profile', 'penalty_input_domain', 'base_precision'}
    if set(options) - allowed or type(options.get('fixture', False)) is not bool:
        raise ValueError('unknown TinyQwen profile option or invalid fixture flag')
    if type(options.get('seed', 7)) is not int or not 0 <= options.get('seed', 7) < 2**63 - 100:
        raise ValueError('integer seed within the shared lifecycle budget required')
    engine_options(options)
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('TinyQwen preview qualifies float32 fixtures only')
    verify_runtime()


def _native(options, *, attach=True):
    # Fixed model/domain gate BEFORE a backend can load weights or attach hooks.
    from particle_sliders.pg094 import validate_pg09_domain
    cap = capabilities(options)
    validate_pg09_domain(cap.response_shape, response_domain=cap.response_domain,
                        **engine_options(options))
    if not options.get('fixture', False):
        raise ValueError('native TinyQwen is unqualified pending explicit continuous admission')
    # A reproducible frozen fixture, independent of caller RNG or training data.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(options.get('seed', 7))
        backend = TinyLLMBackend(dummy=True)
        model_identity = dict(qualification='random CPU fixture only', config=deepcopy(backend.model.config),
            state_sha256=hashlib.sha256(b''.join(t.detach().contiguous().numpy().tobytes()
                for t in backend.model.state_dict().values())).hexdigest())
        slider = TinyQwenHNDLSlider(backend.model, attach=attach)
    try:
        return TinyQwenResponse(backend, slider, model_identity=model_identity)
    except BaseException:
        slider.close()
        raise


def _close(native):
    native.close_checkpointing()
    native.slider.close()


def prepare(options, *, rows):
    native = _native(options)
    try:
        prepared = [native.prepare(AudioRow(tuple(row['neutral_ids']), tuple(row['positive_ids'])))
                    for row in rows]
        conditions = audio_conditions(prepared)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get('seed', 7) + 1)
            critic = TinyQwenHNDLCritic(32, conditions.shape[1])
        game = ConditionalPG094(native, prepared, critic, conditions=conditions,
            particles=native.particles, response_domain='continuous_vector',
            batch_size=options.get('batch', 2), total_steps=options.get('steps', 4),
            seed=options.get('seed', 7), **engine_options(options))
        native_options = {k: options[k] for k in ('fixture', 'seed') if k in options}
        spec = INFERENCE.specification(native, model_identity=native.model_identity,
                                      protocol=PROTOCOL, configuration=native_options)
        return Binding(game, close=lambda: _close(native), inference=spec)
    except BaseException:
        _close(native)
        raise


def validate_inference(options, spec):
    # Configuration comes from the checksummed artifact; local options never rebind it.
    if options:
        raise ValueError('TinyQwen inference accepts no local model/profile overrides')
    if set(spec.configuration) - {'fixture', 'seed'} or spec.configuration.get('fixture') is not True:
        raise ValueError('TinyQwen inference requires its explicit CPU fixture profile')
    validate(spec.configuration)
    # Rebuild the real bounded graph identity without attaching any native forwards.
    # Reject changed model/source/shape before the shared loader attaches or loads state.
    from particle_sliders.inference import native_identity
    native = _native(spec.configuration, attach=False)
    try:
        if native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL) != spec.identity:
            raise ValueError('TinyQwen native model/source/shape differs before attachment')
    finally:
        _close(native)


def prepare_inference(options, spec):
    native = _native(spec.configuration)
    try:
        from particle_sliders.inference import native_identity
        identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
        def response(*, strength, seed, ids):
            del seed
            if not ids or any(type(v) is not int or not 0 <= v < 256 for v in ids):
                raise ValueError('nonempty fixture token ids in [0,256) required')
            return native._response(tuple(ids), scale=strength).detach().clone()
        return InferenceBinding(native.network, identity, lambda: _close(native), response=response)
    except BaseException:
        _close(native)
        raise


BACKEND = ResponseBackend(NAME, capabilities({}), prepare, validate,
                          configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, capabilities({}), prepare_inference, validate_inference,
                             configure_capabilities=capabilities)


def config(*, fixture=False, **options):
    return SessionConfig(NAME, COHORT if fixture else 'tinyqwen-pg094-hypergan-hndl-v1-native-blocked',
                         fixture=fixture, **options)
