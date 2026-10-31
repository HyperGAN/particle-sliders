"""Separate TinyQwen shared PG/HG/HNDL cohort; no historical objective migration.

Only bounded CPU fixtures are executable with the reviewed release-auto core.
Native Qwen3-0.6B hidden width1024 is refused before model loading or hooks.
All losses, penalties, optimizers, continuation and inference belong to core.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import distribution
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLParticleProjection, verified_builder
from particle_sliders.inference import InferenceBackend, InferenceBinding
from particle_sliders.lifecycle import Binding, SessionConfig
from particle_sliders.pg094 import ConditionalPG094, PG_SOURCE_PROFILE, PG_VERSION, engine_options
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter, AudioRow, audio_conditions
from particle_sliders.response import ResponseBackend, ResponseCapabilities

from . import backend as host_source
from .backend import ATTN_CLASS_NAMES, LORA_LINEAR_NAMES, TinyLLMBackend, DummyTinyCausalLM

NAME = 'tinyqwen-shared-preview'
COHORT = 'tinyqwen-pg094-hypergan-hndl-v2-cpu-fixture'
PROTOCOL = 'tinyqwen-neutral-to-positive-last-hidden-v2'
CRITIC_GRAPH = '''joined = concat(response, condition, axis=1)
features = linear(joined, 16, name="hidden")
features = silu(features)
linear(features, 1, name="score")
'''
ROOT = Path(__file__).parent


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _core_install_receipt(lock):
    """Check immutable PEP610 wheel/Git origin, in addition to every source byte."""
    dist = distribution('particle-sliders-core')
    if dist.version != '0.12.0':
        raise ValueError('TinyQwen preview requires its separate exact core0.12 environment')
    try:
        provenance = json.loads(dist.read_text('direct_url.json') or '{}')
    except (ValueError, TypeError) as exc:
        raise ValueError('missing or invalid exact core wheel/Git provenance') from exc
    if not isinstance(provenance, dict):
        raise ValueError('invalid exact core wheel/Git provenance')
    archive = provenance.get('archive_info', {})
    if not isinstance(archive, dict):
        raise ValueError('invalid exact core wheel provenance')
    digest = archive.get('hashes', {}).get('sha256')
    if digest is None and archive.get('hash', '').startswith('sha256='):
        digest = archive['hash'].removeprefix('sha256=')
    if digest == lock['core_wheel_sha256']:
        return dict(kind='wheel', sha256=digest, commit=lock['core_commit'])
    vcs = provenance.get('vcs_info', {})
    if (vcs.get('vcs') == 'git' and vcs.get('commit_id') == lock['core_commit']
            and provenance.get('url') == 'https://github.com/HyperGAN/particle-sliders.git'
            and provenance.get('subdirectory') == 'packages/particle-sliders-core'
            and not provenance.get('dir_info', {}).get('editable', False)):
        return dict(kind='git', commit=vcs['commit_id'])
    raise ValueError('exact core wheel hash or immutable Git origin differs')


def verify_runtime():
    """Verify product/host/core/PG/HG/HNDL closures before native mutation."""
    import particle_sliders
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    install = _core_install_receipt(lock)
    core_root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(core_root)): _digest(p) for p in core_root.rglob('*')
              if p.suffix in ('.py', '.json', '.hndl')}
    if actual != lock['core_sources']:
        raise ValueError('TinyQwen preview core source closure differs; use original cohort wheel')
    product_sources = {p.name: _digest(p) for p in ROOT.glob('*.py')}
    if product_sources != lock['product_sources'] or Path(host_source.__file__).resolve() != (ROOT / 'backend.py').resolve():
        raise ValueError('TinyQwen preview imported product/host source closure differs')
    # The official PG closure is independently checked by the shared domain gate.
    from particle_sliders.pg094 import validate_pg09_domain
    validate_pg09_domain((1, 1), response_domain='continuous_vector', **engine_options({}))
    builder, identity = verified_builder()
    return builder, dict(identity, core_install=install,
        tinyqwen_lock_sha256=_digest(ROOT / 'sources.lock.json'),
        tinyqwen_product_sources=product_sources,
        tinyqwen_critic_graph_sha256=hashlib.sha256(CRITIC_GRAPH.encode()).hexdigest())


def _readout_contract(width, condition_width):
    return dict(response_width=width, condition_width=condition_width,
        response_domain='continuous_vector', graph=CRITIC_GRAPH,
        graph_sha256=hashlib.sha256(CRITIC_GRAPH.encode()).hexdigest(),
        condition_layout=['frozen neutral final hidden', 'neutral token count',
                          'positive token count', 'shared history length', 'training scale'])


def _cleanup_preserving(close, original, owner):
    """Keep the initiating exception and retain failed cleanup owners for retry."""
    try:
        close()
    except BaseException as cleanup_error:
        original.add_note('TinyQwen cleanup also failed; retry through exception.cleanup_owner')
        original.cleanup_owner = owner
        original.cleanup_errors = (cleanup_error,)


def _attempt_all(ledger):
    errors = []
    for index in range(len(ledger) - 1, -1, -1):
        try:
            ledger[index][1]()
        except BaseException as error:
            errors.append(error)
        else:
            del ledger[index]
    return errors


def _restore_forward(module, original, adapter):
    if module.forward == original:
        return
    if module.forward != adapter.forward:
        raise RuntimeError('TinyQwen forward ownership changed; foreign hook retained for retry')
    module.forward = original


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
                              condition_shape=[1, condition_width], plan=CRITIC_GRAPH, readout_contract=_readout_contract(width, condition_width))

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
        self._forward_cleanup = []
        # The slider changes only modes/grad flags; checkpoint handles belong to
        # the shared AR adapter and have their own retry ledger.
        self._state_cleanup = [
            ('mode:' + name, lambda m=module, v=module.training: setattr(m, 'training', v))
            for name, module in model.named_modules()]
        self._state_cleanup += [
            ('grad:' + name, lambda p=parameter, v=parameter.requires_grad: p.requires_grad_(v))
            for name, parameter in model.named_parameters()]
        self._closed = False
        self._close_requested = False
        self._cleanup_errors = ()
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
                original = module.forward
                self._hosts.append((module, original, adapter))
                self._forward_cleanup.append((adapter.lora_name,
                    lambda m=module, o=original, a=adapter: _restore_forward(m, o, a)))
                module.forward = adapter.forward
        except BaseException as original:
            _cleanup_preserving(self.close, original, self)
            raise

    @contextmanager
    def scaled(self, strength):
        if self._close_requested:
            raise RuntimeError('TinyQwen slider is closed or cleanup is pending')
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

    @property
    def cleanup_pending(self):
        return dict(forwards=[label for label, _ in self._forward_cleanup],
                    native_state=[label for label, _ in self._state_cleanup])

    def close(self):
        if self._closed:
            return
        self._close_requested = True
        for adapter in self.adapters.values():
            adapter.multiplier = 0.
        errors = _attempt_all(self._forward_cleanup)
        # Always attempt every saved mode and gradient restoration, even when a
        # foreign forward cannot safely be removed. Failed actions remain owned.
        errors += _attempt_all(self._state_cleanup)
        self._closed = not self._forward_cleanup and not self._state_cleanup
        self._cleanup_errors = tuple(errors)
        if errors:
            raise errors[0]


class _Host(nn.Module):
    """Expose fixture config through the same AR config protocol as native Qwen."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(**model.config) if isinstance(model.config, dict) else model.config


class TinyQwenResponse(ARLastHiddenResponseAdapter):
    family = 'tinyqwen_native_last_hidden_shared_v2'

    def __init__(self, backend, slider, *, model_identity):
        self.backend = backend
        super().__init__(_Host(backend.model), slider, model_identity=model_identity)

    def _hidden(self, ids):
        return self.backend.hidden(list(ids))[:, -1]

    def close(self):
        _close(self)

    @property
    def cleanup_pending(self):
        return dict(checkpoint_handles=len(self._checkpointing), slider=self.slider.cleanup_pending)


def _fixture_width(options):
    width = options.get('fixture_width', 32)
    if type(width) is not int or width not in (16, 32):
        raise ValueError('fixture_width must be the bounded CPU fixture16 or32')
    if not options.get('fixture', False) and 'fixture_width' in options:
        raise ValueError('fixture width cannot alter the native Qwen domain')
    return width


def capabilities(options):
    return ResponseCapabilities('routed_particle_gan', 'text', 'last_caption_token_hidden',
        'continuous_vector', (1, _fixture_width(options) if options.get('fixture', False) else 1024),
        PG_VERSION, PG_SOURCE_PROFILE, strength_min=None, strength_max=None,
        qualification='Bounded random CPU fixture only; native Qwen width1024 blocked; no token generation')


def validate(options):
    allowed = {'fixture', 'steps', 'batch', 'seed', 'ema_decay', 'pg_version',
               'pg_source_profile', 'penalty_input_domain', 'base_precision', 'fixture_width'}
    if set(options) - allowed or type(options.get('fixture', False)) is not bool:
        raise ValueError('unknown TinyQwen profile option or invalid fixture flag')
    if type(options.get('seed', 7)) is not int or not 0 <= options.get('seed', 7) < 2**63 - 100:
        raise ValueError('integer seed within the shared lifecycle budget required')
    _fixture_width(options)
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
        width = _fixture_width(options)
        if backend.model.config['hidden_size'] != width:
            backend.model = DummyTinyCausalLM(hidden_size=width).eval().requires_grad_(False)
        model_identity = dict(qualification='random CPU fixture only', config=deepcopy(backend.model.config),
            readout_contract=_readout_contract(width, width + 4),
            state_sha256=hashlib.sha256(b''.join(t.detach().contiguous().numpy().tobytes()
                for t in backend.model.state_dict().values())).hexdigest())
        slider = TinyQwenHNDLSlider(backend.model, attach=attach)
    try:
        return TinyQwenResponse(backend, slider, model_identity=model_identity)
    except BaseException as original:
        _cleanup_preserving(slider.close, original, slider)
        raise


def _close(native):
    errors = []
    for close in (native.close_checkpointing, native.slider.close):
        try:
            close()
        except BaseException as error:
            errors.append(error)
    native._cleanup_errors = tuple(errors)
    if errors:
        raise errors[0]


def prepare(options, *, rows):
    native = _native(options)
    try:
        prepared = [native.prepare(AudioRow(tuple(row['neutral_ids']), tuple(row['positive_ids'])))
                    for row in rows]
        conditions = audio_conditions(prepared)
        width = int(native.model.config.hidden_size)
        contract = _readout_contract(width, int(conditions.shape[1]))
        if contract != native.model_identity['readout_contract']:
            raise ValueError('observed native response/condition readout contract differs')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get('seed', 7) + 1)
            critic = TinyQwenHNDLCritic(width, conditions.shape[1])
        game = ConditionalPG094(native, prepared, critic, conditions=conditions,
            particles=native.particles, response_domain='continuous_vector',
            batch_size=options.get('batch', 2), total_steps=options.get('steps', 4),
            seed=options.get('seed', 7), **engine_options(options))
        native_options = {k: options[k] for k in ('fixture', 'seed', 'fixture_width') if k in options}
        spec = INFERENCE.specification(native, model_identity=native.model_identity,
                                      protocol=PROTOCOL, configuration=native_options)
        @torch.no_grad()
        def response(*, index, split, strength, seed):
            del seed
            if split != 'train' or not 0 <= index < len(prepared):
                raise ValueError('unavailable TinyQwen evaluation split or row')
            return native._response(prepared[index].row.neutral_ids, scale=strength).detach().clone()
        return Binding(game, close=native.close, inference=spec, response=response)
    except BaseException as original:
        _cleanup_preserving(native.close, original, native)
        raise


def validate_inference(options, spec):
    # Configuration comes from the checksummed artifact; local options never rebind it.
    if options:
        raise ValueError('TinyQwen inference accepts no local model/profile overrides')
    if set(spec.configuration) - {'fixture', 'seed', 'fixture_width'} or spec.configuration.get('fixture') is not True:
        raise ValueError('TinyQwen inference requires its explicit CPU fixture profile')
    validate(spec.configuration)
    # Rebuild the real bounded graph identity without attaching any native forwards.
    # Reject changed model/source/shape before the shared loader attaches or loads state.
    from particle_sliders.inference import native_identity
    native = _native(spec.configuration, attach=False)
    try:
        if native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL) != spec.identity:
            raise ValueError('TinyQwen native model/source/shape differs before attachment')
    except BaseException as original:
        _cleanup_preserving(native.close, original, native)
        raise
    else:
        native.close()


def prepare_inference(options, spec):
    native = _native(spec.configuration)
    try:
        from particle_sliders.inference import native_identity
        identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
        @torch.no_grad()
        def response(*, strength, seed, ids):
            del seed
            if not ids or any(type(v) is not int or not 0 <= v < 256 for v in ids):
                raise ValueError('nonempty fixture token ids in [0,256) required')
            return native._response(tuple(ids), scale=strength).detach().clone()
        return InferenceBinding(native.network, identity, native.close, response=response)
    except BaseException as original:
        _cleanup_preserving(native.close, original, native)
        raise


BACKEND = ResponseBackend(NAME, capabilities({}), prepare, validate,
                          configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, capabilities({}), prepare_inference, validate_inference,
                             configure_capabilities=capabilities)


def config(*, fixture=False, **options):
    return SessionConfig(NAME, COHORT if fixture else 'tinyqwen-pg094-hypergan-hndl-v2-native-blocked',
                         fixture=fixture, **options)
