"""Conditional native-response GAN using released ParticleGAN 0.9.1 components.

Backends own frozen conditioning and differentiable response generation. This
module owns the adversarial game, optimizer lifecycle, sampling and checkpoint
contract. It does not implement ParticleGAN's loss or regularizer mathematics.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path
import platform

import torch
from torch import nn
from particlegan import get_recipe, scale_learning_rates
from particlegan.optim.constraint_geometry import constraint_geometry_backward

PG_VERSION = '0.9.1'
PG_WHEEL_SHA256 = 'b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec'


def _released_sources(profile='release'):
    import particlegan
    root = Path(particlegan.__file__).resolve().parent
    if profile == 'release':
        expected = json.loads(Path(__file__).with_name('pg09_release_sources.json').read_text())
    elif profile == 'alz-continuous-v1':
        document = json.loads(Path(__file__).with_name('pg09_continuous_profile.json').read_text())
        expected = document['python_sources']
    else:
        raise ValueError('unknown PG source profile')
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    if actual != expected:
        raise RuntimeError('installed ParticleGAN sources differ from verified ' + profile + ' source profile')
    return _hash(actual)


def validate_pg09_domain(response_shape, *, response_domain,
                         pg_source_profile='release', penalty_input_domain='auto'):
    """Pre-forward host gate for native shape + exact reviewed PG sources.

    This never installs a package, applies a patch or reshapes a response.
    Callers must declare actual native geometry, then pass the same domain
    options into ConditionalPG09. The optional overlay is ALZ's single patch.
    """
    if version('particlegan') != PG_VERSION:
        raise RuntimeError('ConditionalPG09 requires particlegan==' + PG_VERSION + ' base release')
    if (pg_source_profile, penalty_input_domain) not in (('release', 'auto'), ('alz-continuous-v1', 'continuous')):
        raise ValueError('PG source profile and declared penalty domain must be an explicitly qualified pair')
    sources = _released_sources(pg_source_profile)
    shape = tuple(response_shape)
    if not shape or shape[0] != 1 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError('response requires positive native dimensions and leading singleton batch')
    if response_domain == 'continuous_field':
        if len(shape) not in (4, 5) or penalty_input_domain != 'auto':
            raise ValueError('continuous_field requires genuine native 4D/5D shape on release auto profile')
        resolved = 'image'
    elif response_domain == 'continuous_vector':
        if len(shape) != 2 or (shape[1] >= 170 and penalty_input_domain != 'continuous'):
            raise ValueError('released PG09 cannot admit flat continuous vectors with width >=170')
        resolved = 'continuous'
    else:
        raise ValueError('unsupported response_domain; categorical fallback is forbidden')
    return dict(pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain,
                pg_sources_sha256=sources, resolved_penalty_domain=resolved)


def _implementation_sources(backend, critic, provider=None):
    objects = [backend, backend.network, critic]
    if provider is not None:
        objects.append(provider)
    seen = set()
    # Include wrapped native product adapters and their routed submodules.
    # Module traversal reads classes only; no pretrained parameter copies.
    for _ in range(3):
        for owner in tuple(objects):
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            for name in ('backend', 'runtime', 'model', 'transformer', 'slider', 'adapter'):
                obj = getattr(owner, name, None)
                if obj is not None and not isinstance(obj, (str, bytes, int, float, bool, complex)):
                    objects.append(obj)
            if isinstance(owner, nn.Module):
                objects.extend(owner.modules())
    result = {'particle_sliders.pg09': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for obj in objects:
        cls = type(obj)
        try:
            source = inspect.getsourcefile(cls)
        except TypeError as exc:
            raise ValueError('implementation class must have inspectable source') from exc
        if source is None:
            raise ValueError('implementation class must have inspectable source')
        result[cls.__module__] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return result


def _architectures(backend, critic):
    result, visited = {}, set()
    pending = [('backend', backend), ('critic', critic)]
    for _ in range(4):
        following = []
        for name, owner in pending:
            if id(owner) in visited:
                continue
            visited.add(id(owner))
            if isinstance(owner, nn.Module):
                result[name] = dict(
                    tree=[(n, type(m).__module__ + '.' + type(m).__qualname__, m.extra_repr(), m.training,
                           {key: getattr(m, key) for key in ('backend', 'attention_backend', 'gradient_checkpointing')
                            if isinstance(getattr(m, key, None), (str, int, float, bool))})
                          for n, m in owner.named_modules()],
                    parameters={n: (list(p.shape), str(p.dtype), p.requires_grad)
                                for n, p in owner.named_parameters()},
                    buffers={n: (list(p.shape), str(p.dtype)) for n, p in owner.named_buffers()})
                config = getattr(owner, 'config', None)
                if config is not None and hasattr(config, 'to_dict'):
                    result[name]['config'] = _identity(config.to_dict())
            for attr in ('network', 'backend', 'runtime', 'model', 'transformer', 'slider', 'adapter'):
                value = getattr(owner, attr, None)
                if value is not None and not isinstance(value, (str, bytes, int, float, bool, complex)):
                    following.append((name + '.' + attr, value))
        pending = following
    return result


def _identity(value):
    """Stable frozen-row identity, including conditioning and native geometry."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if is_dataclass(value):
        return {f.name: _identity(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            if any(not isinstance(k, (str, int, float, bool)) for k in value):
                raise TypeError('conditioning dictionaries require scalar keys')
            # Native Transformers configs include integer id2label keys. Keep
            # key types explicit so integer 0 cannot alias the string "0".
            return {'__typed_mapping__': [[type(k).__name__, k, _identity(v)]
                    for k, v in sorted(value.items(), key=lambda item: (type(item[0]).__name__, str(item[0])))]}
        return {k: _identity(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_identity(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError('unsupported frozen conditioning identity: ' + type(value).__name__)


def _hash(value):
    return hashlib.sha256(json.dumps(_identity(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate_module(module, state):
    expected = module.state_dict()
    if state.keys() != expected.keys():
        raise ValueError('checkpoint module keys differ')
    for name, tensor in state.items():
        target = expected[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError('checkpoint tensor schema differs: ' + name)
        if not torch.isfinite(tensor).all():
            raise ValueError('checkpoint has nonfinite tensor: ' + name)


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError('checkpoint contains nonfinite optimizer or regularizer state')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('checkpoint contains nonfinite optimizer or regularizer state')
    elif isinstance(value, dict):
        for v in value.values():
            _finite_tree(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _finite_tree(v)


def _validate_optimizer(optimizer, packet):
    """Bind loaded update laws to the recipe before public optimizer restore.

    Native Adam restores group settings without validating them. The recipe
    owns every setting except the scheduled learning rate; reject changed
    settings even when Adam would accept them or fail only on the next step.
    """
    if (not isinstance(packet, dict) or not isinstance(packet.get('param_groups'), list)
            or not isinstance(packet.get('state'), dict)
            or len(packet['param_groups']) != len(optimizer.param_groups)):
        raise ValueError('checkpoint optimizer groups differ')
    identifiers = set()
    for saved, live in zip(packet['param_groups'], optimizer.param_groups):
        if not isinstance(saved, dict) or saved.keys() != live.keys():
            raise ValueError('checkpoint optimizer group fields differ')
        if not isinstance(saved['params'], list) or len(saved['params']) != len(live['params']):
            raise ValueError('checkpoint optimizer parameter count differs')
        for name in live.keys() - {'params', 'lr'}:
            if saved[name] != live[name]:
                raise ValueError('checkpoint optimizer setting differs: ' + name)
        rate = saved['lr']
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('invalid checkpoint optimizer learning rate')
        for key, parameter in zip(saved['params'], live['params']):
            if type(key) is not int or key in identifiers:
                raise ValueError('invalid checkpoint optimizer parameter IDs')
            identifiers.add(key)
            slots = packet['state'].get(key, {})
            if not isinstance(slots, dict):
                raise ValueError('invalid checkpoint optimizer history')
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'momentum_buffer'):
                if name in slots and (not isinstance(slots[name], torch.Tensor)
                                      or slots[name].shape != parameter.shape
                                      or slots[name].dtype != parameter.dtype):
                    raise ValueError('checkpoint optimizer tensor schema differs')
    if not packet['state'].keys() <= identifiers:
        raise ValueError('unknown checkpoint optimizer parameter history')


class ConditionalPG09:
    """Caller-owned conditional game; ``step`` performs one D and one G update.

    ``backend.generated(prepared)`` must keep the row's sign active through
    checkpoint recomputation, and return its native response shape. ``critic``
    accepts (response, condition) and returns raw [batch] or [batch,1] logits.
    The reference law is fixed Gaussian noise; fake coordinates add the paired
    response residual. No squared-error/reconstruction loss is optimized.
    """

    def __init__(self, backend, prepared_rows, critic, *, conditions=None, particles=None,
                 response_domain='continuous_field', response_scale=1.0,
                 preset='bispf_optimized', batch_size=2, seed=7, total_steps=100,
                 sigma=1.0, recipe_overrides=None, sampling='balanced',
                 pg_source_profile='release', penalty_input_domain='auto'):
        if version('particlegan') != PG_VERSION:
            raise RuntimeError('ConditionalPG09 requires released particlegan==' + PG_VERSION)
        if (pg_source_profile, penalty_input_domain) not in (('release', 'auto'), ('alz-continuous-v1', 'continuous')):
            raise ValueError('PG source profile and declared penalty domain must be an explicitly qualified pair')
        if preset not in ('bispf_optimized', 'bispf'):
            raise ValueError('only explicitly selected public BISPF presets are admitted')
        if type(batch_size) is not int or not 2 <= batch_size <= 512:
            raise ValueError('batch_size must be an integer in [2,512]')
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError('total_steps must be a positive integer')
        if type(seed) is not int or not 0 <= seed < 2**63 - 100:
            raise ValueError('invalid seed')
        if any(not math.isfinite(v) or v <= 0 for v in (sigma, response_scale)):
            raise ValueError('sigma and response_scale must be finite and positive')
        if sampling not in ('balanced', 'replacement'):
            raise ValueError('sampling must be balanced or replacement')
        self.backend, self.network, self.critic = backend, backend.network, critic
        trainables = [p for p in self.network.parameters() if p.requires_grad]
        d_parameters = [p for p in critic.parameters() if p.requires_grad]
        if not trainables or not d_parameters:
            raise ValueError('generator and critic require trainable parameters')
        self.device = trainables[0].device
        if any(p.device != self.device or p.dtype != torch.float32 for p in trainables + d_parameters):
            raise ValueError('all trainables must share float32 dtype and device')
        if set(map(id, trainables)) & set(map(id, critic.parameters())):
            raise ValueError('generator and critic parameter ownership overlaps')
        if any(p.requires_grad is False for p in self.network.parameters()):
            raise ValueError('backend.network must contain adapter trainables only')
        if getattr(critic, 'score_bound', 0) != 0:
            raise ValueError('critic must produce unbounded raw logits')
        self.provider = prepared_rows if callable(getattr(prepared_rows, 'identity', None)) else None
        self.rows = prepared_rows if self.provider is not None else tuple(deepcopy(tuple(prepared_rows)))
        if len(self.rows) < 2:
            raise ValueError('at least two frozen response rows required')
        shape = tuple(self.provider.response_shape) if self.provider is not None else tuple(self.rows[0].target.shape)
        self.response_shape = shape
        domain = validate_pg09_domain(shape, response_domain=response_domain,
            pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
        pg_sources_sha256 = domain['pg_sources_sha256']
        self.targets = None
        if self.provider is None:
            for row in self.rows:
                self._validate_response(row)
            self.targets = torch.cat([r.target for r in self.rows]).detach().to(device=self.device, dtype=torch.float32)
        self.conditions = None
        if conditions is not None:
            self._validate_conditions(conditions, len(self.rows))
            self.conditions = conditions.detach().to(device=self.device, dtype=torch.float32).clone()
        elif self.provider is None or not callable(getattr(self.provider, 'conditions', None)):
            raise ValueError('conditions tensor or lazy provider.conditions(indices) is required')
        self.response_domain, self.response_scale = response_domain, float(response_scale)
        self.batch_size, self.sigma = batch_size, float(sigma)
        # This caller-owned native-response loop does not execute GANTrainer's
        # CUDA convolution autocast policy. Bind the actual host precision even
        # though the released optimized recipe now defaults to BF16 training.
        options = dict(batch_size=batch_size, total_steps=total_steps, training_precision='float32')
        self.prior = None
        if particles is not None:
            if (not isinstance(particles, nn.Parameter) or particles.ndim != 2
                    or min(particles.shape) < 1 or not particles.requires_grad
                    or not any(p is particles for p in trainables)):
                raise ValueError('particles must be the actual routed bank owned by backend.network')
            self.prior = nn.Module()
            self.prior.register_parameter('z', particles)
            options.update(num_particles=len(particles), z_dim=particles.shape[1])
        overrides = dict(recipe_overrides or {})
        for name, value in options.items():
            if name in overrides and overrides[name] != value:
                raise ValueError('recipe override contradicts actual ' + name)
        # Alternative objectives/optimizer laws require a separately reviewed arm.
        allowed = {'num_particles', 'z_dim', 'batch_size', 'total_steps', 'lr',
                   'd_lr_mult', 'prior_lr_mult', 'training_precision'}
        if set(overrides) - allowed:
            raise ValueError('unsupported recipe overrides: ' + str(sorted(set(overrides) - allowed)))
        options.update(overrides)
        self.recipe = get_recipe(preset, **options)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(self.network, critic, self.prior)
        self.loss = self.recipe.make_loss()
        penalty_options = {'input_domain': 'continuous'} if penalty_input_domain == 'continuous' else {}
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True, **penalty_options)
        self.spread = self.recipe.make_prior_regularizer(prior=self.prior) if self.prior else None
        self._check_optimizer_ownership(trainables, d_parameters)
        self.base_lrs = [[g['lr'] for g in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.streams = {name: torch.Generator().manual_seed(seed + offset)
                        for name, offset in (('d_rows', 10), ('d_noise', 20), ('g_rows', 30), ('g_noise', 40))}
        self.sampling = sampling
        self.row_orders = {phase: torch.empty(0, dtype=torch.int64) for phase in ('d', 'g')}
        self.row_cursors = dict(d=0, g=0)
        self.exposures = {phase: torch.zeros(len(self.rows), dtype=torch.int64) for phase in ('d', 'g')}
        self.model_cpu_rng = torch.Generator().manual_seed(seed + 80).get_state()
        self.model_cuda_rng = (torch.Generator(device=self.device).manual_seed(seed + 80).get_state()
                               if self.device.type == 'cuda' else None)
        self.completed_steps = 0
        self._failed = False
        self._export_loaded = False
        self._signature = dict(pg_version=PG_VERSION, pg_release_wheel_sha256=PG_WHEEL_SHA256,
            pg_sources_sha256=pg_sources_sha256, implementation_sources=_implementation_sources(backend, critic, self.provider),
            pg_source_profile=pg_source_profile,
            pg_patch_sha256=(json.loads(Path(__file__).with_name('pg09_continuous_profile.json').read_text())['patch_sha256']
                             if pg_source_profile == 'alz-continuous-v1' else None),
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                         numpy=version('numpy'), deterministic=torch.are_deterministic_algorithms_enabled(),
                         tf32_matmul=torch.backends.cuda.matmul.allow_tf32),
            architectures=_architectures(backend, critic),
            recipe=asdict(self.recipe), backend=_identity(backend.metadata()),
            rows_sha256=_hash(self.provider.identity() if self.provider is not None else self.rows),
            conditions_sha256=(_hash(self.conditions) if self.conditions is not None else 'bound-by-provider-identity'),
            row_count=len(self.rows), sampling=sampling,
            response_domain=response_domain, resolved_penalty_domain=('image' if len(shape) >= 3 else 'continuous'),
            penalty_input_domain=penalty_input_domain, response_shape=list(shape[1:]),
            response_scale=self.response_scale, sigma=self.sigma,
            generator_arm='routed_particle_gan' if self.prior else 'adapter_gan',
            objective='g_adv+public_prior;d_adv+public_penalty',
            network_schema={k: (list(v.shape), str(v.dtype)) for k, v in self.network.state_dict().items()})

    def _check_optimizer_ownership(self, g_parameters, d_parameters):
        for optimizer, expected in ((self.opt_g, g_parameters), (self.opt_d, d_parameters)):
            owned = [p for group in optimizer.param_groups for p in group['params']]
            if len(owned) != len(set(map(id, owned))) or set(map(id, owned)) != set(map(id, expected)):
                raise ValueError('optimizer ownership differs from declared trainables')

    def _validate_response(self, row):
        for tensor in (row.target, row.neutral):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != self.response_shape
                    or tensor.requires_grad or not torch.isfinite(tensor).all()):
                raise ValueError('target and neutral must be frozen finite native responses of equal shape')

    @staticmethod
    def _validate_conditions(conditions, count):
        if (not isinstance(conditions, torch.Tensor) or conditions.requires_grad
                or conditions.ndim != 2 or conditions.shape[0] != count
                or conditions.shape[1] < 1 or not torch.isfinite(conditions).all()):
            raise ValueError('conditions must be frozen finite [rows,K>=1] numeric coordinates')

    def _sample_ids(self, phase):
        generator = self.streams[phase + '_rows']
        if self.sampling == 'replacement':
            return torch.randint(len(self.rows), (self.batch_size,), generator=generator)
        pieces, remaining = [], self.batch_size
        while remaining:
            if self.row_cursors[phase] == len(self.row_orders[phase]):
                self.row_orders[phase] = torch.randperm(len(self.rows), generator=generator)
                self.row_cursors[phase] = 0
            cursor = self.row_cursors[phase]
            count = min(remaining, len(self.rows) - cursor)
            pieces.append(self.row_orders[phase][cursor:cursor + count])
            self.row_cursors[phase] += count
            remaining -= count
        return torch.cat(pieces)

    def _batch(self, phase):
        ids = self._sample_ids(phase)
        selected = {i: self.rows[i] for i in sorted(set(ids.tolist()))}
        for row in selected.values():
            self._validate_response(row)
        targets = torch.cat([selected[i].target for i in ids.tolist()]).detach().to(device=self.device, dtype=torch.float32)
        conditions = (self.conditions[ids.to(self.device)] if self.conditions is not None
                      else self.provider.conditions(ids))
        self._validate_conditions(conditions, len(ids))
        conditions = conditions.detach().to(device=self.device, dtype=torch.float32)
        noise = torch.randn((self.batch_size, *self.response_shape[1:]),
                            generator=self.streams[phase + '_noise']).to(device=self.device, dtype=torch.float32) * self.sigma
        return ids, noise, conditions, targets, selected

    def _generated(self, ids, selected):
        values = {i: self.backend.generated(selected[i]) for i in sorted(set(ids.tolist()))}
        shape = self.response_shape
        if any(v.shape != shape or v.device != self.device or not torch.isfinite(v).all() for v in values.values()):
            raise ValueError('generated response must preserve finite native shape and device')
        return torch.cat([values[i].float() for i in ids.tolist()])

    @contextmanager
    def _model_stream(self):
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.autograd.set_multithreading_enabled(False):
            torch.set_rng_state(self.model_cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self.model_cuda_rng, self.device)
            try:
                yield
            finally:
                self.model_cpu_rng = torch.get_rng_state()
                if devices:
                    self.model_cuda_rng = torch.cuda.get_rng_state(self.device)

    def step(self):
        if self._export_loaded:
            raise RuntimeError('inference export is not a resumable training checkpoint')
        if self._failed:
            raise RuntimeError('failed update: restore a qualified checkpoint before continuing')
        if self.completed_steps >= self.recipe.total_steps:
            raise RuntimeError('execution budget exhausted')
        try:
            with self._model_stream():
                return self._step()
        except Exception:
            self._failed = True
            raise

    def _step(self):
        # Check actual response graph BEFORE any D/optimizer update for every preset.
        g_ids, g_real, g_condition, g_target, g_selected = self._batch('g')
        generated = self._generated(g_ids, g_selected)
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if not generated.requires_grad:
            raise ValueError('detached generator response; adversarial path is inactive')
        connected = torch.autograd.grad(generated.sum(), parameters, retain_graph=True, allow_unused=True)
        if any(gradient is None for gradient in connected):
            raise ValueError('generator trainable or routed bank is disconnected from response')
        scale_learning_rates(self.completed_steps, self.recipe, (self.opt_g, self.opt_d), self.base_lrs, self.prior)
        d_ids, d_real, d_condition, d_target, d_selected = self._batch('d')
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)
        flags = [p.requires_grad for p in self.critic.parameters()]
        try:
            with torch.no_grad():
                d_fake = d_real + (self._generated(d_ids, d_selected) - d_target) / self.response_scale
            d_adv = self.loss.d_loss(self.critic(d_real, d_condition), self.critic(d_fake, d_condition))
            d_pen = self.penalty(self.critic, d_real, d_fake, d_condition)
            d_total = d_adv + d_pen
            if not torch.isfinite(d_total):
                raise FloatingPointError('nonfinite critic loss')
            d_total.backward()
            self._require_finite_gradients(self.critic)
            self.opt_d.step()
            self.opt_d.zero_grad(set_to_none=True)
            self.critic.requires_grad_(False)
            g_fake = g_real + (generated - g_target) / self.response_scale
            g_adv = self.loss.g_loss(self.critic(g_fake, g_condition), self.critic(g_real, g_condition))
            if not g_adv.requires_grad:
                raise ValueError('detached adversarial generator scalar')
            prior_loss = self.spread(self.prior.z) if self.spread else g_adv.new_zeros(())
            g_total = g_adv + prior_loss
            if not torch.isfinite(g_total):
                raise FloatingPointError('nonfinite generator loss')
            if self.prior is not None and hasattr(self.opt_g, 'set_sampled_rows'):
                self.opt_g.set_sampled_rows(self.prior.z, torch.arange(len(self.prior.z), device=self.device))
            constraint_geometry_backward(g_total, self.opt_g, (g_adv,))
            self._require_finite_gradients(self.network)
            self.opt_g.step()
        finally:
            for parameter, flag in zip(self.critic.parameters(), flags):
                parameter.requires_grad_(flag)
        self.completed_steps += 1
        for phase, ids in (('g', g_ids), ('d', d_ids)):
            self.exposures[phase].index_add_(0, ids, torch.ones_like(ids))
        return dict(step=self.completed_steps, g_loss=float(g_total.detach()), g_adv=float(g_adv.detach()),
            prior_regularization=float(prior_loss.detach()), d_loss=float(d_total.detach()),
            d_adv=float(d_adv.detach()), d_pen=float(d_pen.detach()),
            g_rows=g_ids.tolist(), d_rows=d_ids.tolist(),
            g_unique_rows=int(torch.count_nonzero(self.exposures['g'])),
            d_unique_rows=int(torch.count_nonzero(self.exposures['d'])),
            g_row_exposures=int(self.exposures['g'].sum()),
            d_row_exposures=int(self.exposures['d'].sum()), **self.penalty.diagnostics())

    @staticmethod
    def _require_finite_gradients(module):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
            raise FloatingPointError('nonfinite gradients')

    def state_dict(self):
        if self._export_loaded:
            raise RuntimeError('inference export is not a resumable training checkpoint')
        if self._failed:
            raise RuntimeError('cannot checkpoint an incomplete failed update')
        return deepcopy(dict(schema='conditional-pg09-v1', config=self._signature,
            completed_steps=self.completed_steps, network=self.network.state_dict(),
            critic=self.critic.state_dict(), prior=self.prior.state_dict() if self.prior else None,
            opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(),
            spread=self.spread.state_dict() if self.spread else None,
            streams={k: v.get_state() for k, v in self.streams.items()},
            row_orders=self.row_orders, row_cursors=self.row_cursors, exposures=self.exposures,
            model_cpu_rng=self.model_cpu_rng, model_cuda_rng=self.model_cuda_rng))

    def load_state_dict(self, state):
        if state.get('schema') != 'conditional-pg09-v1' or state.get('config') != self._signature:
            raise ValueError('checkpoint objective, recipe, domain, source or conditioning mismatch')
        if type(state['completed_steps']) is not int or not 0 <= state['completed_steps'] <= self.recipe.total_steps:
            raise ValueError('invalid checkpoint step')
        _validate_module(self.network, state['network'])
        _validate_module(self.critic, state['critic'])
        if self.prior:
            _validate_module(self.prior, state['prior'])
            names = [name for name, p in self.network.named_parameters() if p is self.prior.z]
            if not names or not torch.equal(state['network'][names[0]], state['prior']['z']):
                raise ValueError('checkpoint bank alias values disagree')
        elif state['prior'] is not None or state['spread'] is not None:
            raise ValueError('unexpected particle state in adapter-only arm')
        _finite_tree((state['opt_g'], state['opt_d'], state['spread']))
        for optimizer, packet in ((self.opt_g, state['opt_g']), (self.opt_d, state['opt_d'])):
            _validate_optimizer(optimizer, packet)
        if state['streams'].keys() != self.streams.keys():
            raise ValueError('checkpoint sampler streams differ')
        for rng in list(state['streams'].values()) + [state['model_cpu_rng']]:
            torch.Generator().set_state(rng.cpu())
        expected_exposures = state['completed_steps'] * self.batch_size
        for key in ('row_orders', 'row_cursors', 'exposures'):
            if set(state[key]) != {'d', 'g'}:
                raise ValueError('checkpoint sampler phases differ')
        for phase in ('d', 'g'):
            order, cursor, counts = state['row_orders'][phase], state['row_cursors'][phase], state['exposures'][phase]
            if (not isinstance(order, torch.Tensor) or order.dtype != torch.int64 or order.ndim != 1
                    or len(order) not in (0, len(self.rows)) or type(cursor) is not int
                    or not 0 <= cursor <= len(order)):
                raise ValueError('invalid checkpoint sampler order/cursor')
            if len(order) and not torch.equal(order.cpu().sort().values, torch.arange(len(self.rows))):
                raise ValueError('checkpoint sampler order is not a row permutation')
            if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int64
                    or counts.shape != (len(self.rows),) or (counts < 0).any()
                    or int(counts.sum()) != expected_exposures):
                raise ValueError('invalid checkpoint row exposures')
            if self.sampling == 'balanced':
                if not expected_exposures and (len(order) or cursor):
                    raise ValueError('unstarted sampler cannot carry an advanced row order')
                if expected_exposures and (len(order) == 0 or cursor != ((expected_exposures - 1) % len(self.rows)) + 1):
                    raise ValueError('checkpoint cursor differs from completed-step coverage')
                cycles = expected_exposures // len(self.rows)
                expected_counts = torch.full((len(self.rows),), cycles, dtype=torch.int64)
                remainder = expected_exposures % len(self.rows)
                if remainder:
                    expected_counts[order[:remainder].cpu()] += 1
                if not torch.equal(counts.cpu(), expected_counts):
                    raise ValueError('checkpoint exposures differ from balanced row order')
            elif len(order) or cursor:
                raise ValueError('replacement sampler must not carry shuffled order')
        if self.device.type == 'cuda':
            torch.Generator(device=self.device).set_state(state['model_cuda_rng'])
        elif state['model_cuda_rng'] is not None:
            raise ValueError('CUDA checkpoint cannot silently resume on CPU')
        # Public optimizer restore on isolated copies validates BEFORE live mutation.
        # Copies contain adapter/critic parameters only, never a pretrained backend.
        # PG critic restore also publishes per-device thermal state: make it the
        # final preflight after every sampler and other state can no longer reject.
        shadow_g, shadow_d, shadow_spread = deepcopy((self.opt_g, self.opt_d, self.spread))
        shadow_g.load_state_dict(deepcopy(state['opt_g']))
        if shadow_spread:
            shadow_spread.load_state_dict(deepcopy(state['spread']))
        shadow_d.load_state_dict(deepcopy(state['opt_d']))
        self.network.load_state_dict(state['network'], strict=True)
        self.critic.load_state_dict(state['critic'], strict=True)
        if self.prior:
            self.prior.load_state_dict(state['prior'], strict=True)
            self.spread.load_state_dict(deepcopy(state['spread']))
        self.opt_g.load_state_dict(deepcopy(state['opt_g']))
        self.opt_d.load_state_dict(deepcopy(state['opt_d']))
        for name, generator in self.streams.items():
            generator.set_state(state['streams'][name].cpu())
        self.row_orders = {k: v.cpu().clone() for k, v in state['row_orders'].items()}
        self.row_cursors = dict(state['row_cursors'])
        self.exposures = {k: v.cpu().clone() for k, v in state['exposures'].items()}
        self.model_cpu_rng = state['model_cpu_rng'].cpu().clone()
        self.model_cuda_rng = deepcopy(state['model_cuda_rng'])
        self.completed_steps, self._failed, self._export_loaded = state['completed_steps'], False, False

    @torch.no_grad()
    def evaluate(self, index, *, seed=0):
        """Evaluate adapter modules without changing their modes or RNG streams.

        The backend owns evaluation mode for its frozen native model. This
        context owns only ``network`` and restores even mixed adapter modes.
        """
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        modes = [(module, module.training) for module in self.network.modules()]
        try:
            self.network.eval()
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
                if devices:
                    torch.cuda.set_rng_state(torch.Generator(device=self.device).manual_seed(seed).get_state(), self.device)
                return self.backend.generated(self.rows[index]).detach().clone()
        finally:
            for module, training in modes:
                module.training = training

    def export_state_dict(self):
        """Adapter-only payload, including the actual bank through network ownership."""
        if self._failed:
            raise RuntimeError('cannot export an incomplete failed update')
        return deepcopy(dict(schema='conditional-pg09-adapter-v1', config=self._signature,
                             network=self.network.state_dict(), completed_steps=self.completed_steps))

    def load_export_state_dict(self, state):
        if state.get('schema') != 'conditional-pg09-adapter-v1' or state.get('config') != self._signature:
            raise ValueError('adapter export protocol differs')
        if type(state['completed_steps']) is not int or not 0 <= state['completed_steps'] <= self.recipe.total_steps:
            raise ValueError('invalid adapter export step')
        _validate_module(self.network, state['network'])
        self.network.load_state_dict(state['network'], strict=True)
        self.completed_steps = state['completed_steps']
        self._failed = False
        self._export_loaded = True
