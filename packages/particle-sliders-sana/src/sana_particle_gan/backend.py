"""Staged Sana image-velocity binding to the installed shared GAN lifecycle.

Historical Sana UNI is intentionally outside this distribution. The modern
student uses raw neutral-caption velocity; only its frozen positive teacher
uses declared CFG. No negative-caption teacher, model loader or sampler exists
here. The host supplies exact native encodings/masks and staged frozen weights.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import math
import inspect
from pathlib import Path
import re

import torch
from torch import nn

from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.fibo_hndl import HNDLFieldCritic, HNDLParticleProjection, verified_builder
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Backend, Binding, Capabilities, SessionConfig, native_cleanup
from particle_sliders.pg094 import ConditionalPG094, engine_options, validate_pg09_domain
from particle_sliders.pg09_backends import FrozenResponseAdapter, PreparedResponse, ResponseRow
from particle_sliders.pg09_images import ImageResponseAdapter

NAME = 'sana-shared-gan'
COHORT = 'sana-neutral-cfg-velocity-hypergan-hndl-v1'
MODEL_ID = 'Efficient-Large-Model/Sana_600M_512px_diffusers'
PROTOCOL = dict(cohort=COHORT, objective='shared conditional response GAN; not UNI',
    student='raw neutral velocity', teacher='unconditional + guidance*(positive-unconditional)',
    geometry='native [1,C,H,W] velocity; no packing', targets='attn2 Q/K/V/O',
    rank=8, alpha=8, bank=[128, 4], network_profile='hypergan-hndl-v2',
    qualification='CPU software fixtures only; pretrained/GPU unqualified')
CAPABILITIES = Capabilities('routed_particle_gan', render=False,
    base_precisions=('float32',), render_outputs=(),
    operations=('train', 'resume', 'evaluate', 'export', 'response'),
    unavailable_reasons=(('render', 'Sana native pipeline sampling is not qualified by this response binding'),))
_STAGED = ContextVar('sana_staged_model', default=None)


def _source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _verified_core():
    import particle_sliders
    if version('particle-sliders-core') != '0.12.0':
        raise ValueError('exact core0.12 preview required')
    lock_path = Path(__file__).with_name('core012.lock.json')
    lock = json.loads(lock_path.read_text())
    root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    if actual != lock['python_sources']:
        raise ValueError('shared core source differs from the exact candidate wheel')
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def _weight_digest(model):
    """Bind actual borrowed base tensors, including buffers, without storing them."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _targets(model):
    count = getattr(getattr(model, 'config', None), 'num_layers', None)
    if type(count) is not int or count < 1:
        raise ValueError('Sana config.num_layers is required')
    expected = [f'transformer_blocks.{i}.attn2.{leaf}' for i in range(count)
                for leaf in ('to_q', 'to_k', 'to_v', 'to_out.0')]
    actual = dict(model.named_modules())
    result = {}
    for name in expected:
        module = actual.get(name)
        if (not isinstance(module, nn.Linear) or getattr(module.forward, '__self__', None) is not module
                or module.weight.dtype != torch.float32):
            raise ValueError('unmodified float32 native Sana attention Linear required: ' + name)
        result[name] = module
    selected = {name for name in actual if re.fullmatch(
        r'transformer_blocks\.\d+\.attn2\.(to_q|to_k|to_v|to_out\.0)', name)}
    if selected != set(expected) or len({id(m) for m in result.values()}) != len(result):
        raise ValueError('Sana target plan differs or aliases projections')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('freeze the entire staged Sana base before binding')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('this response cohort requires float32 staged base weights')
    if not math.isfinite(float(model.config.timestep_scale)):
        raise ValueError('finite native timestep_scale required')
    return result


def _model_identity(model, identity):
    identity = deepcopy(identity)
    fixture = identity.get('qualification') == 'CPU fixture only'
    if fixture:
        if any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('fixture identity is CPU-only')
    elif (identity.get('model_id') != MODEL_ID
          or not re.fullmatch('[0-9a-f]{40}', identity.get('revision', ''))
          or not isinstance(identity.get('staged_files_sha256'), dict)
          or not identity['staged_files_sha256']
          or any(not re.fullmatch('[0-9a-f]{64}', str(v))
                 for v in identity['staged_files_sha256'].values())):
        raise ValueError('exact staged Sana revision and file hashes required')
    config = model.config.to_dict() if hasattr(model.config, 'to_dict') else vars(model.config)
    source = inspect.getsourcefile(type(model))
    if source is None:
        raise ValueError('inspectable native Sana model source required')
    return dict(declaration=identity, base_state_sha256=_weight_digest(model),
                native_source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                native_config=json.loads(json.dumps(config, sort_keys=True, allow_nan=False)))


def _plan(model):
    return {n: [m.in_features, m.out_features] for n, m in _targets(model).items()}


def configuration(model, *, model_identity, response_shape, steps=4, batch=2, seed=7,
                  guidance=4.5, ema_decay=.9):
    """Inspect staged weights and declare this separate opt-in cohort before hooks."""
    shape = tuple(response_shape)
    if (len(shape) != 4 or shape[0] != 1
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[1] != model.config.in_channels):
        raise ValueError('native Sana [1,C,H,W] shape required')
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError('finite nonnegative teacher CFG required')
    _, identity = verified_builder()
    validate_pg09_domain(shape, response_domain='continuous_field')
    return SessionConfig(NAME, COHORT, steps=steps, batch=batch, seed=seed,
        guidance=float(guidance), ema_decay=ema_decay, base_precision='float32',
        pg_version='0.9.4', pg_source_profile='release-0.9.4', penalty_input_domain='auto',
        response_shape=list(shape), targets=_plan(model), model_identity=_model_identity(model, model_identity),
        binding_source_sha256=_source(), shared_core_lock_sha256=_verified_core(), builder=identity)


@contextmanager
def staged_model(model, *, model_identity):
    """Explicit native-only handoff: caller stages pipeline/model, never this package."""
    token = _STAGED.set((model, deepcopy(model_identity)))
    try:
        yield
    finally:
        _STAGED.reset(token)


class SanaParticleSlider(nn.Module):
    """One shared bank and shared HyperGAN/HNDL projections on frozen attn2 leaves."""
    def __init__(self, model):
        super().__init__()
        targets = _targets(model)
        builder, identity = verified_builder()  # All source gates before mutation.
        self.names = tuple(targets)
        self.scale = 0.
        self._restore = native_cleanup(model)
        self._hosts = [(module, module.forward) for module in targets.values()]
        self._closed = False
        self.particles = nn.Parameter(torch.randn(128, 4, device=next(model.parameters()).device))
        self.adapters = nn.ModuleList()
        try:
            # Build every graph first; attach only after the full target plan succeeds.
            for name, module in targets.items():
                self.adapters.append(HNDLParticleProjection(name, module, self.particles, 8, 8,
                    builder=builder, identity=identity, attach=False).to(self.particles))
            model.eval()
            for (host, _), projection in zip(self._hosts, self.adapters):
                host.forward = projection.forward
        except BaseException:
            self.close()
            raise

    def extra_repr(self):
        return json.dumps(dict(protocol=PROTOCOL, targets=self.names), sort_keys=True)

    @contextmanager
    def scaled(self, scale):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('Sana strength must be in [0,1]')
        old = [p.multiplier for p in self.adapters]
        previous = self.scale
        try:
            self.scale = float(scale)
            for projection in self.adapters:
                projection.multiplier = self.scale
            yield self
        finally:
            self.scale = previous
            for projection, value in zip(self.adapters, old):
                projection.multiplier = value

    def close(self):
        if not self._closed:
            self._closed = True
            for host, forward in self._hosts:
                host.forward = forward
            self._restore()


class SanaResponseAdapter(ImageResponseAdapter):
    """Teacher-only CFG; shared native Sana forward preserves masks/time/channels."""
    def __init__(self, model, slider, identity, guidance):
        self.guidance = float(guidance)
        super().__init__('sana', model, slider, model_identity=identity)

    def prepare(self, row):
        if row.schedule is not None or row.guidance != 1 or row.scale != 1 or row.strength != 1:
            raise ValueError('Sana GAN uses raw neutral velocity; teacher CFG is declared separately')
        # Shared capture owns snapshots, native shape checks and frozen contexts.
        prepared = super().prepare(row)
        with torch.no_grad(), self.network.scaled(0.):
            uncond = self.velocity(prepared.row.negative, prepared.row.latent, prepared.row.timestep, scale=0.)
            target = uncond + self.guidance * (prepared.target - uncond)
        if not torch.isfinite(target).all():
            raise ValueError('nonfinite CFG teacher')
        return PreparedResponse(prepared.row, target.detach(), prepared.neutral)

    def metadata(self):
        result = FrozenResponseAdapter.metadata(self)
        fixture = self.model_identity['declaration'].get('qualification') == 'CPU fixture only'
        result['diffusers_version'] = None if fixture else version('diffusers')
        result.update(protocol=PROTOCOL, target_convention=PROTOCOL['teacher'],
                      teacher_guidance=self.guidance, cfg_adapter_branch='student raw neutral only')
        return result


def _validate(options):
    if options.get('binding_source_sha256') != _source():
        raise ValueError('Sana binding source changed; retain original cohort environment')
    if options.get('base_precision') != 'float32':
        raise ValueError('Sana response preview requires float32')
    if options.get('shared_core_lock_sha256') != _verified_core():
        raise ValueError('shared core source lock changed')
    engine_options(options)
    _, identity = verified_builder()
    if options.get('builder') != identity:
        raise ValueError('Sana builder profile differs')
    shape = tuple(options['response_shape'])
    if len(shape) != 4 or shape[1] != options['model_identity']['native_config']['in_channels']:
        raise ValueError('Sana requires native 4D image fields')
    validate_pg09_domain(shape, response_domain='continuous_field')
    if not math.isfinite(options['guidance']) or options['guidance'] < 0:
        raise ValueError('finite nonnegative teacher CFG required')


def _preflight(options, request, staged):
    model, identity = staged.get('model'), staged.get('model_identity')
    if model is None or identity is None:
        raise ValueError('explicit already-staged model and model_identity required')
    if _model_identity(model, identity) != options['model_identity'] or _plan(model) != options['targets']:
        raise ValueError('staged model weights/identity/targets differ before attachment')
    rows = staged.get('rows')
    if not isinstance(rows, (tuple, list)) or len(rows) < 2:
        raise ValueError('at least two frozen native ResponseRows required')
    for row in rows:
        _row_check(row, options)
        if row.latent.device != next(model.parameters()).device:
            raise ValueError('stage native inputs on the model device')


def _row_check(row, options):
    if not isinstance(row, ResponseRow) or list(row.latent.shape) != options['response_shape']:
        raise ValueError('row native geometry differs')
    if (row.latent.requires_grad or row.latent.dtype != torch.float32 or not torch.isfinite(row.latent).all()
            or row.timestep is None or row.timestep.numel() != 1 or row.timestep.requires_grad
            or not torch.isfinite(row.timestep).all() or not 0 <= row.timestep.item() <= 1000):
        raise ValueError('frozen native latent and one raw scheduler timestep required')
    if row.schedule is not None or row.guidance != 1 or row.scale != 1 or row.strength != 1:
        raise ValueError('raw neutral student with separately declared teacher CFG required')
    for condition in (row.neutral, row.positive, row.negative):
        if not isinstance(condition, dict) or set(condition) != {'embeds', 'mask'}:
            raise ValueError('actual Sana embeddings and attention masks required')
        embeds, mask = condition['embeds'], condition['mask']
        if (embeds.ndim != 3 or embeds.shape[0] != 1 or tuple(mask.shape) != tuple(embeds.shape[:2])
                or embeds.requires_grad or mask.requires_grad or not torch.isfinite(embeds).all()
                or not torch.all((mask == 0) | (mask == 1))):
            raise ValueError('frozen finite single-example encodings and native binary mask required')


def _response(adapter, options, *, latent, timestep, condition, strength):
    row = ResponseRow(condition, condition, condition, latent, timestep=timestep)
    _row_check(row, options)
    return adapter.velocity(condition, latent, timestep, scale=strength)


def _prepare(options, *, model, model_identity, rows):
    device = next(model.parameters()).device
    seed_initialization_rng(options['seed'], [device.index] if device.type == 'cuda' else [])
    slider = SanaParticleSlider(model)
    try:
        adapter = SanaResponseAdapter(model, slider, options['model_identity'], options['guidance'])
        prepared = [adapter.prepare(row) for row in rows]
        if any(list(row.target.shape) != options['response_shape'] for row in prepared):
            raise ValueError('native output geometry differs from declared response')
        # Stable row identity conditions critic on the exact paired native capture.
        conditions = torch.eye(len(rows), device=slider.particles.device)
        critic = HNDLFieldCritic(options['response_shape'][1], len(rows), options['response_shape'][2:]).to(slider.particles)
        game = ConditionalPG094(adapter, prepared, critic, conditions=conditions, particles=slider.particles,
            total_steps=options['steps'], batch_size=options['batch'], seed=options['seed'], **engine_options(options))
        specification = INFERENCE.specification(adapter, model_identity=options['model_identity'],
            protocol=json.dumps(PROTOCOL, sort_keys=True),
            configuration={k: v for k, v in options.items() if k not in ('steps', 'batch', 'seed', 'ema_decay')})
        return Binding(game, slider.close, inference=specification)
    except BaseException:
        slider.close()
        raise


def _infer_validate(options, spec):
    if options:
        raise ValueError('stage native model explicitly; no loader/local options supported')
    _validate(spec.configuration)
    if spec.identity['protocol'] != json.dumps(PROTOCOL, sort_keys=True):
        raise ValueError('native Sana protocol differs before attachment')
    if _STAGED.get() is None:
        raise ValueError('use staged_model for explicit native-only reload')
    model, identity = _STAGED.get()
    config = spec.configuration
    if _model_identity(model, identity) != config['model_identity'] or _plan(model) != config['targets']:
        raise ValueError('native model/targets differ before attachment')


def _infer_prepare(options, spec):
    model, _ = _STAGED.get()
    slider = SanaParticleSlider(model)
    try:
        config = spec.configuration
        adapter = SanaResponseAdapter(model, slider, config['model_identity'], config['guidance'])
        identity = native_identity(adapter, model_identity=config['model_identity'], protocol=json.dumps(PROTOCOL, sort_keys=True))
        def response(*, strength, seed=0, **native):
            return _response(adapter, config, strength=strength, **native)
        return InferenceBinding(slider, identity, slider.close, response=response)
    except BaseException:
        slider.close()
        raise


BACKEND = Backend(NAME, CAPABILITIES, _prepare, _validate, preflight=_preflight)
INFERENCE = InferenceBackend(NAME, CAPABILITIES, _infer_prepare, _infer_validate)
