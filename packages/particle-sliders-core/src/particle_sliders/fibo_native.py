"""Fibo 2.0 Edit Turbo: staged native edit responses, no loader or game on import.

Uses the reviewed TinyParticleProjection and ConditionalPG09 interfaces. The
official pipeline owns image/VGL preprocessing, VAE normalization and sampling.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import re

import torch
from torch import nn

from .pg09_backends import (
    AdapterParameters, FrozenResponseAdapter, PreparedResponse, ResponseRow,
    bind_native_gradient_checkpointing,
)
from .pg09_images import image_condition_features
from .routed_projection import TinyParticleProjection

MODEL_ID = 'briaai/Fibo-2.0-edit-turbo'
MODEL_REVISION = '177f2d4a06b6587144ef5c2b8452a4f1fe2ab4b7'
DIFFUSERS_COMMIT = '60f95274bf07ed705d675eb09d99b6ae125b601e'
FORMAT = 'fibo-edit-turbo-routed-particle-v1'
DEFAULTS = dict(num_inference_steps=4, guidance_scale=1.0)
_TARGET = re.compile(r'^(noise_refiner|layers)\.\d+\.(attention|cross_attention)\.(to_q|to_k|to_v|to_out\.0)$')


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def _identity(identity):
    identity = _canonical(identity)
    if identity.get('qualification') != 'CPU fixture only':
        if (identity.get('model_id') != MODEL_ID or identity.get('model_revision') != MODEL_REVISION
                or identity.get('diffusers_commit') != DIFFUSERS_COMMIT
                or not identity.get('staged_files_sha256')):
            raise ValueError('exact staged Fibo Turbo model/source identity required')
    return identity


def _export_metadata(targets, model_identity, network_profile='legacy'):
    from .reference import RoutedMLP
    result = dict(format=FORMAT, model_identity=_identity(model_identity), targets=list(targets),
                rank=8, alpha=8., particles=128, particle_dim=4, bridge_width=48, router_width=16,
                defaults=DEFAULTS, range=[0., 1.], frozen_base_exported=False,
                implementation=dict(backend_sha256=_digest(__file__),
                    projection_sha256=_digest(inspect.getsourcefile(TinyParticleProjection)),
                    bridge_sha256=_digest(inspect.getsourcefile(RoutedMLP)),
                    provenance_sha256=_digest(Path(__file__).with_name('fibo_provenance.json'))))
    if network_profile != 'legacy':
        from .fibo_hndl import PROFILE, HNDLParticleProjection, verified_builder
        if network_profile != PROFILE:
            raise ValueError('unknown Fibo network profile')
        _, identity = verified_builder()
        result['network_profile'] = identity
        result['implementation'].pop('projection_sha256')
        result['implementation'].pop('bridge_sha256')
        result['implementation'].update(
            hndl_binding_sha256=_digest(inspect.getsourcefile(HNDLParticleProjection)),
            projection_hndl_sha256=identity['projection_sha256'])
    return result


def target_modules(model):
    """Exact native trunk/refiner self- and injection-attention Q/K/V/O leaves.

    Perceiver, VLM, patch embedding, modulation, FFN and final head stay frozen.
    """
    expected = []
    cfg = model.config
    for group, count in [('noise_refiner', cfg.n_refiner_layers), ('layers', cfg.n_layers)]:
        for index in range(count):
            kinds = ['attention']
            if group == 'layers' and index in cfg.injection_layer_ids:
                kinds.append('cross_attention')
            for kind in kinds:
                expected.extend(f'{group}.{index}.{kind}.{leaf}'
                                for leaf in ('to_q', 'to_k', 'to_v', 'to_out.0'))
    actual = {name: module for name, module in model.named_modules() if _TARGET.fullmatch(name)}
    if not expected or set(actual) != set(expected) or any(not isinstance(m, nn.Linear) for m in actual.values()):
        raise ValueError('Fibo attention target architecture differs')
    if len({id(m) for m in actual.values()}) != len(actual):
        raise ValueError('aliased Fibo attention projections')
    from .fibo_hndl import HNDLParticleProjection
    if any(isinstance(getattr(m.forward, '__self__', None),
                      (TinyParticleProjection, HNDLParticleProjection)) for m in actual.values()):
        raise ValueError('Fibo particle slider already attached')
    return {name: actual[name] for name in expected}


class FiboParticleSlider(nn.Module):
    """Same routed rank8/alpha8/128x4 architecture as the reviewed tiny sibling."""
    def __init__(self, model, *, seed=7, network_profile='legacy'):
        super().__init__()
        from .fibo_hndl import PROFILE, HNDLParticleProjection, verified_builder
        if network_profile not in ('legacy', PROFILE):
            raise ValueError('unknown Fibo network profile')
        builder, identity = verified_builder() if network_profile == PROFILE else (None, None)
        self.network_profile = network_profile
        self.network_identity = identity
        targets = target_modules(model)
        model.eval().requires_grad_(False)
        self.target_names = list(targets)
        device = next(model.parameters()).device
        devices = [device] if device.type == 'cuda' else []
        # Attachment does not consume another task's training/evaluation RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.particles = nn.Parameter(torch.randn(128, 4, device=device, dtype=torch.float32))
            self.adapters = nn.ModuleDict()
            for name, module in targets.items():
                key = name.replace('.', '-')
                if network_profile == PROFILE:
                    projection = HNDLParticleProjection(key, module, self.particles, 8, 8.,
                                                        builder=builder, identity=identity)
                else:
                    projection = TinyParticleProjection(key, module, self.particles, 8, 8.)
                self.adapters[key] = projection.to(device=device, dtype=torch.float32)
        self.scale = 0.

    @contextmanager
    def scaled(self, scale):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('Fibo unipolar scale must be in [0,1]')
        previous = self.scale
        self.scale = float(scale)
        for module in self.adapters.values():
            module.multiplier = self.scale
        try:
            yield self
        finally:
            self.scale = previous
            for module in self.adapters.values():
                module.multiplier = previous

    def metadata(self, model_identity):
        return _export_metadata(self.target_names, model_identity, self.network_profile)

    def save(self, path, *, model_identity):
        from safetensors.torch import save_file
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Fibo exports require finite FP32 adapter and particle tensors')
        save_file(state, str(path), metadata={'fibo_slider': json.dumps(self.metadata(model_identity), sort_keys=True)})

    @classmethod
    def load(cls, model, path, *, model_identity, network_profile='legacy'):
        """Preflight the full export before attaching hooks or changing the host."""
        from safetensors import safe_open
        from safetensors.torch import load_file
        from .reference import RoutedMLP
        targets = target_modules(model)
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            meta = json.loads((handle.metadata() or {}).get('fibo_slider', '{}'))
        expected_meta = _export_metadata(targets, model_identity, network_profile)
        if meta != _canonical(expected_meta):
            raise ValueError('Fibo export protocol/model/target identity differs')
        state = load_file(str(path), device='cpu')
        if network_profile == 'legacy':
            with torch.random.fork_rng(devices=[]):
                bridge_shapes = {k: tuple(v.shape) for k, v in RoutedMLP(8, 8).state_dict().items()}
        expected = {'particles': (128, 4)}
        for name, module in targets.items():
            prefix = 'adapters.' + name.replace('.', '-')
            if network_profile != 'legacy':
                from .fibo_hndl import HNDLParticleProjection
                with torch.random.fork_rng(devices=[]):
                    projection = HNDLParticleProjection(prefix, module,
                        nn.Parameter(torch.zeros(128, 4)), 8, 8., attach=False)
                expected.update({prefix + '.' + k: tuple(v.shape) for k, v in projection.state_dict().items()})
                continue
            expected[prefix + '.lora_down.weight'] = (8, module.in_features)
            expected[prefix + '.lora_up.weight'] = (module.out_features, 8)
            expected.update({prefix + '.bridge.' + k: shape for k, shape in bridge_shapes.items()})
        if set(state) != set(expected) or any(tuple(state[k].shape) != shape
                or state[k].dtype != torch.float32 or not torch.isfinite(state[k]).all()
                for k, shape in expected.items()):
            raise ValueError('invalid Fibo FP32 export tensors')
        slider = cls(model, network_profile=network_profile)
        slider.load_state_dict(state, strict=True)
        return slider


def staged_pipeline(folder, manifest, *, device='cpu'):
    """Load only an already approved local snapshot; never fetch or accept terms.

    Native qualification is caller scheduled. This function is not called by
    CPU tests. The external staging receipt must enumerate every local file.
    """
    from diffusers import BriaFibo2Pipeline
    from diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2Transformer2DModel
    from diffusers import FlowMatchEulerDiscreteScheduler
    folder = Path(folder).resolve()
    identity = _identity(manifest)
    if identity.get('qualification') == 'CPU fixture only':
        raise ValueError('fixture identity cannot load pretrained weights')
    lock = json.loads(Path(__file__).with_name('fibo_provenance.json').read_text())
    for cls, key in [(BriaFibo2Pipeline, 'pipeline_bria_fibo2.py'),
                     (BriaFibo2Transformer2DModel, 'transformer_bria_fibo2.py'),
                     (FlowMatchEulerDiscreteScheduler, 'scheduling_flow_match_euler_discrete.py')]:
        if _digest(inspect.getsourcefile(cls)) != lock['diffusers_sources_sha256'][key]:
            raise ValueError('installed Fibo implementation differs from inspected source')
    files = identity['staged_files_sha256']
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual != set(files) or any(_digest(folder / name) != digest for name, digest in files.items()):
        raise ValueError('staged snapshot files/checksums differ')
    # Hub snapshots commonly symlink to the adjacent immutable blobs cache.
    # Verify content against the pinned public tree instead of rejecting that
    # native cache layout or relocating/deleting its resident files.
    official = lock['model_files']
    if set(files) - set(official):
        raise ValueError('staged snapshot contains unrecognized model files')
    required = {name for name in official if name.startswith(
        ('processor/', 'scheduler/', 'text_encoder/', 'transformer/', 'vae/', 'fonts/'))} | {'model_index.json'}
    if required - set(files):
        raise ValueError('staged snapshot is missing official model/processor/font files')
    for name in files:
        path, info = folder / name, official[name]
        if path.stat().st_size != info['bytes']:
            raise ValueError('staged Fibo upstream file size differs')
        if info['is_lfs']:
            # The gated public API redacts LFS SHA256s as stars. Its Git
            # pointer blob OID remains public and authenticates the SHA256
            # computed from an already authorized local weight file.
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{files[name]}\nsize {info['bytes']}\n".encode()
            pointer_oid = hashlib.sha1(f'blob {len(pointer)}\0'.encode() + pointer).hexdigest()
            matches = pointer_oid == info['git_blob_sha1'] and (
                info['lfs_sha256'] is None or files[name] == info['lfs_sha256'])
        else:
            content = path.read_bytes()
            matches = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == info['git_blob_sha1']
        if not matches:
            raise ValueError('staged Fibo content differs from pinned upstream revision')
    model_index = json.loads((folder / 'model_index.json').read_text())
    components = dict(transformer=['diffusers', 'BriaFibo2Transformer2DModel'],
                      scheduler=['diffusers', 'FlowMatchEulerDiscreteScheduler'],
                      vae=['diffusers', 'AutoencoderKLFlux2'],
                      text_encoder=['transformers', 'Qwen3VLForConditionalGeneration'],
                      processor=['transformers', 'Qwen3VLProcessor'])
    if model_index.get('_class_name') != 'BriaFibo2Pipeline' or any(
            model_index.get(key) != value for key, value in components.items()):
        raise ValueError('unrecognized Fibo pipeline component implementation')
    for name in ('Aileron-Regular.otf', 'DejaVuSans.ttf'):
        if 'fonts/' + name not in files:
            raise ValueError('stage official fonts to prevent native font downloads')
    pipe = BriaFibo2Pipeline.from_pretrained(str(folder), local_files_only=True,
                                           use_safetensors=True, dtype=torch.bfloat16)
    return pipe.to(device)


class FiboEditResponseAdapter(FrozenResponseAdapter):
    family = 'fibo_edit_turbo'
    time_coordinates = 'raw scheduler t in [0,1000]; model t=raw/1000; clean reference tokens at t=0'

    def __init__(self, pipeline, slider, *, model_identity):
        self.pipeline, self.transformer, self.slider = pipeline, pipeline.transformer, slider
        identity = _identity(model_identity)
        if list(slider.target_names) != [n.replace('-', '.') for n in slider.adapters]:
            raise ValueError('Fibo target registry differs')
        if any(p.requires_grad for p in self.transformer.parameters()):
            raise ValueError('Fibo base transformer must remain frozen')
        self.transformer.eval()
        for key in ('text_encoder', 'vae'):
            component = getattr(pipeline, key, None)
            if component is not None:
                component.eval().requires_grad_(False)
        network = AdapterParameters(slider.named_parameters(), slider.scaled)
        super().__init__(network, model_identity=identity)
        bind_native_gradient_checkpointing(self.transformer, network)

    def metadata(self):
        result = super().metadata()
        def source_identity(owner):
            try:
                path = inspect.getsourcefile(type(owner))
            except TypeError:
                path = None
            return dict(type=type(owner).__module__ + '.' + type(owner).__qualname__,
                        source_sha256=_digest(path) if path is not None else None)
        scheduler_config = getattr(self.pipeline.scheduler, 'config', {})
        def normalize(value):
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, (set, tuple, list)):
                return [normalize(v) for v in sorted(value)] if isinstance(value, set) else [normalize(v) for v in value]
            return value
        result.update(protocol=FORMAT, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                      diffusers_commit=DIFFUSERS_COMMIT, defaults=DEFAULTS,
                      particle_bank='actual shared dense-routed 128x4 parameter',
                      prediction='native flow velocity, no output negation',
                      geometry='native unpatched [1,C,H,W]; transformer owns patching and context packing',
                      conditioning='native six Qwen3-VL bundles, boolean padding mask, ordered VAE references',
                      render='official edit pipeline; native scheduler and FP32 inter-step latents',
                      training='raw velocity at captured native neutral-trajectory inputs; no local solver',
                      target_policy='unipolar positive teacher; matched references and x/t; no negative branch',
                      native_pipeline=source_identity(self.pipeline),
                      native_scheduler=source_identity(self.pipeline.scheduler),
                      scheduler_config=normalize(dict(scheduler_config)),
                      targets=self.slider.target_names,
                      qualification='CPU contract only; Turbo fine-tuning not recommended by model card')
        if self.slider.network_profile != 'legacy':
            result['custom_networks'] = self.slider.network_identity
        return result

    def _validate_condition(self, condition, latent):
        if not isinstance(condition, dict):
            raise ValueError('Fibo native edit condition dictionary required')
        refs = [key for key in condition if key.startswith('context_')]
        if not 1 <= len(refs) <= 4 or set(condition) != {'embeds', 'mask', *[f'context_{i}' for i in range(len(refs))]}:
            raise ValueError('Fibo EDIT requires 1..4 ordered reference latents')
        embeds, mask = condition['embeds'], condition['mask']
        cfg = self.transformer.config
        if (not isinstance(embeds, torch.Tensor) or embeds.ndim != 4 or embeds.shape[0] != 1
                or embeds.shape[1] != 1 + len(cfg.injection_layer_ids) or embeds.shape[-1] != cfg.cap_feat_dim):
            raise ValueError('Fibo native layer-bundled Qwen3-VL embeddings required')
        if mask is not None and (mask.dtype != torch.bool or mask.shape != (1, embeds.shape[2]) or not mask.any()):
            raise ValueError('Fibo boolean padding mask differs')
        for value in [embeds] + [condition[f'context_{i}'] for i in range(len(refs))]:
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite native edit conditioning')
        for key in refs:
            value = condition[key]
            if (value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != latent.shape[1]
                    or any(d % cfg.patch_size for d in value.shape[-2:])):
                raise ValueError('native reference lattice differs')

    def _velocity(self, condition, latent, timestep):
        self._validate_condition(condition, latent)
        if timestep.numel() != 1 or not torch.isfinite(timestep).all() or not 0 <= float(timestep) <= 1000:
            raise ValueError('Fibo raw scheduler timestep must be in [0,1000]')
        dtype = next(self.transformer.parameters()).dtype
        device = next(self.transformer.parameters()).device
        move = lambda v: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        return self.transformer(
            hidden_states=latent.to(device=device, dtype=dtype),
            timestep=timestep.reshape(1).to(device=device, dtype=torch.float32) / 1000,
            encoder_hidden_states=move(condition['embeds']),
            encoder_attention_mask=None if condition['mask'] is None else move(condition['mask']),
            context_latents=[move(condition[f'context_{i}']) for i in range(len(condition) - 2)],
            return_dict=False,
        )[0].float()

    def prepare(self, row):
        cfg = self.transformer.config
        if (row.schedule is not None or row.timestep is None or row.guidance != 1
                or row.scale != 1 or row.strength != 1):
            raise ValueError('Fibo Turbo training requires raw native velocity, scale1 and guidance1')
        if (row.latent.ndim != 4 or row.latent.shape[:2] != (1, cfg.in_channels)
                or row.latent.dtype != torch.float32 or any(d % cfg.patch_size for d in row.latent.shape[-2:])):
            raise ValueError('Fibo native FP32 spatial response lattice required')
        for condition in (row.neutral, row.positive):
            self._validate_condition(condition, row.latent)
        refs = [k for k in row.neutral if k.startswith('context_')]
        if set(refs) != {k for k in row.positive if k.startswith('context_')} or any(
                not torch.equal(row.neutral[k], row.positive[k]) for k in refs):
            raise ValueError('paired Fibo teachers require identical ordered references')
        return super().prepare(row)

    def _native_call(self, prompt, images, *, scale, seed, mask=None, height=None, width=None, output_type='latent'):
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 4 or any(i is None for i in images):
            raise ValueError('Fibo EDIT requires 1..4 ordered reference images')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('nonempty native edit instruction required')
        # Do not rewrite VGL, or call a prompt service / Gemini implicitly.
        if prompt.lstrip().startswith('{'):
            parsed = json.loads(prompt)
            if not isinstance(parsed.get('edit_instruction'), str) or not parsed['edit_instruction'].strip():
                raise ValueError('VGL edit JSON requires edit_instruction')
        if mask is not None and (len(images) != 1 or mask.size != images[0].size):
            raise ValueError('a native mask requires exactly one same-sized source image')
        if output_type not in ('latent', 'pil'):
            raise ValueError('Fibo output_type must be latent or pil')
        kwargs = dict(prompt=prompt, image=list(images), mask=mask, output_type=output_type, **DEFAULTS)
        if height is not None:
            kwargs['height'] = height
        if width is not None:
            kwargs['width'] = width
        device = next(self.transformer.parameters()).device
        kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)
        scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = deepcopy(scheduler)
        try:
            with torch.no_grad(), self.network.scaled(scale):
                return self.pipeline(**kwargs)
        finally:
            self.pipeline.scheduler = scheduler

    def capture_pair(self, neutral_prompt, positive_prompt, images, *, seed=7, mask=None, height=None, width=None):
        """Four immutable raw-velocity rows from the native neutral trajectory.

        The positive native pass obtains its independently encoded VGL condition.
        Targets are then evaluated at the SAME neutral x/t/reference fields.
        No training, model loading, decode or API prompt expansion occurs here.
        """
        states, conditions = [], []
        def capture(keep_states):
            def hook(module, args, kwargs):
                if args:
                    raise ValueError('expected official Fibo keyword transformer calls')
                if not conditions:
                    condition = dict(embeds=kwargs['encoder_hidden_states'], mask=kwargs['encoder_attention_mask'])
                    condition.update({f'context_{i}': value for i, value in enumerate(kwargs['context_latents'] or [])})
                    conditions.append(_cpu(condition))
                if keep_states:
                    states.append((_cpu(kwargs['hidden_states']).float(), _cpu(kwargs['timestep']).float() * 1000))
            return hook
        captured = []
        for prompt, keep in ((neutral_prompt, True), (positive_prompt, False)):
            conditions.clear()
            handle = self.transformer.register_forward_pre_hook(capture(keep), with_kwargs=True)
            try:
                self._native_call(prompt, images, scale=0., seed=seed, mask=mask, height=height, width=width)
            finally:
                handle.remove()
            if not conditions:
                raise ValueError('native pipeline did not call the Fibo transformer')
            captured.append(conditions[0])
        if len(states) != DEFAULTS['num_inference_steps']:
            raise ValueError('native Fibo Turbo trajectory did not contain four guidance-free steps')
        # Shared condition sketches require three dictionaries. Guidance is
        # exactly one, so the neutral alias is never evaluated as a CFG branch.
        rows = []
        for x, t in states:
            item = self.prepare(ResponseRow(captured[0], captured[1], captured[0], x, timestep=t))
            # The four rows share one frozen encoding pair on CPU. Do not retain
            # four copies of six long VLM bundles, or a dataset on the GPU.
            row = replace(item.row, neutral=captured[0], positive=captured[1], negative=captured[0])
            rows.append(PreparedResponse(row, item.target.cpu(), item.neutral.cpu()))
        return rows

    @torch.no_grad()
    def render_edit(self, prompt, images, *, scale=1., seed=7, mask=None, height=None, width=None, output_type='pil'):
        """Render through the exact same official reference/mask/packing path."""
        return self._native_call(prompt, images, scale=scale, seed=seed, mask=mask, height=height,
                                 width=width, output_type=output_type)


class FiboCachedRows:
    """Immutable checksum-bound row files, selected-row transfers, bounded LRU.

    Constructing a provider loads only its index. Core consumes this via its
    existing lazy-provider API and binds the index, policies and checksums.
    """
    def __init__(self, folder, *, index_sha256, model_identity, device='cpu', capacity=2):
        self.folder = Path(folder).resolve()
        path = self.folder / 'index.json'
        if _digest(path) != index_sha256:
            raise ValueError('Fibo cache index checksum differs')
        self.index = json.loads(path.read_text())
        if (self.index.get('format') != FORMAT or self.index.get('model_identity') != _identity(model_identity)
                or self.index.get('defaults') != DEFAULTS or len(self.index.get('rows', [])) < 2):
            raise ValueError('Fibo cache model/protocol differs')
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('bounded Fibo cache capacity must be in [1,64]')
        self.device, self.capacity = torch.device(device), capacity
        self.response_shape = tuple(self.index['response_shape'])
        self._identity = dict(index_sha256=index_sha256, format=FORMAT,
                             model_identity=self.index['model_identity'], defaults=DEFAULTS,
                             condition_policy=self.index['condition_policy'])
        self.cache = OrderedDict()
        self.context_cache = OrderedDict()

    def __len__(self):
        return len(self.index['rows'])

    def identity(self):
        return deepcopy(self._identity)

    def conditions(self, indices):
        return torch.tensor([self.index['rows'][int(i)]['condition'] for i in indices], dtype=torch.float32, device=self.device)

    def __getitem__(self, index):
        if index not in self.cache:
            record = self.index['rows'][index]
            path = (self.folder / record['file']).resolve()
            if not path.is_relative_to(self.folder) or _digest(path) != record['sha256']:
                raise ValueError('Fibo cached row checksum differs')
            data = torch.load(path, map_location=self.device, weights_only=True)
            for key in ('neutral', 'positive', 'negative'):
                name = data['row'][key]
                if name not in self.context_cache:
                    asset = self.index['contexts'][name]
                    context_path = (self.folder / asset['file']).resolve()
                    if not context_path.is_relative_to(self.folder) or _digest(context_path) != asset['sha256']:
                        raise ValueError('Fibo cached context checksum differs')
                    self.context_cache[name] = torch.load(context_path, map_location=self.device, weights_only=True)
                    if len(self.context_cache) > 2 * self.capacity:
                        self.context_cache.popitem(last=False)
                self.context_cache.move_to_end(name)
                data['row'][key] = self.context_cache[name]
            row = ResponseRow(**data['row'])
            item = PreparedResponse(row, data['target'], data['neutral'])
            if tuple(item.target.shape) != self.response_shape or tuple(item.neutral.shape) != self.response_shape:
                raise ValueError('Fibo cache response lattice differs')
            self.cache[index] = item
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        # Caller mutation must not poison a later lazy fetch or teacher target.
        return deepcopy(self.cache[index])


def write_cache(folder, prepared, *, model_identity):
    from dataclasses import fields
    from .pg09_images import CONDITION_FEATURE_POLICY
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    prepared = tuple(prepared)
    if len(prepared) < 2 or any(p.row.schedule is not None for p in prepared):
        raise ValueError('Fibo cache requires at least two raw velocity rows')
    conditions = image_condition_features(prepared).cpu().tolist()
    shape = tuple(prepared[0].target.shape)
    records, contexts, seen = [], {}, {}
    for index, item in enumerate(prepared):
        if tuple(item.target.shape) != shape or tuple(item.neutral.shape) != shape:
            raise ValueError('Fibo fixed-shape cache cannot mix response lattices')
        row = {field.name: _cpu(getattr(item.row, field.name)) for field in fields(item.row)
               if field.name not in ('neutral', 'positive', 'negative')}
        for key in ('neutral', 'positive', 'negative'):
            from .pg09 import _hash
            value = getattr(item.row, key)
            if id(value) not in seen:
                seen[id(value)] = _hash(value)
            name = seen[id(value)]
            if name not in contexts:
                path = folder / f'context-{name}.pt'
                torch.save(_cpu(value), path)
                contexts[name] = dict(file=path.name, sha256=_digest(path))
            row[key] = name
        path = folder / f'row-{index}.pt'
        torch.save(dict(row=row, target=_cpu(item.target), neutral=_cpu(item.neutral)), path)
        records.append(dict(file=path.name, sha256=_digest(path), condition=conditions[index]))
    index = dict(format=FORMAT, model_identity=_identity(model_identity), defaults=DEFAULTS,
                 condition_policy=CONDITION_FEATURE_POLICY, response_shape=list(shape), rows=records, contexts=contexts)
    path = folder / 'index.json'
    path.write_text(json.dumps(index, sort_keys=True, allow_nan=False) + '\n')
    return _digest(path)


def build_fibo_game(adapter, provider, *, response_scale, total_steps, batch_size=2, seed=7,
                    optimizer_dense_profile='auto'):
    """Build the existing public PG game, preserving its optimized native recipe.

    A resource owner must separately admit any pretrained/GPU execution. This
    function neither loads a model nor schedules a process. The caller supplies
    a declared response scale; no product normalization/loss is introduced.
    """
    from .pg09 import ConditionalPG09, PG_SOURCE_PROFILE
    from .field_critic import FieldCritic, seed_initialization_rng
    if provider.index['model_identity'] != adapter.model_identity:
        raise ValueError('Fibo cache and backend staged model identity differ')
    shape = provider.response_shape
    if len(shape) != 4 or shape[:2] != (1, adapter.transformer.config.in_channels):
        raise ValueError('Fibo cache must retain native spatial response geometry')
    device = next(adapter.network.parameters()).device
    if provider.device != device:
        raise ValueError('Fibo provider and backend devices differ')
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        seed_initialization_rng(seed, devices)
        if adapter.slider.network_profile == 'legacy':
            critic = FieldCritic(shape[1], provider.conditions([0]).shape[1])
        else:
            from .fibo_hndl import HNDLFieldCritic
            critic = HNDLFieldCritic(shape[1], provider.conditions([0]).shape[1], shape[2:])
        critic = critic.to(device=device, dtype=torch.float32)
    return ConditionalPG09(adapter, provider, critic, conditions=None, particles=adapter.slider.particles,
        response_domain='continuous_field', response_scale=response_scale, preset='bispf_optimized',
        batch_size=batch_size, seed=seed, total_steps=total_steps,
        pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain='auto',
        optimizer_dense_profile=optimizer_dense_profile)
