"""Archive-faithful Sana Happy/SD3 Smile response ports and online row contract.

No loaders, worker launchers, optimizer, loss, or training loop. Production use
requires root admission, source/parity qualification and reviewed online semantics.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

from .pg09_backends import PreparedResponse, bind_native_gradient_checkpointing
from .pg09_images import ImageResponseAdapter, image_condition_features

CORE_COMMIT = '3e94b13eb5d6cd1ceee3f018f019a68603f35696'
CORE_SOURCE_SHA256 = '426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256'


def digest(value):
    def encode(v):
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            return dict(shape=list(x.shape), dtype=str(x.dtype),
                sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if is_dataclass(v):
            return encode(asdict(v))
        if isinstance(v, dict):
            if any(not isinstance(k,str) for k in v):
                raise ValueError('campaign identities require string dictionary keys')
            return {k: encode(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [encode(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value):
        return type(value)(**{k: snapshot(v) for k, v in vars(value).items()})
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(v) for v in value)
    return deepcopy(value)


class XattnDeltaNetwork(nn.Module):
    """Same full-attn2 capacity as Sana's148M-parameter archive, no LoRA substitute.

    The frozen original W and trainable FP32 delta implement W+scale*delta by
    functional_call. Base weights never enter optimizers or adapter exports.
    """
    def __init__(self, model):
        super().__init__()
        if getattr(model, 'is_gradient_checkpointing', False):
            raise ValueError('functional full-xattn replay needs separate checkpoint qualification')
        model.requires_grad_(False).eval()
        object.__setattr__(self, 'model', model)
        selected = [(n, p) for n, p in model.named_parameters() if 'attn2' in n]
        if not selected or any(p.dtype != torch.float32 for _, p in selected):
            raise ValueError('archive Sana xattn requires nonempty FP32 attn2 weights')
        self.names = tuple(n for n, _ in selected)
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros_like(p)) for _, p in selected])
        self.scale = 1.

    @contextmanager
    def scaled(self, scale):
        if not torch.isfinite(torch.tensor(scale)):
            raise ValueError('finite strength required')
        previous, self.scale = self.scale, float(scale)
        try:
            yield self
        finally:
            self.scale = previous

    def call(self, *args, **kwargs):
        base = dict(self.model.named_parameters())
        mapping = {n: base[n] if self.scale == 0 else base[n] + self.scale * delta
                   for n, delta in zip(self.names, self.weights)}
        return functional_call(self.model, mapping, args, kwargs, strict=False)


class FunctionalSanaTransformer(nn.Module):
    def __init__(self, network):
        super().__init__()
        object.__setattr__(self, 'network', network)
        object.__setattr__(self, 'model', network.model)

    def forward(self, *args, **kwargs):
        return self.network.call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'model'), name)


@dataclass(frozen=True)
class CampaignRow:
    neutral: dict
    positive: dict
    negative: dict
    latent: torch.Tensor
    timestep: torch.Tensor
    scale: float = 1.
    strength: float = 1.
    guidance: float = 1.
    schedule: object = None
    step_index: int | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    provenance: dict | None = None
    unconditional: dict | None = None


class SanaHappyCampaignBackend:
    """Raw neutral student vs archived frozen CFG4.5 positive teacher, fixed t500."""
    def __init__(self, model, *, model_identity):
        self.transformer = model
        self.network = XattnDeltaNetwork(model)
        self.model_identity = deepcopy(model_identity)

    def response(self, condition, row, scale):
        if float(row.timestep) != 500. or row.latent.ndim != 4:
            raise ValueError('Sana Happy archive uses native4D raw t500')
        embeds = condition['embeds'].to(row.latent.device, torch.float32)
        # Original training backend discarded the tokenizer mask and used ones.
        # Rendering below preserves actual native masks. This known gap is declared.
        mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
        t = row.timestep.reshape(1).to(row.latent.device) * getattr(self.transformer.config, 'timestep_scale', 1.)
        with self.network.scaled(scale):
            value = self.network.call(row.latent.float(), encoder_hidden_states=embeds,
                encoder_attention_mask=mask, timestep=t, return_dict=False)[0]
        if value.shape[1] == row.latent.shape[1] * 2:
            value = value.chunk(2, dim=1)[0]
        return value.float()

    @torch.no_grad()
    def prepare(self, row):
        row = snapshot(row)
        if row.unconditional is None:
            raise ValueError('Sana teacher requires explicit empty-prompt encoding; minus is canary only')
        if row.scale != 1 or row.guidance != 1 or row.strength != 1 or row.schedule is not None:
            raise ValueError('Sana archive student is raw neutral at scale1; teacher CFG is separately4.5')
        neutral = self.response(row.neutral, row, 0.)
        positive = self.response(row.positive, row, 0.)
        empty = self.response(row.unconditional, row, 0.)
        target = empty + 4.5 * (positive - empty)
        return PreparedResponse(row, target.detach(), neutral.detach())

    def generated(self, prepared):
        return self.response(prepared.row.neutral, prepared.row, 1.)

    def metadata(self):
        return dict(protocol='sana-happy-archive-response-v1', model_identity=self.model_identity,
            target='frozen empty +4.5*(frozen positive-frozen empty)', student='raw neutral',
            response='native4D FP32 raw velocity at t500', particles=None,
            architecture='all original attn2 weights represented by FP32 deltas',
            training_mask='ones, matching archived teacher', minus_prompt='canary only; never positive teacher', render_mask='actual native cached mask',
            render_cfg='4.5, both native branches adapted as in archive',
            known_protocol_gap='raw student/CFG teacher + training ones-mask/render true-mask',
            checkpointing='native disabled as in archive; functional replay qualification pending',
            quality_qualified=False)


class SD3SmileCampaignBackend:
    """Keep the native scheduler-step endpoint and BF16 target arithmetic."""
    def __init__(self, model, network, scheduler, *, model_identity, archive_prefix=None):
        self.transformer, self.network = model, network
        self.scheduler = deepcopy(scheduler)
        self.archive_prefix = archive_prefix
        self.model_identity = deepcopy(model_identity)
        bind_native_gradient_checkpointing(model, network)
        self.raw = ImageResponseAdapter('sd3', model, network, model_identity=model_identity)

    def endpoint(self, condition, row, scale):
        if row.step_index is None or row.timesteps is None or row.sigmas is None:
            raise ValueError('SD3 endpoint requires the actual selected scheduler cursor and intervals')
        if not 1 <= row.step_index <= 49 or len(row.timesteps) != 50 or len(row.sigmas) != 51:
            raise ValueError('archive SD3 selects1..49 of50 native steps')
        if not torch.equal(row.timestep.reshape(()), row.timesteps[row.step_index].reshape(())):
            raise ValueError('SD3 raw timestep does not match its captured scheduler index')
        with self.network.scaled(scale):
            velocity = self.raw._velocity(condition, row.latent, row.timestep)
        scheduler = deepcopy(self.scheduler)
        scheduler.timesteps = row.timesteps.detach().clone()
        scheduler.sigmas = row.sigmas.detach().clone()
        scheduler.num_inference_steps = 50
        scheduler._step_index = row.step_index
        # Use the staged native scheduler.step, including its FP32 internal arithmetic.
        out = scheduler.step(velocity.to(row.latent.dtype), row.timestep.reshape(()),
                             row.latent, return_dict=False)[0]
        return out.to(row.latent.dtype)

    @torch.no_grad()
    def prepare(self, row):
        row = snapshot(row)
        if row.scale != 1 or row.guidance != 1 or row.strength != 1 or row.schedule is not None:
            raise ValueError('archive SD3 response training uses scale1/CFG1')
        neutral = self.endpoint(row.neutral, row, 0.)
        positive = self.endpoint(row.positive, row, 0.)
        negative = self.endpoint(row.negative, row, 0.)
        # Do not simplify to positive: BF16 subtraction/addition is not bit-exact cancellation.
        target = neutral + (positive - negative)
        return PreparedResponse(row, target.float().detach(), neutral.float().detach())

    def generated(self, prepared):
        return self.endpoint(prepared.row.neutral, prepared.row, 1.).float()

    def metadata(self):
        return dict(protocol='sd3-smile-archive-endpoint-v1', model_identity=self.model_identity,
            target='native BF16 neutral_endpoint + (positive_endpoint-negative_endpoint)',
            student='native neutral scheduler-step endpoint', particles=None,
            prefix='current adapter, no_grad, selected1..49 of50, detached before paired endpoint',
            response='native4D scheduler-step endpoint; never raw velocity',
            render_cfg='3, both native branches adapted as in archive',
            archive_prefix_source=(hashlib.sha256(Path(inspect.getsourcefile(self.archive_prefix)).read_bytes()).hexdigest()
                                   if self.archive_prefix is not None else None),
            scheduler_source=hashlib.sha256(Path(inspect.getsourcefile(type(self.scheduler))).read_bytes()).hexdigest(),
            scheduler_config={k: sorted(v) if k == '_use_default_values' else v
                              for k, v in self.scheduler.config.items()}, quality_qualified=False)


@torch.no_grad()
def render_archived_case(backend, pipe, conditional, empty, *, family, seed, scale):
    """Call the actual native pipeline with the archived CFG, masks and CPU seed.

    Source/CPU contract only; real frozen teacher/pixel/sampler parity remains a
    root-owned admission gate. No alternative sampler/decoder is implemented.
    """
    common = dict(height=512, width=512, generator=torch.Generator('cpu').manual_seed(seed))
    original = pipe.transformer
    if original is not backend.transformer:
        raise ValueError('render pipeline must reference the same staged base model')
    if family == 'sana':
        kwargs = dict(prompt_embeds=conditional['embeds'].float(),
            prompt_attention_mask=conditional['mask'], negative_prompt_embeds=empty['embeds'].float(),
            negative_prompt_attention_mask=empty['mask'], num_inference_steps=20, guidance_scale=4.5)
        object.__setattr__(pipe, 'transformer', FunctionalSanaTransformer(backend.network))
    elif family == 'sd3':
        kwargs = dict(prompt_embeds=conditional['embeds'], pooled_prompt_embeds=conditional['pooled'],
            negative_prompt_embeds=empty['embeds'], negative_pooled_prompt_embeds=empty['pooled'],
            num_inference_steps=28, guidance_scale=3.)
    else:
        raise ValueError('only the two observed image failure cases are admitted to this proposal')
    try:
        with preserve_evaluation_rng(backend.network), backend.network.scaled(scale):
            return pipe(**common, **kwargs)
    finally:
        object.__setattr__(pipe, 'transformer', original)


class OnlineSnapshotProvider:
    """Proposed online contract: freeze each selected row once per logical update.

    Descriptors/condition table/source/algorithm identity are immutable. Dynamic
    prefix states and targets are derived from the loaded generator at step start,
    not a new prior or a frozen-prefix substitute. Only the active D/G batch union
    is retained. No private evolving RNG is introduced: descriptor assets bind all
    initial-noise bytes/selected time; the builder must consume no global RNG.
    """
    def __init__(self, backend, descriptors, conditions, response_shape, builder, *, protocol_identity):
        self.backend = backend
        self.descriptors = tuple(snapshot(descriptors))
        self._conditions = conditions.detach().clone()
        if conditions.requires_grad or conditions.ndim != 2 or len(conditions) != len(descriptors):
            raise ValueError('complete fixed descriptor condition table required')
        if not torch.isfinite(conditions).all():
            raise ValueError('nonfinite descriptor conditions')
        self.response_shape = tuple(response_shape)
        if len(self.response_shape) != 4 or self.response_shape[0] != 1:
            raise ValueError('these campaign responses are genuinely4D')
        self._builder = builder
        self._identity = dict(schema='online-current-adapter-snapshot-proposal-v1',
            algorithm=deepcopy(protocol_identity), descriptors_sha256=digest(self.descriptors),
            conditions_sha256=digest(self._conditions),
            builder_source=hashlib.sha256(Path(inspect.getsourcefile(builder)).read_bytes()).hexdigest(),
            builder_name=builder.__module__+'.'+builder.__qualname__,
            snapshot='per-logical-update/version-checked/current-adapter-prefix/no-grad',
            mutable_provider_rng=False)
        self._step = None
        self._cache = {}
        self.audit = []

    def __len__(self):
        return len(self.descriptors)

    def identity(self):
        return deepcopy(self._identity)

    def begin_step(self, completed_step):
        if self._step is not None:
            raise RuntimeError('previous online snapshot is still active')
        if type(completed_step) is not int or completed_step < 0:
            raise ValueError('nonnegative logical completed step required')
        self._step = completed_step
        self._versions = tuple(p._version for p in self.backend.network.parameters())
        self._g_version = digest(self.backend.network.state_dict())
        self._cache = {}
        self.audit = []

    def __getitem__(self, index):
        if self._step is None:
            raise RuntimeError('online rows require an explicit step snapshot')
        if tuple(p._version for p in self.backend.network.parameters()) != self._versions:
            raise RuntimeError('generator changed while paired online snapshot was active')
        if index not in self._cache:
            devices = sorted({p.device.index or 0 for p in self.backend.network.parameters()
                              if p.device.type == 'cuda'})
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = {device: torch.cuda.get_rng_state(device).clone() for device in devices}
                import random
                import numpy as np
                python_rng, numpy_rng = random.getstate(), np.random.get_state()
                try:
                    item = self._builder(self.backend, self.descriptors[index], self._step)
                finally:
                    python_changed = random.getstate() != python_rng
                    current_numpy = np.random.get_state()
                    numpy_changed = (current_numpy[0] != numpy_rng[0] or
                        not np.array_equal(current_numpy[1], numpy_rng[1]) or current_numpy[2:] != numpy_rng[2:])
                    random.setstate(python_rng)
                    np.random.set_state(numpy_rng)
                if tuple(p._version for p in self.backend.network.parameters()) != self._versions:
                    raise RuntimeError('generator changed inside online row builder')
                changed = python_changed or numpy_changed or (not torch.equal(cpu_rng, torch.get_rng_state()) or
                    any(not torch.equal(value, torch.cuda.get_rng_state(device))
                        for device, value in cuda_rng.items()))
                if changed:
                    raise RuntimeError('online builder must use fixed descriptor noise, not global RNG')
                if (item.target.requires_grad or item.neutral.requires_grad or
                        tuple(item.target.shape) != self.response_shape):
                    raise ValueError('online builder changed target shape or left a teacher graph')
                item = snapshot(item)
            self._cache[index] = item
            self.audit.append(dict(step=self._step, g_version_sha256=self._g_version, index=index,
                descriptor=digest(self.descriptors[index]), prefix=digest(item.row.latent),
                initial_noise=self.descriptors[index].get('initial_noise_sha256'),
                paired_state=digest(item.row), target=digest(item.target), neutral=digest(item.neutral)))
        return self._cache[index]

    def conditions(self, indices):
        for index in set(indices.tolist()):
            self[index]
        return self._conditions[indices.to(self._conditions.device)]

    def end_step(self):
        self._cache = {}
        self._step = None


def online_campaign_step(game, provider):
    """One core-owned GAN step, not a second optimizer/training loop."""
    if game.provider is not provider or game.backend is not provider.backend:
        raise ValueError('online provider/game ownership mismatch')
    provider.begin_step(game.completed_steps)
    try:
        return game.step()
    finally:
        provider.end_step()


def campaign_conditions(prepared, *, family):
    """Complete global condition table, including empty teacher and step interval."""
    if family not in ('sana','sd3'):
        raise ValueError('only the two observed failure campaign protocols')
    augmented = []
    for item in prepared:
        row = item.row
        extra = dict(teacher_cfg=4.5 if family=='sana' else 1.,student_cfg=1.)
        if row.unconditional is not None:
            extra.update({'unconditional_'+k:v for k,v in row.unconditional.items()})
        if row.step_index is not None:
            extra['selected_step_index'] = float(row.step_index)
            extra['sigma_interval'] = row.sigmas[row.step_index+1]-row.sigmas[row.step_index]
        def condition(value):
            return {**value,**extra}
        bound = replace(row,neutral=condition(row.neutral),positive=condition(row.positive),
                        negative=condition(row.negative))
        augmented.append(PreparedResponse(bound,item.target,item.neutral))
    return image_condition_features(augmented)


def build_sana_snapshot(backend, descriptor, logical_step):
    """Root supplies a source/checksum-bound row with original seeded noise/text."""
    row = snapshot(descriptor['input_row'])
    if digest(row.latent) != descriptor['initial_noise_sha256']:
        raise ValueError('initial noise differs from immutable descriptor')
    return backend.prepare(replace(row, provenance=dict(logical_step=logical_step,
        initial_noise_sha256=descriptor['initial_noise_sha256'],
        protocol='sana-fixed-t500-no-prefix')))


def build_sd3_current_prefix_snapshot(backend, descriptor, logical_step):
    """Reuse the actual archived no-grad prefix helper, with CURRENT adapter on.

    Descriptor input row is initial noise, not a frozen precomputed prefix. The
    parent must stage exact encoded text/noise assets after admission; this code
    neither extracts an old checkpoint nor loads any model/cache file.
    """
    if backend.archive_prefix is None:
        raise ValueError('bind the source-verified original diffusion_sd3 helper before prefix execution')
    row = snapshot(descriptor['input_row'])
    if digest(row.latent) != descriptor['initial_noise_sha256']:
        raise ValueError('initial noise differs from immutable descriptor')
    index = descriptor['selected_step_index']
    if not 1 <= index <= 49:
        raise ValueError('selected SD3 prefix index must be1..49')
    scheduler = deepcopy(backend.scheduler)
    scheduler.set_timesteps(50, device=row.latent.device)
    empty = descriptor['unconditional']
    with torch.no_grad(), backend.network.scaled(1.):
        prefix = backend.archive_prefix(backend.transformer, scheduler, row.latent,
            text_embeddings=torch.cat([empty['embeds'], row.neutral['embeds']]),
            add_text_embeddings=torch.cat([empty['pooled'], row.neutral['pooled']]),
            start_timesteps=0, total_timesteps=index, guidance_scale=1.)
    row = replace(row, latent=prefix.detach(), timestep=scheduler.timesteps[index].detach(),
        step_index=index, timesteps=scheduler.timesteps.detach().clone(), sigmas=scheduler.sigmas.detach().clone(),
        provenance=dict(logical_step=logical_step, initial_noise_sha256=descriptor['initial_noise_sha256'],
                        prefix='current-adapter/native-original-diffusion_sd3/no_grad',
                        prefix_sha256=digest(prefix)))
    return backend.prepare(row)


@contextmanager
def preserve_evaluation_rng(network):
    import random
    import numpy as np
    devices=sorted({p.device.index or 0 for p in network.parameters() if p.device.type=='cuda'})
    python_rng,numpy_rng=random.getstate(),np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


class ImageFailureCritic(nn.Module):
    """All native spatial coordinates plus useful fixed conditions; raw logits."""
    score_bound=0
    def __init__(self,response_shape,conditioning_dim):
        super().__init__()
        import math
        if len(response_shape)!=4 or response_shape[0]!=1:
            raise ValueError('native4D image field shape required')
        self.response_shape=tuple(response_shape[1:])
        self.net=nn.Sequential(nn.Linear(math.prod(self.response_shape)+conditioning_dim,256),
            nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self,response,condition):
        if tuple(response.shape[1:])!=self.response_shape:
            raise ValueError('critic response geometry changed')
        return self.net(torch.cat([response.flatten(1),condition],dim=1)).squeeze(-1)
