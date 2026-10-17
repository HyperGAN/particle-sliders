"""Lazy, read-only MOSS campaign cache bridge into the shared conditional GAN.

No model loader, GPU admission, trainer loss, optimizer or background process.
The coordinator supplies an admitted transformer and optional native decoder.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .pg09_backends import (
    FlowSchedule, MossResponseAdapter, PreparedResponse, ResponseRow,
    checkpoint_with_adapter_scale, moss_pipeline_decoder, save_adapter, load_adapter,
)


def canonical_digest(value):
    # Match moss_sliders.campaign.digest exactly, including JSON separators.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


@dataclass(frozen=True)
class CampaignAddress:
    bundle: int
    path: int
    position: int
    sign: int


class MossCampaignRows:
    """Immutable cache identity with a bounded CPU LRU and lazy native responses.

    Dataset order: bundle(row/seed), neutral/positive trajectory, timestep, +/-.
    The empty text is CFG-negative conditioning; the cached teacher negative
    pole is the neutral caption. This distinction is part of the identity.
    """
    condition_policy = dict(
        version='moss-context-time-sign-v1', pooling='mask-weighted neutral Qwen mean',
        normalization='per-context pooled RMS, epsilon=1e-12',
        suffix=['native_timestep/1000', 'sign', 'strength', 'training_guidance/3', 'positive_path'],
        row_id_feature=False, teacher_negative='neutral caption', cfg_negative='empty prompt',
    )

    def __init__(self, directory, *, split='train', strength=1., device='cpu',
                 bundle_cache_size=4, context_cache_size=32):
        from safetensors import safe_open
        self.directory = Path(directory).resolve()
        if split not in ('train', 'dev') or not math.isfinite(strength) or strength <= 0:
            raise ValueError('train/dev split and positive finite strength required')
        if min(bundle_cache_size, context_cache_size) < 1:
            raise ValueError('bounded nonempty cache sizes required')
        self.split, self.strength, self.device = split, float(strength), torch.device(device)
        self.bundle_cache_size, self.context_cache_size = bundle_cache_size, context_cache_size
        self._bundles, self._contexts, self._pooled, self._times = OrderedDict(), OrderedDict(), {}, {}
        self.recipe = json.loads((self.directory / 'recipe.json').read_text())
        self.recipe_identity = canonical_digest(self.recipe)
        if self.recipe.get('negative_caption_policy') != 'neutral equals negative; target is base +/- (positive-neutral)':
            raise ValueError('cache signed-pole convention is not the reviewed positive-minus-neutral protocol')
        self.rows = self.recipe['training_rows'] + self.recipe['development_rows']
        self.row_by_id = {r['id']: r for r in self.rows}
        if len(self.row_by_id) != len(self.rows):
            raise ValueError('duplicate row identifiers')
        self.prompts = list(dict.fromkeys([r[k] for r in self.rows for k in ('neutral', 'positive')] + ['']))
        self.prompt_index = {p: i for i, p in enumerate(self.prompts)}
        embeddings = self.directory / 'embeddings.safetensors'
        emeta = json.loads((self.directory / 'embeddings.json').read_text())
        if (emeta['prompts_sha256'] != canonical_digest(self.prompts) or
                emeta['sha256'] != file_digest(embeddings)):
            raise ValueError('encoded prompt cache checksum/identity mismatch')
        self._embedding_path = embeddings
        self._embedding_stat = embeddings.stat()
        index = json.loads((self.directory / 'target-index.json').read_text())
        if index['identity'] != self.recipe_identity:
            raise ValueError('target index recipe identity mismatch')
        expected = [(r['id'], seed, 'train') for r in self.recipe['training_rows'] for seed in r['seeds']]
        expected += [(r['id'], self.recipe['evaluation_seed'], 'dev') for r in self.recipe['development_rows']]
        actual = [(r['row'], r['seed'], r['split']) for r in index['records']]
        if actual != expected:
            raise ValueError('target index order, row/seed coverage or split differs')
        self.records = []
        asset_identity = []
        for record in index['records']:
            filename = f"{record['split']}-{record['row']}-{record['seed']}.safetensors"
            if Path(record['path']).name != filename:
                raise ValueError('target filename differs from row identity')
            path = self.directory / 'targets' / filename
            if path.resolve().parent != (self.directory / 'targets').resolve():
                raise ValueError('target path escapes cache directory')
            sidecar = json.loads(path.with_suffix('.json').read_text())
            target_identity = canonical_digest(dict(identity=self.recipe_identity, row=record['row'],
                                                   seed=record['seed'], split=record['split']))
            if sidecar['identity'] != target_identity:
                raise ValueError('target sidecar row/seed/split identity mismatch')
            item = dict(record, path=str(path), sha256=sidecar['sha256'])
            asset_identity.append(dict(row=record['row'], seed=record['seed'], split=record['split'],
                                       sha256=sidecar['sha256'], identity=sidecar['identity']))
            if record['split'] == split:
                self.records.append(item)
        if not self.records:
            raise ValueError('empty frozen response split')
        self.positions = int(self.recipe['settings']['target_steps'])
        with safe_open(self.records[0]['path'], framework='pt', device='cpu') as file:
            shape = file.get_slice('latent').get_shape()
        if len(shape) != 6 or shape[:3] != [2, self.positions, 1]:
            raise ValueError('native cached tensor schema must be [2,T,1,C,H,W]')
        self.response_shape = tuple(shape[2:])
        with safe_open(embeddings, framework='pt', device='cpu') as file:
            self.context_width = file.get_slice('e0').get_shape()[-1]
        self.condition_width = self.context_width + 5
        self._identity = dict(protocol='lazy-moss-campaign-pg090-v1', concept=self.recipe['concept'],
            model_revision=self.recipe['model_revision'], recipe_sha256=self.recipe_identity,
            embeddings_sha256=emeta['sha256'], prompt_order_sha256=emeta['prompts_sha256'],
            target_index_sha256=file_digest(self.directory / 'target-index.json'),
            target_assets_sha256=canonical_digest(asset_identity), split=split, strength=self.strength,
            response_shape=list(self.response_shape), condition_width=self.condition_width,
            condition_policy=self.condition_policy, rows=len(self),
            order='bundle/path/position/sign(+1,-1)', training_guidance=1.,
            target='cached frozen neutral + sign*strength*cached(positive-neutral)',
            target_verification='sidecar identities eagerly; asset SHA256 on every LRU load')

    def __len__(self):
        return len(self.records) * 2 * self.positions * 2

    def identity(self):
        return deepcopy(self._identity)

    def address(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        state, sign = divmod(index, 2)
        bundle, local = divmod(state, 2 * self.positions)
        path, position = divmod(local, self.positions)
        return CampaignAddress(bundle, path, position, 1 if sign == 0 else -1)

    def _bundle(self, bundle):
        from safetensors.torch import load_file
        if bundle in self._bundles:
            self._bundles.move_to_end(bundle)
            return self._bundles[bundle]
        record = self.records[bundle]
        if file_digest(record['path']) != record['sha256']:
            raise ValueError('frozen target checksum mismatch')
        data = load_file(record['path'], device='cpu')
        expected = (2, self.positions, *self.response_shape)
        if (set(data) != {'latent', 'base', 'delta', 'time'} or
                any(tuple(data[k].shape) != expected or data[k].dtype != torch.float32
                    for k in ('latent', 'base', 'delta')) or
                tuple(data['time'].shape) != (self.positions,) or
                any(not t.isfinite().all() for t in data.values())):
            raise ValueError('cached native response schema/dtype/finite values differ')
        self._bundles[bundle] = data
        while len(self._bundles) > self.bundle_cache_size:
            self._bundles.popitem(last=False)
        self._times[bundle] = data['time'].clone()
        return data

    def context(self, prompt):
        from safetensors import safe_open
        if prompt not in self.prompt_index:
            raise ValueError('prompt absent from immutable encoder cache')
        if prompt in self._contexts:
            self._contexts.move_to_end(prompt)
            return self._contexts[prompt]
        stat = self._embedding_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (self._embedding_stat.st_size, self._embedding_stat.st_mtime_ns):
            raise ValueError('encoded prompt cache changed after admission')
        index = self.prompt_index[prompt]
        with safe_open(self._embedding_path, framework='pt', device='cpu') as file:
            embeds, mask = file.get_tensor(f'e{index}'), file.get_tensor(f'm{index}')
        if (embeds.ndim != 3 or embeds.shape[0] != 1 or mask.shape != embeds.shape[:2] or
                embeds.shape[-1] != self.context_width or mask.dtype != torch.bool or
                not mask.any() or not embeds.isfinite().all()):
            raise ValueError('invalid frozen context/mask')
        self._contexts[prompt] = (embeds, mask)
        while len(self._contexts) > self.context_cache_size:
            self._contexts.popitem(last=False)
        return embeds, mask

    def __getitem__(self, index):
        address = self.address(index)
        data = self._bundle(address.bundle)
        source = self.row_by_id[self.records[address.bundle]['row']]
        neutral = tuple(t.to(self.device).clone() for t in self.context(source['neutral']))
        positive = tuple(t.to(self.device).clone() for t in self.context(source['positive']))
        latent = data['latent'][address.path, address.position].to(self.device).clone()
        time = data['time'][address.position].to(self.device).clone()
        row = ResponseRow(neutral, positive, neutral, latent, timestep=time,
                          scale=address.sign, strength=self.strength, guidance=1.)
        base = data['base'][address.path, address.position].clone()
        target = base + address.sign * self.strength * data['delta'][address.path, address.position]
        return PreparedResponse(row, target.detach(), base.detach())

    def conditions(self, indices):
        result = []
        for index in indices:
            address = self.address(int(index))
            source = self.row_by_id[self.records[address.bundle]['row']]
            prompt = source['neutral']
            if prompt not in self._pooled:
                embeds, mask = self.context(prompt)
                pooled = (embeds.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                self._pooled[prompt] = pooled / pooled.square().mean().sqrt().clamp_min(1e-12)
            if address.bundle not in self._times:
                self._bundle(address.bundle)
            time = float(self._times[address.bundle][address.position])
            suffix = torch.tensor([[time / 1000., address.sign, self.strength, 1. / 3., address.path]])
            result.append(torch.cat((self._pooled[prompt], suffix), dim=1))
        if not result:
            return torch.empty((0, self.condition_width), dtype=torch.float32)
        return torch.cat(result).detach()

    def response_scale(self):
        """Training-only frozen edit RMS; diagnostic/calibration, never an MSE loss."""
        if self.split != 'train':
            raise ValueError('response scale calibration requires training split; held-out data is evaluation-only')
        total, count = 0., 0
        for index in range(len(self.records)):
            delta = self._bundle(index)['delta']
            total += float(delta.double().square().sum())
            count += delta.numel()
        scale = math.sqrt(total / count) * self.strength
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('frozen training edit reference has no finite nonzero scale')
        return scale

    def summary(self):
        return dict(self.identity(), bundles=len(self.records), unsigned_states=len(self) // 2,
                    signed_rows=len(self), lru_bundles=len(self._bundles), lru_contexts=len(self._contexts),
                    eager_native_targets=False)


class CampaignFieldCritic(nn.Module):
    """Small native-field critic with context/time/sign FiLM and raw scalar logits."""
    def __init__(self, channels, condition_width, width=32):
        super().__init__()
        self.field = nn.Sequential(nn.Conv2d(channels, width, 1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1))
        self.condition = nn.Linear(condition_width, 2 * width)
        self.score = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, response, condition):
        if response.ndim == 5 and response.shape[2] == 1:
            response = response.squeeze(2)
        elif response.ndim != 4:
            raise ValueError('field critic requires 4D images or 5D fields with singleton temporal axis')
        field = self.field(response)
        gain, bias = self.condition(condition).chunk(2, dim=1)
        field = torch.nn.functional.silu(field * (1 + gain[:, :, None, None]) + bias[:, :, None, None])
        return self.score(field.mean((2, 3))).squeeze(-1)


class MossCampaignBackend(MossResponseAdapter):
    def __init__(self, transformer, slider, provider, *, model_identity, decoder=None):
        self.provider = provider
        identity = dict(model_identity, frozen_cache=provider.identity())
        super().__init__(transformer, slider, decoder=decoder, model_identity=identity)

    def metadata(self):
        return dict(super().metadata(), target_convention='cached base + sign*strength*(positive-neutral)',
                    cache=self.provider.identity(), training_guidance=1.,
                    teacher_negative_role='neutral pole', inference_negative_role='frozen empty-prompt CFG',
                    intentional_render_corrections=['conditional branch only', 'negative layer skipping disabled'])


class _AdapterLoadCandidate:
    """Receive validated adapter tensors without copying a frozen native model."""
    def __init__(self, current):
        self.state = current

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict=True):
        self.state = state


@dataclass
class MossCampaignRun:
    backend: MossCampaignBackend
    rows: MossCampaignRows
    game: object

    def step(self):
        return self.game.step()  # All updates/losses/optimizer state live in shared core.

    def export_adapter(self, path):
        """New immutable adapter file plus resolved GAN/protocol provenance sidecar."""
        # A failed or otherwise unexportable game must not create directories or
        # leave behind an adapter that appears to belong to its previous step.
        packet = self.game.export_state_dict()
        path = Path(path)
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = canonical_digest(self.backend.model_identity)
        save_adapter(self.backend, path, model_fingerprint=fingerprint)
        receipt = dict(schema='moss-campaign-gan-adapter-v1', sha256=file_digest(path),
                       model_fingerprint=fingerprint, completed_steps=packet['completed_steps'],
                       core_config=packet['config'], old_mse_optimizer_resumed=False,
                       quality_qualified=False)
        path.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt

    def load_adapter(self, path):
        path = Path(path)
        receipt = json.loads(path.with_suffix('.json').read_text())
        expected = self.game.export_state_dict()['config']
        if not isinstance(receipt, dict):
            raise ValueError('campaign GAN export receipt must be an object')
        steps = receipt.get('completed_steps')
        if type(steps) is not int or not 0 <= steps <= self.game.recipe.total_steps:
            raise ValueError('invalid campaign GAN export step')
        fingerprint = canonical_digest(self.backend.model_identity)
        if (receipt.get('schema') != 'moss-campaign-gan-adapter-v1' or
                receipt.get('model_fingerprint') != fingerprint or
                canonical_digest(receipt.get('core_config')) != canonical_digest(expected) or receipt.get('sha256') != file_digest(path)):
            raise ValueError('campaign GAN export source, objective, recipe or checksum differs')
        # Reuse shared safetensors validation against the actual live schema,
        # while receiving its result into a candidate rather than live leaves.
        # Only the core's fully validated export load commits the live network.
        candidate = _AdapterLoadCandidate(self.backend.network.state_dict())
        staged = SimpleNamespace(network=candidate, metadata=self.backend.metadata)
        metadata = load_adapter(staged, path, model_fingerprint=fingerprint)
        self.game.load_export_state_dict(dict(schema='conditional-pg09-adapter-v1', config=expected,
                                             network=candidate.state, completed_steps=steps))
        return metadata

    def render_cases(self, pipe=None):
        """Yield matched held-out render specifications; no model call or RNG advance."""
        recipe = self.rows.recipe
        for row in recipe['development_rows']:
            for label, prompt, scale, cfg in (
                ('off', row['neutral'], 0., 3.),
                ('direct-positive', row['positive'], 0., 3.),
                ('direct-positive-cfg1', row['positive'], 0., 1.),
                ('minus', row['neutral'], -1., 3.),
                ('plus', row['neutral'], 1., 3.),
            ):
                yield dict(row_id=row['id'], label=label, prompt=prompt, scale=scale, guidance=cfg,
                           seed=recipe['evaluation_seed'], steps=50,
                           width=recipe['settings']['resolution'], height=recipe['settings']['resolution'],
                           original_negative_adapter='same scale as conditional', original_negative_skip=[22],
                           corrected_negative_adapter=0., corrected_negative_skip=[],
                           teacher_negative='neutral caption', inference_negative='')

    def render(self, pipe, case):
        if pipe.transformer is not self.backend.transformer:
            raise ValueError('render must reuse the same admitted transformer and adapter')
        scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        scheduler.set_timesteps(case['steps'], device=self.rows.device)
        latent = torch.randn((1, *self.rows.response_shape[1:]),
                             generator=torch.Generator('cpu').manual_seed(case['seed'])).to(self.rows.device)
        prompt = tuple(t.to(self.rows.device) for t in self.rows.context(case['prompt']))
        negative = tuple(t.to(self.rows.device) for t in self.rows.context(''))
        row = ResponseRow(prompt, prompt, negative, latent,
                          schedule=FlowSchedule.from_scheduler(scheduler), guidance=case['guidance'])
        old = self.backend.decoder
        self.backend.decoder = moss_pipeline_decoder(pipe)
        try:
            return self.backend.render(row, scale=case['scale'])
        finally:
            self.backend.decoder = old


def build_campaign_run(transformer, provider, *, model_identity, response_scale, batch_size=8, total_steps=3840,
                       seed=None, checkpointing=True, critic_width=32, recipe_overrides=None):
    """Bind already-admitted components; never load weights or choose a GPU."""
    if provider.split != 'train':
        raise ValueError('GAN training requires training split; held-out data is evaluation-only')
    from .pg09 import ConditionalPG09, validate_pg09_domain
    # Reject an unqualified runtime or native geometry before Slider freezes
    # the transformer and attaches hooks that a rejected build would retain.
    validate_pg09_domain(provider.response_shape, response_domain='continuous_field')
    settings = provider.recipe['settings']
    device = next(transformer.parameters()).device
    if not model_identity.get('model') or model_identity.get('revision') != provider.recipe['model_revision']:
        raise ValueError('coordinator-verified model identity must match the frozen cache revision')
    from moss_sliders.lora import Slider
    resolved_seed = settings['seed'] if seed is None else seed
    devices = [device.index or 0] if device.type == 'cuda' else []
    missing = object()
    previous_device = getattr(provider, 'device', missing)
    parameter_flags = [(parameter, parameter.requires_grad) for parameter in transformer.parameters()]
    slider, checkpoint_states = None, []
    try:
        provider.device = device
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(resolved_seed)
            slider = Slider(transformer, rank=settings['rank'], alpha=settings['alpha'])
            backend = MossCampaignBackend(transformer, slider, provider, model_identity=model_identity)
            critic = CampaignFieldCritic(provider.response_shape[1], provider.condition_width, critic_width).to(device)
        game = ConditionalPG09(backend, provider, critic, conditions=None, particles=None,
                              response_scale=response_scale, response_domain='continuous_field',
                              preset='bispf_optimized', batch_size=batch_size, total_steps=total_steps,
                              seed=resolved_seed,
                              recipe_overrides=recipe_overrides, sampling='balanced')
        if checkpointing:
            for block in transformer.layers:
                checkpoint_states.append((block, block.__dict__.get('forward', missing),
                                          block.__dict__.get('_pg09_scale_checkpoint', missing)))
                checkpoint_with_adapter_scale(block, backend.network)
    except Exception:
        for block, forward, marker in reversed(checkpoint_states):
            for name, value in (('forward', forward), ('_pg09_scale_checkpoint', marker)):
                if value is missing:
                    block.__dict__.pop(name, None)
                else:
                    setattr(block, name, value)
        if slider is not None:
            slider.detach()
        for parameter, requires_grad in parameter_flags:
            parameter.requires_grad_(requires_grad)
        if previous_device is missing:
            del provider.device
        else:
            provider.device = previous_device
        raise
    return MossCampaignRun(backend, provider, game)


def coverage_budget(unsigned_states=7680, old_updates=1600, old_batch=8, batch_size=8):
    signed_rows = 2 * unsigned_states
    old_signed_forwards = 2 * old_updates * old_batch
    return dict(unsigned_states=unsigned_states, signed_rows=signed_rows, batch_size=batch_size,
                old_signed_generator_forwards=old_signed_forwards,
                one_full_generator_cycle_updates=math.ceil(signed_rows / batch_size),
                old_generator_forward_matched_updates=math.ceil(old_signed_forwards / batch_size),
                proposed_two_full_cycles_updates=math.ceil(2 * signed_rows / batch_size),
                d_sampling='independent balanced shuffled cycles',
                g_sampling='balanced shuffled cycles; checkpoint order/cursor in shared core',
                wall_time='unknown until coordinator-admitted bounded canary')
