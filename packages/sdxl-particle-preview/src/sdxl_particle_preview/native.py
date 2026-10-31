"""Actual SDXL text_time conditioning and ordinary Attention LoRA via HNDL.

No optimizer or GAN mathematics live here. Frozen native outputs retain NCHW.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn
from particle_sliders.fibo_hndl import verified_builder
from particle_sliders.pg09_images import ImageResponseAdapter
from particle_sliders.lifecycle import native_cleanup

ROOT = Path(__file__).parent
PROTOCOL = 'sdxl-native-epsilon-vprediction-hndl-lora-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate():
    import particle_sliders
    lock = json.loads((ROOT / 'cohort.json').read_text())
    base = Path(particle_sliders.__file__).parent
    actual = {p.name: digest(p) for p in base.iterdir() if p.suffix in ('.py', '.json', '.hndl')}
    if version('particle-sliders-core') != lock['core_version'] or actual != lock['core_sources']:
        raise ValueError('SDXL core source cohort differs; no implicit checkpoint migration')
    if version('diffusers') != '0.35.1' or version('transformers') != '4.57.6':
        raise ValueError('SDXL native dependency cohort differs')
    return verified_builder()


def targets(model):
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import Attention, AttnProcessor
    if type(model) is not UNet2DConditionModel:
        raise ValueError('actual Diffusers SDXL UNet2DConditionModel required')
    cfg = model.config
    if (cfg.addition_embed_type != 'text_time' or cfg.in_channels != 4 or cfg.out_channels != 4
            or not isinstance(cfg.cross_attention_dim, int) or cfg.encoder_hid_dim is not None):
        raise ValueError('SDXL text_time four-channel UNet contract required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('this preview qualifies CPU float32 native responses only')
    if getattr(model, 'gradient_checkpointing', False):
        raise ValueError('SDXL checkpoint recomputation is not qualified in this preview')
    result = {}
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if type(module.processor) is not AttnProcessor:
                raise ValueError('select native AttnProcessor for critic second derivatives')
            found = {key: leaf for key, leaf in module.named_modules() if isinstance(leaf, nn.Linear)}
            if set(found) != {'to_q', 'to_k', 'to_v', 'to_out.0'}:
                raise ValueError('SDXL Attention Q/K/V/O target layout differs')
            for key, leaf in found.items():
                if type(leaf) is not nn.Linear or id(leaf) in seen or 'forward' in leaf.__dict__ or leaf._forward_hooks:
                    raise ValueError('pre-existing/aliased SDXL projection adapter refused')
                seen.add(id(leaf))
                result[name + '.' + key] = leaf
    if not result:
        raise ValueError('empty SDXL attention target set')
    return result


def validate_row(model, scheduler, row, shape):
    from particle_sliders.pg09_backends import ResponseRow
    if not isinstance(row, ResponseRow) or row.schedule is not None:
        raise ValueError('SDXL requires a single native discrete noise-prediction row')
    if tuple(row.latent.shape) != tuple(shape) or len(shape) != 4 or shape[0:2] != (1, 4):
        raise ValueError('SDXL native latent shape differs')
    if row.scale != 1 or row.strength != 1 or not math.isfinite(row.guidance) or row.guidance < 0:
        raise ValueError('new unipolar positive-response target requires scale=strength=1')
    t = row.timestep
    if (not isinstance(t, torch.Tensor) or t.numel() != 1 or not torch.isfinite(t).all()
            or float(t) != int(t) or not 0 <= float(t) < scheduler.config.num_train_timesteps):
        raise ValueError('SDXL discrete scheduler timestep differs')
    pool_width = model.add_embedding.linear_1.in_features - 6 * model.config.addition_time_embed_dim
    for c in (row.neutral, row.positive, row.negative):
        if not isinstance(c, dict) or set(c) != {'embeds', 'pooled', 'time_ids'}:
            raise ValueError('SDXL requires token embeddings, pooled text and six size/crop IDs')
        if (c['embeds'].ndim != 3 or c['embeds'].shape[0] != 1 or c['embeds'].shape[1] < 1
                or c['embeds'].shape[2] != model.config.cross_attention_dim
                or tuple(c['pooled'].shape) != (1, pool_width) or tuple(c['time_ids'].shape) != (1, 6)):
            raise ValueError('SDXL dual-encoder concatenation/pooled/time-ID dimensions differ')
        if any(v.device.type != 'cpu' or v.dtype != torch.float32 or not torch.isfinite(v).all() or v.requires_grad
               for v in c.values()):
            raise ValueError('SDXL conditioning must be frozen finite CPU float32')
        sizes = c['time_ids'][0]
        if (not torch.equal(sizes, sizes.round()) or torch.any(sizes[:2] <= 0)
                or torch.any(sizes[2:4] < 0) or tuple(sizes[4:].tolist()) != (shape[2] * 8, shape[3] * 8)):
            raise ValueError('SDXL original/crop/target size contract differs from native VAE lattice')
    if row.latent.device.type != 'cpu' or row.latent.dtype != torch.float32 or not torch.isfinite(row.latent).all():
        raise ValueError('SDXL latents require finite CPU float32')


def model_identity(model, scheduler):
    def config(value):
        return {k: v for k, v in dict(value).items() if not k.startswith('_')}
    return dict(model='SDXL UNet2DConditionModel', qualification='CPU software conformance only; pretrained quality unqualified',
        weights={k: hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                 for k, v in model.state_dict().items()},
        config=config(model.config), scheduler=config(scheduler.config),
        sources={type(o).__module__: digest(inspect.getsourcefile(type(o))) for o in (model, scheduler)},
        attention_processor='diffusers.models.attention_processor.AttnProcessor',
        prediction_type=scheduler.config.prediction_type)


class HNDLLoRA(nn.Module):
    """Same Linear-down/Linear-up computation and alpha/rank as conceptmod LoRA."""
    def __init__(self, module, rank, alpha, builder, identity):
        super().__init__()
        self.input_width, self.output_width = module.in_features, module.out_features
        self.graph = builder((ROOT / 'lora.hndl').read_text(),
            input_shape={'value': ['B', 1, self.input_width]}, output_shape=['B', 1, self.output_width],
            parameters={'rank': rank, 'output_width': self.output_width})
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=1)
        nn.init.zeros_(self.graph['up'].weight)
        self.factor = alpha / rank
        self.identity = dict(identity, graph_sha256=digest(ROOT / 'lora.hndl'),
                             input_width=self.input_width, output_width=self.output_width, rank=rank, alpha=alpha,
                             plan_sha256=hashlib.sha256(str(self.graph).encode()).hexdigest())

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)

    def forward(self, x):
        # Linear acts independently on native leading token dimensions; no response geometry changes.
        return self.graph(value=x.reshape(-1, 1, self.input_width)).reshape(*x.shape[:-1], self.output_width)


class SDXLNetwork(nn.Module):
    def __init__(self, target_map, rank, alpha, builder, identity):
        super().__init__()
        self.names = tuple(target_map)
        self.layers = nn.ModuleList([HNDLLoRA(m, rank, alpha, builder, identity) for m in target_map.values()])
        self.scale = 0.
        self.handles = []

    def attach(self, target_map):
        try:
            for module, branch in zip(target_map.values(), self.layers):
                def hook(host, args, output, branch=branch):
                    return output if self.scale == 0 else output + branch(args[0]) * (self.scale * branch.factor)
                self.handles.append(module.register_forward_hook(hook))
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, scale):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('SDXL preview strength must be in [0,1]')
        previous = self.scale
        self.scale = float(scale)
        try:
            yield self
        finally:
            self.scale = previous

    def close(self):
        failures = []
        handles, self.handles = self.handles, []
        for handle in handles:
            try:
                handle.remove()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            for extra in failures[1:]:
                failures[0].add_note('additional SDXL hook cleanup failure: ' + repr(extra))
            raise failures[0]


class NativeSDXL(ImageResponseAdapter):
    def __init__(self, model, scheduler, shape, rank, alpha):
        builder, identity = gate()  # Before changing the borrowed native model.
        selected = targets(model)
        from diffusers import DDIMScheduler, DDPMScheduler
        if type(scheduler) not in (DDIMScheduler, DDPMScheduler) or scheduler.config.prediction_type not in ('epsilon', 'v_prediction'):
            raise ValueError('SDXL preview qualifies native DDIM/DDPM input scaling only')
        self.identity = model_identity(model, scheduler)
        self.shape = tuple(shape)
        self._cleanup = native_cleanup(model)
        network = SDXLNetwork(selected, rank, alpha, builder, identity)
        self.slider = network
        try:
            model.eval().requires_grad_(False)
            # LoRA leaves remain independent FP32 trainables; host owns frozen weights only.
            super().__init__('sdxl', model, network, scheduler=scheduler, model_identity=self.identity)
            network.attach(selected)
        except BaseException:
            try:
                network.close()
            finally:
                self._cleanup()
            raise

    def prepare(self, row):
        validate_row(self.transformer, self.scheduler, row, self.shape)
        return super().prepare(row)

    def response(self, *, row, strength=1., seed=0):
        validate_row(self.transformer, self.scheduler, row, self.shape)
        return self._response(row, scale=strength)

    def metadata(self):
        result = super().metadata()
        result.update(protocol=PROTOCOL, arm='adapter_gan', response_domain='continuous_field',
            prediction_type=self.scheduler.config.prediction_type,
            native_contract='NCHW noise prediction; frozen dual-CLIP concatenation, pooled text and six size/crop IDs',
            target_convention='new unipolar frozen positive response; historical prompt_pair.loss remains separate',
            lora_graph_sha256=digest(ROOT / 'lora.hndl'))
        return result

    def close(self):
        try:
            self.network.close()
        finally:
            self._cleanup()
