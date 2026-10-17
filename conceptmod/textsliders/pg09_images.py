"""Native image response bindings; all adversarial updates live in ConditionalPG09.

Constructors accept already-staged frozen models and attached adapters. They never
load weights, dispatch a process, or call a historical trainer.
"""
from contextlib import contextmanager
from copy import deepcopy

import torch

from .pg09_backends import (
    AdapterParameters, FrozenResponseAdapter, _lora_scale_context,
    bind_native_gradient_checkpointing,
)


def image_lora_parameters(model, native_lora=None):
    """Own the existing PEFT or conceptmod LoRA leaves exactly once."""
    model.eval()
    if native_lora is None:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if any('.lora_A.' not in n and '.lora_B.' not in n for n, _ in named):
            raise ValueError('only existing PEFT LoRA A/B trainables are supported')
        context = _lora_scale_context([('model', model)])
    else:
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('freeze base model before attaching conceptmod LoRA')
        layers = list(native_lora.unet_loras)
        if not layers:
            raise ValueError('empty native LoRA')
        named = list(native_lora.named_parameters())
        initial = [float(m.multiplier) for m in layers]

        @contextmanager
        def context(scale):
            previous = [m.multiplier for m in layers]
            try:
                for module, value in zip(layers, initial):
                    module.multiplier = value * scale
                yield
            finally:
                for module, value in zip(layers, previous):
                    module.multiplier = value
    network = AdapterParameters(named, context)
    bind_native_gradient_checkpointing(model, network)
    return network


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(v) for v in value)
    return deepcopy(value)


class ImageResponseAdapter(FrozenResponseAdapter):
    """Raw spatial prediction contract for existing Diffusers image families.

    A row contains frozen native encodings, never strings. DiT/UNet LoRA only;
    trainable text encoders require a separate live-encoding binding. Flux and
    Krea operate on genuine image lattices and use their native reversible
    pack/unpack functions. No arbitrary vector is reinterpreted as an image.
    """
    conventions = {
        'qwen_image21': 'model t=raw_t/1000; native unpatched VAE latent flow velocity',
        'iris': 'raw model t=1000*sigma; pixel-space RGB flow velocity',
        'sana': 'raw scheduler t multiplied by config.timestep_scale; flow velocity',
        'zimage': 'model t=(1000-raw_t)/1000; negate native output to flow velocity',
        'sd': 'raw discrete scheduler t; scheduler input scaling; declared prediction_type',
        'sdxl': 'raw discrete scheduler t; scheduler input scaling; declared prediction_type',
        'sd3': 'raw scheduler t; native flow velocity (no scheduler.step)',
        'flux': 'model t=raw_t/1000; native packed flow velocity restored to image lattice',
        'krea2': 'model t=raw_t/1000; native packed flow velocity restored to VAE lattice',
        'cascade': 'native DDPMWuerstchen continuous ratio r in [0,1]; image-embedding prediction',
    }

    def __init__(self, family, model, network, *, scheduler=None, pipeline=None,
                 decoder=None, model_identity=None):
        if family not in self.conventions:
            raise ValueError('unsupported image family')
        self.family = family
        self.time_coordinates = self.conventions[family]
        self.transformer = model
        self.scheduler = deepcopy(scheduler) if scheduler is not None else None
        self.pipeline = pipeline
        owned = {id(p) for p in network.parameters()}
        if any(p.requires_grad and id(p) not in owned for p in model.parameters()):
            raise ValueError('unowned trainable model parameter')
        if family in ('sd', 'sdxl'):
            if scheduler is None or not callable(getattr(scheduler, 'scale_model_input', None)):
                raise ValueError('SD/SDXL require the actual staged scheduler input scaling')
            if scheduler.config.prediction_type not in ('epsilon', 'v_prediction'):
                raise ValueError('unsupported SD prediction type')
        if family == 'krea2':
            if pipeline is None or pipeline.transformer is not model:
                raise ValueError('Krea requires its native pipeline geometry')
            encoder = getattr(pipeline, 'text_encoder', None)
            if encoder is not None and any(p.requires_grad for p in encoder.parameters()):
                raise ValueError('frozen encodings cannot own text-encoder trainables')
        super().__init__(network, decoder=decoder, model_identity=model_identity)

    @classmethod
    def from_krea2_product(cls, backend, *, model_identity):
        """Reuse LiveKrea2BboxBackend's actual DiT/PEFT leaves and native packing."""
        model = backend.pipe.transformer
        params = image_lora_parameters(model)
        actual = {id(p) for p in backend.trainable_parameters()}
        if actual != {id(p) for p in params.parameters()}:
            raise ValueError('only DiT LoRA is supported; text-encoder binding is pending')
        return cls('krea2', model, params, pipeline=backend.pipe, model_identity=model_identity)

    def prepare(self, row):
        from dataclasses import replace
        if row.schedule is not None and (row.latent.dtype != torch.float32 or
                next(self.transformer.parameters()).dtype != torch.float32):
            raise ValueError('mixed-precision trajectory stepping requires native sampler qualification')
        if row.schedule is not None and self.family in ('sd', 'sdxl', 'cascade', 'iris'):
            raise ValueError('discrete/non-flow scheduler trajectories require native scheduler qualification')
        if row.latent.ndim not in (4, 5):
            raise ValueError('native spatial image/latent field required')
        if self.family in ('krea2', 'qwen_image21'):
            if row.latent.ndim != 5 or row.latent.shape[2] != 1:
                raise ValueError('image VAE field must be [1,C,1,H,W]')
        elif row.latent.ndim != 4:
            raise ValueError('this image family requires [1,C,H,W]')
        row = replace(row, neutral=_snapshot(row.neutral), positive=_snapshot(row.positive),
                      negative=_snapshot(row.negative))
        return super().prepare(row)

    def metadata(self):
        result = super().metadata()
        result.update(protocol='native-image-response-pg090-v1',
                      response_coordinates='native spatial prediction field',
                      text_encoder='frozen precomputed encoding; DiT/UNet LoRA only',
                      particle_bank='none: adapter GAN only',
                      trajectory_contract='explicit FP32 Euler only; native production sampler parity pending',
                      native_scheduler=({k: sorted(v) if k == '_use_default_values' else v
                                         for k, v in self.scheduler.config.items()}
                                        if self.scheduler is not None else None))
        import hashlib
        import inspect
        from importlib.metadata import version
        sources = []
        if self.family == 'flux':
            from diffusers import FluxPipeline
            sources.append(FluxPipeline)
        if self.family == 'krea2':
            sources.append(type(self.pipeline))
            result['native_geometry'] = dict(patch_size=self.pipeline.patch_size,
                                            vae_scale_factor=self.pipeline.vae_scale_factor)
        if self.family == 'qwen_image21':
            from diffusers import QwenImage21Pipeline
            from qwen_image_sliders.teachers import with_encoding
            from qwen_image_sliders.projection import predict
            sources.extend([QwenImage21Pipeline, with_encoding, predict])
        if self.scheduler is not None:
            sources.append(type(self.scheduler))
        from pathlib import Path
        result['helper_source_hashes'] = {obj.__module__: hashlib.sha256(
            Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest() for obj in sources}
        result['diffusers_version'] = version('diffusers')
        return result

    def _velocity(self, condition, latent, timestep):
        if not isinstance(condition, dict):
            raise ValueError('native conditioning dictionary required')
        if timestep.numel() != 1:
            raise ValueError('one timestep per row required')
        limit = 1 if self.family == 'cascade' else 1000
        if torch.any((timestep < 0) | (timestep > limit)):
            raise ValueError('timestep outside declared native coordinate range')
        model = self.transformer
        dtype = next(model.parameters()).dtype
        z = latent.to(dtype)
        t = timestep.reshape(1).to(device=z.device, dtype=torch.float32)
        c = {k: v.to(device=z.device, dtype=dtype) if isinstance(v, torch.Tensor)
             and v.is_floating_point() else v.to(z.device) if isinstance(v, torch.Tensor)
             else v for k, v in condition.items()}
        if self.family in ('sd', 'sdxl'):
            # A private scheduler copy prevents input-scaling cursor changes leaking between rows.
            z = deepcopy(self.scheduler).scale_model_input(z, t[0])
            kwargs = dict(encoder_hidden_states=c['embeds'])
            if self.family == 'sdxl':
                kwargs['added_cond_kwargs'] = dict(text_embeds=c['pooled'], time_ids=c['time_ids'])
            return model(z, t, **kwargs).sample.float()
        if self.family == 'sana':
            out = model(z, encoder_hidden_states=c['embeds'],
                        encoder_attention_mask=c['mask'],
                        timestep=t * model.config.timestep_scale, return_dict=False)[0]
            if out.shape[1] == z.shape[1] * 2:
                out = out.chunk(2, dim=1)[0]
            return out.float()
        if self.family == 'zimage':
            out = model(list(z.unsqueeze(2).unbind(0)), (1000. - t) / 1000.,
                        [c['embeds']], return_dict=False)[0]
            return -torch.stack([v.squeeze(1) for v in out]).float()
        if self.family == 'sd3':
            return model(hidden_states=z, timestep=t, encoder_hidden_states=c['embeds'],
                         pooled_projections=c['pooled'], return_dict=False)[0].float()
        if self.family == 'flux':
            from diffusers import FluxPipeline
            b, channels, height, width = z.shape
            if height % 2 or width % 2:
                raise ValueError('Flux native 2x2 patch lattice requires even spatial dimensions')
            packed = FluxPipeline._pack_latents(z, b, channels, height, width)
            out = model(hidden_states=packed, timestep=t.to(dtype) / 1000.,
                        encoder_hidden_states=c['embeds'], pooled_projections=c['pooled'],
                        txt_ids=c['text_ids'], img_ids=c['image_ids'],
                        guidance=c.get('guidance'), return_dict=False)[0]
            return FluxPipeline._unpack_latents(out, height * 8, width * 8, 8).float()
        if self.family == 'krea2':
            b, channels, _, height, width = z.shape
            pipe = self.pipeline
            if height % pipe.patch_size or width % pipe.patch_size:
                raise ValueError('Krea native patch geometry requires divisible spatial dimensions')
            packed = pipe._pack_latents(z.squeeze(2), b, channels, height, width)
            out = model(hidden_states=packed, timestep=(t / 1000.).to(dtype),
                        encoder_hidden_states=c['embeds'], encoder_attention_mask=c['mask'],
                        position_ids=c['position_ids'], return_dict=False)[0]
            return pipe._unpack_latents(out, height * pipe.vae_scale_factor,
                                       width * pipe.vae_scale_factor).float()
        if self.family == 'qwen_image21':
            from diffusers import QwenImage21Pipeline
            from qwen_image_sliders.teachers import with_encoding
            from qwen_image_sliders.projection import predict
            b, channels, _, height, width = z.shape
            if height % 2 or width % 2:
                raise ValueError('Qwen 2.1 native image geometry requires even dimensions')
            packed = QwenImage21Pipeline._pack_latents(z.squeeze(2), b, channels, height, width)
            encoding = dict(prompt_embeds=c['embeds'], prompt_embeds_mask=c['mask'],
                            image_pad_mask=c['image_pad_mask'])
            native = with_encoding(dict(hidden_states=packed, timestep=t.to(dtype) / 1000.,
                                       img_shapes=[[(1, height, width)]]), encoding)
            out = predict(model, native)
            return QwenImage21Pipeline._unpack_latents(out, height * 8, width * 8, 8).float()
        if self.family == 'iris':
            return model(z, t, c['embeds'], y_mask=c['mask']).x.float()
        # Stage-C predicts a spatial image-embedding field, not RGB and not a VAE latent.
        return model(sample=z, timestep_ratio=t, clip_text=c['embeds'], clip_text_pooled=c['pooled'],
                     clip_img=c['image_embeds']).sample.float()


def image_condition_features(prepared):
    """Declared time/sign/CFG and fixed native text summary coordinates, never row IDs."""
    features = []
    for item in prepared:
        row = item.row
        times = ([float(row.timestep)] if row.schedule is None else list(row.schedule.timesteps))
        values = [sum(times) / len(times) / 1000., row.scale, row.guidance, row.strength]
        for condition in (row.neutral, row.positive, row.negative):
            embeds = condition['embeds'].detach().float()
            values.extend([float(embeds.mean()), float(embeds.square().mean().sqrt())])
        features.append(values)
    return torch.tensor(features, dtype=torch.float32, device=prepared[0].target.device)


class IrisResponseAdapter(ImageResponseAdapter):
    """Reuse the product's actual hooked AdapterBank and the same routed particles.

    The bank's bridge architecture is retained; its old FormulationGame/velocity
    sketch/normalizer is not used. Native RGB response stays [1,3,H,W] through PG.
    """
    def __init__(self, runtime, adapter, particles, *, model_identity=None):
        self.runtime, self.adapter = runtime, adapter
        if not isinstance(particles, torch.nn.Parameter) or tuple(particles.shape) != adapter.particle_shape:
            raise ValueError('bind the actual product particle parameter')
        if any(p.requires_grad for p in runtime.model.parameters()):
            raise ValueError('Iris model must remain frozen')
        named = [('adapter.' + n, p) for n, p in adapter.named_parameters()]
        named.append(('particles', particles))
        super().__init__('iris', runtime.model,
            AdapterParameters(named, lambda scale: adapter.active(particles, strength=scale)),
            model_identity=model_identity)

    @property
    def particles(self):
        return self.network.particles

    def cfg(self, row, latent, timestep, *, scale):
        # Native Iris default CFG interval is open at sigma=0 and sigma=1.
        if float(timestep) in (0., 1000.):
            return self.velocity(row.neutral, latent, timestep, scale=scale)
        return super().cfg(row, latent, timestep, scale=scale)

    def metadata(self):
        result = super().metadata()
        result.update(particle_bank='actual product routed bank',
                      response_coordinates='native RGB pixel velocity [1,3,H,W]',
                      cfg_interval=[0., 1.], cfg_interval_endpoints='open',
                      sampler='raw response only; native FlowDPMSolver rendering pending')
        return result


class QwenImage21ResponseAdapter(IrisResponseAdapter):
    """Reuse Qwen's product hooks and bank; preserve real VAE geometry, no sketches."""
    def __init__(self, runtime, adapter, particles, *, model_identity=None):
        super().__init__(runtime, adapter, particles, model_identity=model_identity)
        self.family = 'qwen_image21'
        self.time_coordinates = self.conventions[self.family]

    def cfg(self, row, latent, timestep, *, scale):
        if row.guidance != 1:
            raise ValueError('Qwen product supports true_cfg_scale=1 only')
        return self.velocity(row.neutral, latent, timestep, scale=scale)

    def metadata(self):
        result = ImageResponseAdapter.metadata(self)
        result.update(particle_bank='actual product routed bank',
                      response_coordinates='native VAE latent velocity [1,C,1,H,W]',
                      true_cfg_scale=1., image_conditioning=False, kv_cache=False)
        return result
