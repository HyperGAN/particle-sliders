"""Opt-in installed response/inference factories using the shared lifecycle."""
import math
from pathlib import Path

import torch
from particle_sliders.lifecycle import Binding
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg094 import ConditionalPG094
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09_images import image_condition_features
from particle_sliders.fibo_hndl import HNDLFieldCritic

from .native import NativeSDXL, PROTOCOL, gate, targets, validate_row, model_identity

NAME = 'sdxl-native-hndl-v1'


def configuration(options):
    shape = tuple(options.get('shape', (1, 4, 8, 8)))
    rank, alpha = options.get('rank', 4), options.get('alpha', 1.)
    if (len(shape) != 4 or shape[:2] != (1, 4) or any(type(v) is not int or v < 1 for v in shape)
            or any(v % 2 for v in shape[2:])):
        raise ValueError('declare genuine SDXL [1,4,H,W] native lattice with even H/W')
    if type(rank) is not int or not 1 <= rank <= 128 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('positive ordinary LoRA rank/alpha required')
    return dict(shape=list(shape), rank=rank, alpha=float(alpha))


def capabilities(options):
    cfg = configuration(options)
    return ResponseCapabilities('adapter_gan', 'sdxl', 'native_scheduler_noise_prediction',
        'continuous_field', tuple(cfg['shape']), '0.9.4', 'release-0.9.4',
        qualification='CPU FP32 random native SDXL UNet conformance only; pretrained quality unqualified')


def validate(options):
    allowed = {'shape', 'rank', 'alpha', 'steps', 'batch', 'seed', 'ema_decay', 'preset',
               'model_dir', 'rows_path', 'pg_version', 'pg_source_profile', 'penalty_input_domain', 'base_precision'}
    if set(options) - allowed:
        raise ValueError('unknown SDXL preview options')
    configuration(options)
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('select a shared BISPF preset')
    gate()


def load_model(folder):
    """Already staged local files only; tokenizers/encoders are never loaded here."""
    from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
    from diffusers.models.attention_processor import AttnProcessor
    import json
    folder = Path(folder)
    scheduler_type = json.loads((folder / 'scheduler/scheduler_config.json').read_text()).get('_class_name')
    scheduler_class = {'DDIMScheduler': DDIMScheduler, 'DDPMScheduler': DDPMScheduler}.get(scheduler_type)
    if scheduler_class is None:
        raise ValueError('staged DDIM/DDPM SDXL scheduler required')
    model = UNet2DConditionModel.from_pretrained(folder / 'unet', local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    # Explicit math attention supports the game's input-gradient second derivatives.
    model.set_attn_processor(AttnProcessor())
    scheduler = scheduler_class.from_pretrained(folder / 'scheduler', local_files_only=True)
    return model, scheduler


def load_rows(path):
    values = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError('at least two explicit frozen SDXL tensor rows required')
    return [ResponseRow(**row) for row in values]


def staged(options, model=None, scheduler=None, rows=None):
    if (model is None) != (scheduler is None):
        raise ValueError('stage both actual UNet and scheduler')
    if model is None:
        model, scheduler = load_model(options['model_dir'])
    rows = list(rows) if rows is not None else load_rows(options['rows_path'])
    targets(model)
    if len(rows) < 2:
        raise ValueError('at least two fixed SDXL rows required')
    for row in rows:
        validate_row(model, scheduler, row, tuple(configuration(options)['shape']))
    return model, scheduler, rows


def preflight(options, request, supplied):
    # Validate staged rows and exact native contracts before any hooks/modes/gradients change.
    if supplied:
        staged(options, **supplied)


def prepare(options, model=None, scheduler=None, rows=None):
    model, scheduler, rows = staged(options, model, scheduler, rows)
    cfg = configuration(options)
    native = NativeSDXL(model, scheduler, tuple(cfg['shape']), cfg['rank'], cfg['alpha'])
    try:
        prepared = tuple(native.prepare(row) for row in rows)
        conditions = image_condition_features(prepared)
        critic = HNDLFieldCritic(4, conditions.shape[1], tuple(cfg['shape'][2:]))
        game = ConditionalPG094(native, prepared, critic, conditions=conditions,
            response_domain='continuous_field', particles=None,
            preset=options.get('preset', 'bispf_optimized'), batch_size=options.get('batch', 2),
            seed=options.get('seed', 7), total_steps=options.get('steps', 4))
        spec = INFERENCE.specification(native, model_identity=native.identity,
            protocol=PROTOCOL, configuration=cfg)
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if set(options) != {'model_dir'}:
        raise ValueError('native-only SDXL inference accepts only an already staged model_dir')
    if configuration(spec.configuration) != spec.configuration:
        raise ValueError('SDXL inference configuration differs')
    gate()


def prepare_inference(options, spec):
    cfg = configuration(spec.configuration)
    model, scheduler = load_model(options['model_dir'])
    # Reject changed base weights, scheduler, runtime and native shape before attaching adapters.
    targets(model)
    if model_identity(model, scheduler) != spec.identity['model']:
        raise ValueError('SDXL frozen model or scheduler identity differs before hook attachment')
    native = NativeSDXL(model, scheduler, tuple(cfg['shape']), cfg['rank'], cfg['alpha'])
    try:
        return InferenceBinding(native.network,
            native_identity(native, model_identity=native.identity, protocol=PROTOCOL),
            native.close, response=native.response)
    except BaseException:
        native.close()
        raise


CAPS = capabilities({})
TRAINING = ResponseBackend(NAME, CAPS, prepare, validate,
                          configure_capabilities=capabilities, preflight=preflight)
INFERENCE = InferenceBackend(NAME, CAPS, prepare_inference, validate_inference,
                            configure_capabilities=capabilities)
