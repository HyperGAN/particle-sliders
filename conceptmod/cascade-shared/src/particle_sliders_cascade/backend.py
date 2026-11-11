"""Cascade Stage C native geometry; shared core owns every GAN update.

This is a new positive-teacher adapter GAN cohort, not a migration of the
historical prompt_pair.loss objective. Stage B and Stage A are never trained.
"""
from contextlib import contextmanager
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

from particle_sliders.fibo_hndl import verified_builder, HNDLFieldCritic
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, Capabilities, checkpoint_cleanup
from particle_sliders.pg09 import ConditionalPG09, PG_VERSION, PG_SOURCE_PROFILE, _hash, _identity
from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter, ResponseRow
from particle_sliders.pg09_images import image_condition_features
from particle_sliders.response import ResponseBackend, ResponseCapabilities

NAME = 'cascade_stage_c'
PROTOCOL = 'cascade-stage-c-spatial-response-hndl-v1'
GRAPH = Path(__file__).with_name('lora.hndl')


def validate(options):
    allowed = {'dummy', 'model_dir', 'cache_file', 'cache_sha256', 'shape', 'rank',
               'steps', 'batch', 'seed', 'ema_decay', 'base_precision'}
    if set(options) - allowed:
        raise ValueError('unknown Cascade options: ' + str(sorted(set(options) - allowed)))
    shape = options.get('shape', [1, 4, 4, 4])
    if (not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1
            or any(type(n) is not int or n < 1 for n in shape)):
        raise ValueError('Stage C requires explicit native [1,C,H,W] shape')
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('dummy must be boolean')
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('only CPU FP32 Stage C is qualified by this cohort')
    if type(options.get('rank', 4)) is not int or not 1 <= options.get('rank', 4) <= 128:
        raise ValueError('rank must be in [1,128]')
    if options.get('dummy', False):
        if shape != [1, 4, 4, 4] or any(k in options for k in ('model_dir', 'cache_file', 'cache_sha256')):
            raise ValueError('dummy fixture has fixed geometry and no staged paths')
    else:
        if not Path(options.get('model_dir', '')).is_dir():
            raise ValueError('stage the local Stage C prior directory first')
        path = Path(options.get('cache_file', ''))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != options.get('cache_sha256'):
            raise ValueError('frozen cache SHA256 differs before native allocation')
    if version('diffusers') != '0.36.0':
        raise ValueError('Cascade requires diffusers==0.36.0')
    verified_builder()  # Check HG/HNDL bytes before native mutation.


def capabilities(options):
    return ResponseCapabilities('adapter_gan', 'image', 'stage_c_image_embedding_prediction',
        'continuous_field', tuple(options.get('shape', [1, 4, 4, 4])), PG_VERSION, PG_SOURCE_PROFILE,
        strength_min=None, strength_max=None,
        qualification='tiny random native Stage C CPU conformance; pretrained and full Cascade rendering unqualified')


def tiny_model():
    from diffusers import StableCascadeUNet
    with torch.random.fork_rng():
        torch.manual_seed(51)
        return StableCascadeUNet(in_channels=4, out_channels=4,
            timestep_ratio_embedding_dim=8, conditioning_dim=8, block_out_channels=(16,),
            num_attention_heads=(2,), down_num_layers_per_block=(1,), up_num_layers_per_block=(1,),
            down_blocks_repeat_mappers=(1,), up_blocks_repeat_mappers=(1,),
            block_types_per_layer=(('SDCascadeResBlock', 'SDCascadeTimestepBlock', 'SDCascadeAttnBlock'),),
            clip_text_in_channels=8, clip_text_pooled_in_channels=8, clip_image_in_channels=8,
            clip_seq=2, dropout=0., timestep_conditioning_type=())


def fixture_rows():
    generator = torch.Generator().manual_seed(61)
    def condition():
        return dict(embeds=torch.randn(1, 3, 8, generator=generator),
                    pooled=torch.randn(1, 1, 8, generator=generator),
                    image_embeds=torch.zeros(1, 1, 8))
    return [ResponseRow(condition(), condition(), condition(),
            torch.randn(1, 4, 4, 4, generator=generator), torch.tensor(.25 + i * .25)) for i in range(2)]


def load_model(options):
    if options.get('dummy', False):
        return tiny_model()
    from diffusers import StableCascadeUNet
    return StableCascadeUNet.from_pretrained(options['model_dir'], local_files_only=True,
                                           torch_dtype=torch.float32, use_safetensors=True)


def model_identity(model):
    return json.loads(json.dumps(dict(stage='C', state_sha256=_hash(_identity(model.state_dict())),
                configuration=dict(model.config), diffusers=version('diffusers'),
                graph_sha256=hashlib.sha256(GRAPH.read_bytes()).hexdigest(),
                graph_runtime=verified_builder()[1], decoder_stages='B/A outside response cohort'), allow_nan=False))


def check_row(row, model, shape):
    if (not isinstance(row, ResponseRow) or row.schedule is not None or tuple(row.latent.shape) != tuple(shape)
            or row.latent.device.type != 'cpu' or row.latent.dtype != torch.float32
            or not torch.isfinite(row.latent).all() or row.scale != 1 or row.strength != 1
            or row.timestep is None or row.timestep.numel() != 1
            or not torch.isfinite(row.timestep).all() or not 0 <= float(row.timestep) <= 1
            or not 0 <= row.guidance < float('inf')):
        raise ValueError('finite CPU Stage C raw response row required; no flow trajectory')
    config = model.config
    if config.in_channels != shape[1] or config.out_channels != shape[1] or config.effnet_in_channels is not None:
        raise ValueError('requires Stage C prior, not Stage B/effnet decoder')
    expected = {'embeds': config.clip_text_in_channels, 'pooled': config.clip_text_pooled_in_channels,
                'image_embeds': config.clip_image_in_channels}
    for condition in (row.neutral, row.positive, row.negative):
        if not isinstance(condition, dict) or set(condition) != set(expected):
            raise ValueError('all native Stage C text/pooled/image conditions required')
        for key, width in expected.items():
            value = condition[key]
            if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
                    or value.shape[-1] != width or value.device.type != 'cpu'
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('invalid native Stage C conditioning: ' + key)


class LoRAGraph(nn.Module):
    def __init__(self, module, rank):
        super().__init__()
        builder, identity = verified_builder()
        self.graph = builder(GRAPH.read_text(), input_shape={'value': ['B', module.in_features]},
            output_shape=['B', module.out_features], parameters=dict(rank=rank, output_width=module.out_features))
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=5 ** .5)
        nn.init.zeros_(self.graph['up'].weight)
        self.identity = dict(identity, graph_sha256=hashlib.sha256(GRAPH.read_bytes()).hexdigest())
        self.multiplier = 1.

    def extra_repr(self):
        return str(sorted(self.identity.items()))

    def delta(self, value):
        shape = value.shape
        return self.graph(value=value.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)


class NativeResponse(FrozenResponseAdapter):
    family = NAME
    time_coordinates = 'native DDPMWuerstchen timestep_ratio in [0,1]; Stage C embedding field'

    def __init__(self, model, network, identity):
        self.model = model
        super().__init__(network, model_identity=identity)

    def _velocity(self, condition, latent, timestep):
        return self.model(sample=latent, timestep_ratio=timestep.reshape(1),
            clip_text=condition['embeds'], clip_text_pooled=condition['pooled'],
            clip_img=condition['image_embeds']).sample.float()


def attach(model, rank):
    """Transactional hooks: restore modes, gradients, buffers and checkpoint settings."""
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.endswith(('to_q', 'to_k', 'to_v', 'to_out.0'))]
    if not targets:
        raise ValueError('native Stage C attention projections missing')
    modes = [(m, m.training) for m in model.modules()]
    gradients = [(p, p.requires_grad, p.grad) for p in model.parameters()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    restore_checkpoint = checkpoint_cleanup(model)
    handles, graphs = [], nn.ModuleDict()
    closed = False
    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        for handle in reversed(handles):
            handle.remove()
        restore_checkpoint()
        for m, mode in modes:
            m.training = mode
        for p, flag, grad in gradients:
            p.requires_grad_(flag)
            p.grad = grad
        with torch.no_grad():
            for b, value in buffers:
                b.copy_(value)
    try:
        # Build all executable graphs before modifying the borrowed model.
        for name, module in targets:
            graphs[name.replace('.', '__')] = LoRAGraph(module, rank)
        model.requires_grad_(False).eval()
        for name, module in targets:
            graph = graphs[name.replace('.', '__')]
            def hook(module, inputs, output, graph=graph):
                if graph.multiplier == 0:
                    return output
                return output + graph.delta(inputs[0]) * graph.multiplier
            handles.append(module.register_forward_hook(hook))
        @contextmanager
        def scaled(scale):
            previous = [g.multiplier for g in graphs.values()]
            try:
                for g in graphs.values():
                    g.multiplier = scale
                yield
            finally:
                for g, value in zip(graphs.values(), previous):
                    g.multiplier = value
        network = AdapterParameters(graphs.named_parameters(), scaled)
        return network, close
    except BaseException:
        close()
        raise


def native(options, *, model=None):
    model = load_model(options) if model is None else model
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('this cohort requires CPU FP32 native base')
    identity = model_identity(model)
    network, close = attach(model, options.get('rank', 4))
    try:
        return NativeResponse(model, network, identity), close
    except BaseException:
        close()
        raise


def prepare(options, *, model=None, rows=None):
    with torch.random.fork_rng():
        torch.manual_seed(options.get('seed', 7))
        return _prepare(options, model=model, rows=rows)


def _prepare(options, *, model=None, rows=None):
    model = load_model(options) if model is None else model
    if rows is None:
        if options.get('dummy', False):
            rows = fixture_rows()
        else:
            cache = torch.load(options['cache_file'], map_location='cpu', weights_only=True)
            if set(cache) != {'model_identity', 'rows'} or cache['model_identity'] != model_identity(model):
                raise ValueError('frozen cache belongs to a different Stage C base')
            rows = [ResponseRow(**row) for row in cache['rows']]
    if len(rows) < 2:
        raise ValueError('at least two frozen rows required')
    for row in rows:
        check_row(row, model, options.get('shape', [1, 4, 4, 4]))
    response, close = native(options, model=model)
    try:
        prepared = [response.prepare(row) for row in rows]
        conditions = image_condition_features(prepared)
        critic = HNDLFieldCritic(prepared[0].target.shape[1], conditions.shape[1],
                                 prepared[0].target.shape[2:], width=8)
        game = ConditionalPG09(response, prepared, critic, conditions=conditions,
            total_steps=options.get('steps', 4), batch_size=options.get('batch', 2), seed=options.get('seed', 7))
        configuration = {k: v for k, v in options.items() if k in ('dummy', 'shape', 'rank')}
        spec = INFERENCE.specification(response, model_identity=response.model_identity,
            protocol=PROTOCOL, configuration=configuration)
        return Binding(game, close, inference=spec)
    except BaseException:
        close()
        raise


def validate_inference(options, spec):
    if set(options) - {'model_dir', 'base_precision'}:
        raise ValueError('native-only inference accepts local base path/precision, no training cache')
    validate(dict(spec.configuration, **options)) if spec.configuration.get('dummy') else _validate_native(options)


def _validate_native(options):
    if not Path(options.get('model_dir', '')).is_dir() or options.get('base_precision', 'float32') != 'float32':
        raise ValueError('local CPU FP32 Stage C base required')
    verified_builder()
    if version('diffusers') != '0.36.0':
        raise ValueError('Cascade requires diffusers==0.36.0')


def prepare_inference(options, spec):
    configuration = dict(spec.configuration, **options)
    model = load_model(configuration)
    if model_identity(model) != spec.identity['model']:
        raise ValueError('native Stage C base/source differs before adapter attachment')
    response, close = native(configuration, model=model)
    try:
        identity = native_identity(response, model_identity=response.model_identity, protocol=PROTOCOL)
        def predict(*, row, strength=1., seed=0, **request):
            if request:
                raise ValueError('unknown Stage C response arguments')
            check_row(row, response.model, spec.configuration.get('shape', [1, 4, 4, 4]))
            return response._response(row, scale=strength).detach().clone()
        return InferenceBinding(response.network, identity, close, response=predict)
    except BaseException:
        close()
        raise


BACKEND = ResponseBackend(NAME, capabilities({}), prepare, validate, configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, Capabilities('adapter_gan', False), prepare_inference,
    validate_inference, configure_capabilities=capabilities)
