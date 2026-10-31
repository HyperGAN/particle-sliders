"""Staged SD3 response factory. Shared core owns updates, packets, EMA and inference."""
from contextlib import contextmanager
import math
import json
from pathlib import Path

import torch

from particle_sliders.lifecycle import Backend, Binding, Capabilities, native_cleanup
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg094 import ConditionalPG094, validate_pg09_domain
from particle_sliders.pg09_backends import AdapterParameters, ResponseRow
from particle_sliders.pg09_images import ImageResponseAdapter, image_condition_features
from particle_sliders.fibo_hndl import HNDLFieldCritic
from .runtime import Adapter, PROTOCOL, admitted_builder, digest, model_identity, restore_all, targets

NAME = 'sd3-shared-hndl010'
CAPABILITIES = Capabilities('adapter_gan', False, base_precisions=('float32',),
    render_outputs=('latent',), operations=('train', 'resume', 'evaluate', 'export', 'response'),
    qualification='CPU SD3 native architecture conformance only; pretrained/model quality and samplers unqualified',
    unavailable_reasons=(('render', 'This raw flow-velocity cohort has no qualified native image sampler'),))
NATIVE_KEYS = ('rank', 'alpha', 'base_sha256')


def validate(options):
    allowed = set(NATIVE_KEYS) | {'model_path', 'rows_path', 'rows_sha256', 'steps', 'batch', 'seed', 'ema_decay',
        'base_precision', 'preset', 'critic_width', 'pg_version', 'pg_source_profile', 'penalty_input_domain'}
    if set(options) - allowed:
        raise ValueError('unknown SD3 cohort options: ' + str(sorted(set(options) - allowed)))
    if type(options.get('rank', 4)) is not int or not 1 <= options.get('rank', 4) <= 128:
        raise ValueError('rank must be an integer in [1,128]')
    alpha = options.get('alpha', 4.)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    if type(options.get('critic_width', 8)) is not int or not 1 <= options.get('critic_width', 8) <= 128:
        raise ValueError('critic_width must be in [1,128]')
    base = options.get('base_sha256')
    if not isinstance(base, str) or len(base) != 64 or any(c not in '0123456789abcdef' for c in base):
        raise ValueError('the actual staged base_sha256 is required')
    if 'rows_path' in options:
        rows_sha = options.get('rows_sha256')
        if not isinstance(rows_sha, str) or len(rows_sha) != 64 or any(c not in '0123456789abcdef' for c in rows_sha):
            raise ValueError('rows_sha256 must bind the fixed training cache')
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('this cohort requires float32')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('only shared public BISPF presets are admitted')
    from particle_sliders.pg094 import engine_options
    engine_options(options)
    admitted_builder()


def load_model(options):
    from diffusers import SD3Transformer2DModel
    path = Path(options['model_path'])
    if not path.is_dir():
        raise ValueError('model_path must be a local staged SD3 transformer directory')
    return SD3Transformer2DModel.from_pretrained(str(path), local_files_only=True, torch_dtype=torch.float32)


def check_model(model, options):
    identity = model_identity(model)
    if identity['base_sha256'] != options['base_sha256']:
        raise ValueError('staged SD3 base tensors/config differ; no native mutation performed')
    return identity, targets(model)


def check_condition(condition, model, device):
    if not isinstance(condition, dict) or set(condition) != {'embeds', 'pooled'}:
        raise ValueError('SD3 needs both native joint text embeddings and pooled projections')
    embeds, pooled = condition['embeds'], condition['pooled']
    for value in (embeds, pooled):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device != device or not torch.isfinite(value).all():
            raise ValueError('native conditioning must be finite staged float32 tensors')
    if (embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[1] < 1 or
            embeds.shape[2] != model.config.joint_attention_dim or
            pooled.shape != (1, model.config.pooled_projection_dim)):
        raise ValueError('SD3 text/pooled conditioning has incompatible native dimensions')


def check_row(row, model):
    if not isinstance(row, ResponseRow) or row.schedule is not None or row.scale != 1 or row.strength != 1 or row.guidance != 1:
        raise ValueError('SD3 cohort rows require one raw timestep, scale/strength/CFG=1; no historical objective alias')
    latent = row.latent
    if not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32 or latent.device.type != 'cpu' or latent.ndim != 4:
        raise ValueError('SD3 requires genuine CPU float32 [1,C,H,W] latents')
    if latent.shape[:2] != (1, model.config.in_channels) or not torch.isfinite(latent).all():
        raise ValueError('SD3 latent channels/batch/nonfinite values differ')
    patch = model.config.patch_size
    if any(v < patch or v % patch for v in latent.shape[2:]):
        raise ValueError('SD3 spatial lattice must respect native patch size')
    maximum = model.config.pos_embed_max_size
    if maximum is not None and any(v // patch > maximum for v in latent.shape[2:]):
        raise ValueError('SD3 lattice exceeds native positional geometry')
    if model.config.out_channels != model.config.in_channels:
        raise ValueError('SD3 flow response must retain latent channels')
    t = row.timestep
    if not isinstance(t, torch.Tensor) or t.numel() != 1 or not torch.isfinite(t).all() or not 0 <= float(t) <= 1000:
        raise ValueError('SD3 requires a finite raw scheduler timestep in [0,1000]')
    for condition in (row.neutral, row.positive, row.negative):
        check_condition(condition, model, latent.device)
    validate_pg09_domain(tuple(latent.shape), response_domain='continuous_field')


def load_rows(path):
    values = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError('at least two fixed SD3 row dictionaries required')
    allowed = {'neutral', 'positive', 'negative', 'latent', 'timestep'}
    if any(not isinstance(v, dict) or set(v) != allowed for v in values):
        raise ValueError('SD3 fixed rows require neutral/positive/negative/latent/timestep only')
    return [ResponseRow(**v) for v in values]


@contextmanager
def observation(model):
    restore = native_cleanup(model)
    try:
        model.eval()
        yield
    finally:
        restore()


def bind(model, options, identity, selected):
    # All model/row/runtime gates have run before this first native mutation.
    builder, graph_identity = admitted_builder()
    restore = native_cleanup(model)
    adapter = None
    try:
        model.requires_grad_(False)
        model.eval()
        adapter = Adapter(selected, options.get('rank', 4), options.get('alpha', 4.), builder, graph_identity)
        network = AdapterParameters(adapter.named_parameters(), adapter.scaled)
        adapter.attach(selected)
        response = ImageResponseAdapter('sd3', model, network, model_identity=identity)
        response.adapter = adapter
        def close():
            restore_all([adapter.close, restore])
        return response, close
    except BaseException:
        restore_all(([adapter.close] if adapter is not None else []) + [restore])
        raise


def native_callback(response):
    def predict(*, strength=1., seed=0, latent, embeds, pooled, timestep):
        condition = dict(embeds=embeds, pooled=pooled)
        row = ResponseRow(condition, condition, condition, latent, timestep=torch.as_tensor(timestep))
        check_row(row, response.transformer)
        return response.velocity(condition, latent, row.timestep, scale=strength)
    return predict


def prepare(options, *, model=None, rows=None):
    torch.default_generator.manual_seed(options.get('seed', 7))
    if rows is None and digest(options['rows_path']) != options['rows_sha256']:
        raise ValueError('fixed SD3 training cache identity differs before native mutation')
    model = load_model(options) if model is None else model
    identity, selected = check_model(model, options)
    rows = load_rows(options['rows_path']) if rows is None else list(rows)
    if len(rows) < 2:
        raise ValueError('two or more fixed rows required')
    for row in rows:
        check_row(row, model)
    if len({tuple(row.latent.shape) for row in rows}) != 1:
        raise ValueError('one native spatial geometry per cohort required')
    response, close = bind(model, options, identity, selected)
    try:
        prepared = [response.prepare(row) for row in rows]
        conditions = image_condition_features(prepared)
        channels, height, width = rows[0].latent.shape[1:]
        critic = HNDLFieldCritic(channels, conditions.shape[1], (height, width), options.get('critic_width', 8))
        game = ConditionalPG094(response, prepared, critic, conditions=conditions,
            batch_size=options.get('batch', 2), total_steps=options.get('steps', 4),
            seed=options.get('seed', 7), preset=options.get('preset', 'bispf_optimized'))
        configuration = {key: options.get(key, {'rank': 4, 'alpha': 4.}.get(key)) for key in NATIVE_KEYS}
        spec = INFERENCE.specification(response, model_identity=identity, protocol=PROTOCOL, configuration=configuration)
        return Binding(game, close, inference=spec, observation=lambda: observation(model))
    except BaseException:
        close()
        raise


def validate_inference(options, spec):
    if set(options) - {'model_path', 'base_precision'}:
        raise ValueError('native inference accepts only local model_path and float32 precision')
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('native inference requires float32')
    validate(spec.configuration)


def prepare_inference(options, spec):
    settings = dict(spec.configuration, model_path=options['model_path'])
    model = load_model(settings)
    identity, selected = check_model(model, settings)
    if spec.identity['protocol'] != PROTOCOL or spec.identity['model'] != json.loads(json.dumps(identity)):
        raise ValueError('native SD3 model/protocol specification differs before native mutation')
    rank = settings.get('rank', 4)
    shapes = [shape for _, module in selected for shape in ([rank, module.in_features], [module.out_features, rank])]
    schema = {f'weights.{i}': [shape, 'torch.float32'] for i, shape in enumerate(shapes)}
    if spec.identity['network_schema'] != schema:
        raise ValueError('native SD3 rank/target schema differs before native mutation')
    tree = spec.identity['architectures'].get('backend.adapter', {}).get('tree', [])
    projections = [entry for entry in tree if entry[1] == __package__ + '.runtime.Projection']
    if len(projections) != len(selected):
        raise ValueError('native SD3 projection count differs before native mutation')
    for index, (entry, (_, module)) in enumerate(zip(projections, selected)):
        saved = json.loads(entry[2])
        expected = dict(rank=rank, alpha=settings.get('alpha', 4.),
                        input_width=module.in_features, output_width=module.out_features)
        if entry[0] != f'layers.{index}' or any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('native SD3 projection configuration differs before native mutation')
    response, close = bind(model, settings, identity, selected)
    try:
        return InferenceBinding(response.network,
            native_identity(response, model_identity=identity, protocol=PROTOCOL), close,
            response=native_callback(response), observation=lambda: observation(model))
    except BaseException:
        close()
        raise


BACKEND = Backend(NAME, CAPABILITIES, prepare, validate)
INFERENCE = InferenceBackend(NAME, CAPABILITIES, prepare_inference, validate_inference)
