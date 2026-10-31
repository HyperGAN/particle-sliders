"""Product preparation only. Shared core owns losses, optimizers and lifecycle.

Only staged local Diffusers ZImageTransformer2DModel FP32 CPU bundles are admitted
by this preview. No model download, production sampler or media rendering runs.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import importlib
import inspect
import json
import math
import struct
from pathlib import Path

import torch
from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

from particle_sliders.fibo_hndl import HNDLFieldCritic, verified_builder
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, native_cleanup, _restore_all
from particle_sliders.pg094 import ConditionalPG094
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09_images import ImageResponseAdapter, image_condition_features
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from .graphs import GRAPH_SHA256, ZImageLoRA, targets

NAME = 'embedded-zimage-shared'
COHORT = 'embedded-zimage-adapter-gan-pg094-hg-hndl010-v3'
PROTOCOL = 'embedded-zimage-native-4d-raw-velocity-v2'
CORE_COMMIT = '7a8a321d04323d49319d0b08e6093a0ddce36b27'
CORE_HASHES = {
    'pg09.py': 'f4fe77d6b417c023355c9398c1d000072f35072fb5c1b923c60c5799983d2b56',
    'lifecycle.py': 'a584c2ab2c97eb41c2329aa92d8935da4d7d6773e5babf66aea6a492cd855a05',
    'inference.py': 'bd51ec7ae8b512b57f2964f5d5e80a2bd1280083529e17c64726e6ab04494065',
}


INITIALIZATION_POLICY = 'scoped-cpu-initialization-streams-v1'


def initialization_identity(seed):
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError('invalid initialization seed')
    return dict(policy=INITIALIZATION_POLICY, seed=seed, device='cpu',
        streams=dict(adapter=seed, critic=seed + 1, native_loader=seed + 2, preparation=seed + 3))


def validate_initialization(identity):
    if not isinstance(identity, dict) or identity != initialization_identity(identity.get('seed')):
        raise ValueError('initialization policy/seed/streams differ')


@contextmanager
def initialization_scope(identity, component):
    """CPU-only profile: fork/seed only its admitted CPU initialization stream.

    No CUDA devices are admitted or discovered. CUDA RNG state is untouched;
    any future accelerator profile requires its own explicit device policy.
    """
    validate_initialization(identity)
    if component not in identity['streams']:
        raise ValueError('unknown initialization stream')
    with torch.random.fork_rng(devices=[]):
        seed_initialization_rng(identity['streams'][component], [])
        yield


HASH_CHUNK_BYTES = 1024 * 1024


def _stream_sha256(stream):
    digest = hashlib.sha256()
    while block := stream.read(HASH_CHUNK_BYTES):
        digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    """Hash complete local file bytes with bounded reads, without tensor loading."""
    with Path(path).open('rb') as stream:
        return _stream_sha256(stream)


def digest_state(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(',', ':')).encode())
        data = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')
        for offset in range(0, len(data), HASH_CHUNK_BYTES):
            digest.update(data[offset:offset + HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _safetensors_state_digest(path):
    """Match digest_state from header/offsets and streamed bytes; deserialize no weights.

    The native loader still validates the safetensors format and loads the base
    once. Finite FP32 values are checked on that actual model before attachment.
    """
    dtypes = {'F32': ('torch.float32', 4), 'I64': ('torch.int64', 8),
              'I32': ('torch.int32', 4), 'I16': ('torch.int16', 2),
              'I8': ('torch.int8', 1), 'U8': ('torch.uint8', 1), 'BOOL': ('torch.bool', 1)}
    with Path(path).open('rb') as stream:
        size = Path(path).stat().st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('invalid staged safetensors header')
        header_size = struct.unpack('<Q', prefix)[0]
        if not 2 <= header_size <= min(16 * HASH_CHUNK_BYTES, size - 8):
            raise ValueError('invalid staged safetensors header length')
        blocks, remaining = [], header_size
        while remaining:
            block = stream.read(min(HASH_CHUNK_BYTES, remaining))
            if not block:
                raise ValueError('truncated staged safetensors header')
            blocks.append(block)
            remaining -= len(block)
        header = json.loads(b''.join(blocks))
        if not isinstance(header, dict):
            raise ValueError('invalid staged safetensors tensor schema')
        entries = {}
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if (not isinstance(tensor, dict) or set(tensor) != {'dtype', 'shape', 'data_offsets'}
                    or not isinstance(tensor['dtype'], str) or tensor['dtype'] not in dtypes or not isinstance(tensor['shape'], list)
                    or any(type(v) is not int or v < 0 for v in tensor['shape'])
                    or not isinstance(tensor['data_offsets'], list) or len(tensor['data_offsets']) != 2
                    or any(type(v) is not int for v in tensor['data_offsets'])):
                raise ValueError('staged Z-Image requires FP32/integer native safetensors schema')
            begin, end = tensor['data_offsets']
            dtype, itemsize = dtypes[tensor['dtype']]
            if not 0 <= begin <= end <= size - 8 - header_size or end - begin != math.prod(tensor['shape']) * itemsize:
                raise ValueError('invalid staged safetensors offsets')
            entries[name] = (tensor['shape'], dtype, begin, end)
        if not entries:
            raise ValueError('empty staged safetensors model')
        cursor = 0
        for _, _, begin, end in sorted(entries.values(), key=lambda value: (value[2], value[3])):
            if begin != cursor:
                raise ValueError('noncontiguous staged safetensors offsets')
            cursor = end
        if cursor != size - 8 - header_size:
            raise ValueError('unbound staged safetensors trailing bytes')
        digest = hashlib.sha256()
        for name, (shape, dtype, begin, end) in sorted(entries.items()):
            digest.update(json.dumps([name, shape, dtype], separators=(',', ':')).encode())
            stream.seek(8 + header_size + begin)
            remaining = end - begin
            while remaining:
                block = stream.read(min(HASH_CHUNK_BYTES, remaining))
                if not block:
                    raise ValueError('truncated staged safetensors tensor')
                digest.update(block)
                remaining -= len(block)
        return digest.hexdigest()


def bundle_identity(directory):
    """Bind exact local config/weight file bytes without deserializing weights."""
    path = Path(directory)
    if not path.is_dir() or not (path / 'config.json').is_file() or not (path / 'diffusion_pytorch_model.safetensors').is_file():
        raise ValueError('staged local unsharded Z-Image config/safetensors bundle required')
    config = json.loads((path / 'config.json').read_text())
    if config.get('_class_name') != 'ZImageTransformer2DModel':
        raise ValueError('model bundle is not the native Z-Image transformer')
    if config.get('all_patch_size') != [2] or config.get('all_f_patch_size') != [1]:
        raise ValueError('preview requires Z-Image native image patch2/temporal1 geometry')
    return dict(model='ZImageTransformer2DModel', identity_profile='streamed-local-safetensors-v2',
        files={name:file_sha256(path / name) for name in ('config.json', 'diffusion_pytorch_model.safetensors')},
        state_sha256=_safetensors_state_digest(path / 'diffusion_pytorch_model.safetensors'), config=config,
        diffusers=version('diffusers'), native_source_sha256=hashlib.sha256(
            Path(inspect.getsourcefile(ZImageTransformer2DModel)).read_bytes()).hexdigest(),
        graph_sha256=GRAPH_SHA256, core_commit=CORE_COMMIT, rank=16, alpha=16,
        qualification='random tiny native API CPU fixtures only; pretrained/native quality unqualified')


def source_closure(root):
    root = Path(root)
    return {p.relative_to(root).as_posix():file_sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.hndl', '.json')}


def _core_gate():
    import particle_sliders
    root = Path(particle_sliders.__file__).parent
    if version('particle-sliders-core') != '0.12.0' or version('diffusers') != '0.42.1':
        raise ValueError('isolated core0.12/diffusers0.42.1 preview required')
    profile = json.loads(Path(__file__).with_name('source_profile.json').read_text())
    actual = source_closure(root)
    if (profile.get('core_closure') != 'recursive-relative-path-sha256-v1'
            or profile['core_commit'] != CORE_COMMIT or actual != profile['core_sources']) or any(
            actual.get(name) != digest for name,digest in CORE_HASHES.items()):
        raise ValueError('core lifecycle/source differs from the pinned CPU-tested candidate')
    for module_name,digest in profile['native_sources'].items():
        if hashlib.sha256(Path(importlib.import_module(module_name).__file__).read_bytes()).hexdigest() != digest:
            raise ValueError('native Diffusers source differs from the qualified CPU closure')
    if hashlib.sha256(Path(__file__).with_name('graphs.py').read_bytes()).hexdigest() != profile['product_graph_source_sha256']:
        raise ValueError('product HNDL graph source differs from the declared cohort')
    verified_builder()


def _shape(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or value[0] != 1 or any(type(v) is not int or v < 1 for v in value):
        raise ValueError('declare native response shape [1,C,H,W]; vectors/temporal reinterpretation refused')
    if value[2] % 2 or value[3] % 2:
        raise ValueError('native patch2 requires even height/width')
    return tuple(value)


def capabilities(options):
    return ResponseCapabilities('adapter_gan', 'image', 'raw_flow_velocity', 'continuous_field',
        _shape(options.get('response_shape', [1,4,4,4])), '0.9.4', 'release-0.9.4',
        base_precisions=('float32',), qualification='tiny native API CPU fixtures only; pretrained/GPU unqualified')


def validate(options):
    allowed = {'model_dir','model_identity','rows','rows_sha256','response_shape','steps','batch','seed','preset',
               'ema_decay','base_precision','pg_version','pg_source_profile','penalty_input_domain'}
    if set(options) - allowed:
        raise ValueError('unknown embedded Z-Image preview option')
    _core_gate()
    shape = _shape(options.get('response_shape'))
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('this preview admits float32 CPU only')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('choose an existing shared BISPF preset')
    actual = bundle_identity(options['model_dir'])
    if actual != options.get('model_identity') or shape[1] != actual['config']['in_channels']:
        raise ValueError('native model/source/geometry identity differs before attachment')
    if not isinstance(options.get('rows'), str):
        raise ValueError('explicit frozen preencoded rows file required')
    digest = options.get('rows_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('complete frozen rows_sha256 required')


def read_rows(options):
    with Path(options['rows']).open('rb') as stream:
        if _stream_sha256(stream) != options.get('rows_sha256'):
            raise ValueError('complete frozen rows file hash differs before attachment')
        stream.seek(0)
        raw = torch.load(stream, map_location='cpu', weights_only=True)
        stream.seek(0)
        if _stream_sha256(stream) != options['rows_sha256']:
            raise ValueError('frozen rows file changed while reading')
    if not isinstance(raw, dict) or set(raw) != {'schema','model_identity','rows'} or raw['schema'] != PROTOCOL or raw['model_identity'] != options['model_identity']:
        raise ValueError('frozen rows model/protocol identity differs')
    if not isinstance(raw['rows'], list) or len(raw['rows']) < 2:
        raise ValueError('at least two frozen native rows required')
    result = []
    for item in raw['rows']:
        if not isinstance(item, dict) or set(item) != {'neutral','positive','negative','latent','timestep'}:
            raise ValueError('rows require three native encodings, latent and raw timestep')
        row = ResponseRow(**item, guidance=1.)  # raw velocity, historical g=0; no CFG override.
        check_request(row.neutral, row.latent, row.timestep, options['response_shape'], options['model_identity'])
        for condition in (row.positive, row.negative):
            check_request(condition, row.latent, row.timestep, options['response_shape'], options['model_identity'])
        result.append(row)
    return result


def check_request(condition, latent, timestep, shape, identity):
    if (not isinstance(latent, torch.Tensor) or tuple(latent.shape) != tuple(shape) or latent.dtype != torch.float32
            or latent.device.type != 'cpu' or not torch.isfinite(latent).all()):
        raise ValueError('finite float32 CPU native latent field required')
    if not isinstance(condition, dict) or set(condition) != {'embeds'}:
        raise ValueError('one frozen native caption embedding dictionary required')
    embeds = condition['embeds']
    if (not isinstance(embeds, torch.Tensor) or embeds.ndim != 2 or embeds.shape[0] < 1
            or embeds.shape[1] != identity['config']['cap_feat_dim'] or embeds.dtype != torch.float32
            or embeds.device.type != 'cpu' or not torch.isfinite(embeds).all()):
        raise ValueError('finite float32 native [tokens,caption_width] encoding required')
    if (not isinstance(timestep, torch.Tensor) or timestep.numel() != 1 or timestep.device.type != 'cpu'
            or not torch.isfinite(timestep).all() or not 0 <= timestep.item() <= 1000):
        raise ValueError('one native raw timestep in [0,1000] required')


def preflight(options, request, staged):
    read_rows(options)  # Complete dataset before hooks/freeze/mode mutation.
    if staged.get('model') is not None:
        _model_gate(staged['model'], options['model_identity'])


def _model_gate(model, identity):
    if type(model) is not ZImageTransformer2DModel:
        raise ValueError('borrowed native model identity differs')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('native CPU float32 qualification only')
    if digest_state(model.state_dict()) != identity['state_sha256']:
        raise ValueError('borrowed native model identity differs')
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('staged native model must contain finite FP32 CPU weights')
    if dict(model.config) != identity['config']:
        # Diffusers adds private load metadata; it cannot change model geometry.
        a = {k:v for k,v in model.config.items() if not k.startswith('_')}
        b = {k:v for k,v in identity['config'].items() if not k.startswith('_')}
        if json.loads(json.dumps(a)) != b:
            raise ValueError('borrowed native model config differs')
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('native model already has hooks; overlapping adapter ownership refused')
    for module in model.modules():
        processor = getattr(module, 'processor', None)
        if processor is not None and (type(processor) is not ZSingleStreamAttnProcessor or
                processor._attention_backend is not None or processor._parallel_config is not None):
            raise ValueError('only default native CPU attention processor is qualified')
    targets(model)


def _native(options, identity, model=None, *, initialization=None):
    initialization = initialization_identity(options.get('seed', 7)) if initialization is None else initialization
    validate_initialization(initialization)
    if model is None:
        with initialization_scope(initialization, 'native_loader'):
            model = ZImageTransformer2DModel.from_pretrained(
                options['model_dir'], local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    if any(file_sha256(Path(options['model_dir']) / name) != digest for name,digest in identity['files'].items()):
        raise ValueError('native model file identity changed before attachment')
    _model_gate(model, identity)
    restore = native_cleanup(model)
    network = None
    def close():
        _restore_all(([network.close] if network is not None else []) + [restore])
    try:
        model.requires_grad_(False).eval()
        with initialization_scope(initialization, 'adapter'):
            network = ZImageLoRA(model)
        adapter = ImageResponseAdapter('zimage', model, network, model_identity=identity)
        return adapter, close
    except BaseException:
        close()
        raise


def _observe(adapter):
    @contextmanager
    def observe():
        restore = native_cleanup(adapter.transformer)
        try:
            yield
        finally:
            restore()
    return observe


def prepare(options, *, model=None):
    initialization = initialization_identity(options.get('seed', 7))
    with initialization_scope(initialization, 'preparation'):
        return _prepare(options, model=model, initialization=initialization)


def _prepare(options, *, model, initialization):
    frozen_rows = read_rows(options)  # Recheck complete file before native attachment.
    adapter, close = _native(options, options['model_identity'], model, initialization=initialization)
    try:
        rows = [adapter.prepare(row) for row in frozen_rows]
        features = image_condition_features(rows)
        with initialization_scope(initialization, 'critic'):
            critic = HNDLFieldCritic(options['response_shape'][1], features.shape[1], tuple(options['response_shape'][2:]))
        game = ConditionalPG094(adapter, rows, critic, conditions=features,
            preset=options.get('preset', 'bispf_optimized'), total_steps=options.get('steps',4),
            batch_size=options.get('batch',2), seed=options.get('seed',7), response_domain='continuous_field')
        configuration = dict(model_identity=options['model_identity'], response_shape=options['response_shape'],
            initialization=initialization)
        spec = INFERENCE_BACKEND.specification(adapter, model_identity=options['model_identity'],
            protocol=PROTOCOL, configuration=configuration)
        return Binding(game, close, inference=spec, observation=_observe(adapter))
    except BaseException:
        close()
        raise


def validate_inference(options, spec):
    if set(options) != {'model_dir'}:
        raise ValueError('native-only inference accepts just a staged model_dir')
    _core_gate()
    if bundle_identity(options['model_dir']) != spec.configuration['model_identity']:
        raise ValueError('native inference base/source identity differs before attachment')
    _shape(spec.configuration['response_shape'])
    validate_initialization(spec.configuration.get('initialization'))
    source = hashlib.sha256(Path(__file__).with_name('graphs.py').read_bytes()).hexdigest()
    if spec.identity['sources'].get('embedded_zimage_shared.graphs') != source:
        raise ValueError('saved native graph source differs before attachment')


def inference_capabilities(configuration):
    return capabilities(configuration)


def prepare_inference(options, spec):
    initialization = spec.configuration['initialization']
    with initialization_scope(initialization, 'preparation'):
        return _prepare_inference(options, spec, initialization=initialization)


def _prepare_inference(options, spec, *, initialization):
    identity = spec.configuration['model_identity']
    adapter, close = _native(options, identity, initialization=initialization)
    try:
        def response(*, condition, latent, timestep, strength, seed):
            check_request(condition, latent, timestep, spec.configuration['response_shape'], identity)
            return adapter.velocity(condition, latent, timestep, scale=strength).detach().clone()
        return InferenceBinding(adapter.network, native_identity(adapter, model_identity=identity, protocol=PROTOCOL),
            close, response=response, observation=_observe(adapter))
    except BaseException:
        close()
        raise


BACKEND = ResponseBackend(NAME, capabilities({'response_shape':[1,4,4,4]}), prepare, validate,
    configure_capabilities=capabilities, preflight=preflight)
INFERENCE_BACKEND = InferenceBackend(NAME, capabilities({'response_shape':[1,4,4,4]}),
    prepare_inference, validate_inference, configure_capabilities=inference_capabilities)
