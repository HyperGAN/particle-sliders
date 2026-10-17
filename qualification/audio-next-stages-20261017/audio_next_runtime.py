"""Import/config/tokenizer-only audio next-stage checks; no model constructors."""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import torch
import yaml

from inspect_audio_pg09_assets import bootstrap_yue2_product, runtime_identity
from particle_sliders.pg09 import validate_pg09_domain

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--family', choices=('yue2', 'music3'), required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = Path('/mnt/nvme/sliders/audio-adapters-pg09-20261017')
sources = {}


def bind(path):
    path = Path(path).resolve()
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


result = dict(family=args.family, runtime=runtime_identity(), models_constructed=False,
              pretrained_tensors_loaded=False, audio_generated=False)
result['domain'] = validate_pg09_domain((1, 2048 if args.family == 'yue2' else 4096),
    response_domain='continuous_vector', pg_source_profile='alz-continuous-v1',
    penalty_input_domain='continuous')
rows = yaml.safe_load((ROOT / 'conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml').read_text())['rows']
if args.family == 'yue2':
    native = bootstrap_yue2_product('/mnt/nvme/sliders/audio-yue2-source-20261017')
    from yue2.pipeline import YuE2Pipeline
    from yue2.protocol import SongRequest, Sampling, GenerationConfig, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.modeling_vae import YuE2VAE
    import yue2.nar
    import soundfile
    manifest = json.loads(Path('/mnt/nvme/sliders/qualification/models/yue2-cached/qualification-manifest.json').read_text())
    tokenizer = YuE2TextTokenizer(Path(manifest['records']['model']['path']) / 'qwen.tiktoken')
    sampling = Sampling(min_tokens=32, max_tokens=200)
    generation = GenerationConfig(semantic=sampling)
    result['native_render_api'] = {name: str(inspect.signature(getattr(YuE2Pipeline, name)))
        for name in ('__init__', 'from_pretrained', 'plan', 'generate_semantic', 'synthesize', 'decode', 'effective_config')}
    result['native_decoder_api'] = {name: str(inspect.signature(getattr(YuE2VAE, name)))
        for name in ('from_pretrained', 'decode', 'decode_tiled')}
    result['native_nar_api'] = str(inspect.signature(yue2.nar.synthesize))
    result['constructor_not_called'] = True
    result['protocol'] = dict(cot='off', cfg_scale=1.0, semantic_sampling=asdict(sampling),
                              generation_config=generation.to_dict(), sample_rate=48000,
                              backend='torch-eager', quantization='none', offload_ar=False)
    result['heldout_prefixes'] = []
    for index, row in enumerate(rows):
        variants = {}
        for label in ('neutral', 'positive'):
            request = SongRequest(style=row[label], lyrics=row['lyrics'], cot='off',
                                  cfg_scale=1.0, seed=1709 + index, id=f'case-{index}-{label}')
            tokens = token_prefixes(request, tokenizer)
            assert len(tokens) + sampling.max_tokens < generation.context
            variants[label] = dict(tokens=len(tokens), token_sha256=hashlib.sha256(
                json.dumps(tokens).encode()).hexdigest())
        result['heldout_prefixes'].append(dict(index=index, seed=1709 + index, variants=variants))
    for path in Path(native.__file__).parent.glob('*.py'):
        bind(path)
    result['soundfile'] = dict(version=soundfile.__version__, file=soundfile.__file__)
    bind(soundfile.__file__)
else:
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from conceptmod.textsliders.train_lm_slider_music3 import _assemble, _tokenize, _assert_last_token_is_audio_start
    from conceptmod.textsliders.music3_particle_bridge import Music3Backend, ParticleNetwork
    model_dir = Path('/mnt/nvme/sliders/qualification/models/music3-bd348f9')
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / 'tokenizer'), local_files_only=True)
    result['heldout_prefixes'] = []
    for index, row in enumerate(rows):
        variants = {}
        for label in ('neutral', 'positive'):
            ids, mask = _tokenize(tokenizer, _assemble(row[label], row['lyrics']), 'cpu')
            _assert_last_token_is_audio_start(ids, mask, tokenizer, where='offline-next-stage')
            variants[label] = dict(tokens=int(mask.sum()), last_real_token=int(ids[0][mask[0].bool()][-1]))
        result['heldout_prefixes'].append(dict(index=index, seed=1709 + index, variants=variants))
    for cls in (Qwen3ForCausalLM, Music3Backend, ParticleNetwork):
        bind(inspect.getsourcefile(cls))
    try:
        import diffusers
        pipeline = getattr(diffusers, 'MiniMaxMusic3ModularPipeline')
        result['native_audio_pipeline_api'] = {name: str(inspect.signature(getattr(pipeline, name)))
            for name in ('from_pretrained', 'load_components', 'update_components', '__call__')}
        bind(inspect.getsourcefile(pipeline))
        for path in Path(inspect.getsourcefile(pipeline)).parent.glob('*.py'):
            bind(path)
        result['diffusers'] = dict(version=diffusers.__version__, file=diffusers.__file__)
    except Exception as error:
        result['native_audio_pipeline_runtime_error'] = type(error).__name__ + ': ' + str(error)
assert not torch.cuda.is_initialized()
result.update(cuda_initialized=False, sources=sources)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('runtime', 'sources')}, indent=2))
