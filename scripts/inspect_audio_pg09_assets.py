"""Offline tokenizer/asset/runtime preflight. Never loads model weight tensors."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_yue2_product(product):
    """Keep the installed native runtime, add only product-only submodules."""
    product = Path(product).resolve()
    if str(product) in sys.path:
        raise ValueError('omit product root from PYTHONPATH; bootstrap native runtime first')
    runtime = importlib.import_module('yue2')
    if Path(runtime.__file__).resolve().is_relative_to(product):
        raise ValueError('product package shadows the official native runtime')
    for name in ('backend.py', 'slider.py', 'defaults.py', 'prompts.py'):
        if any((Path(root) / name).exists() for root in runtime.__path__):
            raise ValueError('runtime/product submodule collision: ' + name)
    runtime.__path__.append(str(product / 'yue2'))
    return runtime


def runtime_identity():
    import torch
    import transformers
    import particlegan
    root = Path(particlegan.__file__).resolve().parent
    return dict(sys_executable=sys.executable,
        torch_version=torch.__version__, torch_file=torch.__file__,
        transformers_version=transformers.__version__, transformers_file=transformers.__file__,
        particlegan_version=importlib.metadata.version('particlegan'),
        particlegan_file=particlegan.__file__, cuda_initialized=torch.cuda.is_initialized(),
        pg_sources={str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob('*.py'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('yue2', 'music3'), required=True)
    parser.add_argument('--product', type=Path)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--train-prompts', type=Path, required=True)
    parser.add_argument('--heldout-prompts', type=Path, required=True)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml
    if torch.cuda.is_initialized():
        raise RuntimeError('asset inspection must not initialize CUDA')
    manifest = json.loads(args.manifest.read_text())
    identity = runtime_identity()
    if identity['particlegan_version'] != '0.9.0' or len(identity['pg_sources']) != 49:
        raise ValueError('expected released PG0.9 source inventory before domain review')
    if args.family == 'yue2':
        if args.product is None:
            parser.error('--product required for yue2')
        runtime = bootstrap_yue2_product(args.product)
        from yue2.protocol import SongRequest, token_prefixes
        from yue2.tokenization_yue2 import YuE2TextTokenizer
        from yue2.backend import YuE2Backend
        tokenizer = YuE2TextTokenizer(args.model_dir / 'qwen.tiktoken')
        def encode(style, lyrics):
            return token_prefixes(SongRequest(style=style, lyrics=lyrics, cot='off'), tokenizer)
        config = args.model_dir / 'config.json'
        identity.update(yue2_runtime_file=runtime.__file__, yue2_product_backend_file=sys.modules[YuE2Backend.__module__].__file__,
            yue2_runtime_distribution=json.loads(importlib.metadata.distribution('yue2-infer').read_text('direct_url.json')))
        weight = args.model_dir / 'model.safetensors'
        recorded = manifest['model_identity']['files']['model.safetensors']
        if weight.stat().st_size != recorded['bytes'] or file_sha(config) != manifest['model_identity']['config_sha256']:
            raise ValueError('staged YuE2 weight size or config identity differs')
        checked_weights = [dict(path=str(weight.resolve()), bytes=weight.stat().st_size,
                                sha256_from_existing_manifest=recorded['sha256'])]
    else:
        from transformers import AutoTokenizer
        from conceptmod.textsliders.train_lm_slider_music3 import (
            _assemble, _tokenize, _assert_last_token_is_audio_start,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / 'tokenizer'), local_files_only=True)
        def encode(style, lyrics):
            ids, mask = _tokenize(tokenizer, _assemble(style, lyrics), 'cpu')
            _assert_last_token_is_audio_start(ids, mask, tokenizer, where='offline PG0.9 asset preflight')
            return ids[0][mask[0].bool()].tolist()
        config = args.model_dir / 'language_model/config.json'
        checked_weights = []
        for recorded in manifest['files']:
            path = args.model_dir / recorded['filename']
            if not path.is_file() or path.stat().st_size != recorded['bytes']:
                raise ValueError('missing or changed staged Music3 asset: ' + str(path))
            if path.suffix == '.safetensors':
                checked_weights.append(dict(path=str(path.resolve()), bytes=recorded['bytes'],
                    sha256_from_existing_manifest=recorded['sha256']))
            elif path.name == 'config.json' and file_sha(path) != recorded['sha256']:
                raise ValueError('staged Music3 config hash differs: ' + str(path))
    rows = []
    for split, path in (('train', args.train_prompts), ('heldout', args.heldout_prompts)):
        prompts = yaml.safe_load(path.read_text())['rows']
        for i, row in enumerate(prompts):
            neutral = encode(row['neutral'], row['lyrics'])
            positive = encode(row['positive'], row['lyrics'])
            if max(len(neutral), len(positive)) > args.max_seq_len:
                raise ValueError(f'{split} row{i} exceeds declared token budget; no truncation allowed')
            rows.append(dict(split=split, index=i, neutral_tokens=len(neutral), positive_tokens=len(positive),
                context_sha256=hashlib.sha256(json.dumps(dict(neutral=neutral, positive=positive)).encode()).hexdigest()))
    assert not torch.cuda.is_initialized()
    receipt = dict(status='assets_and_tokenizers_prepared_not_admitted', family=args.family,
        runtime=identity, config=dict(path=str(config), sha256=file_sha(config),
            hidden_size=json.loads(config.read_text())['hidden_size']),
        existing_asset_manifest_sha256=file_sha(args.manifest),
        weight_checks='existence and byte sizes; SHA values inherited from existing verified manifest',
        weights=checked_weights, prompt_rows=rows, max_seq_len=args.max_seq_len,
        proposed_update_budgets=[3, 20], proposed_max_wall_seconds=900,
        domain_qa_approved=False, gpu_admitted=False, loaded_model_weights=False,
        commands_executed='CPU tokenizer and file metadata inspection only')
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'family', 'config', 'prompt_rows', 'loaded_model_weights')}, indent=2))


if __name__ == '__main__':
    main()
