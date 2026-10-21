"""Installed CLI for registered image backends; all models must be locally staged."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .lifecycle import SessionConfig, atomic_save, atomic_json, backend, open_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('capabilities', 'train', 'evaluate', 'render', 'infer'))
    parser.add_argument('--interface', choices=('image', 'response'), default='image')
    parser.add_argument('--config', type=Path, help='training/session configuration; not used by infer')
    parser.add_argument('--runtime', type=Path, help='native-only local runtime options JSON for infer')
    parser.add_argument('--mode', choices=('render', 'response'), default='render', help='native-only inference operation')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--export-format', choices=('v1', 'v2', 'both'), default='v1',
                        help='train exports: v1 adapter.pt, v2 native-only adapter-v2.pt, or both')
    parser.add_argument('--selection', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--native', type=Path, help='JSON native render arguments; images are local paths')
    parser.add_argument('--split', default='train', help='product-owned read-only evaluate split')
    args = parser.parse_args(argv)
    if args.operation != 'evaluate' and args.split != 'train':
        parser.error('--split applies only to evaluate')
    if args.operation != 'train' and args.export_format != 'v1':
        parser.error('--export-format applies only to train')
    if args.operation == 'infer':
        return infer(args, parser)
    if args.config is None:
        parser.error('--config is required for session operations')
    config = SessionConfig.from_dict(json.loads(args.config.read_text()))
    resolve, open_registered = backend, open_session
    if args.interface == 'response':
        from .response import response_backend, open_response
        resolve, open_registered = response_backend, open_response
    if args.operation == 'capabilities':
        print(json.dumps(asdict(resolve(config.backend).describe(config)), indent=2))
        return
    if args.out is None:
        parser.error('--out is required')
    if args.operation == 'train' and args.out.exists() and any(args.out.iterdir()):
        raise ValueError('training output must be a new empty directory to preserve existing receipts')
    native = json.loads(args.native.read_text()) if args.native else {}
    references = native.get('images')
    with open_registered(config, resume=args.resume, artifact=args.artifact,
                      operation=args.operation, strength=args.strength, references=bool(references), output_type=native.get('output_type'), native_request=native) as session:
        if args.operation == 'train':
            if args.export_format in ('v2', 'both'):
                from .inference import InferenceSpec
                from .lifecycle import UnsupportedOperation
                if not isinstance(session.binding.inference, InferenceSpec):
                    raise UnsupportedOperation('v2 export requires a product native-only InferenceSpec before training')
            args.out.mkdir(parents=True, exist_ok=True)
            with (args.out / 'metrics.jsonl').open('a') as stream:
                session.train(stop_after=args.stop_after,
                    on_step=lambda metrics: stream.write(json.dumps(metrics, allow_nan=False) + '\n'))
            session.checkpoint(args.out / 'training.pt')
            names = ['training.pt', 'metrics.jsonl']
            if args.export_format in ('v1', 'both'):
                session.export(args.out / 'adapter.pt', selection=args.selection)
                names.append('adapter.pt')
            if args.export_format in ('v2', 'both'):
                session.export_inference(args.out / 'adapter-v2.pt', selection=args.selection)
                names.append('adapter-v2.pt')
            files = {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in names}
            atomic_json(args.out / 'manifest.json', dict(manifest=session.manifest,
                completed_steps=session.game.completed_steps, files=files))
        elif args.operation == 'evaluate':
            atomic_save(args.out, {'kind': getattr(session.capabilities, 'response_kind', 'native velocity response'), 'quality_validated': False,
                'split': args.split, 'row': args.row, 'seed': args.seed, 'strength': args.strength,
                'response': session.evaluate(args.row, seed=args.seed, strength=args.strength, split=args.split).cpu()})
        else:
            from PIL import Image
            if references:
                native['images'] = []
                for path in references:
                    with Image.open(path) as image:
                        native['images'].append(image.copy())
            output = session.render(strength=args.strength, seed=args.seed, **native)
            output = getattr(output, 'images', output)
            if isinstance(output, list):
                if len(output) != 1:
                    raise ValueError('CLI output requires one image; use session.render for batches')
                output = output[0]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(output, torch.Tensor):
                atomic_save(args.out, {'kind': 'native latent', 'value': output.cpu()})
            else:
                output.save(args.out)


def infer(args, parser):
    """Native-only CLI; deliberately accepts no session/training configuration."""
    from .inference import open_inference
    if args.config is not None or args.resume is not None:
        parser.error('infer accepts no training config/resume; use --artifact and --runtime')
    if args.artifact is None or args.out is None:
        parser.error('infer requires --artifact and --out')
    options = json.loads(args.runtime.read_text()) if args.runtime else {}
    native = json.loads(args.native.read_text()) if args.native else {}
    references = native.get('images')
    output_type = native.pop('output_type', None)
    with open_inference(args.artifact, options=options, operation=args.mode,
                        strength=args.strength, references=bool(references), output_type=output_type, native_request=native) as session:
        if references:
            from PIL import Image
            native['images'] = []
            for path in references:
                with Image.open(path) as image:
                    native['images'].append(image.copy())
        if args.mode == 'render':
            output = session.render(strength=args.strength, seed=args.seed, output_type=output_type, **native)
        else:
            output = session.response(strength=args.strength, seed=args.seed, **native)
        output = getattr(output, 'images', output)
        if isinstance(output, list):
            if len(output) != 1:
                raise ValueError('CLI requires one output; use open_inference for batches')
            output = output[0]
        receipt = dict(selection=session.selection, artifact_sha256=session.manifest['sha256'],
                       origin=session.manifest['origin'], kind='native ' + args.mode)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(output, torch.Tensor):
            atomic_save(args.out, dict(receipt, value=output.detach().cpu()))
        else:
            output.save(args.out)
        atomic_json(str(args.out) + '.json', receipt)


if __name__ == '__main__':
    main()
