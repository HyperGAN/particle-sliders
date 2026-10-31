"""Executable opt-in shared lifecycle over explicitly staged native inputs."""
import argparse
import json
from pathlib import Path

import torch
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from .backend import COHORT, NAME, bundle_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--model-dir', required=True)
    train.add_argument('--rows', required=True)
    train.add_argument('--shape', nargs=4, type=int, required=True, metavar=('B','C','H','W'))
    train.add_argument('--steps', type=int, default=4)
    train.add_argument('--stop-after', type=int)
    train.add_argument('--batch', type=int, default=2)
    train.add_argument('--seed', type=int, default=7)
    train.add_argument('--preset', choices=['bispf','bispf_optimized'], default='bispf_optimized')
    train.add_argument('--ema-decay', type=float, default=.9)
    train.add_argument('--resume')
    train.add_argument('--checkpoint', required=True)
    train.add_argument('--export', required=True)
    train.add_argument('--selection', choices=['raw','ema'], default='ema')
    response = commands.add_parser('response')
    response.add_argument('--model-dir', required=True)
    response.add_argument('--artifact', required=True)
    response.add_argument('--request', required=True, help='weights-only tensor dict: condition, latent, timestep')
    response.add_argument('--strength', type=float, default=1.)
    response.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'train':
        config = SessionConfig(NAME, COHORT, model_dir=str(Path(args.model_dir).resolve()),
            model_identity=bundle_identity(args.model_dir), rows=str(Path(args.rows).resolve()),
            response_shape=args.shape, steps=args.steps, batch=args.batch, seed=args.seed,
            preset=args.preset, ema_decay=args.ema_decay, base_precision='float32')
        with open_response(config, resume=args.resume) as session:
            metrics = session.train(stop_after=args.stop_after)
            session.checkpoint(args.checkpoint)
            session.export_inference(args.export, selection=args.selection)
            print(json.dumps(dict(cohort=COHORT, metrics=metrics, checkpoint=args.checkpoint, export=args.export)))
    else:
        request = torch.load(args.request, map_location='cpu', weights_only=True)
        if not isinstance(request, dict) or set(request) != {'condition','latent','timestep'}:
            raise ValueError('native request requires condition, latent and timestep only')
        with open_inference(args.artifact, options={'model_dir':args.model_dir}, operation='response') as session:
            value = session.response(strength=args.strength, **request)
            torch.save(value, args.output)
            print(json.dumps(dict(shape=list(value.shape), output=args.output, selection=session.selection)))


if __name__ == '__main__':
    main()
