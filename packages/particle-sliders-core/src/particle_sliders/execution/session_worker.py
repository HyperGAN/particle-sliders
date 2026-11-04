"""Standalone worker: import the selected environment's installed core, never replace it.

Only this process imports Torch/native backends. The update remains session.train.
"""
import argparse
import hashlib
import json
import inspect
import os
from pathlib import Path


def write(path, value):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interface', choices=('image', 'response'), default='image')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=32)
    parser.add_argument('--selection', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--export-format', choices=('v1', 'v2', 'both'), default='v1')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.checkpoint_every < 1:
        parser.error('checkpoint interval must be positive')
    from particle_sliders.lifecycle import SessionConfig, backend, open_session
    if args.interface == 'response':
        from particle_sliders.response import response_backend as backend, open_response as open_session
    config = SessionConfig.from_dict(json.loads(args.config.read_text()))
    contract = backend(config.backend)
    if args.check:
        contract.describe(config).check('train', precision=config.options.get('base_precision'))
        contract.validate(config.options)
        print(json.dumps(dict(status='checked', backend=config.backend)))
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('worker output must be new/empty')
    resume = args.resume
    if resume and resume.is_dir():
        resume = resume / 'training.pt'
    with open_session(config, resume=resume, operation='train') as session:
        if args.export_format in ('v2', 'both'):
            from particle_sliders.inference import InferenceSpec
            if not isinstance(session.binding.inference, InferenceSpec):
                raise ValueError('v2 export requires native InferenceSpec before training')
        if args.selection == 'ema' and session.ema.decay is None:
            raise ValueError('EMA selection requires configured EMA')
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.stop_after or session.game.recipe.total_steps
        final_checkpoint = None
        def checkpoint():
            nonlocal final_checkpoint
            directory = args.output / 'checkpoints' / f'step-{session.game.completed_steps:06d}'
            directory.mkdir(parents=True, exist_ok=True)
            session.checkpoint(directory / 'training.pt')
            write(directory / 'checkpoint.json', dict(completed_steps=session.game.completed_steps,
                  sha256=hashlib.sha256((directory / 'training.pt').read_bytes()).hexdigest()))
            final_checkpoint = directory
            write(args.output / 'latest.json', dict(checkpoint=str(directory), completed_steps=session.game.completed_steps))
        def on_step(metrics):
            step = session.game.completed_steps
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + '\n')
                stream.flush()
            if step % args.checkpoint_every == 0 or step == target:
                checkpoint()
            write(args.output / 'progress.json', dict(status='training', completed_steps=step, target_steps=target,
                  checkpoint=str(final_checkpoint) if final_checkpoint else None))
            print(f'step {step}/{target}', flush=True)
        if session.game.completed_steps < target:
            if 'on_step' in inspect.signature(session.train).parameters:
                session.train(stop_after=target, on_step=on_step)
            else:
                # Older pinned product cores own the same update math, without
                # the callback convenience. Advance through their public train
                # API; never import/rebind a newer core into that environment.
                while session.game.completed_steps < target:
                    metrics = session.train(stop_after=session.game.completed_steps + 1)
                    if not isinstance(metrics, dict):
                        metrics = getattr(session, 'last_metrics', None)
                    if not isinstance(metrics, dict):
                        metrics = {'step': session.game.completed_steps}
                    on_step(metrics)
        else:
            if session.game.completed_steps != target:
                raise ValueError('resume is past requested target')
            checkpoint()
        paths = []
        if args.export_format in ('v1', 'both'):
            session.export(args.output / 'adapter.pt', selection=args.selection)
            paths.append(args.output / 'adapter.pt')
        if args.export_format in ('v2', 'both'):
            session.export_inference(args.output / 'adapter-v2.pt', selection=args.selection)
            paths.append(args.output / 'adapter-v2.pt')
        write(args.output / 'manifest.json', dict(manifest=session.manifest, completed_steps=target,
              files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
        write(args.output / 'result.json', dict(status='completed', completed_steps=target,
              checkpoint=str(final_checkpoint), adapter=str(paths[0]), rendered_images=0, lr_search=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
