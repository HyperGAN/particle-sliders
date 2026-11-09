import argparse
import json
import os
from pathlib import Path
import sys
import time

from .runner import Run, atomic_json, audit, plan, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded model-owned hyperparameter search; no default GPU allocation')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--allow-native', action='store_true', help='explicitly permit the backend-declared device; native admission still applies')
        if name == 'run':
            command.add_argument('--out', required=True)
            command.add_argument('--media-review', help='optional external media diagnostic; never a ranking or confirmation gate')
    command = commands.add_parser('audit')
    command.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'audit':
            result = audit(args.out)
        elif args.command == 'plan':
            result = plan(read_json(args.config), allow_native=args.allow_native)
        else:
            run = Run(read_json(args.config), args.out, allow_native=args.allow_native)
            with run.lock():
                try:
                    report = run.execute(read_json(args.media_review) if args.media_review else None)
                except BaseException as exc:
                    atomic_json(Path(args.out) / 'interruption.json', dict(error=type(exc).__name__, reason=str(exc),
                        retained='resume the identical command; actual saved steps are in progress.json',
                        no_confirmed_result=True))
                    raise
                # A later successful resume supersedes the failure receipt;
                # retain it under a non-canonical name instead of deleting it.
                interruption = Path(args.out) / 'interruption.json'
                if interruption.exists():
                    os.replace(interruption, interruption.with_name(f'interruption-resolved-{time.time_ns()}.json'))
            result = dict(run=report['run'], results=str(Path(args.out) / 'results.json'), nominations=report['nominations'])
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
