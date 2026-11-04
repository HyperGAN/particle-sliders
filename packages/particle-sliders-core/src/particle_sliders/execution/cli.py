"""One training command with explicit product profiles and isolated executors."""
import argparse
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import sys

from .request import REQUEST_SCHEMA, read_json, resolve_request


def should_dispatch(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return False
    if argv[0] in ('--help', '-h', 'plan', 'profiles', 'status', 'logs', 'cancel', 'fetch'):
        return True
    if argv[0] != 'train':
        return False
    if any(v in argv for v in ('--help', '-h')):
        return True
    if any(v in argv for v in ('--profile', '--runner', '--check', '--checkpoint-every', '--device', '--runtime-python', '--reporting-config', '--no-reporting')):
        return True
    try:
        i = argv.index('--config')
        path = Path(argv[i + 1])
        config = read_json(path)
        return (config.get('schema') == REQUEST_SCHEMA or
                set(config) != {'backend', 'cohort', 'options'} and
                (path.resolve().parent / 'training_profile.json').is_file())
    except (ValueError, IndexError, OSError, AttributeError):
        return False


def runner_named(name):
    matches = list(entry_points(group='particle_sliders.runners', name=name))
    if len(matches) != 1:
        raise ValueError(f'runner {name!r} is not installed; install its runner plugin')
    value = matches[0].load()
    if not callable(getattr(value, 'submit', None)):
        raise TypeError('runner must expose submit(request)')
    return value



def _reporting_file(path):
    path = Path(path).expanduser().resolve()
    reporting = read_json(path)
    if isinstance(reporting, dict):
        for key in ('runtime_path', 'runtime_python'):
            if isinstance(reporting.get(key), str) and (key == 'runtime_path' or '/' in reporting[key]):
                reporting[key] = (str((path.parent / reporting[key]).resolve()) if key == 'runtime_path'
                                  else os.path.abspath(path.parent / os.path.expanduser(reporting[key])))
    return reporting


def default_reporting_config():
    explicit = os.environ.get('PARTICLE_SLIDER_REPORTING_CONFIG')
    if explicit:
        return _reporting_file(explicit)
    base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    path = base / 'particle-slider/reporting.json'
    return _reporting_file(path) if path.is_file() else None

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for op in ('train', 'plan'):
        command = commands.add_parser(op)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--profile', help='product training_profile.json path or installed profile name')
        command.add_argument('--runner', help='local or installed executor such as modal')
        command.add_argument('--out', type=Path)
        command.add_argument('--resume')
        command.add_argument('--checkpoint-every', type=int)
        command.add_argument('--device')
        command.add_argument('--runtime-python', help='existing isolated product interpreter; does not install or repin dependencies')
        reporting = command.add_mutually_exclusive_group()
        reporting.add_argument('--reporting-config', type=Path, help='explicit pinned HyperGAN develop reporting runtime JSON')
        reporting.add_argument('--no-reporting', action='store_true', help='disable configured reporting for this run')
        command.add_argument('--selection', choices=('raw', 'ema'))
        command.add_argument('--export-format', choices=('v1', 'v2', 'both'))
        command.add_argument('--stop-after', type=int)
        command.add_argument('--check', action='store_true', help='run product validation without training')
    commands.add_parser('profiles')
    for op in ('status', 'logs', 'cancel', 'fetch'):
        command = commands.add_parser(op)
        command.add_argument('--job', type=Path, required=True, help='JSON handle returned by remote submission')
        if op == 'fetch':
            command.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == 'profiles':
            result = {'installed': [{'name': p.name, 'value': p.value} for p in entry_points(group='particle_sliders.training_profiles')],
                      'file_profiles': 'pass --profile path/to/training_profile.json; products remain isolated'}
        elif args.operation in ('status', 'logs', 'cancel', 'fetch'):
            handle = read_json(args.job)
            runner = runner_named(handle.get('runner', 'modal'))
            action = getattr(runner, args.operation)
            if args.operation == 'fetch':
                result = action(handle, args.out)
            elif args.operation == 'logs':
                for item in action(handle):
                    print(json.dumps(item, allow_nan=False), flush=True)
                return 0
            else:
                result = action(handle)
        else:
            config_path = args.config.resolve()
            reporting = _reporting_file(args.reporting_config) if args.reporting_config else None
            default_reporting = None if args.no_reporting or args.reporting_config else default_reporting_config
            request = resolve_request(read_json(config_path), base=config_path.parent,
                profile=args.profile, runner=args.runner, output=args.out, resume=args.resume,
                checkpoint_every=args.checkpoint_every, device=args.device, selection=args.selection,
                export_format=args.export_format, stop_after=args.stop_after, runtime_python=args.runtime_python, reporting=reporting, no_reporting=args.no_reporting, default_reporting=default_reporting)
            if args.operation == 'plan':
                result = request
            elif args.check:
                if request['execution']['runner'] != 'local':
                    raise ValueError('--check is local-only; use plan for a remote request')
                from .local import check_request
                result = check_request(request)
            elif request['execution']['runner'] == 'local':
                from .local import execute_resolved_request
                result = execute_resolved_request(request)
            else:
                result = runner_named(request['execution']['runner']).submit(request)
                # Keep remote submission distinct from training/completion.
                if result.get('status') not in ('submitted', 'queued'):
                    raise ValueError('runner submit must return submitted/queued status')
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
