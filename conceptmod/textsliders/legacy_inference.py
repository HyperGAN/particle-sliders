"""Product-side paths and provenance for reporting archived native adapters.

No training, model loader, optimizer or export is part of this surface.
"""
import hashlib
from contextlib import contextmanager
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4


def source_digest(source):
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def resolve_export(path, *, label):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != '.safetensors':
            raise ValueError(f'{label} must be an explicit .safetensors export')
        return path
    if path.is_dir():
        candidates = sorted(path.glob('*.safetensors'))
    else:
        candidates = [p for p in (Path(str(path) + '_last.safetensors'),
                                  Path(str(path) + '.safetensors')) if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f'{label} requires one unambiguous export; select an explicit file '
                         f'(found {len(candidates)} under {path})')
    return candidates[0].resolve()


def report_destination(save_dir, name, source):
    source = Path(source).resolve()
    directory = Path(save_dir).resolve()
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('inference report name must be a simple filename')
    receipt = directory / f'{name}_inference.json'
    if receipt in (source, source.with_suffix('.json')):
        raise ValueError('inference receipt would overwrite source metadata')
    if receipt.exists() or (directory / 'report').exists():
        raise FileExistsError('inference report destination already exists; choose a fresh name/directory')
    return directory, receipt


def inference_receipt(args, source, metadata, *, family, report=None, expected_sha256=None):
    source = Path(source).resolve()
    actual = source_digest(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('source export changed during loading')
    return dict(operation='inference_only', backend=family, name=args.name,
                steps=0, history=[], training_performed=False, optimizer_constructed=False,
                source_export=str(source), source_sha256=actual,
                source_metadata=metadata, source_weights_kind=metadata.get('weights_kind', 'unspecified'),
                source_completed_steps=metadata.get('completed_steps', metadata.get('step', metadata.get('steps'))),
                package_version=version('particle-sliders-core'),
                quality_qualified=False, dummy=bool(args.dummy), report=report)


def save_receipt(path, receipt):
    payload = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a concurrent report from being replaced.
    with path.open('x') as handle:
        handle.write(payload)


def _finish_receipt(path, receipt):
    payload = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    current = json.loads(path.read_text())
    if current.get('operation_id') != receipt['operation_id']:
        raise ValueError('inference receipt ownership changed')
    pending = path.with_name(path.name + '.' + receipt['operation_id'] + '.pending')
    with pending.open('x') as handle:
        handle.write(payload)
    pending.replace(path)


@contextmanager
def reporting_operation(path, receipt):
    """Reserve receipt ownership before any report output, then record outcome."""
    path = Path(path)
    receipt.update(status='running', operation_id=uuid4().hex)
    save_receipt(path, receipt)
    try:
        yield receipt
        if source_digest(receipt['source_export']) != receipt['source_sha256']:
            raise ValueError('source export changed during reporting')
    except BaseException as exc:
        receipt.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        _finish_receipt(path, receipt)
        raise
    else:
        receipt['status'] = 'completed'
        _finish_receipt(path, receipt)
