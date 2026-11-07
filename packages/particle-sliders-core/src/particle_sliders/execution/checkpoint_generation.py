"""Publish immutable groups of artifacts before atomically advancing a receipt.

Standard-library only. A failed staging directory remains evidence and is never
adopted as a committed generation. No elapsed training or campaign limit.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

MANIFEST_SCHEMA = 'particle-sliders-artifact-generation-v1'
REFERENCE_SCHEMA = 'particle-sliders-artifact-reference-v1'


def _file_identity(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=digest, bytes=Path(path).stat().st_size)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'missing or invalid artifact receipt: {path}') from error
    if not isinstance(value, dict):
        raise ValueError(f'invalid artifact receipt: {path}')
    return value


@contextmanager
def _publication_lock(root):
    with (root/'.artifact-generation.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'artifact publication is already owned: {root}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'artifact generations cannot contain symlinks: {path}')
        if path.is_file() and path != directory/'generation.json':
            result[str(path.relative_to(directory))] = _file_identity(path)
        elif not path.is_dir() and path != directory/'generation.json':
            raise ValueError(f'artifact is not a regular file: {path}')
    return result


def verify_generation(directory, *, identity=None):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError(f'artifact generation cannot be a symlink: {directory}')
    directory = directory.resolve()
    if directory.name.startswith('.'):
        raise ValueError(f'uncommitted staging generation cannot be adopted: {directory}')
    if (directory/'generation.json').is_symlink():
        raise ValueError(f'generation manifest cannot be a symlink: {directory}')
    manifest = _read_json(directory/'generation.json')
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('generation') != directory.name:
        raise ValueError(f'invalid artifact generation: {directory}')
    if not isinstance(manifest.get('identity'), dict) or not manifest['identity']:
        raise ValueError(f'artifact generation has no identity: {directory}')
    if identity is not None and manifest.get('identity') != json.loads(json.dumps(identity,sort_keys=True,allow_nan=False)):
        raise ValueError(f'artifact generation identity changed: {directory}')
    if not manifest.get('files') or manifest['files'] != _inventory(directory):
        raise ValueError(f'artifact generation is incomplete or changed: {directory}')
    return manifest


def verify_reference(path, *, identity=None):
    """Verify exactly this receipt, anchored at its containing directory.

    Relative generation paths must stay within that root. This does not search
    other roots, infer a latest generation, or adopt an orphan automatically.
    Caller-owned reference metadata is returned unchanged; consumers must check
    any role/path aliases against the verified generation before using them.
    """
    path = Path(path).resolve()
    receipt = _read_json(path)
    if receipt.get('schema') != REFERENCE_SCHEMA:
        raise ValueError(f'not a committed artifact reference: {path}')
    name = receipt.get('generation')
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'invalid generation reference: {path}')
    directory = path.parent/name
    if directory.is_symlink() or not directory.resolve().is_relative_to(path.parent):
        raise ValueError(f'generation reference escaped its artifact root: {path}')
    if (directory/'generation.json').is_symlink():
        raise ValueError(f'generation manifest cannot be a symlink: {directory}')
    try:
        manifest_digest = _file_identity(directory/'generation.json')['sha256']
    except OSError as error:
        raise ValueError(f'missing or invalid generation manifest: {path}') from error
    if receipt.get('manifest_sha256') != manifest_digest:
        raise ValueError(f'artifact generation receipt changed: {path}')
    manifest = verify_generation(directory, identity=identity)
    return directory, manifest, receipt


def publish_generation(root, name, *, identity, required, build, reference_fields):
    """Commit one new generation and its latest.json reference.

    build(directory) writes exactly required relative filenames. reference_fields
    receives the final directory and supplies caller-owned receipt metadata.
    Staging, committed orphan generations and failed pointer temporaries remain
    on failure; an existing generation is never overwritten or silently reused.
    """
    root = Path(root).resolve()
    if not name or name.startswith('.') or Path(name).name != name:
        raise ValueError('generation name must be one non-hidden path component')
    required = set(required)
    if not isinstance(identity, dict) or not identity:
        raise ValueError('artifact generation identity is required')
    identity = json.loads(json.dumps(identity,sort_keys=True,allow_nan=False))
    if not required or any(Path(item).is_absolute() or '..' in Path(item).parts or
                           item == 'generation.json' for item in required):
        raise ValueError('required artifacts must be relative files inside the generation')
    root.mkdir(parents=True, exist_ok=True)
    generations = root/'generations'
    generations.mkdir(exist_ok=True)
    if generations.is_symlink():
        raise ValueError(f'generation root cannot be a symlink: {generations}')
    final = generations/name
    with _publication_lock(root):
        if final.exists() or final.is_symlink():
            raise FileExistsError(f'artifact generation is immutable: {final}')
        staging = Path(tempfile.mkdtemp(dir=generations, prefix='.'+name+'-', suffix='.partial'))
        build(staging)
        files = _inventory(staging)
        if set(files) != required:
            raise ValueError(f'publication requires exactly {sorted(required)}; got {sorted(files)}')
        for relative in files:
            with (staging/relative).open('rb') as stream:
                os.fsync(stream.fileno())
        manifest = dict(schema=MANIFEST_SCHEMA, generation=name, identity=identity, files=files)
        _write_json(staging/'generation.json', manifest)
        for directory in sorted((p for p in staging.rglob('*') if p.is_dir()), reverse=True):
            _sync_directory(directory)
        _sync_directory(staging)
        os.rename(staging, final)
        _sync_directory(generations)
        receipt = dict(schema=REFERENCE_SCHEMA, generation=str(final.relative_to(root)),
                       manifest_sha256=_file_identity(final/'generation.json')['sha256'])
        fields = json.loads(json.dumps(reference_fields(final),sort_keys=True,allow_nan=False))
        if not isinstance(fields, dict):
            raise ValueError('caller reference metadata must be a JSON object')
        if set(fields) & set(receipt):
            raise ValueError('caller metadata cannot replace artifact reference identity')
        receipt.update(fields)
        temporary = root/('.latest-'+uuid.uuid4().hex+'.partial')
        _write_json(temporary, receipt)
        os.replace(temporary, root/'latest.json')
        _sync_directory(root)
        return receipt
