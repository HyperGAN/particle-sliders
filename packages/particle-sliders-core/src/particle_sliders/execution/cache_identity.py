"""Content-bound tensor caches; callers own model/runtime identity and execution."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = 'particle-sliders-tensor-cache-v1'


def fingerprint_file(path):
    """Hash actual bytes, refusing an input that changes while being read."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError('cache identity input changed while hashing: ' + str(path))
    return dict(size=after.st_size, sha256=digest.hexdigest())


def _document(identity, request):
    if not isinstance(identity, dict) or not identity:
        raise ValueError('an explicit nonempty cache identity is required')
    # Snapshot JSON values, reject nonfinite settings, and preserve exact floats.
    return json.loads(json.dumps(dict(schema=SCHEMA, identity=identity, request=request),
                                 sort_keys=True, allow_nan=False))


def cache_key(identity, request):
    encoded = json.dumps(_document(identity, request), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def cache_path(directory, identity, request):
    return Path(directory) / ('identity-v1-' + cache_key(identity, request) + '.pt')


def _tensor(value):
    import torch
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or not value.numel() or not torch.isfinite(value).all()):
        raise ValueError('cached tensor must be nonempty, floating and finite')
    return value.detach().cpu().contiguous()


def load_tensor(path, identity, request):
    import torch
    expected = _document(identity, request)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != set(expected) | {'tensor'}
            or any(payload[key] != value for key, value in expected.items())):
        raise ValueError('cache schema or identity differs; preserve the old artifact and regenerate')
    return _tensor(payload['tensor'])


def save_tensor(path, tensor, identity, request):
    """Publish a complete cache exclusively; never overwrite an existing file."""
    import torch
    path = Path(path)
    document = _document(identity, request)
    tensor = _tensor(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return load_tensor(path, identity, request)
    descriptor, temporary = tempfile.mkstemp(prefix='.tensor-cache-', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(document, tensor=tensor), temporary)
        try:
            os.link(temporary, path)  # Atomic publication without replacing a competing writer.
        except FileExistsError:
            return load_tensor(path, identity, request)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return tensor
