"""Integrity receipts for historical distillation artifacts; standard library only.

These receipts identify the historical protocol. They do not qualify its metric
as the current RAW FP64 equal-example relative-SSE protocol.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "historical-distillation-adoption-v1"


def snapshot(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def require_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"verified SHA-256 identity required for {label}")


def file_identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": sha, "bytes": path.stat().st_size}


def source_identity(paths):
    return {str(name): file_identity(path) for name, path in sorted(paths.items())}


@contextmanager
def attempt_lock(directory):
    """Exclude simultaneous writers without imposing any elapsed-time budget."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".distillation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"distillation attempt is already owned: {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_new_json(path, value):
    """Publish once; an interrupted or stale artifact is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable distillation evidence, retained at {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid distillation evidence, retained at {path}")
    return value


def bind_identity(path, identity, *, evidence=()):
    path = Path(path)
    if path.exists():
        saved = read_json(path)
        if saved != {"schema": SCHEMA, "identity": identity}:
            raise ValueError(f"distillation identity changed; retain evidence and use a new output directory: {path}")
    else:
        if any(Path(item).exists() for item in evidence):
            raise ValueError(f"distillation evidence has no identity; retain it and use a new output directory: {path}")
        write_new_json(path, {"schema": SCHEMA, "identity": identity})


def seal(payload, identity):
    if "adoption" in payload:
        raise ValueError("reserved adoption field")
    return dict(payload, adoption={"schema": SCHEMA, "identity": identity,
                                   "payload_sha256": digest(payload)})


def verify_record(path, identity):
    saved = read_json(path)
    receipt = saved.pop("adoption", None)
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError(f"distillation evidence has no adoption identity; retained at {path}")
    if receipt.get("identity") != identity:
        raise ValueError(f"distillation identity changed; retained at {path}")
    if receipt.get("payload_sha256") != digest(saved):
        raise ValueError(f"distillation evidence changed; retained at {path}")
    return saved


def _artifact_names(directory, paths):
    directory = Path(directory).resolve()
    result = {}
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"distillation artifact must be a regular file inside {directory}: {path}")
        name = str(path.resolve().relative_to(directory))
        result[name] = file_identity(path)
    return result


def publish_completion(path, identity, payload, outputs):
    path = Path(path)
    payload = dict(payload, artifacts=_artifact_names(path.parent, outputs))
    write_new_json(path, seal(payload, identity))


def verify_completion(path, identity, outputs):
    path = Path(path)
    saved = verify_record(path, identity)
    if saved.get("artifacts") != _artifact_names(path.parent, outputs):
        raise ValueError(f"distillation output missing or changed; retained at {path}")
    return saved
