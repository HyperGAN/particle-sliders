"""Immutable runtime identity includes weights, source, scheduler and environment."""
import importlib.metadata
import json
from pathlib import Path

from .contracts import digest, file_hash

ROOT = Path(__file__).resolve().parents[1]


def cache_runtime_compatibility(model_dir, recorded, current, cache_fingerprint):
    """Return evidence hash for explicitly verified reuse of older frozen targets.

    Source changes normally invalidate caches. A narrowly scoped certificate may
    attest that an exact set of existing single-image teacher caches is unchanged
    by a runtime correction. The original cache identities and bytes stay intact.
    """
    if recorded == current:
        return None
    verified = single_image_runtime_compatibility(model_dir, recorded, current)
    proof = json.loads((Path(model_dir) / "runtime-compatibility.json").read_text())
    if cache_fingerprint not in proof.get("cache_fingerprints", []):
        raise ValueError("Runtime compatibility evidence does not cover this cache")
    return verified


def single_image_runtime_compatibility(model_dir, recorded, current):
    """Verify exact base-image equivalence after the recorded batch-mask fix."""
    if recorded == current:
        return None
    path = Path(model_dir) / "runtime-compatibility.json" if model_dir else None
    if path is None or not path.exists():
        raise ValueError("Cached teacher runtime differs; no compatibility verification")
    proof = json.loads(path.read_text())
    unchanged = lambda identity: {k: v for k, v in identity.items() if k != "runtime_sha256"}
    if (unchanged(recorded) != unchanged(current) or proof.get("before") != recorded
            or proof.get("after") != current or not proof.get("passed")
            or proof.get("purpose") != "single-image-teacher-cache-reuse"
            or proof.get("single_image_max_abs") != 0
            or proof.get("positions") != list(range(10))
            or proof.get("batch_sizes") != [1, 2, 4]):
        raise ValueError("Runtime compatibility evidence does not cover this cache")
    return file_hash(path)


def model_identity(model_dir):
    lock = json.loads((Path(model_dir) / "anima-lock.json").read_text())
    # Verify the original config hash at load, but remove installation paths
    # from the portable identity so identical weights can move pod-to-desktop.
    index = json.loads((Path(model_dir) / "modular_model_index.json").read_text())
    def portable(value):
        if isinstance(value, dict):
            return {k: "$MODEL_ROOT" if k == "pretrained_model_name_or_path" else portable(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value
    lock["files"]["modular_model_index.json"] = digest(portable(index))
    return dict(model=lock["model"], sha256=digest(lock), identity_schema="portable-model-v1",
                environment_sha256=file_hash(ROOT / "configs/anima/requirements.lock"),
                runtime_sha256=digest({name: file_hash(ROOT / "lumen_studio" / name)
                    for name in ("runtime.py", "backends/anima.py", "contracts.py", "particles.py", "vendor/reference.py")}))


def verify_environment():
    from packaging.requirements import Requirement
    for line in (ROOT / "configs/anima/requirements.lock").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        dist = importlib.metadata.distribution(requirement.name)
        if requirement.specifier and dist.version not in requirement.specifier:
            raise ValueError(f"Pinned environment mismatch: {requirement.name} {dist.version}")
        if requirement.url and "git+" in requirement.url:
            expected = requirement.url.rsplit("@", 1)[-1]
            source = json.loads(dist.read_text("direct_url.json") or "{}")
            if source.get("vcs_info", {}).get("commit_id") != expected:
                raise ValueError(f"Source revision mismatch: {requirement.name}")
