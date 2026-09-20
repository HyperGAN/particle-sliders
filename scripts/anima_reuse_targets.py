"""Reuse only byte-verified frozen targets whose complete paired row is unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    import os
    import torch
    from lumen_studio.contracts import Definition, VARIATIONS, atomic_json, digest, file_hash
    from lumen_studio.dataset import compile_manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--variation", choices=VARIATIONS, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--publish", action="store_true", help="Without this flag, validate and report only")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination caches must be separate")
    bundle = json.loads(args.catalog.read_text())
    if digest({k: v for k, v in bundle.items() if k != "sha256"}) != bundle["sha256"]:
        raise ValueError("Candidate catalog hash changed")
    definitions = [Definition(**d) for d in bundle["definitions"]]
    manifest = compile_manifest(args.split, definitions=definitions, characters=bundle["characters"])
    if manifest != bundle[args.split]:
        raise ValueError("Expanded candidate prompts do not match the shared-field compiler")
    index = json.loads((source / "index.json").read_text())
    original_identity = index["identity"]
    source_input = digest(original_identity)
    if (source_input != index["input_fingerprint"]
            or digest(dict(input=source_input, shards=index["shards"])) != index["fingerprint"]):
        raise ValueError("Source index identity changed")
    expected = dict(variation=args.variation, split=args.split, resolution=512, steps=10, cfg=1,
                    layout="both-trajectories/all-positions/full-velocity/shared-state-v1")
    if any(original_identity.get(k) != value for k, value in expected.items()):
        raise ValueError("Source rendering settings are incompatible")
    qualification = None
    if args.publish:
        if args.qualification is None:
            raise ValueError("Publishing targets requires the completed reference qualification")
        qualification = json.loads(args.qualification.read_text())
        if not (qualification.get("passed") is True
                and qualification.get("stage") == "definition_reference_screen"
                and qualification.get("candidate") == bundle["sha256"]
                and qualification.get("manifest_sha256") == bundle["train"]["sha256"]
                and qualification.get("model_identity") == original_identity["model"]):
            raise ValueError("Qualification does not match these prompts and the frozen model")
    identity = dict(original_identity, manifest_sha256=manifest["sha256"])
    fingerprint = digest(identity)
    rows = {r["id"]: r for r in manifest["rows"] if r["variation"] == args.variation}
    if args.publish:
        destination.mkdir(parents=True, exist_ok=True)
        identity_path = destination / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Destination identity changed")
        atomic_json(identity_path, identity)
    started, compatible, changed, published, existing = time.monotonic(), [], [], [], []
    for shard in index["shards"]:
        name = shard["path"]
        if Path(name).name != name or file_hash(source / name) != shard["sha256"]:
            raise ValueError("Source shard bytes differ from their immutable index")
        data = torch.load(source / name, map_location="cpu", weights_only=True)
        if (data["fingerprint"] != source_input or data["row"]["id"] != shard["row"]
                or data["seed"] != shard["seed"] or shard["count"] != 20
                or [(r["trajectory"], r["position"]) for r in data["records"]]
                   != [(path, t) for path in ("neutral", "positive") for t in range(10)]):
            raise ValueError("Invalid source trajectory layout or row")
        row = rows.get(shard["row"])
        if row != data["row"] or shard["seed"] not in row["seeds"]:
            changed.append(name)
            continue
        compatible.append(name)
        if not args.publish:
            continue
        result = dict(data, fingerprint=fingerprint)
        path = destination / name
        if path.exists():
            if not equal(result, torch.load(path, map_location="cpu", weights_only=True)):
                raise ValueError("Existing destination tensors do not match verified reuse")
            existing.append(name)
            continue
        temp = destination / (name + f".{os.getpid()}.tmp")
        try:
            torch.save(result, temp)
            if not equal(result, torch.load(temp, map_location="cpu", weights_only=True)):
                raise ValueError("Reused full fields or embeddings changed during serialization")
            # Never overwrite a shard committed concurrently by the preparer.
            try:
                os.link(temp, path)
            except FileExistsError:
                if not equal(result, torch.load(path, map_location="cpu", weights_only=True)):
                    raise ValueError("Concurrent target differs from the verified frozen fields")
                existing.append(name)
            else:
                published.append(name)
        finally:
            temp.unlink(missing_ok=True)
    report = dict(source_cache=index["fingerprint"], source_manifest=original_identity["manifest_sha256"],
        destination_manifest=manifest["sha256"], candidate=bundle["sha256"], model=original_identity["model"],
        qualification_sha256=file_hash(args.qualification) if qualification else None,
        compatible=compatible, changed=changed, published=published, existing=existing,
        full_fields_unchanged=True, seconds=time.monotonic() - started)
    if args.publish:
        atomic_json(destination / "reuse.json", report)
    print(json.dumps({k: len(report[k]) for k in ("compatible", "changed", "published", "existing")}
                     | dict(seconds=report["seconds"], dry_run=not args.publish)))


if __name__ == "__main__":
    main()
