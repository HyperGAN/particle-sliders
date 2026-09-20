"""Reuse verified compatible neutral paths without changing any positive teacher target."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from lumen_studio.contracts import atomic_json, digest, file_hash
    from lumen_studio.dataset import compile_manifest
    from lumen_studio.provenance import cache_runtime_compatibility, model_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variation", choices=("candlelit", "moonlit", "theatrical"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root.resolve()
    identity = model_identity(root / "model")
    destination = root / "targets/neutral-cache"
    destination.mkdir(exist_ok=True)
    prepared, provenance = [], []
    start = time.monotonic()
    # Validate source provenance and every already-populated destination before
    # publishing any additional paths. No model is allocated by this script.
    for split in ("train", "dev"):
        directory = args.source / split
        index = json.loads((directory / "index.json").read_text())
        assert index["identity"]["split"] == split and index["identity"]["variation"] == args.variation
        assert (index["identity"]["resolution"], index["identity"]["steps"], index["identity"]["cfg"]) == (512, 10, 1)
        assert digest(index["identity"]) == index["input_fingerprint"]
        assert digest(dict(input=index["input_fingerprint"], shards=index["shards"])) == index["fingerprint"]
        certificate = cache_runtime_compatibility(root / "model", index["identity"]["model"], identity,
                                                  index["fingerprint"])
        rows = {r["id"]: r for r in compile_manifest(split)["rows"] if r["variation"] == args.variation}
        provenance.append(dict(split=split, source_cache=index["fingerprint"], compatibility_certificate=certificate))
        for shard in index["shards"]:
            source = directory / shard["path"]
            assert file_hash(source) == shard["sha256"], "Source shard changed"
            data = torch.load(source, map_location="cpu", weights_only=True)
            assert data["fingerprint"] == index["input_fingerprint"]
            row = rows[data["row"]["id"]]
            assert data["row"]["neutral"] == row["neutral"] and data["row"]["shared"] == row["shared"]
            assert data["seed"] in row["seeds"] and data["seed"] == shard["seed"]
            key = digest(dict(model=identity, prompt=row["neutral"], seed=data["seed"], resolution=512, steps=10))
            target = destination / (key + ".pt")
            records = [r for r in data["records"] if r["trajectory"] == "neutral"]
            assert [r["position"] for r in records] == list(range(10))
            if target.exists():
                existing = torch.load(target, map_location="cpu", weights_only=True)
                assert len(existing) == 10
                for a, b in zip(existing, records):
                    assert a["timestep"] == b["timestep"]
                    assert all(torch.equal(a[k], b[k]) for k in ("latent", "neutral")), "Neutral runtime parity failed"
            prepared.append((source, target, target.exists()))
    created = 0
    for source, target, existed in reversed(prepared):
        if target.exists():
            continue
        data = torch.load(source, map_location="cpu", weights_only=True)
        records = [{k: r[k] for k in ("timestep", "latent", "neutral")}
                   for r in data["records"] if r["trajectory"] == "neutral"]
        temporary = target.with_suffix(f".prefill-{os.getpid()}")
        try:
            torch.save(records, temporary)
            try:
                # Publish only if still absent; never overwrite the live preparer.
                os.link(temporary, target)
                created += 1
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    report = dict(passed=True, model=identity, sources=provenance, paths=len(prepared),
                  existing_paths_verified=sum(existed for _, _, existed in prepared),
                  created=created, seconds=time.monotonic() - start,
                  formulation_unchanged=True, positive_teachers_recomputed=True)
    atomic_json(root / "migrations/neutral-reuse.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
