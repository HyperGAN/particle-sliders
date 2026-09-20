"""Verify the singleton-mask correction before reusing frozen teacher caches."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    import hashlib
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.cache import TargetCache
    from lumen_studio.contracts import atomic_json, digest, file_hash
    root = ROOT / "artifacts/anima"
    evidence = root / "runtime-fixes/padding-batch-v1"
    before_source = (evidence / "anima-before.py").read_text()
    after_source = (ROOT / "lumen_studio/backends/anima.py").read_text()
    old = "padding_mask=torch.zeros(latent.shape[0], 1, latent.shape[-2] * self.scale,"
    new = "padding_mask=torch.zeros(1, 1, latent.shape[-2] * self.scale,"
    assert before_source.count(old) == 1 and before_source.replace(old, new) == after_source
    source_files = ("runtime.py", "backends/anima.py", "contracts.py", "particles.py", "vendor/reference.py")
    old_hashes = {name: file_hash(ROOT / "lumen_studio" / name) for name in source_files}
    old_hashes["backends/anima.py"] = hashlib.sha256(before_source.encode()).hexdigest()
    original_hash = digest(old_hashes)
    runtime = TurboRuntime(root / "model", "cuda:0")
    report = dict(passed=False, purpose="single-image-teacher-cache-reuse", after=runtime.identity,
        source_change="singleton zero padding mask; Cosmos repeats across batch",
        old_source_sha256=old_hashes["backends/anima.py"],
        new_source_sha256=file_hash(ROOT / "lumen_studio/backends/anima.py"),
        cache_fingerprints=[], single_image_max_abs=0., positions=list(range(10)), batch_sizes=[1, 2, 4],
        checks=[], batch_atol=.02, batch_rtol=.02)
    try:
        for index_path in sorted((root / "targets").glob("*/*/index.json")):
            cache = TargetCache(index_path.parent)
            before = cache.index["identity"]["model"]
            assert before["runtime_sha256"] == original_hash
            assert {k: v for k, v in before.items() if k != "runtime_sha256"} == {
                k: v for k, v in runtime.identity.items() if k != "runtime_sha256"}
            report["before"] = before
            report["cache_fingerprints"].append(cache.index["fingerprint"])
            shard = torch.load(index_path.parent / cache.index["shards"][0]["path"], map_location="cpu", weights_only=True)
            records = shard["records"]
            check = dict(variation=cache.index["identity"]["variation"], split=cache.index["identity"]["split"],
                         single_image_errors=[], batch_errors={})
            with torch.no_grad():
                for record in records:
                    for side, key in (("neutral", "embedding"), ("positive", "positive_embedding")):
                        value = runtime.predict(record["latent"].to(runtime.device), record["timestep"], shard[key])
                        error = float((value.cpu() - record[side]).abs().max())
                        check["single_image_errors"].append(error)
                        assert error == 0, (index_path, side, record["position"], error)
                for size in (2, 4):
                    selected = records[:size]
                    latent = torch.cat([r["latent"] for r in selected]).to(runtime.device)
                    ts = torch.tensor([r["timestep"] for r in selected], device=runtime.device)
                    embed = shard["embedding"].expand(size, -1, -1).to(runtime.device)
                    value = runtime.predict(latent, ts, embed).cpu()
                    expected = torch.cat([r["neutral"] for r in selected])
                    check["batch_errors"][str(size)] = dict(max_abs=float((value - expected).abs().max()),
                        rms=float((value - expected).square().mean().sqrt()),
                        parity_passed=bool(torch.allclose(value, expected, atol=.02, rtol=.02)))
            report["checks"].append(check)
            atomic_json(evidence / "verification.json", report)
            print(json.dumps(check), flush=True)
        assert len(report["cache_fingerprints"]) == 6
        report["passed"] = True
        atomic_json(evidence / "verification.json", report)
        atomic_json(root / "model/runtime-compatibility.json", report)
        print("All single-image teachers unchanged. Batch shapes work; numerical parity is reported separately.", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
