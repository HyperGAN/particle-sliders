"""Compare immutable EMA snapshots on fixed training rows without updating weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variation", choices=("candlelit", "moonlit", "theatrical"))
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 400, 800])
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from lumen_studio import execution
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import atomic_json, digest, file_hash
    from lumen_studio.coordinator import GpuLease
    from lumen_studio.metrics import make_fixtures, residual_breakdown, swd
    from lumen_studio.particles import ParticleAdapter
    from lumen_studio.provenance import cache_runtime_compatibility
    from lumen_studio.runtime import create_runtime
    from lumen_studio.store import Store

    root = args.root.resolve()
    run_dir = root / "runs" / args.variation
    run = json.loads((run_dir / "run.json").read_text())
    cache = TargetCache(root / "targets" / args.variation / "train")
    if cache.index["fingerprint"] != run["cache"] or cache.index["identity"]["split"] != "train":
        raise ValueError("Training diagnostic cache identity differs from run")
    normalization = torch.load(run_dir / "normalization.pt", map_location="cpu", weights_only=True)
    if normalization["provenance"]["cache_fingerprint"] != cache.index["fingerprint"]:
        raise ValueError("Training diagnostic normalization differs from cache")
    # First fixed seed for every training row, both paths and all ten positions.
    indices = [i for i, (shard, _) in enumerate(cache.offsets) if shard["seed"] == 1000]
    records = [cache[i] for i in indices]
    if len(records) != 480 or len({r["row"]["id"] for r in records}) != 24:
        raise ValueError("Expected all 24 training rows, both paths, all ten positions")
    scale = torch.stack([normalization["scale"][r["position"]] for r in records])
    teachers = torch.stack([(r["positive"] - r["neutral"]).flatten() for r in records])
    directory = root / "diagnostics/training-generalization" / args.variation
    directory.mkdir(parents=True, exist_ok=True)
    fixture_path = directory / "fixtures.pt"
    identity = dict(cache=cache.index["fingerprint"], indices=indices,
                    normalization_sha256=file_hash(run_dir / "normalization.pt"))
    if fixture_path.exists():
        fixtures = torch.load(fixture_path, map_location="cpu", weights_only=True)
        if fixtures["identity"] != identity:
            raise ValueError("Fixed training diagnostic fixtures changed")
    else:
        fixtures = dict(identity=identity, **make_fixtures(scale.shape[-1], len(indices), seed=9013))
        save_tensor_file(fixture_path, fixtures)

    def metrics(student):
        import numpy as np
        student, teacher = student.float(), teachers.float()
        error = (student - teacher) / scale
        s, t = student / scale, teacher / scale
        gain = (s * t).sum(-1) / t.square().sum(-1).clamp_min(1e-12)
        orthogonal = s - gain[:, None] * t
        # torch.quantile rejects more than 2**24 elements. This full training
        # fixture has 480 velocity fields; NumPy uses the same linear quantile.
        p95 = float(np.quantile(error.abs().numpy(), .95, method="linear"))
        return dict(residual_rms=float(error.square().mean().sqrt()), residual_p95=p95,
            edit_cosine=float(torch.nn.functional.cosine_similarity(s, t, dim=-1).mean()),
            edit_gain=float(gain.mean()), orthogonal_rms=float(orthogonal.square().mean().sqrt()),
            teacher_swd_256=swd(s, t, fixtures["projections"]),
            game_swd={str(level): swd(fixtures["noise"] * level + error,
                fixtures["noise"] * level, fixtures["projections"]) for level in (.5, 1., 2.)},
            breakdown=residual_breakdown(error, records))

    report = dict(schema=1, purpose="training-fit diagnostic; not development selection",
                  run=run, fixture_sha256=file_hash(fixture_path), identity=identity,
                  baseline=metrics(torch.zeros_like(teachers)), checkpoints=[])
    checkpoint_paths = [run_dir / f"ema-{step:06}.safetensors" for step in args.steps]
    if any(not path.exists() for path in checkpoint_paths):
        raise ValueError("Every requested immutable checkpoint must exist")
    key = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    store = Store(root / "studio/studio.sqlite3")
    with GpuLease(Path("/tmp"), key):
        runtime = None
        try:
            store.control(owner=dict(pid=os.getpid(), gpu=args.gpu, state="fixed training-fit diagnostic"))
            runtime = create_runtime(root / "model", "cuda:0", checkpointing=False)
            if runtime.identity != run["model"]:
                raise ValueError("Diagnostic runtime differs from training runtime")
            cache_runtime_compatibility(root / "model", cache.index["identity"]["model"],
                                        runtime.identity, cache.index["fingerprint"])
            adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
            runtime.mixer.add(args.variation, adapter)
            for step, path in zip(args.steps, checkpoint_paths, strict=True):
                start = time.monotonic()
                metadata = adapter.load_export(path, model_identity=runtime.identity)
                if (metadata["step"] != step or metadata["weights"] != "ema"
                        or metadata["normalization"] != normalization["provenance"]
                        or metadata["manifest_sha256"] != cache.index["identity"]["manifest_sha256"]):
                    raise ValueError("Diagnostic checkpoint provenance differs")
                students = []
                with torch.no_grad(), runtime.mixer.scales({args.variation: 1.}):
                    for record in records:
                        prediction = runtime.predict(record["latent"].to(runtime.device),
                            record["timestep"], record["embedding"])
                        students.append((prediction.cpu() - record["neutral"]).flatten())
                result = dict(step=step, checkpoint_sha256=file_hash(path),
                              **metrics(torch.stack(students)), seconds=time.monotonic() - start)
                dev_path = run_dir / f"probe-{step:06}.json"
                if dev_path.exists():
                    result["development_residual_rms"] = json.loads(dev_path.read_text())["residual_rms"]
                report["checkpoints"].append(result)
                report["sha256"] = digest({k: v for k, v in report.items() if k != "sha256"})
                atomic_json(directory / "report.json", report)
                print(json.dumps({k: v for k, v in result.items() if k != "breakdown"}), flush=True)
        finally:
            if runtime is not None:
                runtime.close()
            store.control(owner=None)


if __name__ == "__main__":
    main()
