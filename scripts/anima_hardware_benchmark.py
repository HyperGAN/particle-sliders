"""Time the qualified execution mode on a new host and compare its full state."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from lumen_studio import execution
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.cache import TargetCache
    from lumen_studio.contracts import atomic_json, file_hash
    from lumen_studio.coordinator import GpuLease
    from lumen_studio.training import Trainer

    root, reference = args.root.resolve(), args.reference.resolve()
    directory = root / "benchmarks/migration-hardware"
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(reference / "normalization.pt", directory / "normalization.pt")
    expected_identity = json.loads((reference / "run.json").read_text())
    key = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid",
                                   "--format=csv,noheader"], text=True).strip()
    runtime = trainer = None
    started = time.monotonic()
    report = dict(status="running", warmup_updates=20, timed_updates=50, gpu_uuid=key, completed=0)
    with GpuLease(Path("/tmp"), key):
        try:
            runtime = TurboRuntime(root / "model", "cuda:0",
                                   checkpointing=expected_identity["execution"]["checkpointing"])
            cache = TargetCache(root / "targets/candlelit/train", pin_memory=True)
            trainer = Trainer(runtime, cache, directory, "candlelit", seed=expected_identity["seed"],
                              microbatch=expected_identity["execution"]["microbatch"])
            if trainer.identity != expected_identity:
                raise ValueError("Hardware benchmark changed the qualified training configuration")
            elapsed = []
            reference_updates = [json.loads(line) for line in
                                 (reference / "updates.jsonl").read_text().splitlines()]
            for i in range(70):
                if i == 20:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.monotonic()
                update = trainer.update()
                torch.cuda.synchronize()
                expected_update = reference_updates[i]
                mismatches = {key: dict(expected=value, actual=update.get(key))
                              for key, value in expected_update.items()
                              if key != "seconds" and update.get(key) != value}
                if mismatches:
                    report.update(completed=i + 1, update_differences=mismatches)
                    raise ValueError(f"Hardware update parity failed at update {i + 1}")
                if i >= 20:
                    elapsed.append(time.monotonic() - start)
                report.update(completed=i + 1, updated_at=time.time())
                atomic_json(directory / "report.json", report)
            trainer.save()
            expected = torch.load(reference / "state-000070.pt", map_location="cpu", weights_only=True)
            actual = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True)
            differences, count = [], 0

            def compare(a, b, path="state"):
                nonlocal count
                if isinstance(a, torch.Tensor):
                    count += 1
                    if not torch.equal(a, b):
                        differences.append(path)
                elif isinstance(a, dict):
                    if a.keys() != b.keys():
                        differences.append(path + ".keys")
                    else:
                        for k in a:
                            compare(a[k], b[k], path + "." + str(k))
                elif isinstance(a, (list, tuple)):
                    if len(a) != len(b):
                        differences.append(path + ".length")
                    else:
                        for i, (x, y) in enumerate(zip(a, b)):
                            compare(x, y, path + "." + str(i))
                elif a != b:
                    differences.append(path)

            compare(expected, actual)
            report.update(status="completed" if not differences else "failed", parity_passed=not differences,
                tensors_compared=count, differences=len(differences), first_differences=differences[:20],
                mean_seconds=statistics.mean(elapsed), median_seconds=statistics.median(elapsed),
                peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                total_seconds=time.monotonic() - started,
                reference_sha256=file_hash(reference / "state-000070.pt"), run=trainer.identity)
            atomic_json(directory / "report.json", report)
            print(json.dumps(report), flush=True)
            if differences:
                raise ValueError("Seventy-update hardware parity failed")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(directory / "report.json", report)
            raise
        finally:
            if trainer is not None:
                trainer.close()
            del trainer
            if runtime is not None:
                runtime.close()


if __name__ == "__main__":
    main()
