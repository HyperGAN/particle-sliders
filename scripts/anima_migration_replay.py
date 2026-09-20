"""Verify four exact training updates across GPU hosts without advancing a run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--run", type=Path, help="Quiesced source run, required for reference")
    parser.add_argument("--evidence", type=Path, required=True)
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
    from lumen_studio.store import Store

    root, evidence = args.root.resolve(), args.evidence.resolve()
    scratch = evidence / args.mode
    if scratch.exists():
        raise FileExistsError("Migration evidence is immutable; use a fresh directory")
    key = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid",
                                   "--format=csv,noheader"], text=True).strip()
    started = time.monotonic()
    runtime = trainer = None
    store = Store(root / "studio/studio.sqlite3") if (root / "studio/studio.sqlite3").exists() else None
    with GpuLease(Path("/tmp"), key):
        try:
            if store is not None:
                store.control(owner=dict(pid=os.getpid(), gpu=args.gpu, state="verifying GPU migration checkpoint"))
            if args.mode == "reference":
                if args.run is None:
                    raise ValueError("A quiesced source run is required")
                source = args.run.resolve()
                (evidence / "input").mkdir(parents=True)
                for name in ("run.json", "normalization.pt", "resume.pt"):
                    shutil.copy2(source / name, evidence / "input" / name)
            scratch.mkdir(parents=True)
            for name in ("run.json", "normalization.pt", "resume.pt"):
                shutil.copy2(evidence / "input" / name, scratch / name)
            identity = json.loads((scratch / "run.json").read_text())
            variation = identity["variation"]
            runtime = TurboRuntime(root / "model", "cuda:0",
                checkpointing=identity["execution"]["checkpointing"])
            cache = TargetCache(root / "targets" / variation / "train", pin_memory=True)
            trainer = Trainer(runtime, cache, scratch, variation, seed=identity["seed"],
                              microbatch=identity["execution"]["microbatch"])
            before = trainer.step
            # Both cached trajectories, all ten positions, with adapters bypassed.
            with torch.no_grad():
                shard = torch.load(cache.directory / cache.index["shards"][0]["path"],
                                   map_location="cpu", weights_only=True)
                embeddings = {side: runtime.encode(shard["row"][side]) for side in ("neutral", "positive")}
                embedding_errors = {side: float((embeddings[side].cpu() - shard[
                    "embedding" if side == "neutral" else "positive_embedding"]).abs().max())
                    for side in embeddings}
                teacher_errors = [float((runtime.predict(cache[i]["latent"].to(runtime.device),
                    cache[i]["timestep"], cache[i]["embedding"]).cpu() - cache[i]["neutral"]).abs().max())
                    for i in range(20)]
                positive_errors = [float((runtime.predict(cache[i]["latent"].to(runtime.device),
                    cache[i]["timestep"], embeddings["positive"]).cpu() - cache[i]["positive"]).abs().max())
                    for i in range(20)]
            if any(teacher_errors + positive_errors + list(embedding_errors.values())):
                raise ValueError("Frozen teacher or text encoder predictions changed across hosts")
            rows = [trainer.update() for _ in range(4)]
            trainer.save()
            from lumen_studio.dataset import compile_manifest
            row = next(r for r in compile_manifest("dev")["rows"] if r["variation"] == variation)
            with trainer.use_ema(), runtime.mixer.scales({variation: 1.}):
                image = runtime.render(row["neutral"], 29001, width=768, height=768, steps=10)
            image.save(evidence / (args.mode + ".png"))
            report = dict(mode=args.mode, step_before=before, step_after=trainer.step,
                input_sha256=file_hash(evidence / "input/resume.pt"),
                output_sha256=file_hash(scratch / "resume.pt"),
                teacher_max_abs=teacher_errors, positive_teacher_max_abs=positive_errors,
                text_embedding_max_abs=embedding_errors,
                image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                image_settings=dict(case=row["id"], seed=29001, width=768, height=768, steps=10, energy=1.),
                updates=[{k: v for k, v in row.items() if k != "seconds"} for row in rows],
                seconds=time.monotonic() - started, gpu_uuid=key, passed=None)
            if args.mode == "verify":
                reference = json.loads((evidence / "reference.json").read_text())
                expected = torch.load(evidence / "reference/resume.pt", map_location="cpu", weights_only=True)
                actual = torch.load(scratch / "resume.pt", map_location="cpu", weights_only=True)
                differences, count = [], 0

                def compare(a, b, path="state"):
                    nonlocal count
                    if isinstance(a, torch.Tensor):
                        count += 1
                        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
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
                report.update(tensors_compared=count, differences=len(differences), first_differences=differences[:20],
                    passed=not differences and report["updates"] == reference["updates"]
                    and report["image_pixel_sha256"] == reference["image_pixel_sha256"]
                    and report["image_settings"] == reference["image_settings"]
                    and report["input_sha256"] == reference["input_sha256"]
                    and file_hash(evidence / "reference/resume.pt") == reference["output_sha256"])
            atomic_json(evidence / (args.mode + ".json"), report)
            print(json.dumps({k: v for k, v in report.items() if k != "updates"}), flush=True)
            if report["passed"] is False:
                raise ValueError("Cross-host checkpoint replay differed")
        finally:
            if trainer is not None:
                trainer.close()
            del trainer
            if runtime is not None:
                runtime.close()
            if store is not None:
                owner = store.controls()["owner"]
                if isinstance(owner, dict) and owner.get("pid") == os.getpid():
                    store.control(owner=None)


if __name__ == "__main__":
    main()
