"""python -m lumen_studio --help"""
import argparse
import json
import os
from pathlib import Path
import signal

os.environ.setdefault("OMP_NUM_THREADS", "4")

from .contracts import VARIATIONS, atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description="NTC Image Studio and paired particle training")
    parser.add_argument("--root", type=Path, default=Path("artifacts/anima"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-model")
    sub.add_parser("compile")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8876)
    worker = sub.add_parser("worker")
    worker.add_argument("--gpu", default="1")
    worker.add_argument("--once", action="store_true")
    refs = sub.add_parser("references")
    refs.add_argument("--split", choices=["train", "dev"], default="train")
    cache = sub.add_parser("cache")
    cache.add_argument("variation", choices=VARIATIONS)
    cache.add_argument("--split", choices=["train", "dev"], default="train")
    cache.add_argument("--gpu", default="1")
    train = sub.add_parser("train")
    train.add_argument("variation", choices=VARIATIONS)
    train.add_argument("--until", type=int, choices=[200, 1600], default=200)
    train.add_argument("--microbatch", type=int, choices=[1, 2, 4], default=1)
    train.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--seed", type=int, default=7)
    bench = sub.add_parser("benchmark")
    bench.add_argument("variation", choices=VARIATIONS)
    bench.add_argument("--gpu", default="1")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-model":
        from .prepare import convert
        print(convert(root / "model"))
    elif args.command == "compile":
        from .dataset import write_manifests
        write_manifests(root / "manifests")
        print(root / "manifests")
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        model = root / "model"
        uvicorn.run(create_app(root / "studio", model if model.exists() else None),
                    host=args.host, port=args.port, timeout_graceful_shutdown=5)
    elif args.command == "worker":
        # Device visibility is set before importing torch or any model code.
        if args.gpu != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        from . import execution  # Set deterministic CUDA policy before model allocation.
        from .coordinator import Coordinator
        coordinator = Coordinator(root / "studio", root / "model", gpu=args.gpu)
        def stop(*_):
            coordinator.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        coordinator.run(once=args.once)
    elif args.command == "references":
        from .sampling import reference_payloads
        from .store import Store
        from .provenance import model_identity
        identity = model_identity(root / "model")
        result = Store(root / "studio/studio.sqlite3").enqueue(reference_payloads(identity, args.split))
        print(json.dumps(result))
    elif args.command == "cache":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        from . import execution  # The same deterministic policy as training and Studio.
        from .cache import prepare_targets
        from .coordinator import GpuLease
        from .backends.anima import TurboRuntime
        from .dataset import compile_manifest
        import subprocess
        key = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
            "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        manifest = compile_manifest(args.split)
        if args.split == "train":
            qualification_path = root / "manifests/qualification.json"
            if not qualification_path.exists():
                raise ValueError("Inspect paired reference renders and record manifest qualification before caching")
            qualification = json.loads(qualification_path.read_text())
            if qualification.get("manifest_sha256") != manifest["sha256"] or not qualification.get("passed"):
                raise ValueError("Definition qualification does not match the training manifest")
            from .provenance import model_identity
            if qualification.get("model_identity") != model_identity(root / "model"):
                raise ValueError("Definition references use a different model runtime")
        with GpuLease(Path("/tmp"), key):
            runtime = TurboRuntime(root / "model", "cuda:0")
            try:
                index = prepare_targets(runtime, manifest, args.variation,
                    root / "targets" / args.variation / args.split,
                    progress=lambda done, total: print(f"{done}/{total} paired trajectories", flush=True))
                print(json.dumps(index, default=str))
            finally:
                runtime.close()
    elif args.command == "train":
        from .store import Store
        store = Store(root / "studio/studio.sqlite3")
        config = dict(variation=args.variation, until=args.until, seed=args.seed, microbatch=args.microbatch,
            checkpointing=args.checkpointing,
            directory=str(root / "runs" / args.variation),
            train_cache=str(root / "targets" / args.variation / "train"),
            dev_cache=str(root / "targets" / args.variation / "dev"))
        for name in ("train_cache", "dev_cache"):
            if not (Path(config[name]) / "index.json").exists():
                raise ValueError(f"Prepare {name} first")
        if args.until == 1600:
            from .audits import require_pilot_qualification
            require_pilot_qualification(config["directory"])
        store.add_run(f"{args.variation}-{args.until}", config)
        print(f"Queued {args.variation} through update {args.until}; start the coordinator with worker")
    elif args.command == "benchmark":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        from .benchmark import benchmark
        print(json.dumps(benchmark(root, args.variation, args.gpu), indent=2))


if __name__ == "__main__":
    main()
