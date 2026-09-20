"""Benchmark prepared targets, then run the three gated 200-update pilots."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--verified-benchmark", action="store_true",
                        help="Reuse a completed benchmark, then verify GPU resume before the pilots")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from lumen_studio.benchmark import benchmark
    from lumen_studio.contracts import VARIATIONS, atomic_json
    from lumen_studio.coordinator import Coordinator
    root = args.root.resolve()
    worker = Coordinator(root / "studio", root / "model", gpu=args.gpu)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        worker.stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (root / "campaign.pid").write_text(str(os.getpid()))
    try:
        for variation in VARIATIONS:
            for split in ("train", "dev"):
                if not (root / "targets" / variation / split / "index.json").exists():
                    raise ValueError(f"Missing prepared targets: {variation}/{split}")
        atomic_json(root / "campaign.json", dict(stage="benchmark", updated_at=time.time()))
        if args.verified_benchmark:
            report = json.loads((root / "benchmarks/candlelit/report.json").read_text())
            if report.get("status") != "completed":
                raise ValueError("Only a completed benchmark may be reused")
        else:
            report = benchmark(root, "candlelit", args.gpu, should_stop=lambda: stopping)
        if stopping:
            return
        selected = next((c for c in report["candidates"] if c["name"] == report["accepted"]), None)
        if selected is None or not selected["parity_passed"]:
            raise ValueError("No benchmark configuration passed correctness checks")
        from lumen_studio.preflight import verify_gpu_resume
        atomic_json(root / "campaign.json", dict(stage="verifying GPU resume", updated_at=time.time()))
        verify_gpu_resume(root, selected, args.gpu, should_stop=lambda: stopping)
        if stopping:
            return
        for variation in VARIATIONS:
            ident = f"{variation}-200"
            config = dict(variation=variation, until=200, seed=7,
                microbatch=selected["microbatch"], checkpointing=selected["checkpointing"],
                directory=str(root / "runs" / variation),
                train_cache=str(root / "targets" / variation / "train"),
                dev_cache=str(root / "targets" / variation / "dev"))
            existing = next((r for r in worker.store.runs() if r["id"] == ident), None)
            if existing is not None:
                if existing["config"] != config:
                    raise ValueError(f"Existing pilot configuration differs: {variation}")
            else:
                worker.store.add_run(ident, config)
        atomic_json(root / "campaign.json", dict(stage="pilots", benchmark=report["accepted"],
            until=200, updated_at=time.time(), next="Pilot quality review; full training is gated"))
        (root / "worker.pid").write_text(str(os.getpid()))
        worker.run()
    except Exception as exc:
        atomic_json(root / "campaign.json", dict(stage="interrupted" if stopping else "failed",
            error=f"{type(exc).__name__}: {exc}", updated_at=time.time()))
        raise
    finally:
        worker.release_runtime()
        worker.store.control(owner=None)


if __name__ == "__main__":
    main()
