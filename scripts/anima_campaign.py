"""Prepare qualified targets on one GPU, serving base-model requests between shards."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TargetRuntime:
    """Resolve the current frozen runtime after each rendering handoff."""
    def __init__(self, worker):
        self.worker = worker

    def __getattr__(self, name):
        return getattr(self.worker.load_runtime(), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--variation", action="append", choices=("candlelit", "moonlit", "theatrical"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from lumen_studio import execution
    from lumen_studio.cache import prepare_targets
    from lumen_studio.contracts import VARIATIONS, atomic_json
    from lumen_studio.coordinator import Coordinator, GpuLease
    from lumen_studio.dataset import compile_manifest
    from lumen_studio.models import ANIMA
    from lumen_studio.runtime import Cancelled
    root = args.root.resolve()
    qualification = json.loads((root / "manifests/qualification.json").read_text())
    train = compile_manifest("train")
    if not qualification.get("passed") or qualification["manifest_sha256"] != train["sha256"]:
        raise ValueError("Training definitions have not passed visual qualification")
    worker = Coordinator(root / "studio", root / "model", gpu=args.gpu)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (root / "campaign.pid").write_text(str(os.getpid()))
    key = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    start = time.monotonic()
    prepared = []
    counter, last_burst = 0, -20
    with GpuLease(Path("/tmp"), key):
        worker.store.recover()
        runtime = TargetRuntime(worker)
        worker.loaded_checkpoints = {}
        if runtime.identity != qualification["model_identity"]:
            raise ValueError("Reference gallery and target runtime differ")
        try:
            for variation in (args.variation or VARIATIONS):
                for split in ("train", "dev"):
                    def progress(done, total):
                        nonlocal counter, last_burst
                        counter += 1
                        if stopping:
                            raise Cancelled("Stopped after committing a complete target shard")
                        worker.store.control(owner=dict(pid=os.getpid(), gpu=args.gpu,
                            state=f"preparing {variation} {split} {done}/{total}"))
                        atomic_json(root / "preparation.json", dict(stage="targets", variation=variation,
                            split=split, done=done, total=total, elapsed_seconds=time.monotonic() - start))
                        if done % 16 == 0 or done == total:
                            print(f"{variation}/{split}: {done}/{total}", flush=True)
                        paused = worker.store.controls()["paused"]
                        if counter - last_burst >= 20 or paused:
                            count = 0
                            if paused or any(j["status"] == "queued" for j in worker.store.jobs()):
                                worker.release_runtime()
                                try:
                                    while count < 4 or paused:
                                        if stopping:
                                            raise Cancelled("Stopped between target shards")
                                        if any(j["status"] == "queued" for j in worker.store.jobs()):
                                            worker.render_one()
                                            count += 1
                                            last_burst = counter
                                        elif paused:
                                            time.sleep(.5)
                                        else:
                                            break
                                        paused = worker.store.controls()["paused"]
                                finally:
                                    # TargetRuntime lazily reloads a fresh frozen
                                    # model before the next teacher prediction.
                                    worker.release_runtime()
                    index = prepare_targets(runtime, compile_manifest(split), variation,
                        root / "targets" / variation / split, progress=progress)
                    prepared.append(dict(variation=variation, split=split, fingerprint=index["fingerprint"],
                        seconds=index["elapsed_seconds"], bytes=index["storage_bytes"]))
            atomic_json(root / "preparation.json", dict(stage="complete", targets=prepared,
                elapsed_seconds=time.monotonic() - start, model=runtime.identity))
        finally:
            worker.release_runtime()
            worker.store.control(owner=None, render_burst=0, training_updates=20)
    with (root / "worker.log").open("a") as log:
        proc = subprocess.Popen([sys.executable, "-m", "lumen_studio", "--root", str(root), "worker", "--gpu", args.gpu],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / "worker.pid").write_text(str(proc.pid))


if __name__ == "__main__":
    main()
