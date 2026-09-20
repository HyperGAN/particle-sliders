"""Real-model resume and renderer-parity gate before expensive pilot updates."""
import os
import json
from pathlib import Path
import subprocess
import time

from .contracts import atomic_json, digest, file_hash


def verify_gpu_resume(root, configuration, gpu="0", should_stop=None, profile_update=False):
    from .backends.anima import TurboRuntime
    from .cache import TargetCache
    from .coordinator import GpuLease
    from .store import Store
    from .training import Trainer
    root = Path(root)
    directory = root / "correctness/gpu-resume"
    report_path = root / "correctness/gpu-resume.json"
    if directory.exists():
        from .provenance import model_identity
        from .execution import DETERMINISM
        proof = json.loads(report_path.read_text()) if report_path.exists() else {}
        engine = digest({name: file_hash(Path(__file__).parent / name) for name in (
            "training.py", "game.py", "cache.py", "metrics.py", "execution.py", "vendor/grad_regularizers.py")})
        expected = dict(microbatch=configuration["microbatch"], checkpointing=configuration["checkpointing"],
                        determinism=DETERMINISM)
        if (proof.get("passed") and proof.get("optimizer_counters_on_cpu") and proof.get("frozen_base")
                and proof.get("run", {}).get("model") == model_identity(root / "model")
                and proof["run"].get("engine_sha256") == engine and proof["run"].get("execution") == expected
                and proof["run"].get("cache") == json.loads((root / "targets/candlelit/train/index.json").read_text())["fingerprint"]
                and file_hash(directory / "state-000003.pt") == proof.get("checkpoint_sha256")):
            return proof
        archived = directory.with_name(f"gpu-resume-archived-{time.time_ns()}")
        directory.rename(archived)
        if report_path.exists():
            report_path.rename(archived / "verification-summary.json")
    key = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    store = Store(root / "studio/studio.sqlite3")
    runtime = trainer = None
    start = time.monotonic()
    with GpuLease(Path("/tmp"), key):
        try:
            store.control(owner=dict(pid=os.getpid(), gpu=gpu, state="verifying GPU checkpoint resume"))
            runtime = TurboRuntime(root / "model", "cuda:0", checkpointing=configuration["checkpointing"])
            cache = TargetCache(root / "targets/candlelit/train", pin_memory=True)
            trainer = Trainer(runtime, cache, directory, "candlelit", microbatch=configuration["microbatch"])
            import torch
            with torch.no_grad():
                teacher_errors = [float((runtime.predict(cache[i]["latent"].to(runtime.device),
                    cache[i]["timestep"], cache[i]["embedding"]).cpu() - cache[i]["neutral"]).abs().max())
                    for i in range(20)]
            if any(teacher_errors):
                raise ValueError(f"Deterministic runtime changed frozen teacher predictions: {teacher_errors}")
            for _ in range(3):
                if should_stop and should_stop():
                    raise InterruptedError("Resume check stopped between updates")
                trainer.update()
            # Update four exercises the active exact second-derivative cap.
            report = trainer.verify_resume()
            assert all(s["step"].device.type == "cpu" for optimizer in (trainer.game.g, trainer.game.d)
                       for s in optimizer.state.values())
            assert all(v.device == runtime.device for v in trainer.ema.values())
            assert all(not p.requires_grad and p.grad is None for p in runtime.transformer.parameters())
            report.update(seconds=time.monotonic() - start, optimizer_counters_on_cpu=True,
                          frozen_base=True, configuration=configuration["name"], teacher_max_abs_by_state=teacher_errors)
            atomic_json(root / "correctness/gpu-resume.json", report)
            # Profile one replayed active-cap update after correctness passes.
            # Profiling is diagnostic; its availability is not a training gate.
            if profile_update:
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        trainer.update()
                    averages = profile.key_averages()
                    (directory / "update-profile.txt").write_text(
                        averages.table(sort_by="self_cpu_time_total", row_limit=40) + "\n\n"
                        + averages.table(sort_by="self_cuda_time_total", row_limit=40))
                except Exception as exc:
                    atomic_json(directory / "profiling-error.json", dict(error=f"{type(exc).__name__}: {exc}"))
                finally:
                    trainer.restore(directory / "resume.pt")
            return report
        finally:
            if trainer is not None:
                trainer.close()
            del trainer
            if runtime is not None:
                runtime.close()
            store.control(owner=None)
