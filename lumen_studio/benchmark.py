"""Measured, parity-gated eager microbatch/checkpointing benchmark."""
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .cache import TargetCache, fit_normalization, save_tensor_file
from .contracts import atomic_json
from .coordinator import GpuLease
from .backends.anima import TurboRuntime
from .training import Trainer


def benchmark(root, variation, gpu="0", should_stop=None):
    root = Path(root)
    key = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    report = dict(warmup_updates=20, timed_updates=50, parity_updates=4, seed=7,
                  candidates=[], accepted=None, attention="pinned default SDPA for generator; plain math for critic",
                  compilation="not attempted before eager qualification", cpu_threads=torch.get_num_threads(),
                  status="running", variation=variation, pid=os.getpid(), candidate_count=6)
    output = root / "benchmarks" / variation
    output.mkdir(parents=True, exist_ok=True)
    from .store import Store
    store = Store(root / "studio/studio.sqlite3")
    started = time.monotonic()

    def progress(phase, done=0, total=0):
        report.update(phase=phase, phase_done=done, phase_total=total,
                      updated_at=time.time(), elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "report.json", report)
        store.control(owner=dict(pid=os.getpid(), gpu=gpu, state=f"benchmark: {phase} {done}/{total}"))
        if should_stop is not None and should_stop():
            raise InterruptedError("Benchmark stopped at an update boundary")

    progress("checking target caches")
    cache = TargetCache(root / "targets" / variation / "train", pin_memory=True)
    dev = TargetCache(root / "targets" / variation / "dev", pin_memory=True)
    progress("fitting normalization")
    normalization = fit_normalization(cache)
    reference = None
    with GpuLease(Path("/tmp"), key):
        for checkpointing, microbatch in ((True, 1), (True, 2), (True, 4), (False, 1), (False, 2), (False, 4)):
            name = f"checkpoint-{int(checkpointing)}-micro-{microbatch}"
            directory = output / name
            if directory.exists():
                raise FileExistsError("Benchmark fixtures must start fresh; choose a new output root")
            directory.mkdir()
            save_tensor_file(directory / "normalization.pt", normalization)
            runtime = trainer = renderer = None
            candidate = dict(name=name, microbatch=microbatch, checkpointing=checkpointing)
            report["current_candidate"] = candidate
            try:
                progress("loading model")
                start = time.monotonic()
                runtime = TurboRuntime(root / "model", "cuda:0", checkpointing=checkpointing)
                candidate["load_seconds"] = time.monotonic() - start
                trainer = Trainer(runtime, cache, directory, variation, microbatch=microbatch, dev_cache=dev)
                progress("parity checks", 0, 4)
                for index in range(4):
                    trainer.update()
                    progress("parity checks", index + 1, 4)
                with runtime.mixer.scales({variation: 1.}), torch.no_grad():
                    predictions = torch.cat([trainer.predict_residual(list(range(start, min(start + microbatch, 4))))
                                             for start in range(0, 4, microbatch)]).detach().cpu()
                state = {k: v.detach().cpu().clone() for k, v in trainer.adapter.state_dict().items()}
                gradients = {k: v.grad.detach().cpu().clone() for k, v in trainer.adapter.named_parameters() if v.grad is not None}
                if reference is None:
                    reference = dict(state=state, gradients=gradients, predictions=predictions)
                # Mixed-precision matmul batch sizes change rounding; record the
                # discrepancy and reject candidates beyond explicit tolerances.
                candidate["prediction_max_abs"] = float((predictions - reference["predictions"]).abs().max())
                candidate["update_max_abs"] = max(float((v - reference["state"][k]).abs().max()) for k, v in state.items())
                candidate["gradient_max_abs"] = max(float((v - reference["gradients"][k]).abs().max()) for k, v in gradients.items())
                candidate["parity_passed"] = (
                    torch.allclose(predictions, reference["predictions"], atol=.02, rtol=.02)
                    and all(torch.allclose(v, reference["state"][k], atol=2e-5, rtol=2e-3) for k, v in state.items())
                    and all(torch.allclose(v, reference["gradients"][k], atol=2e-3, rtol=.02) for k, v in gradients.items()))
                if not candidate["parity_passed"]:
                    candidate["status"] = "rejected: parity"
                    continue
                # Restore identical initial sampler and optimizer streams for the
                # actual 20 warmup + 50 timed sequence, independent of parity runs.
                trainer.close()
                del trainer
                trainer = None
                (directory / "updates.jsonl").rename(directory / "parity-updates.jsonl")
                runtime.mixer.close()
                trainer = Trainer(runtime, cache, directory, variation, microbatch=microbatch, dev_cache=dev)
                progress("warmup", 0, 20)
                for index in range(20):
                    trainer.update()
                    if index == 3:
                        repeated_state = {k: v.detach().cpu() for k, v in trainer.adapter.state_dict().items()}
                        repeated_gradients = {k: v.grad.detach().cpu() for k, v in trainer.adapter.named_parameters()
                                              if v.grad is not None}
                        candidate["repeatability_passed"] = (
                            all(torch.equal(v, repeated_state[k]) for k, v in state.items())
                            and all(torch.equal(v, repeated_gradients[k]) for k, v in gradients.items()))
                        if not candidate["repeatability_passed"]:
                            raise RuntimeError("Identically seeded repeat changed model state or gradients")
                    progress("warmup", index + 1, 20)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                progress("timed updates", 0, 50)
                for index in range(50):
                    start = time.monotonic()
                    trainer.update()
                    torch.cuda.synchronize()
                    elapsed.append(time.monotonic() - start)
                    progress("timed updates", index + 1, 50)
                candidate.update(status="measured", mean_seconds=statistics.mean(elapsed),
                    median_seconds=statistics.median(elapsed), updates_per_second=50 / sum(elapsed),
                    peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
                start = time.monotonic()
                trainer.save(snapshot=True)
                candidate["save_seconds"] = time.monotonic() - start
                progress("development probe")
                candidate["probe_seconds"] = trainer.probe()["seconds"]
                # Use a separate Studio queue and immutable EMA export for
                # rendering cost, keeping benchmark samples out of the main UI.
                from .api import Generation, generation_payloads
                from .coordinator import Coordinator
                from .dataset import compile_manifest
                row = next(r for r in compile_manifest("dev")["rows"] if r["variation"] == variation)
                trainer.close()
                del trainer
                trainer = None
                runtime.mixer.close()
                renderer = Coordinator(directory / "studio", root / "model", gpu=gpu,
                    runtime_factory=lambda *a, **k: runtime)
                sha = renderer.store.register_checkpoint(directory / "ema-000070.safetensors", variation)
                progress("rendering comparisons", 0, 4)
                rendered = 0
                for size in (512, 768):
                    request = Generation(prompt=row["neutral"], seed=29001, width=size, height=size,
                        mix={variation: 1.}, checkpoints={variation: sha})
                    jobs = renderer.store.enqueue(generation_payloads(request, renderer.store, runtime.identity, [0., 1.]))
                    for ident in jobs["ids"]:
                        renderer.render_one()
                        job = renderer.store.job(ident)
                        if job["status"] != "completed":
                            raise RuntimeError(f"Benchmark Studio render failed: {job['error']}")
                        rendered += 1
                        progress("rendering comparisons", rendered, 4)
                candidate["renders"] = [{k: i["metadata"][k] for k in ("width", "height", "steps", "energy", "render_seconds")}
                                       for i in renderer.store.history()]
                renderer.runtime = None  # The outer scope still owns this runtime.
                renderer = None
            except torch.cuda.OutOfMemoryError:
                candidate["status"] = "rejected: out of memory"
            except Exception as exc:
                candidate.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                report.update(status="interrupted" if isinstance(exc, InterruptedError) else "failed",
                              error=candidate["error"])
                raise
            finally:
                start = time.monotonic()
                if trainer is not None:
                    trainer.close()
                del trainer
                if renderer is not None:
                    renderer.runtime = None
                renderer = None
                if runtime is not None and hasattr(runtime, "pipe"):
                    runtime.close()
                del runtime
                gc.collect()
                torch.cuda.empty_cache()
                candidate["release_seconds"] = time.monotonic() - start
                report["candidates"].append(candidate)
                atomic_json(output / "report.json", report)
    accepted = [c for c in report["candidates"] if c.get("status") == "measured"]
    if accepted:
        chosen = min(accepted, key=lambda c: c["mean_seconds"])
        report["accepted"] = chosen["name"]
        report["three_variation_training_hours_estimate"] = 4800 * chosen["mean_seconds"] / 3600
    indexes = [json.loads(p.read_text()) for p in (root / "targets").glob("*/*/index.json")]
    report["target_preparation_seconds"] = sum(i.get("elapsed_seconds", 0) for i in indexes)
    report["target_storage_bytes"] = sum(i.get("storage_bytes", 0) for i in indexes)
    if accepted:
        costs = {size: statistics.mean(r["render_seconds"] for r in chosen["renders"] if r["width"] == size)
                 for size in (512, 768)}
        # Per variation: 4 smoke images, twelve 16-image rounds, four
        # 32-image rounds, and baseline + sixteen development probes.
        sample_512, sample_768 = 3 * (4 + 12 * 16), 3 * 4 * 32
        audit_images = 512
        render_seconds = sample_512 * costs[512] + (sample_768 + audit_images) * costs[768]
        probe_seconds = 3 * 17 * chosen["probe_seconds"]
        # Four renders per burst, with both render and training runtimes
        # reloaded between bursts. This is an estimate, not a timed campaign.
        bursts = (sample_512 + sample_768) // 4
        handoff_seconds = bursts * (2 * chosen["load_seconds"] + chosen["save_seconds"] + 2 * chosen["release_seconds"])
        report["campaign_estimate"] = dict(training_seconds=4800 * chosen["mean_seconds"],
            probe_seconds=probe_seconds, rendering_seconds=render_seconds,
            handoff_seconds=handoff_seconds, scheduled_images=sample_512 + sample_768,
            final_audit_images=audit_images, total_remaining_hours=(4800 * chosen["mean_seconds"]
                + probe_seconds + render_seconds + handoff_seconds) / 3600,
            excludes=["human review time", "interactive requests", "unmeasured cache reload overhead"])
    report.update(status="completed" if accepted else "failed", updated_at=time.time(),
                  elapsed_seconds=time.monotonic() - started)
    atomic_json(output / "report.json", report)
    return report
