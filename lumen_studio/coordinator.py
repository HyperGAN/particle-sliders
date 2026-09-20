"""One process owns Anima GPU work, with update-boundary training handoffs."""
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .contracts import atomic_json, file_hash
from .store import Store
from .models import ANIMA, get_model


class OwnershipError(RuntimeError):
    pass


class GpuLease:
    def __init__(self, directory, device):
        self.directory, self.device = Path(directory), device
        self.handle = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = (self.directory / f"anima-{self.device.replace(':', '-')}.lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise OwnershipError("Another Anima coordinator owns this GPU") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def choose_work(controls, has_images, has_training):
    if controls["paused"]:
        return "render" if has_images else None
    if has_images and (not has_training or controls["render_burst"] < 4) and (
            not has_training or controls["training_updates"] >= 20):
        return "render"
    if has_training:
        return "train"
    return "render" if has_images else None


class Coordinator:
    def __init__(self, directory, model_dir, *, gpu="1", runtime_factory=None):
        self.directory, self.model_dir = Path(directory).resolve(), Path(model_dir).resolve()
        self.store = Store(self.directory / "studio.sqlite3")
        self.gpu = str(gpu)
        self.device = "cpu" if gpu == "cpu" else "cuda:0"
        self.runtime_factory = runtime_factory
        self.runtime = None
        self.loaded_checkpoints = None
        self.loaded_model = None
        self.stopping = False
        (self.directory / "images").mkdir(parents=True, exist_ok=True)

    def load_runtime(self, training=False, model_id=ANIMA.id, checkpointing=None):
        if self.runtime is not None and self.loaded_model != model_id:
            self.release_runtime()
        if self.runtime is None:
            from .runtime import create_runtime
            factory = self.runtime_factory or create_runtime
            self.runtime = factory(self.model_dir, self.device, backend=get_model(model_id).backend,
                                   checkpointing=training if checkpointing is None else checkpointing)
            self.loaded_model = model_id
        return self.runtime

    def release_runtime(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.loaded_checkpoints = None
        self.loaded_model = None
        gc.collect()
        if self.device != "cpu":
            import torch
            torch.cuda.empty_cache()

    def render_one(self):
        from .particles import ParticleAdapter
        from .inference_controls import set_inference_mix
        from .runtime import Cancelled
        from PIL.PngImagePlugin import PngInfo
        job = self.store.claim()
        if job is None:
            return False
        ident, payload = job["id"], job["payload"]
        try:
            checkpoints = payload["checkpoints"]
            if self.runtime is not None and checkpoints != self.loaded_checkpoints:
                self.release_runtime()
            runtime = self.load_runtime(model_id=payload.get("model", ANIMA.id))
            if runtime.identity != payload["model_identity"]:
                if checkpoints:
                    raise ValueError("Queued model identity changed")
                from .provenance import single_image_runtime_compatibility
                single_image_runtime_compatibility(self.model_dir, payload["model_identity"], runtime.identity)
            if checkpoints != self.loaded_checkpoints:
                catalog = {c["sha256"]: c for c in self.store.catalog()}
                for variation, sha in checkpoints.items():
                    checkpoint = catalog[sha]
                    if file_hash(checkpoint["path"]) != sha:
                        raise ValueError("Selected checkpoint file changed")
                    adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
                    adapter.load_export(checkpoint["path"], model_identity=runtime.identity)
                    adapter.eval().requires_grad_(False)
                    runtime.mixer.add(variation, adapter)
                self.loaded_checkpoints = checkpoints.copy()
            strengths = set_inference_mix(runtime.mixer, payload["mix"], payload["energy"])
            if strengths != payload["strengths"]:
                raise ValueError("Queued mixing contract changed")
            start = time.monotonic()
            image = runtime.render(payload["prompt"], payload["seed"], payload["width"], payload["height"],
                payload["steps"], progress=lambda p: self.store.progress(ident, p),
                cancelled=lambda: self.store.job(ident)["status"] == "cancelling")
            metadata = dict(**payload, effective_prompt=payload["prompt"], job_id=ident,
                            group_id=job["group_id"], render_seconds=time.monotonic() - start)
            if runtime.identity != payload["model_identity"]:
                metadata.update(requested_model_identity=payload["model_identity"], model_identity=runtime.identity)
            path = self.directory / "images" / f"{ident}.png"
            temp = path.with_suffix(".tmp")
            png = PngInfo()
            png.add_text("anima", json.dumps(metadata, sort_keys=True))
            image.save(temp, format="PNG", pnginfo=png)
            temp.replace(path)
            if not self.store.finish(ident, path, metadata):
                path.unlink(missing_ok=True)
            else:
                atomic_json(path.with_suffix(".json"), metadata)
        except Cancelled as exc:
            self.store.fail(ident, exc, cancelled=True)
        except Exception as exc:
            self.store.fail(ident, f"{type(exc).__name__}: {exc}")
            self.release_runtime()
        controls = self.store.controls()
        self.store.control(render_burst=controls["render_burst"] + 1)
        return True

    def train_quantum(self, run):
        from .cache import TargetCache
        from .training import Trainer
        self.release_runtime()
        config = run["config"]
        runtime = trainer = None
        try:
            runtime = self.load_runtime(training=True, checkpointing=config.get("checkpointing", True))
            cache = TargetCache(config["train_cache"], pin_memory=self.device != "cpu")
            dev = TargetCache(config["dev_cache"], pin_memory=self.device != "cpu")
            trainer = Trainer(runtime, cache, config["directory"], config["variation"],
                microbatch=config.get("microbatch", 1), seed=config.get("seed", 7), dev_cache=dev)
            target = config.get("until", 200)
            if target > 200:
                from .audits import require_pilot_qualification
                require_pilot_qualification(config["directory"])
            controls = self.store.controls()
            # Switching from any render burst starts a fresh 20-update quota.
            updates = 0 if controls["render_burst"] else controls["training_updates"]
            self.store.control(render_burst=0, training_updates=updates,
                               owner=dict(pid=os.getpid(), gpu=self.gpu, state="training"))
            if trainer.step == 0:
                self.store.update_run(run["id"], "running", 0, trainer.probe())
            while trainer.step < target and not self.stopping:
                trainer.update()
                updates += 1
                self.store.control(training_updates=updates)
                self.store.update_run(run["id"], "running", trainer.step)
                if trainer.step == 20 or trainer.step % 100 == 0:
                    trainer.save(snapshot=True)
                    snapshot = Path(config["directory"]) / f"ema-{trainer.step:06}.safetensors"
                    sha = self.store.register_checkpoint(snapshot, config["variation"])
                    if trainer.step % 100 == 0:
                        self.store.update_run(run["id"], "running", trainer.step, trainer.probe())
                    from .sampling import enqueue_cadence
                    enqueue_cadence(self.store, runtime.identity, config["variation"], sha, trainer.step)
                controls = self.store.controls()
                has_images = any(j["status"] == "queued" for j in self.store.jobs())
                if controls["paused"] or (has_images and updates >= 20):
                    break
            trainer.save(snapshot=trainer.step == target)
            if trainer.step == target == 200:
                trainer.verify_resume()
            status = "pilot_review" if trainer.step == target == 200 else (
                "completed" if trainer.step >= target else "queued")
            self.store.update_run(run["id"], status, trainer.step)
        except Exception as exc:
            # Only completed updates are resumable. Save in normal paths, never a
            # half-finished D/G update after an exception.
            self.store.update_run(run["id"], "failed", trainer.step if trainer else run["step"],
                                  error=f"{type(exc).__name__}: {exc}")
        finally:
            if trainer is not None:
                trainer.close()
            del trainer
            del runtime
            self.release_runtime()

    def enough_memory(self, training):
        if self.gpu == "cpu" or self.runtime_factory:
            return True
        if self.runtime is None:
            # Diagnostics can release their last references after Runtime.close
            # returns. Return this process's unused allocator cache before
            # treating it as another application's occupied GPU memory.
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        result = subprocess.run(["nvidia-smi", "-i", self.gpu,
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
        # Do not evict or terminate another application's allocations.
        return int(result.stdout.strip()) >= (22000 if training else 10000)

    def run(self, once=False):
        if self.gpu != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        # A lock shared across all Studio data directories, keyed by physical GPU.
        lease_key = self.gpu
        if self.gpu != "cpu":
            lease_key = subprocess.check_output(["nvidia-smi", "-i", self.gpu,
                "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        with GpuLease(Path("/tmp"), lease_key):
            self.store.recover()
            self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="idle"))
            try:
                while not self.stopping:
                    queued = any(j["status"] == "queued" for j in self.store.jobs())
                    runs = [r for r in self.store.runs() if r["status"] == "queued"]
                    # Finish a new or revised pilot before spending more on
                    # already-qualified full runs. Existing updates stay atomic.
                    runs.sort(key=lambda r: (r["config"].get("until", 200) != 200, r["id"]))
                    work = choose_work(self.store.controls(), queued, bool(runs))
                    if work and self.runtime is None and not self.enough_memory(work == "train"):
                        self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="waiting for GPU memory"))
                        if once:
                            return
                        time.sleep(2)
                        continue
                    if work == "render":
                        self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="rendering"))
                        self.render_one()
                    elif work == "train":
                        self.train_quantum(runs[0])
                    else:
                        if self.runtime is not None:
                            self.release_runtime()
                        owner = self.store.controls()["owner"]
                        if not isinstance(owner, dict) or owner.get("state") != "idle":
                            self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="idle"))
                        if once:
                            return
                        time.sleep(.5)
                    if once:
                        return
            finally:
                self.release_runtime()
                self.store.control(owner=None)
