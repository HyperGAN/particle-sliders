"""Resumable update-boundary trainer; runtime allocations owned by coordinator."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .cache import TargetCache, fit_normalization, save_tensor_file
from .contracts import atomic_json, digest, file_hash
from .game import Game, RECIPE
from .metrics import make_fixtures, measure, residual_breakdown
from .particles import ParticleAdapter, init_ema, update_ema
from .execution import DETERMINISM


class Trainer:
    def __init__(self, runtime, cache, directory, variation, *, seed=7, microbatch=1, dev_cache=None):
        self.runtime, self.cache = runtime, cache
        self.prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumen-targets")
        self.prefetched = {}
        self.directory, self.variation = Path(directory), variation
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dev_cache = dev_cache
        identity = cache.index["identity"]
        if identity["split"] != "train" or identity["variation"] != variation:
            raise ValueError("Training cache does not match this run")
        from .provenance import cache_runtime_compatibility
        compatibility = cache_runtime_compatibility(getattr(runtime, "model_dir", None),
            identity["model"], runtime.identity, cache.index["fingerprint"])
        self.identity = dict(recipe=RECIPE, model=runtime.identity, cache=cache.index["fingerprint"],
                             seed=seed, variation=variation,
            execution=dict(microbatch=microbatch,
                           determinism=DETERMINISM,
                           checkpointing=bool(getattr(runtime.transformer, "is_gradient_checkpointing", False))),
            engine_sha256=digest({name: file_hash(Path(__file__).parent / name) for name in (
                "training.py", "game.py", "cache.py", "metrics.py", "execution.py", "vendor/grad_regularizers.py")}))
        if compatibility is not None:
            self.identity["teacher_runtime_compatibility_sha256"] = compatibility
        if (self.directory / "run.json").exists():
            if json.loads((self.directory / "run.json").read_text()) != self.identity:
                raise ValueError("Run identity changed on resume")
        else:
            atomic_json(self.directory / "run.json", self.identity)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        norm_path = self.directory / "normalization.pt"
        if norm_path.exists():
            self.normalization = torch.load(norm_path, map_location="cpu", weights_only=True)
        else:
            self.normalization = fit_normalization(cache)
            save_tensor_file(norm_path, self.normalization)
        if self.normalization["provenance"]["cache_fingerprint"] != cache.index["fingerprint"]:
            raise ValueError("Normalization provenance mismatch")
        self.adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
        runtime.mixer.add(variation, self.adapter)
        self.game = Game(self.adapter, self.normalization, len(cache), seed, microbatch)
        self.ema = init_ema(self.adapter)
        self.step = 0
        self.best_residual = None
        self.positions = [i % 10 for i in range(len(cache))]
        self.fixtures = None
        if dev_cache is not None:
            if dev_cache.index["identity"]["split"] != "dev":
                raise ValueError("Only matching development data may be probed during training")
            cache_runtime_compatibility(getattr(runtime, "model_dir", None),
                dev_cache.index["identity"]["model"], runtime.identity, dev_cache.index["fingerprint"])
            # One seed per definition/character row, both trajectories, all ten positions.
            self.dev_indices = [i for i, (s, _) in enumerate(dev_cache.offsets)
                                if s["seed"] == 29001]
            fixture_path = self.directory / "fixtures.pt"
            if fixture_path.exists():
                self.fixtures = torch.load(fixture_path, map_location="cpu", weights_only=True)
                if self.fixtures["cache"] != dev_cache.index["fingerprint"] or self.fixtures["indices"] != self.dev_indices:
                    raise ValueError("Development fixtures changed")
            else:
                self.fixtures = dict(cache=dev_cache.index["fingerprint"], indices=self.dev_indices,
                    **make_fixtures(self.normalization["scale"].shape[-1], len(self.dev_indices)))
                save_tensor_file(fixture_path, self.fixtures)
        if (self.directory / "resume.pt").exists():
            self.restore(self.directory / "resume.pt")

    def predict_residual(self, indices):
        records = [self.prefetched[i].result() if i in self.prefetched else self.cache[i] for i in indices]
        # Exact embedding lengths are bucketed, so batching cannot add unmasked
        # padding tokens to Anima's cross-attention context.
        groups = {}
        results = [None] * len(records)
        for i, record in enumerate(records):
            groups.setdefault(tuple(record["embedding"].shape[1:]), []).append((i, record))
        for group in groups.values():
            z = torch.cat([r["latent"] for _, r in group]).to(self.runtime.device, non_blocking=True)
            embeddings = torch.cat([r["embedding"] for _, r in group]).to(self.runtime.device, non_blocking=True)
            ts = torch.tensor([r["timestep"] for _, r in group], device=self.runtime.device)
            v = self.runtime.predict(z, ts, embeddings)
            for j, (index, record) in enumerate(group):
                # Shared neutral cancels exactly. No large-state subtraction twice.
                results[index] = (v[j:j+1] - record["positive"].to(v.device)).flatten(1)
        return torch.cat(results)

    def prefetch(self, indices):
        self.prefetched = {i: self.prefetch_pool.submit(self.cache.__getitem__, i) for i in dict.fromkeys(indices)}

    def close(self):
        self.prefetch_pool.shutdown(wait=True)
        self.prefetched.clear()

    def update(self):
        if self.step >= 1600:
            raise ValueError("The pinned campaign horizon is 1600 updates")
        start = time.monotonic()
        with self.runtime.mixer.scales({self.variation: 1.}):
            result = self.game.update(self.predict_residual, self.positions, self.step + 1, self.prefetch)
        self.step += 1
        update_ema(self.ema, self.adapter)
        result["seconds"] = time.monotonic() - start
        with (self.directory / "updates.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        if self.step % 100 == 0:
            self.save(snapshot=True)
        return result

    @contextmanager
    def use_ema(self):
        live = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
        self.adapter.load_state_dict(self.ema)
        try:
            yield
        finally:
            self.adapter.load_state_dict(live)

    @torch.no_grad()
    def probe(self):
        start = time.monotonic()
        if self.dev_cache is None:
            raise ValueError("Development cache required")
        students, teachers, scales, records = [], [], [], []
        with self.use_ema(), self.runtime.mixer.scales({self.variation: 1.}):
            for index in self.dev_indices:
                rec = self.dev_cache[index]
                pred = self.runtime.predict(rec["latent"].to(self.runtime.device), rec["timestep"], rec["embedding"])
                students.append((pred.cpu() - rec["neutral"]).flatten())
                teachers.append((rec["positive"] - rec["neutral"]).flatten())
                scales.append(self.normalization["scale"][rec["position"]])
                records.append(rec)
        s, t, scale = torch.stack(students), torch.stack(teachers), torch.stack(scales)
        result = measure(s, t, scale, self.fixtures)
        residual = result["residual_rms"]
        self.best_residual = residual if self.best_residual is None else min(self.best_residual, residual)
        result.update(step=self.step, full_strength_raw_R=residual, best_so_far_R=self.best_residual,
                      breakdown=residual_breakdown((s - t) / scale, records), seconds=time.monotonic() - start)
        atomic_json(self.directory / f"probe-{self.step:06}.json", result)
        return result

    def save(self, snapshot=False):
        state = dict(identity=self.identity, step=self.step, adapter=self.adapter.state_dict(),
            ema=self.ema, game=self.game.state_dict(), best_residual=self.best_residual,
            torch_rng=torch.get_rng_state(), cuda_rng=(torch.cuda.get_rng_state(self.runtime.device)
                if self.runtime.device.type == "cuda" else None), random_rng=random.getstate(),
            numpy_rng=dict(name=np.random.get_state()[0], keys=np.random.get_state()[1].tolist(),
                pos=np.random.get_state()[2], gauss=np.random.get_state()[3], cached=np.random.get_state()[4]))
        save_tensor_file(self.directory / "resume.pt", state)
        if snapshot:
            save_tensor_file(self.directory / f"state-{self.step:06}.pt", state)
            path = self.directory / f"ema-{self.step:06}.safetensors"
            if not path.exists():
                self.adapter.export(path, model_identity=self.runtime.identity,
                    normalization=self.normalization["provenance"],
                    manifest_hash=self.cache.index["identity"]["manifest_sha256"], step=self.step, ema=self.ema)

    def restore(self, path):
        # Adam's non-capturable step counters must remain on CPU. Loading the
        # whole checkpoint onto CUDA silently moves them too; optimizer loading
        # already places parameter moments on their correct devices.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != self.identity:
            raise ValueError("Resume identity mismatch")
        self.adapter.load_state_dict(saved["adapter"])
        self.ema = {k: v.to(self.runtime.device) for k, v in saved["ema"].items()}
        self.game.load_state_dict(saved["game"])
        self.step, self.best_residual = saved["step"], saved["best_residual"]
        # A probe may commit after the periodic state save but before a crash.
        # Retain that committed minimum so the best-so-far chart cannot rise.
        committed = [json.loads(p.read_text())["full_strength_raw_R"]
                     for p in self.directory.glob("probe-*.json")
                     if int(p.stem.split("-")[-1]) <= self.step]
        if committed:
            self.best_residual = min(committed + ([self.best_residual] if self.best_residual is not None else []))
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), self.runtime.device)
        random.setstate(saved["random_rng"])
        n = saved["numpy_rng"]
        np.random.set_state((n["name"], np.array(n["keys"], dtype=np.uint32), n["pos"], n["gauss"], n["cached"]))
        # A crash can leave logs beyond the last committed update. Remove those
        # speculative rows so charts and restart state describe the same run.
        log = self.directory / "updates.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            log.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= self.step))

    def verify_resume(self):
        """Compare the next live update with a restored update; retain this step."""
        self.save(snapshot=True)
        step = self.step
        path = self.directory / "resume.pt"

        def capture():
            return copy.deepcopy(dict(adapter=self.adapter.state_dict(), ema=self.ema,
                game=self.game.state_dict(), gradients={k: p.grad for k, p in self.adapter.named_parameters()}))

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return isinstance(b, torch.Tensor) and torch.equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b

        try:
            first = self.update()
            expected = capture()
            self.restore(path)
            second = self.update()
            actual = capture()
            mismatches = []
            def compare(a, b, path="state"):
                if isinstance(a, torch.Tensor):
                    if not torch.equal(a, b):
                        mismatches.append(dict(path=path, before_device=str(a.device), after_device=str(b.device),
                            max_abs=float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max())))
                elif isinstance(a, dict):
                    for key in a:
                        compare(a[key], b[key], f"{path}.{key}")
                elif isinstance(a, (list, tuple)):
                    for index, (x, y) in enumerate(zip(a, b)):
                        compare(x, y, f"{path}.{index}")
            compare(expected, actual)
            passed = equal(expected, actual) and equal(
                {k: v for k, v in first.items() if k != "seconds"},
                {k: v for k, v in second.items() if k != "seconds"})
            self.restore(path)
            records = [self.cache[i] for i in range(10)]
            with self.use_ema(), self.runtime.mixer.scales({self.variation: 1.}), torch.no_grad():
                predictions = [self.runtime.predict(r["latent"].to(self.runtime.device), r["timestep"],
                               r["embedding"]).cpu() for r in records]
            exported = ParticleAdapter(self.runtime.transformer).to(self.runtime.device)
            exported.load_export(self.directory / f"ema-{step:06}.safetensors", model_identity=self.runtime.identity)
            exported.eval().requires_grad_(False)
            self.runtime.mixer.adapters[self.variation] = exported
            try:
                with self.runtime.mixer.scales({self.variation: 1.}), torch.no_grad():
                    errors = [float((self.runtime.predict(r["latent"].to(self.runtime.device), r["timestep"],
                        r["embedding"]).cpu() - p).abs().max()) for r, p in zip(records, predictions)]
                passed = passed and all(error == 0 for error in errors)
            finally:
                self.runtime.mixer.adapters[self.variation] = self.adapter
                del exported
        finally:
            self.restore(path)
        report = dict(passed=passed, step=step, checked_update=step + 1, run=self.identity,
                      unequal_tensors=len(mismatches), tensor_mismatches=mismatches[:20],
                      resume_tensor_max_abs=max((m["max_abs"] for m in mismatches), default=0.),
                      checkpoint_sha256=file_hash(self.directory / f"state-{step:06}.pt"),
                      studio_prediction_max_abs_by_timestep=errors)
        atomic_json(self.directory / "resume-check.json", report)
        if not passed:
            raise ValueError("Live and restored next updates differ")
        return report
