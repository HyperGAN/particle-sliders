"""Measure late-checkpoint movement and bounded frozen-opponent responses.

Run under anima_cuda_host.py on the verified GPU policy. The regular coordinator
must release its GPU lease first. This diagnostic serves up to four queued Studio
images between units, and writes only diagnostics plus GPU ownership status.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima/remote")
    parser.add_argument("--variations", nargs="+", default=["candlelit", "moonlit"], choices=["candlelit", "moonlit"])
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--steps", nargs="+", type=int, default=[1200, 1300, 1400, 1500, 1600])
    parser.add_argument("--response-steps", nargs="+", type=int, default=[1400, 1600])
    parser.add_argument("--response-updates", type=int, default=16)
    parser.add_argument("--trial-seeds", nargs="+", type=int, default=[811, 947])
    args = parser.parse_args()
    if not os.environ.get("LUMEN_KERNEL_POLICY"):
        raise ValueError("Use the verified anima_cuda_host.py --kernel-policy launcher")
    if (sorted(set(args.steps)) != args.steps or any(s < 1200 or s > 1600 or s % 100 for s in args.steps)
            or not set(args.response_steps).issubset(args.steps) or not 4 <= args.response_updates <= 64):
        raise ValueError("Use saved late 100-update checkpoints and bounded response budgets")
    from lumen_studio import execution
    import subprocess
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import atomic_json, digest, file_hash
    from lumen_studio.convergence import movement, objective_rows, paired_improvement, response_step
    from lumen_studio.coordinator import Coordinator, GpuLease
    from lumen_studio.game import Game, RECIPE, Sampler
    from lumen_studio.particles import ParticleAdapter
    from lumen_studio.provenance import cache_runtime_compatibility
    from lumen_studio.runtime import create_runtime
    from lumen_studio.store import Store
    from lumen_studio.vendor.reference import noise_std

    root = args.root.resolve()
    store = Store(root / "studio/studio.sqlite3")
    gpu_key = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    started = time.monotonic()

    def serve_pending():
        coordinator = Coordinator(root / "studio", root / "model", gpu=args.gpu)
        try:
            for _ in range(4):
                if not any(j["status"] == "queued" for j in store.jobs()):
                    break
                store.control(owner=dict(pid=os.getpid(), gpu=args.gpu, state="rendering between convergence checks"))
                coordinator.render_one()
        finally:
            coordinator.release_runtime()

    with GpuLease(Path("/tmp"), gpu_key):
        try:
            for variation in args.variations:
                run_dir = root / "runs" / variation
                output = root / "diagnostics/convergence" / variation
                output.mkdir(parents=True, exist_ok=True)
                run = json.loads((run_dir / "run.json").read_text())
                train = TargetCache(root / "targets" / variation / "train", pin_memory=True, max_shards=20)
                dev = TargetCache(root / "targets" / variation / "dev", max_shards=12)
                if (run["recipe"] != RECIPE or run["cache"] != train.index["fingerprint"]
                        or train.index["identity"]["split"] != "train" or dev.index["identity"]["split"] != "dev"
                        or dev.index["identity"]["variation"] != variation
                        or run["execution"]["microbatch"] != 1 or run["execution"]["checkpointing"]):
                    raise ValueError("Run, cache or execution differs from the accepted recipe")
                indices = [i for i, (shard, _) in enumerate(dev.offsets) if shard["seed"] == 29001]
                records = [dev[i] for i in indices]
                if not records or {r["position"] for r in records} != set(range(10)):
                    raise ValueError("Complete fixed development timestep coverage is required")
                positions = [r["position"] for r in records]
                groups = [r["row"]["id"] for r in records]
                normalization = torch.load(run_dir / "normalization.pt", map_location="cpu", weights_only=True)
                if normalization["provenance"]["cache_fingerprint"] != run["cache"]:
                    raise ValueError("Normalization differs from the training cache")
                if any(noise_std(s-1, start=float(r)/.28, decay_steps=1600, hold=1.) != 1.
                       for s in args.steps for r in normalization["edit_rms"]):
                    raise ValueError("Late response diagnostics require the unchanged sigma=1 hold")
                sources = [run_dir / "run.json", run_dir / "normalization.pt", run_dir / "resume.pt"]
                sources += [run_dir / f"state-{s:06}.pt" for s in args.steps]
                hashes = {str(p): file_hash(p) for p in sources}
                identity = dict(schema=1, run=run, source_hashes=hashes,
                    dev_fingerprint=dev.index["fingerprint"], dev_indices=indices,
                    steps=args.steps, response_steps=args.response_steps, response_updates=args.response_updates,
                    trial_seeds=args.trial_seeds, evaluation_seed=3803, vic_seed=3804, noise_repeats=2,
                    gpu_policy=os.environ["LUMEN_KERNEL_POLICY"],
                    code_sha256=digest({p.name: file_hash(p) for p in [Path(__file__), ROOT / "lumen_studio/convergence.py"]}))
                identity_hash = digest(identity)
                identity_path = output / "identity.json"
                if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                    raise ValueError("Diagnostic identity changed; archive the prior result before changing fixtures")
                atomic_json(identity_path, identity)
                report_path = output / "report.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else dict(
                    schema=1, variation=variation, identity_sha256=identity_hash, run_cache=run["cache"],
                    status="running", stage="preparing", prediction_measurements=[], movements=[], responses=[],
                    fixture=dict(split="dev", prompt_rows=len(set(groups)), velocity_fields=len(records),
                                 timesteps=10, trajectories=["neutral", "positive"], energy=1., resolution=512),
                    limitations=["Finite local tests do not certify global convergence or image quality.",
                        "Response trials optimize training rows only; evaluation uses fixed development rows and common noise.",
                        "Live weights and their saved critic are tested together; EMA is measured separately for movement.",
                        "No final-test data, production updates, hyperparameter changes or checkpoint promotions."])

                def publish(stage):
                    report.update(stage=stage, updated_at=time.time(), elapsed_seconds=time.monotonic()-started)
                    atomic_json(report_path, report)
                    store.control(owner=dict(pid=os.getpid(), gpu=args.gpu, state=f"convergence: {variation} · {stage}"))
                    print(json.dumps(dict(variation=variation, stage=stage, status=report["status"])), flush=True)

                @contextmanager
                def session(state, mode="adapter", with_game=False):
                    runtime = create_runtime(root / "model", "cuda:0", checkpointing=False)
                    game = adapter = None
                    try:
                        if runtime.identity != run["model"]:
                            raise ValueError("Runtime differs from the trained model")
                        for cache in (train, dev):
                            cache_runtime_compatibility(root / "model", cache.index["identity"]["model"],
                                                        runtime.identity, cache.index["fingerprint"])
                        versions = {n: (p._version, p.requires_grad) for n, p in runtime.transformer.named_parameters()}
                        if any(v[1] for v in versions.values()):
                            raise ValueError("The base transformer must remain frozen")
                        adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
                        adapter.load_state_dict(state[mode])
                        runtime.mixer.add(variation, adapter)
                        if with_game:
                            game = Game(adapter, normalization, len(train), seed=run["seed"], microbatch=1)
                            game.load_state_dict(state["game"])
                        with runtime.mixer.scales({variation: 1.}):
                            yield runtime, adapter, game
                        if versions != {n: (p._version, p.requires_grad) for n, p in runtime.transformer.named_parameters()}:
                            raise ValueError("Frozen base changed during the diagnostic")
                    finally:
                        runtime.close()
                        del game, adapter, runtime
                        gc.collect()
                        torch.cuda.empty_cache()

                def predict(runtime, rows):
                    with torch.no_grad():
                        return torch.stack([runtime.predict(r["latent"].to(runtime.device), r["timestep"],
                            r["embedding"]).detach().cpu().float().flatten() for r in rows])

                teacher = torch.stack([(r["positive"]-r["neutral"]).flatten() for r in records])
                positive = torch.stack([r["positive"].flatten() for r in records])
                neutral = torch.stack([r["neutral"].flatten() for r in records])
                scales = torch.stack([normalization["scale"][p] for p in positions])
                for step in args.steps:
                    path = run_dir / f"state-{step:06}.pt"
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    if state["identity"] != run or state["step"] != step:
                        raise ValueError("Checkpoint identity or step differs")
                    for mode in ("adapter", "ema"):
                        cache_path = output / f"predictions-{step:06}-{mode}.pt"
                        publish(f"predictions {step} {'live' if mode == 'adapter' else 'EMA'}")
                        if not cache_path.exists():
                            with session(state, mode) as (runtime, adapter, _):
                                values = predict(runtime, records)
                                repeated = predict(runtime, records[:10])
                                difference = float((values[:10]-repeated).abs().max())
                                if difference != 0:
                                    raise ValueError("Fixed prediction repeatability failed")
                            del runtime, adapter
                            gc.collect(); torch.cuda.empty_cache()
                            save_tensor_file(cache_path, dict(identity_sha256=identity_hash, predictions=values,
                                checkpoint_sha256=hashes[str(path)], repeat_max_abs=difference))
                        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                        if cached["identity_sha256"] != identity_hash or cached["checkpoint_sha256"] != hashes[str(path)]:
                            raise ValueError("Prediction cache identity differs")
                        entry = dict(step=step, weights="live" if mode == "adapter" else "EMA",
                            repeat_max_abs=cached["repeat_max_abs"], prediction_sha256=file_hash(cache_path))
                        report["prediction_measurements"] = [r for r in report["prediction_measurements"]
                            if (r["step"], r["weights"]) != (step, entry["weights"])] + [entry]
                        publish(f"predictions {step} {entry['weights']} saved")
                        serve_pending()
                    del state
                report["movements"] = []
                for mode in ("adapter", "ema"):
                    predictions = {s: torch.load(output / f"predictions-{s:06}-{mode}.pt",
                        map_location="cpu", weights_only=True)["predictions"] - neutral for s in args.steps}
                    for previous, step in zip(args.steps, args.steps[1:]):
                        report["movements"].append(dict(start=previous, end=step, weights="live" if mode == "adapter" else "EMA",
                            **movement(predictions[previous], predictions[step], teacher, scales, records)))
                    report["movements"].append(dict(start=args.steps[0], end=args.steps[-1],
                        weights="live" if mode == "adapter" else "EMA",
                        **movement(predictions[args.steps[0]], predictions[args.steps[-1]], teacher, scales, records)))
                publish("prediction movement measured")

                for step in args.response_steps:
                    state = torch.load(run_dir / f"state-{step:06}.pt", map_location="cpu", weights_only=True)
                    fixed = torch.load(output / f"predictions-{step:06}-adapter.pt", map_location="cpu", weights_only=True)
                    baseline_residuals = fixed["predictions"] - positive
                    baseline_path = output / f"objective-{step:06}.json"
                    if not baseline_path.exists():
                        publish(f"fixed objective {step}")
                        with session(state, with_game=True) as (runtime, adapter, game):
                            baseline = objective_rows(game, baseline_residuals, positions)
                        del runtime, adapter, game
                        gc.collect(); torch.cuda.empty_cache()
                        atomic_json(baseline_path, dict(identity_sha256=identity_hash, objectives=baseline))
                    saved = json.loads(baseline_path.read_text())
                    if saved["identity_sha256"] != identity_hash:
                        raise ValueError("Baseline objective identity differs")
                    baseline = saved["objectives"]
                    for seed in args.trial_seeds:
                        for side in ("d", "g"):
                            result_path = output / f"response-{step:06}-{side}-{seed}.json"
                            if not result_path.exists():
                                publish(f"{step} {'critic' if side == 'd' else 'slider'} response · seed {seed}")
                                unit_started = time.monotonic()
                                with session(state, with_game=True) as (runtime, adapter, game):
                                    game.sampler = Sampler(len(train), seed)
                                    train_positions = [i % 10 for i in range(len(train))]
                                    memo = {}

                                    def residual(ids):
                                        values = []
                                        for index in ids:
                                            if side == "d" and index in memo:
                                                values.append(memo[index].to(runtime.device))
                                                continue
                                            r = train[index]
                                            v = runtime.predict(r["latent"].to(runtime.device), r["timestep"], r["embedding"])
                                            value = (v-r["positive"].to(runtime.device)).flatten(1)
                                            if side == "d":
                                                memo[index] = value.detach().cpu()
                                            values.append(value)
                                        return torch.cat(values)

                                    opponent = adapter if side == "d" else game.critic
                                    frozen = {k: v.detach().cpu().clone() for k, v in opponent.state_dict().items()}
                                    updates = []
                                    for offset in range(1, args.response_updates+1):
                                        updates.append(response_step(game, residual, train_positions, step+offset, side))
                                        if offset % 4 == 0:
                                            publish(f"{step} {side.upper()} response {seed} · {offset}/{args.response_updates}")
                                    if any(not torch.equal(v, frozen[k]) for k, v in ((k, v.detach().cpu()) for k, v in opponent.state_dict().items())):
                                        raise ValueError("Frozen opponent changed")
                                    after_residuals = baseline_residuals if side == "d" else predict(runtime, records)-positive
                                    after = objective_rows(game, after_residuals, positions)
                                del runtime, adapter, game, residual, opponent
                                gc.collect(); torch.cuda.empty_cache()
                                result = dict(identity_sha256=identity_hash, step=step, side=side, seed=seed,
                                    updates=updates, frozen_opponent_unchanged=True,
                                    seconds=time.monotonic()-unit_started, before=baseline, after=after,
                                    improvement=paired_improvement(baseline[f"{side}_total"], after[f"{side}_total"], groups),
                                    adversarial_improvement=paired_improvement(baseline[f"{side}_adversarial"], after[f"{side}_adversarial"], groups))
                                atomic_json(result_path, result)
                            result = json.loads(result_path.read_text())
                            if result["identity_sha256"] != identity_hash:
                                raise ValueError("Response trial identity differs")
                            item = {k: result[k] for k in ("step", "side", "seed", "seconds", "improvement", "adversarial_improvement", "frozen_opponent_unchanged")}
                            report["responses"] = [r for r in report["responses"]
                                if (r["step"], r["side"], r["seed"]) != (step, side, seed)] + [item]
                            publish(f"response {step} {side.upper()} {seed} saved")
                            serve_pending()
                    del state
                for path, expected in hashes.items():
                    if file_hash(path) != expected:
                        raise ValueError("A production checkpoint or run file changed during the diagnostic")
                final_responses = [r for r in report["responses"] if r["step"] == args.response_steps[-1]]
                found = any(r["improvement"]["ci95"][0] > 0 for r in final_responses)
                report.update(status="completed", source_files_unchanged=True,
                    conclusion="Further objective improvement found against a frozen opponent; convergence is not established."
                        if found else "No clear objective improvement found within this bounded search; convergence remains unproven.")
                publish("complete")
        except BaseException as exc:
            if 'report' in locals() and report.get("status") == "running":
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=time.time())
                atomic_json(report_path, report)
            raise
        finally:
            store.control(owner=None)


if __name__ == "__main__":
    main()
