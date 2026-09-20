"""Run the unchanged recipe on a CPU fixture to check sustained learning end to end."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/lumen_studio"))


def main():
    import torch
    from conftest import TinyRuntime, tiny_cache
    from lumen_studio.cache import TargetCache
    from lumen_studio.contracts import atomic_json
    from lumen_studio.training import Trainer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/anima/diagnostics/tiny-learning")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = TinyRuntime()
    cache_path = args.output / "targets"
    cache = TargetCache(cache_path) if cache_path.exists() else tiny_cache(cache_path, runtime)
    frozen = {k: v.clone() for k, v in runtime.transformer.state_dict().items()}
    trainer = Trainer(runtime, cache, args.output / "run", "candlelit", seed=7, microbatch=1)
    path = args.output / "report.json"
    report = json.loads(path.read_text()) if path.exists() else dict(
        diagnostic_only=True, runtime="CPU tiny nonlinear frozen transformer",
        recipe=trainer.identity["recipe"], records=len(cache), metrics=[],
        interpretation="Training-fixture learning check; does not qualify Anima image quality or alter production")
    start = time.monotonic()

    @torch.no_grad()
    def measure():
        residuals, teachers = [], []
        with trainer.use_ema(), runtime.mixer.scales({"candlelit": 1.}):
            for i in range(len(cache)):
                rec = cache[i]
                pred = runtime.predict(rec["latent"], rec["timestep"], rec["embedding"])
                scale = trainer.normalization["scale"][rec["position"]]
                residuals.append((pred - rec["positive"]).flatten() / scale)
                teachers.append((rec["positive"] - rec["neutral"]).flatten() / scale)
        value = dict(step=trainer.step, residual_rms=float(torch.stack(residuals).square().mean().sqrt()),
                     teacher_rms=float(torch.stack(teachers).square().mean().sqrt()))
        report["metrics"] = [m for m in report["metrics"] if m["step"] != trainer.step] + [value]
        report["finite_gradients"] = True
        report["frozen_weights_unchanged"] = all(torch.equal(v, frozen[k]) for k, v in runtime.transformer.state_dict().items())
        report["elapsed_seconds_this_session"] = time.monotonic() - start
        report["complete"] = trainer.step == 1600
        atomic_json(path, report)
        runtime.calls.clear()
        print(json.dumps(value), flush=True)

    try:
        measure()
        while trainer.step < 1600:
            trainer.update()
            runtime.calls.clear()
            if trainer.step % 100 == 0:
                measure()
        trainer.save()
    finally:
        trainer.close()
        runtime.close()


if __name__ == "__main__":
    main()
