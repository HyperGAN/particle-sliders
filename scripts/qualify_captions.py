# ruff: noqa: E402
"""Native1200-update CPU qualification with exact recovery; no source objective changes."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from conceptmod.textsliders import (
    train_lora_tiny_llm as tiny,
    train_lora_bonsai_gguf as bonsai,
)
from conceptmod.textsliders.caption_recovery import (
    CaptionState,
    FrozenReadouts,
    copy_cpu,
    finite,
    digest_fixed,
)
from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
from caption_quality import HELDOUT, ema_parameters, predict, evaluate

ROOT = Path("/mnt/nvme/sliders/qualification")


def atomic(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


class Observer:
    def __init__(self, recipe, run, until):
        self.recipe, self.run, self.until = recipe, run, until
        self.model = recipe["model"]
        self.started = time.monotonic()
        self.stopping = False
        self.completed = 0
        self.state = CaptionState(
            run / "checkpoints",
            hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        )
        self.progress = dict(
            status="running",
            pid=os.getpid(),
            latest_step=0,
            planned_steps=1200,
            model=self.model,
            quality_qualified=False,
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.stop)

    def stop(self, *_):
        self.stopping = True

    def event(self, event, **values):
        self.progress.update(
            values,
            wall_seconds=time.monotonic() - self.started,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        atomic(self.run / "progress.json", self.progress)
        with (self.run / "events.jsonl").open("a") as f:
            f.write(
                json.dumps(dict(event=event, **self.progress), allow_nan=False) + "\n"
            )
        print(
            json.dumps(dict(event=event, **self.progress), allow_nan=False), flush=True
        )

    def ready(self, c):
        self.c = c
        self.completed = self.state.restore(c)
        if not (self.state.root / f"step-{self.completed:06d}.json").exists():
            self.state.save(self.completed, c)
        self.event(
            "native_ready",
            latest_step=self.completed,
            committed_step=self.completed,
            fixed_target_sha256=digest_fixed(c["fixed"]),
        )
        self.eval(self.completed)
        self.start_step = self.completed
        self.train_started = time.monotonic()
        return self.completed

    def before_step(self, step):
        if self.stopping:
            raise InterruptedError("Own caption worker stopped before update")

    def after_step(self, step, metrics):
        finite(self.c["network"].state_dict())
        finite(self.c["critic"].state_dict())
        finite(self.c["ema"])
        self.completed = step
        self.event("optimizer_updated", latest_step=step, metrics=metrics)
        if step % 20 == 0 or step == self.until or self.stopping:
            self.state.save(step, self.c)
            self.event("checkpoint_committed", committed_step=step)
        if step in (20, 100, 400, 800, 1200):
            self.eval(step)
        if step % 20 == 0 or step == self.until:
            self.event(
                "measured_budget",
                projected_1200_update_seconds=(time.monotonic() - self.train_started)
                / (step - self.start_step)
                * 1200,
            )
        return step < self.until and not self.stopping

    @torch.no_grad()
    def eval(self, step):
        cpu, py, numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
        network = self.c["network"]
        directory = self.run / "evaluation" / f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with ema_parameters(self.c):
                metrics, zero, expected = evaluate(self.c, self.model)
                metadata = dict(
                    step=step,
                    steps=step,
                    planned_steps=1200,
                    base_identity=self.recipe["base_identity"],
                    source_commit=self.recipe["code_commit"],
                    weights_kind="EMA",
                    model_id=self.recipe["model_id"],
                )
                path = directory / "adapter.safetensors"
                network.save(path, metadata)
                if self.model == "bonsai":
                    loaded, record = BonsaiParticleHead.load(path)
                    for row, wanted in zip(HELDOUT, expected):
                        frozen = self.c["backend"].hidden(
                            self.c["backend"].encode(row["neutral"])
                        )
                        with loaded.scaled(1):
                            if not torch.equal(wanted, loaded(frozen)):
                                raise ValueError("Bonsai EMA replay differs")
                else:
                    from safetensors.torch import load_file

                    saved = copy_cpu(network.state_dict())
                    for branch in network.adapters.values():
                        branch.lora_up.weight.zero_()
                    for row, wanted in zip(HELDOUT, zero):
                        if not torch.equal(
                            wanted, predict(self.c, self.model, row["neutral"], 1)
                        ):
                            raise ValueError("Tiny zero differs from cleared base")
                    network.load_state_dict(load_file(str(path)), strict=True)
                    if any(
                        not torch.equal(v, network.state_dict()[k])
                        for k, v in saved.items()
                    ):
                        raise ValueError("Tiny EMA tensor replay differs")
                    for row, wanted in zip(HELDOUT, expected):
                        if not torch.equal(
                            wanted, predict(self.c, self.model, row["neutral"], 1)
                        ):
                            raise ValueError("Tiny EMA output replay differs")
                if (
                    self.model == "tiny"
                    and self.recipe.get("generate_controls", True)
                    and step in (0, 100, 400, 800, 1200)
                ):
                    outputs = []
                    backend = self.c["backend"]
                    for i, row in enumerate(HELDOUT):
                        for scale in (0.0, 0.5, 1.0):
                            ids = backend.encode(row["neutral"]).ids
                            tokens = torch.tensor([ids])
                            mask = torch.ones_like(tokens)
                            with network.scaled(scale):
                                result = backend.model.generate(
                                    input_ids=tokens,
                                    attention_mask=mask,
                                    max_new_tokens=32,
                                    do_sample=False,
                                    pad_token_id=backend.tokenizer.eos_token_id,
                                )
                            outputs.append(
                                dict(
                                    case=i,
                                    scale=scale,
                                    prompt=row["neutral"],
                                    completion=backend.tokenizer.decode(
                                        result[0][len(ids) :], skip_special_tokens=True
                                    ),
                                )
                            )
                    atomic(directory / "generation-controls.json", outputs)
                metrics.update(
                    step=step,
                    planned_steps=1200,
                    export_reload_exact=True,
                    quality_qualified=False,
                    generation_effect_supported=self.model == "tiny",
                )
                atomic(directory / "metrics.json", metrics)
                self.event(
                    "heldout_evaluated",
                    evaluated_step=step,
                    readout_transfer_gate_passed=metrics[
                        "readout_transfer_gate_passed"
                    ],
                    zero_exact=True,
                    ema_export_reload_exact=True,
                )
        finally:
            torch.set_rng_state(cpu)
            random.setstate(py)
            np.random.set_state(numpy)

    def finished(self, sidecar):
        if sidecar["steps"] != self.completed:
            raise ValueError("Native export stamped wrong completed step")
        self.event(
            "stage_completed",
            status="stopped"
            if self.stopping
            else ("canary_complete" if self.until == 100 else "native_budget_complete"),
            latest_step=self.completed,
            committed_step=self.completed,
            quality_qualified=False,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("tiny", "bonsai"), required=True)
    p.add_argument("--until", choices=(100, 1200), type=int, required=True)
    a = p.parse_args()
    p.error('legacy native caption campaigns are retired; use the reviewed shared-package '
            'ConditionalPG09 worker and a new source-bound cohort')
    run = ROOT / ("runs/" + a.model + "-caption-native-1200-20261017")
    recipe = json.loads((run / "recipe.json").read_text())
    actual = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != recipe["code_commit"]:
        raise ValueError("Caption source pin changed")
    subprocess.run(["git", "-C", str(SOURCE), "diff", "--quiet", "HEAD"], check=True)
    for name, version in recipe["environment"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Caption CPU environment changed:" + name)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    trainer = tiny if a.model == "tiny" else bonsai
    argslist = [
        "--steps",
        "1200",
        "--device",
        "cpu",
        "--seed",
        "7",
        "--name",
        a.model + "-caption-native",
        "--no_report",
        "--save_dir",
        str(run / "native-export"),
    ]
    if a.model == "tiny":
        argslist += ["--model_id", recipe["model_dir"]]
        backend = TinyLLMBackend(
            device="cpu", model_id=recipe["model_dir"], allow_hub=False, dummy=False
        )
    else:
        argslist += ["--weights", recipe["weights"]]
        backend = FrozenReadouts(
            None,
            ROOT / "models/bonsai-caption-readouts-20261017",
            recipe["base_identity"],
        )
    observer = Observer(recipe, run, a.until)
    try:
        trainer.train(trainer.parse_args(argslist), backend=backend, observer=observer)
    except BaseException as exc:
        observer.event(
            "worker_failed",
            status="stopped" if observer.stopping else "failed",
            error=type(exc).__name__ + ": " + str(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
