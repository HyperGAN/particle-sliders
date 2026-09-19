"""Exercise the queue's stopping boundary, recovery and health gate without GPUs."""
import threading
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget_campaign as worker
from common import read, write, sha


@pytest.fixture
def queue(tmp_path, monkeypatch):
    work, models = tmp_path / "work", tmp_path / "models"
    monkeypatch.setattr(worker, "WORK", work)
    monkeypatch.setattr(worker, "MODELS", models)
    monkeypatch.setattr(worker.original, "WORK", work)
    monkeypatch.setattr(worker.original, "STOP", threading.Event())
    monkeypatch.setattr(worker, "BUDGET", {"targets": {"example": 2000}})
    item = dict(id="example", gpu=0)
    run = models / "example/example-fresh3400"
    calls = []
    health = {"catastrophic": False}
    monkeypatch.setattr(worker.original, "warmup", lambda _: (tmp_path / "warmup.pt", tmp_path / "warmup.safetensors"))

    def checkpoint(step):
        run.mkdir(parents=True, exist_ok=True)
        weights = run / f"{run.name}_step{step}.safetensors"
        state = run / f"state-step{step}.pt"
        weights.write_bytes(f"weights-{step}".encode())
        state.write_bytes(f"full-state-{step}".encode())
        write(run / f"audit-step{step}.json", dict(completed=step, finite=True, export_matches_state=True,
            weights_sha256=sha(weights), state_sha256=sha(state)))

    def samples(step):
        write(work / "samples" / f"example-step{step}.json", dict(step=step, complete=True, **health))

    def command(_, argv, log, phase):
        argv = list(map(str, argv))
        train = "--until" in argv
        step = int(argv[argv.index("--until" if train else "--step") + 1])
        calls.append(("train" if train else "sample", step))
        (checkpoint if train else samples)(step)

    monkeypatch.setattr(worker.original, "command", command)
    return item, run, calls, health, checkpoint, samples


def test_resumed_run_stops_at_2000_and_restart_does_no_more_work(queue):
    item, run, calls, _, checkpoint, samples = queue
    checkpoint(1000); samples(1000)
    worker.execute(item)
    assert calls == [("train", 2000), ("sample", 2000)]
    job = read(worker.original.job_path(item))
    assert job["status"] == "complete" and job["completed"] == job["target"] == 2000
    assert job["extension_requires_listening"] and sha(job["full_state"]) == job["full_state_sha256"]
    worker.execute(item)
    assert calls == [("train", 2000), ("sample", 2000)]


def test_checkpoint_alone_is_not_complete_until_samples_exist(queue):
    item, _, calls, _, checkpoint, samples = queue
    checkpoint(1000); samples(1000); checkpoint(2000)
    worker.execute(item)
    assert calls == [("sample", 2000)]
    assert read(worker.original.job_path(item))["status"] == "complete"


def test_health_failure_retains_state_and_stops_before_next_milestone(queue):
    item, run, calls, health, _, _ = queue
    health["catastrophic"] = True
    worker.execute(item)
    assert calls == [("train", 1000), ("sample", 1000)]
    assert read(worker.original.job_path(item))["status"] == "needs_listening"
    assert (run / "state-step1000.pt").exists()


def test_later_explicit_extension_uses_existing_2000_checkpoint(queue):
    item, _, calls, _, _, _ = queue
    worker.execute(item)
    calls.clear()
    worker.BUDGET["targets"][item["id"]] = 3400
    worker.execute(item)
    assert calls == [("train", 3000), ("sample", 3000), ("train", 3400), ("sample", 3400)]
    assert read(worker.original.job_path(item))["completed"] == 3400
    calls.clear()
    worker.execute(item)
    assert calls == []


def test_changed_full_state_is_rejected_before_any_more_training(queue):
    item, run, calls, _, checkpoint, samples = queue
    checkpoint(1000); samples(1000)
    (run / "state-step1000.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="full state changed"):
        worker.execute(item)
    assert calls == []
