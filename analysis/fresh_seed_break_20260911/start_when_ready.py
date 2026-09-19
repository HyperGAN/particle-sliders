"""Wait for the 1000→2000 run, then start the 2000→5000 continuation.

Not named queue.py: a module here shadows the stdlib for every script in this folder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
PROTOCOL_PATH = WORK / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
PARENT_RUN = Path(PROTOCOL["source_state"]).parent
PARENT_WORK = PARENT_RUN.parent
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
TRAIN_UNIT = "music-fresh-break-train-20260911"
FINISH_UNIT = "music-fresh-break-finish-20260911"
START = PROTOCOL["source_steps"]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def note(phase, **extra):
    write(WORK / "queue-status.json", dict(phase=phase, updated=time.time(),
        parent=read(PARENT_RUN / "status.json", {}), **extra))


def active(unit):
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def launch(unit, script, **environment):
    if active(unit):
        return "already running"
    command = ["systemd-run", "--user", f"--unit={unit}",
        "--working-directory=/ml2/music/sliders-conceptmod"]
    for key, value in environment.items():
        command.append(f"--setenv={key}={value}")
    command += [PY, "-u", str(WORK / script)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return "started"


def wait_for(phase, ready):
    while True:
        failure = read(PARENT_RUN / "failure.json")
        if failure:
            raise RuntimeError(f"The parent run failed before step {START}: {failure}")
        if ready():
            return
        note(phase)
        time.sleep(30)


def parent_finished():
    status = read(PARENT_RUN / "status.json", {})
    return (status.get("status") == "complete" and status.get("completed") == START
        and Path(PROTOCOL["source_state"]).exists() and Path(PROTOCOL["source_weights"]).exists())


def reference_measured():
    return read(Path(PROTOCOL["evaluation"]["reference_scores"]), {}).get("status") == "complete"


def main():
    note(f"Waiting for the parent run to reach step {START:,}")
    wait_for(f"Waiting for the parent run to reach step {START:,}", parent_finished)
    # Let the parent's final checkpoint settle on disk before hashing it.
    time.sleep(5)
    state, weights = Path(PROTOCOL["source_state"]), Path(PROTOCOL["source_weights"])
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if not protocol["source_state_sha256"]:
        protocol["source_state_sha256"] = sha(state)
        protocol["source_weights_sha256"] = sha(weights)
        write(PROTOCOL_PATH, protocol)
    parent_completion = read(PARENT_RUN / "completion-audit.json", {})
    write(WORK / "setup-audit.json", dict(
        parent_state=str(state), parent_state_sha256=protocol["source_state_sha256"],
        parent_weights=str(weights), parent_weights_sha256=protocol["source_weights_sha256"],
        parent_manifest_sha256=sha(PARENT_RUN / "manifest.json"),
        parent_status=read(PARENT_RUN / "status.json", {}),
        parent_completion_audit=parent_completion,
        recorded_not_predeclared="Hashes are observed at queue time; train.py independently re-verifies weights, critic, both optimizers, sampler and RNG against this state and asserts completed==2000.",
        first_seed=1_000_001 + START + 1 - 601,
        seed_continues_parent=(1_000_001 + START + 1 - 601) == PROTOCOL["continuation_seed_start"]))
    assert (1_000_001 + START + 1 - 601) == PROTOCOL["continuation_seed_start"]
    note("Starting training", train=launch(TRAIN_UNIT, "train.py",
        CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    note("Waiting for the parent's step-2000 reference measurement")
    wait_for("Waiting for the parent's step-2000 reference measurement", reference_measured)
    note("Starting evaluation", finish=launch(FINISH_UNIT, "finish.py",
        HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    note("Handed off; both services are running")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(WORK / "queue-failure.json", dict(type=type(error).__name__, message=str(error)))
        note("Stopped — see queue-failure.json")
        raise
