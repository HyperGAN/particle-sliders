"""Run approved stopping budgets using the original frozen trainer and queues.

Only orchestration changes. The original manifest, trainer, fixtures, schedule,
checkpoint signatures, sampling seeds and rendering protocol stay frozen. A
2000-update budget completes after its samples and leaves a resumable full state.
"""
from common import WORK, MODELS, PY, read, sha, verify
from budget import load_budget, milestones
from gpu_routing import load_routes
import campaign as original

BUDGET = None
ROUTED_CATALOG = None
ORIGINAL_LANE = original.lane


def routed_lane(gpu, catalog):
    # Only physical placement changes. Children still read the frozen catalog
    # and restore the original signature, optimizer, sampler and RNG states.
    return ORIGINAL_LANE(gpu, ROUTED_CATALOG)


def checkpoint(run, step):
    audit = read(run / f"audit-step{step}.json")
    weights = run / f"{run.name}_step{step}.safetensors"
    if not audit or audit["completed"] != step or not audit["finite"] or not audit["export_matches_state"]:
        raise ValueError(f"Step {step} checkpoint audit did not pass")
    if sha(weights) != audit["weights_sha256"] or sha(run / f"state-step{step}.pt") != audit["state_sha256"]:
        raise ValueError(f"Step {step} weights or full state changed")
    return weights, audit


def execute(item):
    target = BUDGET["targets"][item["id"]]
    status = read(original.job_path(item), {})
    if status.get("status") == "complete" and status["completed"] >= target:
        if sha(status["weights"]) != status["weights_sha256"]:
            raise ValueError("Completed checkpoint changed")
        return
    original.update(item, target=target, extension_requires_listening=target == 2000)
    source, initial = original.warmup(item)
    run = MODELS / item["id"] / f"{item['id']}-fresh3400"
    for step in milestones(target):
        if original.STOP.is_set():
            raise InterruptedError("Campaign stopping")
        if not read(run / f"audit-step{step}.json"):
            original.command(item, [PY, "-u", WORK / "train.py", "--id", item["id"],
                "--source-state", source, "--source-weights", initial, "--until", str(step)],
                WORK / "logs" / f"{item['id']}-to-{step}.log", f"Fresh continuations to {step}")
        weights, audit = checkpoint(run, step)
        original.update(item, completed=step, checkpoint_audit=audit)
        sampled = read(WORK / "samples" / f"{item['id']}-step{step}.json")
        if not sampled or not sampled.get("complete"):
            original.command(item, [PY, "-u", WORK / "render.py", "--id", item["id"], "--step", str(step),
                "--weights", weights, "--warmup-weights", initial],
                WORK / "logs" / f"{item['id']}-samples-{step}.log", f"Sampling and health check at {step}")
            sampled = read(WORK / "samples" / f"{item['id']}-step{step}.json")
        if not sampled or not sampled.get("complete") or sampled.get("step") != step:
            raise ValueError(f"Step {step} sample check did not complete")
        if sampled["catastrophic"]:
            original.update(item, status="needs_listening", phase=f"Audio health flag at {step}",
                error="Near-silence, severe clipping or every clip under 4 seconds. Samples retained; review before continuing.")
            return
    original.update(item, status="complete", phase=f"{target} updates; samples ready for listening",
        completed=target, target=target, weights=str(weights), weights_sha256=sha(weights),
        full_state=str(run / f"state-step{target}.pt"), full_state_sha256=audit["state_sha256"],
        extension_requires_listening=target == 2000, error=None)


def main():
    global BUDGET, ROUTED_CATALOG
    verify()
    BUDGET = load_budget()
    ROUTED_CATALOG = load_routes()
    for item in ROUTED_CATALOG:
        status = read(original.job_path(item), {})
        if status.get("status") != "complete" and status.get("gpu") != item["gpu"]:
            original.update(item, gpu=item["gpu"], previous_gpu=status.get("gpu"),
                routing_reason="Remaining jobs moved to the available GPU; saved training state retained.",
                gpu_memory_mib=None)
    # Reuse the established GPU locks, warmup/resume, shutdown and queue handling.
    original.execute = execute
    original.lane = routed_lane
    original.main()


if __name__ == "__main__":
    main()
