"""User-approved stopping budgets, separate from immutable training state."""
from common import WORK, read, sha


def load_budget():
    budget = read(WORK / "budget.json")
    if not budget or budget.get("version") != 1:
        raise ValueError("Missing or invalid approved stopping budget")
    if budget["training_manifest_sha256"] != sha(WORK / "manifest.json"):
        raise ValueError("Stopping budget belongs to a different training campaign")
    ids = {i["id"] for i in read(WORK / "catalog.json")["sliders"]}
    targets = budget["targets"]
    if set(targets) != ids or any(type(t) is not int or t not in (2000, 3400) for t in targets.values()):
        raise ValueError("Every slider needs an explicit 2000 or 3400 stopping budget")
    return budget


def milestones(target):
    return [step for step in (1000, 2000, 3000, 3400) if step <= target]
