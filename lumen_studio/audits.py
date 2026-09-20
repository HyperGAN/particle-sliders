"""Development-only selection and a single immutable held-out audit campaign."""
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

from .contracts import VARIATIONS, atomic_json, digest, file_hash
from .dataset import compile_manifest, compatible_manifest_hashes
from .sampling import select_checkpoint, selection_rows


def all_images(store):
    with store.connect() as db:
        return [store.unpack(r) for r in db.execute("SELECT * FROM images ORDER BY created")]


def major_regression(review):
    return (review.get("major_regression", False)
            or any(not review.get(k, True) for k in ("character", "outfit", "pose", "composition", "medium", "quality"))
            or review.get("unwanted_objects", False) or review.get("recurring_features", False))


def active_checkpoint(checkpoint, variation):
    # Paths may have moved between hosts. An archived retry can share its
    # targets/manifest with the current seed but cannot supply its reviews.
    parts = Path(checkpoint["path"]).parts
    return len(parts) >= 3 and parts[-3:-1] == ("runs", variation)


def selection_report(store, root, variation):
    if variation not in VARIATIONS:
        raise ValueError("Unknown variation")
    reviews = {r["image_id"]: r for r in store.reviews()}
    images = all_images(store)
    train_hashes = compatible_manifest_hashes(root, "train", variation)
    dev_hashes = compatible_manifest_hashes(root, "dev", variation)
    expected = {r["id"] for r in selection_rows(variation)}
    candidates, incomplete = [], []
    for checkpoint in store.catalog():
        step = checkpoint["metadata"]["step"]
        if (checkpoint["variation"] != variation or step == 0 or step % 400
                or not active_checkpoint(checkpoint, variation)):
            continue
        if checkpoint["metadata"].get("manifest_sha256") not in train_hashes:
            continue
        cases = {}
        for image in images:
            m = image["metadata"]
            if (m.get("purpose") == "development" and m.get("energy") == 1
                    and m.get("sampling_step") == step and m.get("case") in expected
                    and m.get("checkpoints") == {variation: checkpoint["sha256"]}
                    and m.get("manifest_sha256") in dev_hashes
                    and m.get("width") == m.get("height") == 768 and m.get("seed") == 29001):
                cases[m["case"]] = image
        rated = [reviews[i["id"]] for i in cases.values() if i["id"] in reviews
                 and reviews[i["id"]]["data"].get("atmosphere") in ("win", "tie", "loss")
                 and reviews[i["id"]]["data"].get("blind")]
        probe = Path(root) / "runs" / variation / f"probe-{step:06}.json"
        if len(rated) != 8 or set(cases) != expected or not probe.exists():
            incomplete.append(dict(step=step, sha256=checkpoint["sha256"], rated=len(rated), required=8))
            continue
        residual = json.loads(probe.read_text())["full_strength_raw_R"]
        if not math.isfinite(residual):
            raise ValueError("Nonfinite development residual")
        candidates.append(dict(step=step, sha256=checkpoint["sha256"], cases=8,
            atmosphere_wins=sum(r["data"]["atmosphere"] == "win" for r in rated),
            major_regressions=sum(major_regression(r["data"]) for r in rated), residual=residual,
            review_ids=sorted(r["id"] for r in rated)))
    return dict(variation=variation, candidates=candidates, incomplete=incomplete,
                selected=select_checkpoint(candidates))


def enqueue_final(store, identity, root):
    from .api import Generation, generation_payloads
    root = Path(root)
    path = root / "audits/final.json"
    if path.exists():
        campaign = json.loads(path.read_text())
        if digest({k: v for k, v in campaign.items() if k != "sha256"}) != campaign["sha256"]:
            raise ValueError("Final audit manifest changed")
        if campaign["model"] != identity:
            raise ValueError("Final audit model changed")
    else:
        selections = {v: selection_report(store, root, v) for v in VARIATIONS}
        if any(s["selected"] is None for s in selections.values()):
            raise ValueError("Every variation needs a qualifying blind development selection")
        if any({c["step"] for c in s["candidates"]} != {400, 800, 1200, 1600} for s in selections.values()):
            raise ValueError("Review all four 400-update checkpoints before opening the final test")
        # Test prompts are expanded only after all selections are frozen.
        manifest = compile_manifest("test")
        campaign = dict(model=identity, selections=selections, manifest=manifest,
                        width=768, height=768, steps=10, energies=[0, .25, .5, 1.])
        campaign["sha256"] = digest(campaign)
        atomic_json(path, campaign)
    group = "final-" + campaign["sha256"]
    with store.connect() as db:
        existing = [r[0] for r in db.execute("SELECT id FROM jobs WHERE group_id=? ORDER BY priority", (group,))]
    if existing:
        return dict(ids=existing, group=group, resumed=True)
    checkpoints = {v: s["selected"]["sha256"] for v, s in campaign["selections"].items()}
    payloads = []
    for row in campaign["manifest"]["rows"]:
        variation = row["variation"]
        for seed in row["seeds"]:
            req = Generation(prompt=row["neutral"], seed=seed, mix={variation: 1.},
                             checkpoints={variation: checkpoints[variation]})
            for p in generation_payloads(req, store, identity, campaign["energies"]):
                p.update(purpose="final_test", audit=campaign["sha256"], case=row["id"],
                    character=row["character"], definition=row["definition"], bare=row["bare"], family=variation)
                payloads.append(p)
    # Eight untouched characters, one fixed fresh seed, all equal pairwise and
    # three-way mixtures. Each family gets a matched Off reference.
    mixes = [list(p) for p in itertools.combinations(VARIATIONS, 2)] + [list(VARIATIONS)]
    for row in campaign["manifest"]["rows"][:8]:
        for names in mixes:
            req = Generation(prompt=row["neutral"], seed=row["seeds"][0],
                mix={v: 1. for v in names}, checkpoints={v: checkpoints[v] for v in names})
            for p in generation_payloads(req, store, identity, campaign["energies"]):
                p.update(purpose="mixture_audit", audit=campaign["sha256"], case=row["id"],
                    character=row["character"], family="+".join(names), bare=row["bare"])
                payloads.append(p)
    return store.enqueue(payloads, group=group)


def audit_report(store, root):
    campaign = json.loads((Path(root) / "audits/final.json").read_text())
    images = [i for i in all_images(store) if i["metadata"].get("audit") == campaign["sha256"]]
    reviews = {r["image_id"]: r["data"] for r in store.reviews()}
    groups, families = defaultdict(list), defaultdict(list)
    for image in images:
        m = image["metadata"]
        groups[(m["family"], m["case"], m["seed"])].append(image)
        families[m["family"]].append(image)
    ordering, unrated = [], []
    for key, rows in groups.items():
        rows.sort(key=lambda i: i["metadata"]["energy"])
        ratings = [reviews.get(i["id"], {}).get("atmosphere_strength") for i in rows]
        if len(rows) != 4 or any(r is None for r in ratings):
            unrated.append(list(key))
        elif any(b < a for a, b in zip(ratings, ratings[1:])):
            ordering.append(dict(family=key[0], case=key[1], seed=key[2], ratings=ratings))
    summaries = {}
    for family, rows in families.items():
        full = [i for i in rows if i["metadata"]["energy"] == 1]
        rated = [reviews[i["id"]] for i in full if i["id"] in reviews
                 and reviews[i["id"]].get("atmosphere") in ("win", "tie", "loss")]
        summaries[family] = dict(completed_images=len(rows), full_strength_cases=len(full), rated=len(rated),
            atmosphere_wins=sum(r["atmosphere"] == "win" for r in rated),
            major_regressions=sum(major_regression(r) for r in rated))
    report = dict(audit=campaign["sha256"], expected_images=512, completed_images=len(images),
        families=summaries, energy_ordering_failures=ordering, energy_ordering_unrated=unrated,
        mixture_failures=[dict(image=i["id"], family=i["metadata"]["family"], energy=i["metadata"]["energy"])
            for i in images if i["metadata"]["purpose"] == "mixture_audit"
            and i["id"] in reviews and (major_regression(reviews[i["id"]])
                                        or reviews[i["id"]].get("atmosphere") == "loss")])
    atomic_json(Path(root) / "audits/report.json", report)
    return report


def qualify_pilot(store, root, variation):
    if variation not in VARIATIONS:
        raise ValueError("Unknown variation")
    directory = Path(root) / "runs" / variation
    run = json.loads((directory / "run.json").read_text())
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    updates = [r for r in updates if r["step"] <= 200]
    finite = len(updates) == 200 and [r["step"] for r in updates] == list(range(1, 201)) and all(
        math.isfinite(r[key]) for r in updates for key in ("d_adv", "g_adv", "vic", "d_penalty", "g_grad_norm", "d_grad_norm"))
    probes = [json.loads(p.read_text()) for p in directory.glob("probe-*.json") if int(p.stem.split('-')[-1]) <= 200]
    baseline = next((p["full_strength_raw_R"] for p in probes if p["step"] == 0), None)
    residuals = [p["full_strength_raw_R"] for p in probes if p["step"] > 0]
    improved = bool(baseline is not None and residuals and all(math.isfinite(r) for r in residuals)
                    and min(residuals) < baseline * (1 - 1e-6))
    check_path = directory / "resume-check.json"
    check = json.loads(check_path.read_text()) if check_path.exists() else {}
    state = directory / "state-000200.pt"
    checkpoint_sha = file_hash(state) if state.exists() else None
    resumed = (check.get("passed") is True and check.get("step") == 200 and check.get("run") == run
               and checkpoint_sha is not None and check.get("checkpoint_sha256") == checkpoint_sha
               and check.get("studio_prediction_max_abs_by_timestep") == [0.] * 10)
    reviews = {r["image_id"]: r for r in store.reviews()}
    catalog = {c["sha256"] for c in store.catalog() if c["variation"] == variation
               and 100 <= c["metadata"]["step"] <= 200
               and active_checkpoint(c, variation)
               and c["metadata"].get("normalization", {}).get("cache_fingerprint") == run["cache"]}
    latest = {}
    for image in all_images(store):
        m = image["metadata"]
        if (m.get("purpose") == "development" and m.get("energy") == 1
                and m.get("checkpoints", {}).get(variation) in catalog and m.get("sampling_step", 0) <= 200):
            latest[m["character"]] = image
    usable = len(latest) == 4 and all(i["id"] in reviews
        and reviews[i["id"]]["data"].get("atmosphere") in ("win", "tie", "loss")
        and not major_regression(reviews[i["id"]]["data"]) for i in latest.values())
    checks = dict(finite_gradients=finite, improved_held_out_residual=improved,
                  exact_resume_and_studio_parity=resumed, usable_character_renders=usable)
    result = dict(passed=all(checks.values()), checks=checks, run_sha256=digest(run),
        checkpoint_sha256=checkpoint_sha, normalization_sha256=file_hash(directory / "normalization.pt"),
        baseline_R=baseline, best_R=min(residuals) if residuals else None,
        review_ids=[reviews[i["id"]]["id"] for i in latest.values() if i["id"] in reviews])
    atomic_json(directory / "pilot-qualification.json", result)
    return result


def require_pilot_qualification(directory):
    directory = Path(directory)
    path = directory / "pilot-qualification.json"
    if not path.exists():
        raise ValueError("The 200-update pilot must pass before continuing")
    result = json.loads(path.read_text())
    if (not result.get("passed") or result.get("run_sha256") != digest(json.loads((directory / "run.json").read_text()))
            or result.get("checkpoint_sha256") != file_hash(directory / "state-000200.pt")
            or result.get("normalization_sha256") != file_hash(directory / "normalization.pt")):
        raise ValueError("Pilot qualification failed or its evidence changed")
