"""Locked listening protocol and durable state for one shared listening session."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import statistics
import sys
import threading
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
DATA = Path(os.environ.get("LOFI_TEST_DATA", str(WORK / "data")))
LOCK = threading.RLock()
STAGES = ("shortlist", "strength", "confirmation", "full")
DIMENSIONS = ("style", "enjoyment", "words")
ISSUES = ("repetition", "pitch", "artifacts", "structure", "ending", "missing_words")


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_id(step, scale=1.):
    return f"step{step}-s{scale:g}"


def label(config):
    if config["kind"] == "off":
        return "Adapter off"
    if config["kind"] == "caption":
        return "Style caption · adapter off"
    return f"Step {config['step']:,} · strength {config['scale']:g}"


def load():
    return read(DATA / "state.json")


def save(state, event, detail=None):
    state["updated"] = time.time()
    state["events"].append(dict(time=state["updated"], event=event, detail=detail))
    write(DATA / "state.json", state)


def add_stage(state, name, fixtures, configs):
    rng = random.Random(secrets.randbits(128))
    cases = []
    for fixture in fixtures:
        order = list(configs)
        rng.shuffle(order)
        takes = []
        for i, config in enumerate(order):
            identity = digest(dict(fixture=fixture, config=state["configs"][config],
                                   protocol=state["protocol_hash"]))
            prior = next((x for x in state["clips"].values() if x["identity"] == identity), None)
            if prior:
                clip_id = prior["id"]
            else:
                clip_id = secrets.token_hex(12)
                state["clips"][clip_id] = dict(id=clip_id, identity=identity, fixture=fixture,
                                               config=config, stage=name)
            takes.append(dict(id=secrets.token_hex(8), clip=clip_id, config=config,
                              letter=chr(65 + i)))
        cases.append(dict(id=secrets.token_hex(8), fixture=fixture, takes=takes))
    state["stages"][name] = dict(cases=cases, revealed=False, created=time.time())


def ready(clip_id):
    meta = read(DATA / "audio" / f"{clip_id}.json")
    if not meta:
        return None
    return meta if all((DATA / "audio" / f"{clip_id}.{ext}").is_file() for ext in ("wav", "mp3")) else None


def rated(value):
    return value is not None and all(type(value.get(k)) is int and 1 <= value[k] <= 5 for k in DIMENSIONS)


def validate_note(note):
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    if _artist_name_hit("", note):
        raise ValueError("Describe the instruments, voice or room instead of naming a performer or recording")


def stage_complete(state, name):
    stage = state["stages"].get(name)
    return bool(stage) and all(ready(t["clip"]) and rated(state["ratings"].get(t["id"]))
                               for c in stage["cases"] for t in c["takes"])


def summaries(state, name):
    groups = {}
    for case in state["stages"][name]["cases"]:
        for take in case["takes"]:
            value = state["ratings"].get(take["id"])
            if rated(value):
                groups.setdefault(take["config"], []).append(value)
    rows = []
    for config, values in groups.items():
        rows.append(dict(id=config, label=label(state["configs"][config]),
            config=state["configs"][config], count=len(values),
            **{k:statistics.mean(v[k] for v in values) for k in DIMENSIONS},
            issue_count=sum(bool(v.get("issues")) for v in values)))
    return sorted(rows, key=lambda r: (-r["enjoyment"], -r["style"], r["config"].get("step", 0)))


def public():
    with LOCK:
        state = load()
        stages = {}
        for name, stage in state["stages"].items():
            cases = []
            for n, case in enumerate(stage["cases"]):
                f = case["fixture"]
                takes = []
                for t in case["takes"]:
                    meta = ready(t["clip"])
                    value = dict(id=t["id"], letter=t["letter"], ready=bool(meta),
                        preview=f"/audio/{t['clip']}.mp3", original=f"/audio/{t['clip']}.wav",
                        rating=state["ratings"].get(t["id"], {}))
                    if meta:
                        value["duration"] = meta["duration"]
                    if stage["revealed"]:
                        value.update(label=label(state["configs"][t["config"]]), config=t["config"],
                                     diagnostics=meta)
                    takes.append(value)
                cases.append(dict(id=case["id"], number=n+1, title=f["title"],
                    variation=f["variation"], brief=f["brief"], lyrics=f["row"]["lyrics"], takes=takes))
            stages[name] = dict(cases=cases, revealed=stage["revealed"],
                complete=stage_complete(state, name),
                summary=summaries(state, name) if stage["revealed"] else [],
                rated=sum(rated(t["rating"]) for c in cases for t in c["takes"]),
                ready=sum(t["ready"] for c in cases for t in c["takes"]),
                total=sum(len(c["takes"]) for c in cases))
        return dict(stages=stages, selections=state["selections"],
            study_id=state["study_id"], updated=state["updated"],
            render=read(DATA / "worker.json", {"status":"idle"}),
            notes=state.get("notes", {}))


def rate(take_id, values):
    with LOCK:
        state = load()
        matches = [(name, t) for name, s in state["stages"].items()
                   for c in s["cases"] for t in c["takes"] if t["id"] == take_id]
        if len(matches) != 1:
            raise ValueError("Unknown recording")
        name, take = matches[0]
        if state["stages"][name]["revealed"]:
            raise ValueError("This stage is locked after reveal")
        if not ready(take["clip"]):
            raise ValueError("This recording is still rendering")
        if set(values) - {*DIMENSIONS, "issues", "note"}:
            raise ValueError("Unknown rating field")
        for key in DIMENSIONS:
            if key in values and (type(values[key]) is not int or not 1 <= values[key] <= 5):
                raise ValueError("Ratings must be 1 through 5")
        if not isinstance(values.get("note", ""), str) or len(values.get("note", "")) > 4000:
            raise ValueError("Note is too long")
        validate_note(values.get("note", ""))
        issues = values.get("issues", [])
        if not isinstance(issues, list) or any(x not in ISSUES for x in issues):
            raise ValueError("Unknown issue")
        state["ratings"][take_id] = dict(values, saved=time.time())
        save(state, "rating", take_id)


def reveal(name):
    with LOCK:
        state = load()
        if name not in state["stages"] or not stage_complete(state, name):
            raise ValueError("Rate sound, enjoyment and words for every recording before revealing")
        state["stages"][name]["revealed"] = True
        save(state, "reveal", name)


def choose(name, choices):
    with LOCK:
        state = load()
        if name not in ("shortlist", "strength") or not state["stages"].get(name, {}).get("revealed"):
            raise ValueError("Finish and reveal the preceding stage first")
        if name in state["selections"]:
            raise ValueError("These choices are already frozen")
        if not isinstance(choices, list) or len(choices) != 2 or len(set(choices)) != 2:
            raise ValueError("Select exactly two finalists")
        available = {t["config"] for c in state["stages"][name]["cases"] for t in c["takes"]}
        if any(c not in available or state["configs"][c]["kind"] != "adapter" for c in choices):
            raise ValueError("Choose two adapters")
        steps = [state["configs"][c]["step"] for c in choices]
        if len(set(steps)) != 2:
            raise ValueError("Choose one strength for each of the two different checkpoints")
        state["selections"][name] = choices
        if name == "shortlist":
            configs = [config_id(step, scale) for step in steps for scale in (.5, .75, 1.)]
            add_stage(state, "strength", state["monitoring"], ["off", "caption", *configs])
        else:
            add_stage(state, "confirmation", state["confirmation"], ["off", "caption", *choices])
            add_stage(state, "full", state["full"], ["off", "caption", *choices])
        save(state, "freeze_finalists", dict(stage=name, choices=choices))


def export():
    with LOCK:
        state = load()
        result = dict(study_id=state["study_id"], protocol_hash=state["protocol_hash"],
                      selections=state["selections"], notes=state.get("notes", {}), stages={})
        for name, stage in state["stages"].items():
            if stage["revealed"]:
                result["stages"][name] = dict(summary=summaries(state, name), cases=stage["cases"],
                    ratings={t["id"]:state["ratings"][t["id"]] for c in stage["cases"] for t in c["takes"]})
        return result
