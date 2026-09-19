"""Immutable benchmark preparation and blinded listening sessions."""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

from .features import CONCEPTS, digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ("direction", "lyrics", "quality", "preservation", "keep")


def prepare(out: Path, folders: list[Path] | None = None) -> dict:
    if (out / "manifest.json").exists():
        raise ValueError("Benchmark already exists; use a new output directory")
    labels = json.loads((ROOT / "eval/lm_score_labels.json").read_text())["labels"]
    if folders is None:
        folders = [ROOT / "eval/listen" / name for name in labels]
        folders += sorted((ROOT / "eval/listen/gan-bcap-repair").glob("*"))
    records, skipped = [], []
    for folder in sorted(set(p.resolve() for p in folders if p.is_dir())):
        source = folder / "lm_scores.json"
        if not source.exists():
            skipped.append({"folder": str(folder), "reason": "missing lm_scores.json"})
            continue
        score = json.loads(source.read_text())
        if not score.get("lyrics") or score.get("seed") is None:
            skipped.append({"folder": str(folder), "reason": "missing lyrics or seed"})
            continue
        channels = {float(c["scale"]): c for c in score["channels"].values()}
        required = [-1.0, 1.0] if score["kind"] == "bipolar" else [1.0]
        if any(scale not in channels for scale in [0.0] + required):
            skipped.append({"folder": str(folder), "reason": "no exact zero/unit setting on every pole"})
            continue
        concepts = {scale: score["plus_label"] if scale > 0 else score["minus_label"] for scale in required}
        if any(label.lower() not in CONCEPTS for label in concepts.values()):
            skipped.append({"folder": str(folder), "reason": "unsupported frozen concept description"})
            continue
        paths = {scale: folder / channels[scale]["file"] for scale in [0.0] + required}
        if not all(path.is_file() for path in paths.values()):
            skipped.append({"folder": str(folder), "reason": "missing audio"})
            continue
        rel = str(folder.relative_to(ROOT / "eval/listen"))
        hashes = {scale: file_hash(path) for scale, path in paths.items()}
        lyric_family = digest(re.findall(r"[a-z]+", score["lyrics"].lower()))
        # Historical caption provenance is incomplete. Conservatively group all
        # examples with the same lyrics; never pretend each seed is a new prompt.
        recipe = rel.split("/")[0]
        if recipe == "gan-bcap-repair":
            # "repaired-tx-smoke" contains the substring "paired" too.
            recipe += "/smoke" if folder.name.startswith("repaired-tx-smoke") else "/paired"
        record = {"id": digest([rel, hashes])[:16], "folder": rel,
                  "candidate": re.sub(r"(?:-s\d+|-seed\d+)$", "", rel),
                  "recipe_family": recipe, "prompt_family": lyric_family,
                  "lyric_family": lyric_family, "seed": score["seed"],
                  "lyrics": score["lyrics"], "requested_s": score["requested_s"],
                  "baseline": str(paths[0.0]), "audio_hashes": list(hashes.values()),
                  "frozen_verdict": score["verdict"], "frozen_gates": score["gates_fired"],
                  "historical_folder_label": labels.get(rel, {}).get("verdict"),
                  "score_source": str(source), "score_sha256": file_hash(source),
                  "clips": [{"path": str(paths[scale]), "scale": scale,
                             "concept": concepts[scale], "transcript": channels[scale]["text"],
                             "same_song": channels[scale]["same_song"]} for scale in required]}
        evaluation = folder / "evaluation.json"
        if evaluation.is_file():
            declared = json.loads(evaluation.read_text())
            for key in ("candidate", "recipe_family", "prompt_family"):
                if not isinstance(declared.get(key), str) or not declared[key]:
                    raise ValueError(f"Invalid explicit benchmark grouping: {evaluation}")
                record[key] = {"gan-smoke": "gan-bcap-repair/smoke", "gan-paired": "gan-bcap-repair/paired"}.get(declared[key], declared[key]) if key == "recipe_family" else declared[key]
            record["evaluation_sha256"] = file_hash(evaluation)
        records.append(record)
    if not records:
        raise ValueError("No complete supported ladders")
    manifest = {"schema": 1, "scope": "LM unit endpoints; intermediate control and long-form untested",
                "records": records, "skipped": skipped}
    manifest["id"] = digest(manifest)
    write_json(out / "manifest.json", manifest)
    return manifest


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value["id"]
    if digest({key: item for key, item in value.items() if key != "id"}) != expected:
        raise ValueError("Manifest changed since it was frozen")
    return value


def build_session(manifest: dict, out: Path, seed: int = 17, max_preferences: int = 2) -> dict:
    """Private key stays outside the public directory; no metric-derived labels."""
    if (out / "session-key.json").exists():
        raise ValueError("Session already exists; do not overwrite collected answers")
    public = out / "public"
    public.mkdir(parents=True, exist_ok=True)
    (public / "clips").mkdir(exist_ok=True)
    rng = random.Random(seed)
    session_id = digest([manifest["id"], seed, max_preferences])[:20]
    trials, key = [], {}

    def copy_audio(path: str) -> str:
        source = Path(path)
        name = digest([session_id, file_hash(source)])[:24] + ".wav"
        target = public / "clips" / name
        if not target.exists():
            shutil.copyfile(source, target)
        return "clips/" + name

    def add(record: dict, kind: str = "ladder", repeat_of: str | None = None,
            clips: list[dict] | None = None) -> str:
        trial_id = digest([session_id, len(trials)])[:16]
        presented = clips if clips is not None else record["clips"]
        trials.append({"id": trial_id, "kind": "ladder", "lyrics": record["lyrics"],
                       "original": copy_audio(record["baseline"]),
                       "settings": [{"url": copy_audio(clip["path"]), "scale": clip["scale"],
                                     "concept": clip["concept"],
                                     "description": CONCEPTS[clip["concept"].lower()][0]}
                                    for clip in presented]})
        key[trial_id] = {"record_id": record["id"], "kind": kind, "repeat_of": repeat_of}
        return trial_id

    records = manifest["records"]
    for record in records:
        actual = {file_hash(Path(path)) for path in [record["baseline"]] + [clip["path"] for clip in record["clips"]]}
        if "audio_hashes" in record and actual != set(record["audio_hashes"]):
            raise ValueError("Source audio changed before the listening session was built")
    originals = {record["id"]: add(record) for record in records}
    # Equal-audio control measures whether expectation alone earns a good label.
    control = records[0]
    add(control, "noop_control", clips=[{**clip, "path": control["baseline"]} for clip in control["clips"]])
    # Real silence, not a metadata-only pretend attack.
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(control["baseline"], always_2d=True)
    silence = out / "silence.wav"
    sf.write(silence, np.zeros_like(data), sr, subtype="PCM_16")
    add(control, "silence_control", clips=[{**clip, "path": str(silence)} for clip in control["clips"]])
    for record in rng.sample(records, min(2, len(records))):
        add(record, "repeat", repeat_of=originals[record["id"]])
    # Same source audio, requested concept and operating settings; no metric
    # values influence which pair gets judged. Prefer the known recent contrast
    # in the pilot, while allowing new checkpoint families in later suites.
    pairs = []
    for i, record in enumerate(records):
        for rival in records[i+1:]:
            if (record.get("candidate", record["id"]) == rival.get("candidate", rival["id"])
                    or record["seed"] != rival["seed"]
                    or [(c["concept"].lower(), c["scale"]) for c in record["clips"]]
                    != [(c["concept"].lower(), c["scale"]) for c in rival["clips"]]
                    or file_hash(Path(record["baseline"])) != file_hash(Path(rival["baseline"]))):
                continue
            pairs.append((record, rival))
    rng.shuffle(pairs)
    pairs.sort(key=lambda pair: not (any("repaired-tx-smoke" in r["folder"] for r in pair)
                                    and any("gender-paired-mined" in r["folder"] for r in pair)))
    for record, rival in pairs[:max_preferences]:
        # Preference UI compares a single positive endpoint. Full bipolar
        # usability is still labeled on the complete ladder trials above.
        a, b = rng.sample([record, rival], 2)
        a_clip = next(clip for clip in a["clips"] if clip["scale"] > 0)
        b_clip = next(clip for clip in b["clips"] if clip["scale"] > 0)
        tid = digest([session_id, len(trials)])[:16]
        trials.append({"id": tid, "kind": "preference", "lyrics": record["lyrics"],
                       "description": CONCEPTS[a_clip["concept"].lower()][0],
                       "original": copy_audio(record["baseline"]),
                       "a": copy_audio(a_clip["path"]), "b": copy_audio(b_clip["path"])})
        key[tid] = {"kind": "preference", "a": a["id"], "b": b["id"]}
    rng.shuffle(trials)
    public_session = {"id": session_id, "trials": trials}
    write_json(public / "session.json", public_session)
    private = {"id": session_id, "manifest_id": manifest["id"], "trials": key,
               "public_sha256": file_hash(public / "session.json")}
    write_json(out / "session-key.json", private)
    shutil.copyfile(Path(__file__).with_name("listen.html"), public / "index.html")
    return public_session


def validate_response(response: dict, session: dict) -> None:
    if response.get("session_id") != session["id"]:
        raise ValueError("Response belongs to a different session")
    tid = response.get("trial_id")
    if tid not in session["trials"]:
        raise ValueError("Unknown trial")
    trial = session["trials"][tid]
    if trial["kind"] == "preference":
        if response.get("preference") not in {"a", "b", "tie", "neither", "unsure"}:
            raise ValueError("Missing preference")
    else:
        ratings = response.get("ratings", {})
        if set(ratings) != set(RATINGS) or any(rating not in {"yes", "no", "unsure"} for rating in ratings.values()):
            raise ValueError("Answer every rating or mark unsure")
    if not isinstance(response.get("played"), list) or not response["played"]:
        raise ValueError("Listen to the audio before answering")


def read_labels(session_path: Path, responses: Path) -> tuple[dict[str, int], dict]:
    session = json.loads((session_path / "session-key.json").read_text())
    public_path = session_path / "public/session.json"
    if file_hash(public_path) != session["public_sha256"]:
        raise ValueError("Public trial mapping changed")
    public = {trial["id"]: trial for trial in json.loads(public_path.read_text())["trials"]}
    latest = {}
    for line in responses.read_text().splitlines():
        if line.strip():
            response = json.loads(line)
            validate_response(response, session)
            trial = public[response["trial_id"]]
            needed = {trial["original"]}
            needed.update([trial["a"], trial["b"]] if trial["kind"] == "preference"
                          else [clip["url"] for clip in trial["settings"]])
            if not needed.issubset(set(response["played"])):
                raise ValueError("Incomplete audio playback in response")
            latest[response["trial_id"]] = response

    def outcome(response):
        values = list(response["ratings"].values())
        if "no" in values:
            return 0
        return None if "unsure" in values else 1

    labels, controls, repeats, preferences = {}, {}, [], []
    for tid, response in latest.items():
        key = session["trials"][tid]
        if key["kind"] == "preference":
            preferences.append({**key, "choice": response["preference"]})
        elif key["kind"] == "ladder":
            label = outcome(response)
            if label is not None:
                labels[key["record_id"]] = label
        elif key["kind"].endswith("control"):
            controls[key["kind"]] = (response["ratings"]["direction"] == "no"
                                      and outcome(response) == 0)
        elif key["kind"] == "repeat" and key["repeat_of"] in latest:
            a, b = outcome(response), outcome(latest[key["repeat_of"]])
            repeats.append(a is not None and b is not None and a == b)
    quality_ok = (set(controls) == {"noop_control", "silence_control"}
                  and all(controls.values()) and len(repeats) >= 2 and all(repeats))
    return labels, {"answered": len(latest), "total": len(session["trials"]),
                    "controls": controls, "repeat_agreement": repeats,
                    "listener_checks_pass": quality_ok, "preferences": preferences,
                    "manifest_id": session["manifest_id"],
                    "labels_source_sha256": file_hash(responses)}
