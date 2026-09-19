#!/usr/bin/env python3
"""Reproducible survey audit and explicitly exploratory component probes.

This never modifies responses, pilot features, the first fitted model, or gates.
All feature choices here were made after the pilot and need fresh validation.
"""
from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from slider_selection.dataset import read_labels, load_manifest
from slider_selection.features import file_hash, lyric_features, write_json
from slider_selection.model import regression, sigmoid


def auc(values, labels):
    pos = [p for p, y in zip(values, labels) if y == 1]
    neg = [p for p, y in zip(values, labels) if y == 0]
    return float(np.mean([(a>b)+.5*(a==b) for a in pos for b in neg])) if pos and neg else None


def metrics(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return {"n": len(y), "positives": int(y.sum()), "brier": float(np.mean((p-y)**2)),
            "auc": auc(p, y), "false_accepts_at_half": int(((p>=.5)&(y==0)).sum()),
            "usable_recalled_at_half": int(((p>=.5)&(y==1)).sum())}


def predict(train, test, names, label, ridge=1.):
    labeled = [r for r in train if r["ratings"].get(label) in {"yes", "no"}]
    y = np.array([r["ratings"][label] == "yes" for r in labeled], dtype=float)
    if len(y) == 0 or len(set(y)) < 2:
        return np.full(len(test), (y.sum()+.5)/(len(y)+1))
    x = np.array([[r["features"][name] for name in names] for r in labeled])
    xx = np.array([[r["features"][name] for name in names] for r in test])
    center, scale = x.mean(0), np.maximum(x.std(0), .01)
    w = regression((x-center)/scale, y, ridge=ridge)
    return sigmoid(np.column_stack([np.ones(len(xx)), (xx-center)/scale]) @ w)


def main():
    root = ROOT / "eval/slider-quality/pilot"
    manifest = load_manifest(root / "manifest.json")
    dataset = json.loads((root / "features.json").read_text())
    acoustic = json.loads((root / "acoustic-probe.json").read_text())
    labels, audit = read_labels(root / "listening", root / "listening/responses.jsonl")
    if not audit["listener_checks_pass"]:
        raise ValueError("Listening checks failed")
    key = json.loads((root / "listening/session-key.json").read_text())
    answers = {r["trial_id"]: r for r in map(json.loads, (root / "listening/responses.jsonl").read_text().splitlines())}
    ratings = {info["record_id"]: answers[tid]["ratings"] for tid, info in key["trials"].items()
               if info["kind"] == "ladder" and tid in answers}
    audio = {}
    protocol = dataset["protocol"]["audio_provenance"]
    from slider_selection.features import digest
    for path in (ROOT / "eval/slider-quality/audio-cache").glob("*.json"):
        item = json.loads(path.read_text())
        if digest(item["provenance"]) == protocol:
            audio[item["sha256"]] = item
    measured = {r["id"]: r for r in dataset["records"]}
    rows = []
    for r in manifest["records"]:
        parts = []
        for clip in r["clips"]:
            a = audio[file_hash(Path(clip["path"]))]
            b = audio[file_hash(Path(r["baseline"]))]
            concept = clip["concept"].lower()
            chunks = [s for s in acoustic["audio"][clip["path"]]["segments"] if s["kind"] == "chunk"]
            whole = acoustic["audio"][clip["path"]]["segments"][0]
            lexical = [s for s in chunks if s["lexical_tokens"] >= 3]
            # Missing lexical evidence is explicitly low confidence, never good.
            confidence = min([s["mean_logprob"] for s in lexical], default=-5.)
            phrase = lyric_features(r["lyrics"], " ".join(s["text"] for s in chunks))
            parts.append({"concept_absolute_mean": float(np.mean([s["concept"][concept] for s in a["windows"]])),
                          "concept_mean_delta": float(np.mean([s["concept"][concept] for s in a["windows"]])
                                                      - np.mean([s["concept"][concept] for s in b["windows"]])),
                          "whole_logprob": whole["mean_logprob"] if whole["mean_logprob"] is not None else -5.,
                          "chunk_logprob": confidence, "chunk_phrase_accuracy": phrase["phrase_accuracy"],
                          "chunk_lyric_precision": phrase["precision"], "chunk_lyric_recall": phrase["recall"]})
        features = {**measured[r["id"]]["features"],
                    **{name: min(p[name] for p in parts) for name in parts[0]}}
        rows.append({"id": r["id"], "candidate": r["candidate"], "seed": r["seed"],
                     "recipe_family": r["recipe_family"], "ratings": ratings[r["id"]],
                     "joint_label": labels.get(r["id"]), "features": features,
                     "frozen_verdict": r["frozen_verdict"], "frozen_gates": r["frozen_gates"]})

    counts = {name: dict(Counter(r["ratings"][name] for r in rows)) for name in rows[0]["ratings"]}
    correlations = {}
    for label, names in {"direction": ["concept_delta", "concept_mean_delta", "concept_absolute_mean"],
                         "lyrics": ["lyric_recall", "lyric_precision", "phrase_accuracy", "whole_logprob",
                                    "chunk_logprob", "chunk_phrase_accuracy", "chunk_lyric_precision", "chunk_lyric_recall"]}.items():
        subset = [r for r in rows if r["ratings"][label] != "unsure"]
        correlations[label] = {name: auc([r["features"][name] for r in subset],
                                         [r["ratings"][label] == "yes" for r in subset]) for name in names}

    # Two independent hypotheses suggested by the component ratings: use the
    # rendered concept itself; add raw ASR confidence instead of text coverage.
    variants = {
        "absolute_concept_and_whole_asr": (["concept_absolute_mean"], ["whole_logprob"]),
        "absolute_concept_and_chunk_asr": (["concept_absolute_mean"], ["chunk_logprob"]),
        "mean_shift_and_chunk_asr": (["concept_mean_delta"], ["chunk_logprob"]),
        "absolute_concept_and_chunk_text": (["concept_absolute_mean"], ["chunk_phrase_accuracy"]),
    }
    probes = {}
    for name, (direction_features, lyric_features_) in variants.items():
        for ridge in (1., 10.):
            predictions = []
            for group in sorted({r["recipe_family"] for r in rows}):
                train = [r for r in rows if r["recipe_family"] != group]
                test = [r for r in rows if r["recipe_family"] == group and r["joint_label"] is not None]
                if not test:
                    continue
                d = predict(train, test, direction_features, "direction", ridge)
                l = predict(train, test, lyric_features_, "lyrics", ridge)
                predictions.extend({"id": r["id"], "label": r["joint_label"], "probability": float(a*b),
                                    "direction_probability": float(a), "lyrics_probability": float(b)}
                                   for r, a, b in zip(test, d, l))
            probes[f"{name}_ridge{ridge:g}"] = {**metrics([p["probability"] for p in predictions], [p["label"] for p in predictions]),
                                               "features": [direction_features, lyric_features_], "predictions": predictions}
    definite = [r for r in rows if r["joint_label"] is not None]
    gate_confusion = {"accepted_by_listener_rejected_by_gates": [r["id"] for r in definite if r["joint_label"] == 1 and r["frozen_gates"]],
                      "rejected_by_listener_passed_gates": [r["id"] for r in definite if r["joint_label"] == 0 and r["frozen_verdict"] == "PASS"]}
    report = {"scope": "Post-pilot development, not independent validation; shared lyrics across recipe folds",
              "audit": audit, "ratings": counts, "joint": dict(Counter(str(r["joint_label"]) for r in rows)),
              "asr_measurement_id": acoustic["measurement_id"], "feature_aucs_in_pilot": correlations,
              "component_probes": probes, "gate_confusion": gate_confusion, "records": rows}
    write_json(root / "survey-analysis.json", report)
    print("ratings", counts)
    print("feature AUC", correlations)
    for name, result in probes.items():
        print(name, {k: v for k, v in result.items() if k not in {"features", "predictions"}})
    print("gate_confusion", gate_confusion)


if __name__ == "__main__":
    main()
