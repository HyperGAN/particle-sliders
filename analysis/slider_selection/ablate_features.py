#!/usr/bin/env python3
"""Development ablation against new human labels; never certifies a metric.

Leave one recipe family out. Lyrics can overlap, so this is explicitly not
the independent validation required to unlock the optimizer score.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from slider_selection.dataset import read_labels
from slider_selection.features import FEATURE_NAMES, write_json
from slider_selection.model import matrix, regression, sigmoid


def main():
    root = ROOT / "eval/slider-quality/pilot"
    dataset = json.loads((root / "features.json").read_text())
    labels, audit = read_labels(root / "listening", root / "listening/responses.jsonl")
    if not audit["listener_checks_pass"]:
        raise ValueError("Listening checks must pass before comparing feature sets")
    records = [record for record in dataset["records"] if record["id"] in labels]
    y = np.array([labels[record["id"]] for record in records])
    x = matrix(records)
    groups = [record["recipe_family"] for record in records]
    subsets = {"all": list(FEATURE_NAMES),
               "lyrics": ["lyric_recall", "lyric_precision", "phrase_accuracy"],
               "audio": [key for key in FEATURE_NAMES if not key.startswith("lyric") and key != "phrase_accuracy"],
               "legacy_geometry_level": ["lyric_recall", "song_distance", "level_excursion"]}
    report = {"scope": "Development: recipe heldout, shared lyric domains. Not terminal validation.",
              "labels_sha256": audit["labels_source_sha256"], "measurement_id": dataset["measurement_id"], "ablations": {}}
    subsets = {"constant_train_prevalence": [], **subsets,
               "concept_enjoyment": ["concept_delta", "enjoyment"]}
    for name, names in subsets.items():
        indices = [list(FEATURE_NAMES).index(key) for key in names]
        predictions, actual, fold_data = [], [], []
        for held in sorted(set(groups)):
            test = np.array([group == held for group in groups])
            train = ~test
            if len(set(y[train])) < 2:
                continue
            center = x[train][:, indices].mean(axis=0)
            scale = np.maximum(x[train][:, indices].std(axis=0), .01)
            z = (x[:, indices]-center)/scale
            if names:
                weights = regression(z[train], y[train])
                p = sigmoid(np.column_stack([np.ones(test.sum()), z[test]]) @ weights)
            else:
                p = np.full(test.sum(), y[train].mean())
            predictions.extend(p.tolist())
            actual.extend(y[test].tolist())
            fold_data.append({"recipe": held, "record_ids": [record["id"] for record, flag in zip(records, test) if flag],
                              "probabilities": p.tolist(), "labels": y[test].tolist()})
        p, target = np.array(predictions), np.array(actual)
        positive, negative = p[target == 1], p[target == 0]
        auc = float(np.mean([(a>b)+.5*(a==b) for a in positive for b in negative])) if len(positive) and len(negative) else None
        result = {"features": names, "n": len(p), "brier": float(np.mean((p-target)**2)) if len(p) else None,
                  "auc": auc, "false_accepts_at_half": int(((p>=.5)&(target==0)).sum()),
                  "usable_recalled_at_half": int(((p>=.5)&(target==1)).sum()), "folds": fold_data}
        report["ablations"][name] = result
        print(name, {key:result[key] for key in ["n","brier","auc","false_accepts_at_half","usable_recalled_at_half"]})
    write_json(root / "feature-ablation.json", report)


if __name__ == "__main__":
    main()
