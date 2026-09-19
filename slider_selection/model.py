"""Regularized probability model and conservative experiment selection.

Historical PASS/FAIL labels never enter fitting. Only responses to the current
blind study can train this model. Predictions remain exploratory until a
separate, disjoint listening suite passes the predeclared validation rules.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from .features import FEATURE_NAMES, digest


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def regression(x, y, ridge=1.0):
    """Penalized logistic Newton solve, including a lightly regularized intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.01] + [ridge] * x.shape[1])
    weights = np.zeros(design.shape[1])
    for _ in range(60):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (probability - y) + penalty @ weights
        hessian = design.T @ ((probability * (1 - probability))[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        # Backtracking protects heavily separated small calibration sets.
        loss = np.logaddexp(0, design @ weights).sum() - y @ (design @ weights) + .5 * weights @ penalty @ weights
        factor = 1.0
        while factor > 1e-6:
            proposed = weights - factor * step
            next_loss = np.logaddexp(0, design @ proposed).sum() - y @ (design @ proposed) + .5 * proposed @ penalty @ proposed
            if next_loss <= loss:
                weights = proposed
                break
            factor *= .5
        if np.linalg.norm(factor * step) < 1e-7:
            break
    return weights


def matrix(records):
    result = np.array([[record["features"][name] for name in FEATURE_NAMES] for record in records], dtype=float)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Missing or nonfinite feature measurements")
    return result


def provenance(records):
    return {name: sorted({value for record in records for value in
                         (record[name] if name == "audio_hashes" else [record[name]])})
            for name in ("recipe_family", "prompt_family", "lyric_family", "audio_hashes")}


def overlap(record, seen):
    def recipe_key(value):
        # The immutable pilot grouped both GAN checkpoints under /paired
        # because "repaired" contains "paired". Preserve that conservative
        # common family when checking newer, explicitly named benchmark rows.
        return "gan-bcap-repair" if value.startswith("gan-bcap-repair/") else value

    shared = ["recipe_family"] if recipe_key(record["recipe_family"]) in {
        recipe_key(value) for value in seen["recipe_family"]} else []
    return shared + [name for name in ("prompt_family", "lyric_family") if record[name] in seen[name]] + (
        ["audio_hashes"] if set(record["audio_hashes"]) & set(seen["audio_hashes"]) else [])


def fit(dataset: dict, labels: dict[str, int], audit: dict, draws=100) -> dict:
    if audit["manifest_id"] != dataset["manifest_id"]:
        raise ValueError("Labels and features belong to different manifests")
    if not audit["listener_checks_pass"]:
        raise ValueError("Complete the two controls and repeated trials before fitting")
    records = [record for record in dataset["records"] if record["id"] in labels]
    y = np.array([labels[record["id"]] for record in records], dtype=float)
    if len(records) < 8 or min(int(y.sum()), int(len(y) - y.sum())) < 3:
        raise ValueError("Need at least eight labeled ladders, including three usable and three rejected")
    x = matrix(records)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
    z = (x - center) / scale
    weights = regression(z, y)
    rng, ensemble = np.random.default_rng(391), []
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record["recipe_family"]].append(i)
    keys = sorted(groups)
    for _ in range(draws):
        selected = np.concatenate([groups[key] for key in rng.choice(keys, len(keys), replace=True)])
        ensemble.append(regression(z[selected], y[selected]).tolist())
    model = {"schema": 1, "status": "exploratory", "feature_names": list(FEATURE_NAMES),
             "measurement_id": dataset["measurement_id"], "center": center.tolist(),
             "scale": scale.tolist(), "weights": weights.tolist(), "ensemble": ensemble,
             "training_manifest": dataset["manifest_id"], "training_records": [record["id"] for record in records],
             "seen": provenance(records), "listener_audit": audit,
             "coverage": {"ladders": len(records), "usable": int(y.sum()),
                          "recipe_families": len(keys),
                          "prompt_families": len({record["prompt_family"] for record in records})},
             "note": "Raw logistic probabilities; calibration and selection validity require an independent suite."}
    model["fit_id"] = digest(model)
    return model


def predictions(model, records, all_models=False):
    if model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Feature schema mismatch; refit rather than silently reorder")
    z = (matrix(records) - np.array(model["center"])) / np.array(model["scale"])
    design = np.column_stack([np.ones(len(z)), z])
    coefficients = np.array(model["ensemble"] if all_models else [model["weights"]])
    return sigmoid(design @ coefficients.T)


def upper_wilson(k, n, z=1.2815515655446004):
    if n == 0:
        return 1.0
    p = k / n
    return (p + z*z/(2*n) + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def validate(model: dict, dataset: dict, labels: dict, audit: dict) -> dict:
    """No fitting here; fail closed on shared prompts, recipes, lyrics or audio."""
    if dataset["measurement_id"] != model["measurement_id"]:
        raise ValueError("Measurement protocol mismatch")
    if audit["manifest_id"] != dataset["manifest_id"]:
        raise ValueError("Labels belong to a different benchmark")
    records = [record for record in dataset["records"] if record["id"] in labels]
    leaked = {record["id"]: overlap(record, model["seen"]) for record in records}
    leaked = {key: value for key, value in leaked.items() if value}
    if leaked:
        return {"status": "rejected_overlap", "overlap": leaked, "passed": False}
    if not records:
        return {"status": "no_labels", "passed": False}
    y = np.array([labels[record["id"]] for record in records])
    raw_p = predictions(model, records)[:, 0]
    # Validate the gated selector, not just the probability branch. The pilot
    # exposed accepted audio rejected by old gates; a good raw classifier must
    # not certify a selector that can never return those accepted examples.
    p = np.array([0. if record["frozen_gates"] or record["frozen_verdict"] == "UNSCOREABLE" else value
                  for record, value in zip(records, raw_p)])
    negatives, positives = int((y == 0).sum()), int(y.sum())
    false_accepts = int(((p >= .5) & (y == 0)).sum())
    true_accepts = int(((p >= .5) & (y == 1)).sum())
    probabilities = dict(zip([record["id"] for record in records], p))
    choices = [item for item in audit["preferences"] if item["choice"] in {"a", "b"}
               and item["a"] in probabilities and item["b"] in probabilities]
    wins = sum((probabilities[item["a"]] > probabilities[item["b"]]) == (item["choice"] == "a")
               and probabilities[item["a"]] != probabilities[item["b"]] for item in choices)
    checks = {"listener_checks": audit["listener_checks_pass"],
              "enough_labeled_ladders": len(records) >= 30 and negatives >= 10 and positives >= 10,
              "prompt_diversity": len({record["prompt_family"] for record in records}) >= 3,
              "recipe_diversity": len({record["recipe_family"] for record in records}) >= 3,
              "false_accept_bound": upper_wilson(false_accepts, negatives) <= .20,
              "usable_recall": positives > 0 and true_accepts / positives >= .50,
              "brier_beats_constant": float(np.mean((p-y)**2)) < float(np.mean((y.mean()-y)**2)),
              "preference_agreement": len(choices) >= 10 and wins / max(1, len(choices)) >= .70}
    return {"status": "validated" if all(checks.values()) else "insufficient_or_failed",
            "passed": all(checks.values()), "checks": checks,
            "n": len(records), "brier": float(np.mean((p-y)**2)),
            "raw_probability_brier": float(np.mean((raw_p-y)**2)),
            "usable_accepted": true_accepts, "usable_examples": positives,
            "false_accepts": false_accepts, "rejected_examples": negatives,
            "false_accept_upper90": upper_wilson(false_accepts, negatives),
            "preference_wins": int(wins), "preference_pairs": len(choices),
            "manifest_id": dataset["manifest_id"], "labels_source": audit["labels_source_sha256"],
            "validation_protocol": "gated-selector-v2",
            "note": "Development criteria frozen before a new validation suite; not proof under unlimited optimization."}


def score(model: dict, dataset: dict, draws=1000) -> dict:
    if dataset["measurement_id"] != model["measurement_id"]:
        raise ValueError("Measurement protocol mismatch")
    by_candidate = defaultdict(list)
    for record in dataset["records"]:
        by_candidate[record["candidate"]].append(record)
    required_slots = {(record["prompt_family"], record["seed"]) for record in dataset["records"]}
    result = []
    for candidate, records in sorted(by_candidate.items()):
        probs = predictions(model, records, all_models=True)
        groups = defaultdict(list)
        for i, record in enumerate(records):
            groups[record["prompt_family"]].append(i)
        keys = sorted(groups)
        # Fixed shared resampling seed helps paired candidates use the same draws.
        rng, samples = np.random.default_rng(218), []
        for _ in range(draws):
            model_index = rng.integers(probs.shape[1])
            row_scores = []
            for key in rng.choice(keys, len(keys), replace=True):
                selected = rng.choice(groups[key], len(groups[key]), replace=True)
                row_scores.append(float(probs[selected, model_index].mean()))
            samples.append(float(np.mean(row_scores)))
        lower = float(np.quantile(samples, .1) * 100)
        failures = sorted({gate for record in records for gate in record["frozen_gates"]})
        reason = []
        slots = [(record["prompt_family"], record["seed"]) for record in records]
        if len(slots) != len(set(slots)): reason.append("duplicate_prompt_seed")
        if set(slots) != required_slots: reason.append("unequal_benchmark_coverage")
        if model["status"] != "validated": reason.append("model_not_validated")
        if len(groups) < 3: reason.append("fewer_than_three_prompt_families")
        if any(len({records[i]["seed"] for i in group}) < 3 for group in groups.values()):
            reason.append("fewer_than_three_seeds_per_prompt")
        if any(overlap(record, model["seen"]) for record in records): reason.append("overlap_with_calibration")
        feature_array = matrix(records)
        z = np.abs((feature_array - np.array(model["center"])) / np.array(model["scale"]))
        if np.any(z > 5): reason.append("features_outside_calibration_range")
        if any(record["frozen_verdict"] == "UNSCOREABLE" for record in records): reason.append("incomplete_frozen_measurements")
        numeric = (-1 - len(failures)/(1+len(failures))) if failures else lower
        result.append({"candidate": candidate, "optimizer_score": None if reason else numeric,
                       "preview_lower90": lower, "frozen_failures": failures,
                       "status": "UNSCOREABLE" if reason else ("FAIL" if failures else "SCORED"),
                       "reasons": reason, "ladders": len(records), "prompt_families": len(groups)})
    return {"model_fit_id": model["fit_id"], "manifest_id": dataset["manifest_id"], "candidates": result}
