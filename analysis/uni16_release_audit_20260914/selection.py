"""Quality shortlist with an explicit preference for later close candidates.

Measurements and the original blend remain frozen in worker outputs. This
separate decision layer never uses style, lyrics or a blended quality score.
"""
from collections import Counter
from copy import deepcopy
import math
import statistics

from rank import summarize

METRICS = ("enjoyment", "production")


def later_key(row):
    # Legacy 660 is a separate run, not an extension of new step 600.
    return (row["label"].startswith("step"), row["step"])


def eligible_at(rows, anchors, tolerance):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Quality tolerance must be finite and nonnegative")
    return [r for r in rows if r["risk"] == 0 and all(
        anchors[k]["value"] - r["mean"][k] <= tolerance + 1e-12 for k in METRICS)]


def paired_quality(chosen, reference):
    paired = {c["fixture"]: c for c in reference["clips"]}
    axes = {}
    for key in METRICS:
        differences = [dict(fixture=c["fixture"], row=c["row"],
            delta=c["candidate"][key] - paired[c["fixture"]]["candidate"][key])
            for c in chosen["clips"]]
        by_prompt = {str(row): statistics.mean(d["delta"] for d in differences if d["row"] == row)
                     for row in sorted({d["row"] for d in differences})}
        axes[key] = dict(mean_delta=statistics.mean(d["delta"] for d in differences),
            clip_wins=sum(d["delta"] > 0 for d in differences), cases=len(differences),
            prompt_deltas=by_prompt, differences=differences)
    return dict(chosen=chosen["label"], reference=reference["label"], metrics=axes)


def select(candidates, protocol, policy, waveform_flags=()):
    # Reuse frozen fixture validation and technical aggregation, discard its
    # historical ordering and all composite fields before making any decision.
    validated = deepcopy(candidates)
    for candidate in validated:
        if not candidate.get("error") and any(
                not isinstance(c["candidate"].get(k), (int, float))
                or not math.isfinite(c["candidate"][k])
                for c in candidate["clips"] for k in METRICS):
            candidate["error"] = "Invalid or missing quality measurement"
    rows = summarize(validated, protocol)["ranking"]
    waveform_counts = Counter(f["checkpoint"] for f in waveform_flags)
    for row in rows:
        for key in ("score", "worst_score", "prompt_scores"):
            row.pop(key, None)
        for clip in row["clips"]:
            clip.pop("score", None)
            clip.pop("components", None)
        if waveform_counts[row["label"]] and row["risk"] < 3:
            row["flags"]["near_identical_waveforms"] = waveform_counts[row["label"]]
            row["risk"] = max(2, row["risk"])
            row["status"] = "avoid_pending_review"
        if row["clips"]:
            row["worst_quality"] = {k: min(c["candidate"][k] for c in row["clips"]) for k in METRICS}
            row["prompt_quality"] = {str(p): {k: statistics.mean(c["candidate"][k]
                for c in row["clips"] if c["row"] == p) for k in METRICS}
                for p in sorted({c["row"] for c in row["clips"]})}

    clean = [r for r in rows if r["risk"] == 0]
    anchors = {}
    if clean:
        for key in METRICS:
            best = max(clean, key=lambda r: (r["mean"][key], later_key(r)))
            anchors[key] = dict(label=best["label"], value=best["mean"][key])
    near = eligible_at(clean, anchors, policy["tolerance"])
    labels = {r["label"] for r in near}
    for row in rows:
        row["within_quality_tolerance"] = row["label"] in labels
        row["quality_shortfall"] = ({k: anchors[k]["value"] - row["mean"][k] for k in METRICS}
            if anchors and row["risk"] < 3 else None)
        row["quality_group"] = ("review_flags" if row["risk"] else "close_quality" if row["label"] in labels
                                else "outside_tolerance" if near else "quality_tradeoff")
    # The shortlisted group is ordered by the user's later-checkpoint preference.
    # Remaining clean rows are chronological, not falsely ordered by quality.
    rows.sort(key=lambda r: (r["risk"], not r["within_quality_tolerance"],
                             -int(later_key(r)[0]), -r["step"]))
    chosen = max(near, key=later_key) if near else None
    recommendation = None
    comparisons = []
    if chosen:
        recommendation = {k: chosen[k] for k in ("label", "step", "weights", "weights_sha256")}
        recommendation.update(status="provisional_quality_shortlist", close_call=len(near) > 1,
            reason="Latest new-run candidate within the quality tolerance on both measures."
                if chosen["label"].startswith("step") else "Only the legacy release meets both quality tolerances.",
            quality_shortfall=chosen["quality_shortfall"], tolerance=policy["tolerance"])
        for reference_label in dict.fromkeys(a["label"] for a in anchors.values()):
            reference = next(r for r in rows if r["label"] == reference_label)
            comparisons.append(paired_quality(chosen, reference))
    sensitivity = []
    for tolerance in policy["sensitivity_tolerances"]:
        group = sorted(eligible_at(clean, anchors, tolerance), key=later_key, reverse=True)
        sensitivity.append(dict(tolerance=tolerance, recommendation=group[0]["label"] if group else None,
                                eligible=[r["label"] for r in group]))
    if recommendation:
        recommendation["tolerance_sensitive"] = any(s["recommendation"] != chosen["label"] for s in sensitivity)
    return dict(ranking=rows, recommendation=recommendation, quality_anchors=anchors,
        eligible_checkpoints=[r["label"] for r in sorted(near, key=later_key, reverse=True)],
        quality_comparisons=comparisons, sensitivity=sensitivity,
        decision_status="selected" if chosen else "quality_tradeoff" if clean else "review_flags" if rows else "waiting")
