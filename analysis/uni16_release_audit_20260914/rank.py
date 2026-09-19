"""Rank matched recordings, retaining uncertainty and technical warnings."""
from collections import Counter, defaultdict
import statistics


def summarize(candidates, protocol):
    rows = []
    fixture_set = None
    for candidate in candidates:
        if candidate.get("error"):
            rows.append(dict(label=candidate["label"], step=candidate["step"], status="audit_error",
                risk=3, score=None, flags=[candidate["error"]], clips=[], weights=candidate.get("weights")))
            continue
        clips = candidate["clips"]
        fixtures = {c["fixture"] for c in clips}
        if len(fixtures) != 4 or len(clips) != 4:
            raise ValueError("Every ranked candidate needs four unique matched cases")
        if fixture_set is None: fixture_set = fixtures
        if fixtures != fixture_set: raise ValueError("Unmatched checkpoint comparisons")
        failures = Counter(f for c in clips for f in c["technical_flags"])
        lyrics = Counter(f for c in clips for f in c["lyric_flags"])
        duplicates = []
        for row in sorted({c["row"] for c in clips}):
            same_prompt = [c for c in clips if c["row"] == row]
            if len({c["candidate"]["sha256"] for c in same_prompt}) < len(same_prompt):
                duplicates.append(row)
        if duplicates: failures["identical_output_across_seeds"] = len(duplicates)
        hard = {"near_silence", "short_output", "excess_clipping", "identical_output_across_seeds", "identical_to_off"}
        risk = 2 if hard & set(failures) else 1 if failures else 0
        scores = [c["score"] for c in clips]
        prompts = defaultdict(list)
        for c in clips: prompts[c["row"]].append(c["score"])
        rows.append(dict(label=candidate["label"], step=candidate["step"], weights=candidate["weights"],
            weights_sha256=candidate["integrity"]["weights_sha256"], integrity=candidate["integrity"],
            status="avoid_pending_review" if risk == 2 else "inspect_flagged_audio" if risk else "screen_passed",
            risk=risk, flags=dict(failures), lyric_flags=dict(lyrics), score=statistics.mean(scores),
            worst_score=min(scores), prompt_scores={str(k):statistics.mean(v) for k,v in prompts.items()},
            mean={key:statistics.mean(c["candidate"][key] for c in clips) if all(c["candidate"][key] is not None for c in clips) else None
                  for key in ("concept", "enjoyment", "production", "lyrics")},
            mean_style_gain=statistics.mean(c["candidate"]["concept"]-c["baseline"]["concept"] for c in clips),
            positive_reference_reach=sum(c["diagnostics"]["reaches_positive_description_reference"] for c in clips),
            clips=clips))
    rows.sort(key=lambda r:(r["risk"], -(r["score"] if r["score"] is not None else -1e9), r["step"]))
    available = [r for r in rows if r["score"] is not None]
    best = next((r for r in available if r["risk"] == 0), None)
    comparison = None
    if len(available) > 1:
        a, b = available[:2]
        paired = {c["fixture"]:c["score"] for c in b["clips"]}
        differences = [c["score"]-paired[c["fixture"]] for c in a["clips"]]
        prompt_diffs = [a["prompt_scores"][k]-b["prompt_scores"][k] for k in a["prompt_scores"]]
        comparison = dict(first=a["label"], second=b["label"], gap=a["score"]-b["score"],
            wins=sum(d > 0 for d in differences), cases=len(differences),
            prompt_wins=sum(d > 0 for d in prompt_diffs), prompts=len(prompt_diffs), differences=differences,
            close_call=a["risk"] == b["risk"] and (abs(a["score"]-b["score"]) < protocol["close_call_gap"]
                       or any(d <= 0 for d in prompt_diffs)))
    recommendation = None if not best else dict(label=best["label"], step=best["step"], weights=best["weights"],
        weights_sha256=best["weights_sha256"], status="provisional_shortlist",
        close_call=bool(comparison and comparison["first"] == best["label"] and comparison["close_call"]),
        reason="Highest style/quality score among checkpoints with no flagged technical regressions on these four clips.")
    return dict(ranking=rows, recommendation=recommendation, comparison=comparison,
        coverage=dict(prompts=2, seeds_per_prompt=2, clips_per_candidate=4, strength=1., seconds=20.),
        musical_quality_validated=False, diversity_status="insufficient_seeds_for_diversity_clearance")
