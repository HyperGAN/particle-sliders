"""Score recorded visual observations with Jev; never claims direct image review.

Consumes observations.json next to immutable anima_review_grid.py directories.
Keeps requests, raw answers and image provenance outside git. Does not mutate the
Studio review database, checkpoint selection, prompts or training formulation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
CHOICES = {
    "A": "Image A is better supported by the observations.",
    "B": "Image B is better supported by the observations.",
    "tie": "No meaningful preference is supported; both are similarly effective.",
    "insufficient": "The observations do not support this judgment.",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Immutable review file changed; use a new output directory: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def build_request(rows, variation, observer, limitations):
    cases = {str(r["case"]): r["observations"] for r in rows}
    if len(cases) != len(rows) or not cases or any(not x.strip() for x in cases.values()):
        raise ValueError("Unique cases with explicit observations are required")
    questions = {}
    for case in cases:
        evidence = f"Using only `cases.{case}` and `goal`, "
        questions[f"{case}_appeal"] = dict(type="choice", criteria=CHOICES,
            instructions=evidence + "which image would be a more compelling, visually coherent illustration? "
            "Consider dramatic contrast, color and atmosphere, while retaining readable subjects and intentional-looking rendering. "
            "Do not assume darker is always better or invent details absent from the observations.")
        questions[f"{case}_atmosphere"] = dict(type="choice", criteria=CHOICES,
            instructions=evidence + "which image more strongly conveys the requested lighting atmosphere? "
            "Judge atmosphere separately from subject preservation or overall appeal.")
        questions[f"{case}_drift"] = dict(type="score",
            instructions=evidence + "how substantial are the described differences unrelated to lighting or atmosphere "
            "between A and B? Consider character appearance, outfit, pose, framing, scene layout and illustration medium. "
            "Color/exposure shifts caused by illumination alone are not drift.",
            criteria=["Same character, outfit, pose, composition and medium; only lighting differences are described.",
                      "Small line, fold or facial-detail changes; same recognizable subject, outfit, framing and medium.",
                      "Noticeable changes in outfit details, face, expression, crop or background layout; main subject and medium remain recognizable.",
                      "Substantial clothing redesign, changed identity, major composition change or illustration-to-photoreal medium shift."])
        questions[f"{case}_focus"] = dict(type="choice",
            instructions=evidence + "what is the most useful concern to examine in the actual images before selecting a checkpoint?",
            criteria={"atmosphere": "Lighting effect is too weak or does not clearly express the requested mood.",
                      "outfit": "Unintended clothing redesign or added/removed garments.",
                      "identity": "Substantial change to character appearance or identity cues.",
                      "composition": "Unintended framing, pose or background layout change.",
                      "medium": "Unintended change between coherent illustration and photoreal or inconsistent rendering.",
                      "none": "No material concern is described; compare other cases and energy levels.",
                      "insufficient": "Descriptions are inadequate; inspect images again."})
    return dict(model=MODEL, state=dict(
        goal=f"Visually compelling {variation} sinister chiaroscuro: cinematic contrast, sculpted shadows and rich atmosphere. "
             "Prefer a visible appealing effect; mere minimal pixel change is not the goal. Keep recognizable characters and coherent images.",
        evidence_source=observer, limitations=limitations,
        instruction="You receive textual observations, not images. Judge the supplied evidence; do not claim to see pixels. "
                    "A/B order is randomized. Checkpoint identities, training steps, On/Off labels and residuals are withheld.",
        cases=cases), questions=questions)


def validate_response(request, response):
    if response.get("model") != MODEL or set(response.get("answers", {})) != set(request["questions"]):
        raise ValueError("Unexpected model or missing/extra answers")
    for name, question in request["questions"].items():
        answer = response["answers"][name]
        if answer.get("type") != question["type"]:
            raise ValueError(f"Answer type differs: {name}")
        expected = set(question["criteria"]) if question["type"] == "choice" else set(map(str, range(len(question["criteria"]))))
        probabilities = answer["probabilities"]
        if set(probabilities) != expected or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError(f"Invalid probabilities: {name}")
        if abs(sum(probabilities.values()) - 1) > .03:
            raise ValueError(f"Probabilities do not sum to one: {name}")
        confidence = answer["confidence"]
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence: {name}")
        if question["type"] == "choice" and answer["choice"] not in expected:
            raise ValueError(f"Invalid choice: {name}")
        if question["type"] == "score" and (not math.isfinite(answer["score"]) or not 0 <= answer["score"] <= len(expected)-1):
            raise ValueError(f"Invalid score: {name}")


def evaluate(request, key):
    body = json.dumps(request).encode()
    for attempt in range(3):
        req = urllib.request.Request(ENDPOINT, data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 529) or attempt == 2:
                # Do not echo request headers, credentials or arbitrary error bodies.
                raise RuntimeError(f"TypeSafe HTTP {error.code}") from None
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.observations.read_text())
    output = args.output or args.observations.parent / "jev"
    output.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("TYPESAFE_API_KEY")
    if not args.dry_run and not key:
        raise ValueError("TYPESAFE_API_KEY is required; never put it in review files")
    summaries = []
    for group, rows in source["groups"].items():
        if Path(group).name != group:
            raise ValueError("Group must name a direct sibling directory")
        folder = args.observations.parent / group
        mapping = json.loads((folder / "mapping.json").read_text())
        by_case = {r["case"]: r for r in mapping}
        if {r["case"] for r in rows} != set(by_case):
            raise ValueError("Observations must cover the complete comparison group")
        provenance = dict(observations_sha256=sha(args.observations),
                          mapping_sha256=sha(folder / "mapping.json"), images={})
        for pair in mapping:
            a, b = (pair[x]["metadata"] for x in ("A", "B"))
            for field in ("prompt", "seed", "width", "height", "steps", "model_identity", "case"):
                if a[field] != b[field]:
                    raise ValueError(f"Unmatched rendering field: {field}")
            if a.get("purpose") != "development" or b.get("purpose") != "development" or {a["energy"], b["energy"]} != {0., 1.}:
                raise ValueError("Only matched Off/Energy-1 development pairs are accepted")
            for label in ("A", "B"):
                ident = pair[label]["id"]
                provenance["images"][ident] = sha(folder / (ident + ".png"))
        variation = group.split("-")[0]
        request = build_request(rows, variation, source["observer"], source["limitations"])
        save(output / f"{group}.provenance.json", provenance)
        request_path = output / f"{group}.request.json"
        save(request_path, request)
        if args.dry_run:
            print(json.dumps(dict(group=group, cases=len(rows), questions=len(request["questions"]), dry_run=True)), flush=True)
            continue
        response_path = output / f"{group}.response.json"
        if response_path.exists():
            recorded = json.loads(response_path.read_text())
            if recorded["request_sha256"] != sha(request_path):
                raise ValueError("Cached response belongs to another request")
        else:
            started = time.monotonic()
            response = evaluate(request, key)
            validate_response(request, response)
            recorded = dict(request_sha256=sha(request_path), seconds=time.monotonic()-started, response=response)
            save(response_path, recorded)
        response = recorded["response"]
        validate_response(request, response)
        results = []
        for row in rows:
            case = row["case"]
            pair = by_case[case]
            active = next(x for x in ("A", "B") if pair[x]["metadata"]["energy"] == 1.)
            answers = {k: response["answers"][f"{case}_{k}"] for k in ("appeal", "atmosphere", "drift", "focus")}
            result = dict(case=case, on_label=active, image_id=pair[active]["id"], answers=answers)
            for dimension in ("appeal", "atmosphere"):
                choice = answers[dimension]["choice"]
                result[dimension] = ("win" if choice == active else "loss") if choice in ("A", "B") else choice
            results.append(result)
        summary = dict(group=group, cases=len(results), results=results,
            appeal={k: sum(r["appeal"] == k for r in results) for k in ("win", "tie", "loss", "insufficient")},
            atmosphere={k: sum(r["atmosphere"] == k for r in results) for k in ("win", "tie", "loss", "insufficient")},
            mean_drift=sum(r["answers"]["drift"]["score"] for r in results)/len(results),
            usage=response["usage"], seconds=recorded["seconds"])
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    if not args.dry_run:
        save(output / "summary.json", dict(model=MODEL, evidence="Jev judgments of Codex visual observations; not direct vision or independent validation", groups=summaries))


if __name__ == "__main__":
    main()
