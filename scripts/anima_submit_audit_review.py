"""Validate and submit completed blind final/mixture ratings without exposing energies."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request


CHECKS = ("character", "outfit", "pose", "composition", "medium", "quality",
          "unwanted_objects", "recurring_features", "major_regression")
MATCHED = ("prompt", "seed", "width", "height", "steps", "model_identity",
           "checkpoints", "case", "family", "audit", "purpose", "character")


def prepare(ratings_path):
    ratings_path = Path(ratings_path)
    judgments = json.loads(ratings_path.read_text())
    raw = (ratings_path.parent / "mapping.json").read_bytes()
    if judgments.get("mapping_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Ratings belong to a different immutable blind mapping")
    mapping = json.loads(raw)
    groups, reviews = mapping["groups"], judgments["reviews"]
    if (len(groups) != 128 or len(reviews) != 128
            or len({g["case"] for g in groups}) != 128
            or {r["case"] for r in reviews} != {g["case"] for g in groups}):
        raise ValueError("Complete all 128 audit cases exactly once")
    by_case = {r["case"]: r for r in reviews}
    prepared, ids = [], set()
    for group in groups:
        review = by_case[group["case"]]
        if review.get("blind") is not True or review.get("family") != group["family"]:
            raise ValueError("Explicit blind review of the matching family is required")
        columns, ratings = group["columns"], review["ratings"]
        if set(columns) != set("ABCD") or set(ratings) != set("ABCD"):
            raise ValueError("Rate all four concealed columns")
        metadata = [i["metadata"] for i in columns.values()]
        if sorted(m["energy"] for m in metadata) != [0., .25, .5, 1.]:
            raise ValueError("Each case requires one complete energy sweep")
        baseline = next(letter for letter, i in columns.items() if i["metadata"]["energy"] == 0.)
        for letter, item in columns.items():
            m, rating = item["metadata"], ratings[letter]
            if (m.get("audit") != mapping["audit"] or m.get("family") != group["family"]
                    or m.get("purpose") not in ("final_test", "mixture_audit")
                    or any(m.get(k) != metadata[0].get(k) for k in MATCHED)):
                raise ValueError("Audit review requires matched immutable render settings")
            if item["id"] in ids:
                raise ValueError("An image appears more than once")
            ids.add(item["id"])
            strength = rating.get("atmosphere_strength")
            if (type(strength) not in (int, float) or not math.isfinite(strength)
                    or not 0 <= strength <= 5):
                raise ValueError("Every image needs an explicit atmosphere strength in [0,5]")
            if (any(type(rating.get(k)) is not bool for k in CHECKS)
                    or not isinstance(rating.get("notes"), str) or not rating["notes"].strip()):
                raise ValueError("Complete every preservation, quality and leakage judgment")
        off_strength = ratings[baseline]["atmosphere_strength"]
        for letter, item in columns.items():
            rating = ratings[letter]
            strength = rating["atmosphere_strength"]
            atmosphere = "win" if strength > off_strength else "loss" if strength < off_strength else "tie"
            prepared.append((item["id"], dict(atmosphere=atmosphere, blind=True,
                atmosphere_strength=strength, notes=rating["notes"],
                **{k: rating[k] for k in CHECKS})))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    prepared = prepare(args.ratings)
    if args.validate_only:
        print(json.dumps(dict(validated=len(prepared), submitted=0)))
        return
    for index, (ident, value) in enumerate(prepared, 1):
        request = urllib.request.Request(args.url + "/api/images/" + ident + "/review",
            data=json.dumps(value).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            json.load(response)
        if index % 32 == 0:
            print(json.dumps(dict(submitted=index, total=len(prepared))), flush=True)


if __name__ == "__main__":
    main()
