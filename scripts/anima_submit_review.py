"""Submit explicitly completed A/B ratings without revealing image identities."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", type=Path, help="Completed copy of ratings.template.json")
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    args = parser.parse_args()
    judgments = json.loads(args.ratings.read_text())
    mapping_path = args.ratings.parent / "mapping.json"
    if hashlib.sha256(mapping_path.read_bytes()).hexdigest() != judgments["mapping_sha256"]:
        raise ValueError("Ratings belong to a different blind grid")
    mapping = {row["case"]: row for row in json.loads(mapping_path.read_text())}
    rows = judgments["reviews"]
    if len(rows) != len(mapping) or {r["case"] for r in rows} != set(mapping):
        raise ValueError("Complete every case exactly once")
    checks = ("blind", "character", "outfit", "pose", "composition", "medium", "quality",
              "unwanted_objects", "recurring_features", "major_regression")
    prepared = []
    for review in rows:
        if review.get("preferred") not in ("A", "B", "tie"):
            raise ValueError("Each case needs an explicit A/B/tie atmosphere judgment")
        if any(type(review.get(key)) is not bool for key in checks) or not review.get("notes", "").strip():
            raise ValueError("Every preservation/leakage check and review note must be explicit")
        pair = mapping[review["case"]]
        active = next(side for side in ("A", "B") if pair[side]["metadata"]["energy"] == 1.)
        off = "B" if active == "A" else "A"
        a, b = pair[active]["metadata"], pair[off]["metadata"]
        if b["energy"] != 0 or any(a.get(k) != b.get(k) for k in
                ("prompt", "seed", "width", "height", "steps", "model_identity", "checkpoints", "case", "family")):
            raise ValueError("Review requires a matched full-strength/Off comparison")
        atmosphere = "tie" if review["preferred"] == "tie" else ("win" if review["preferred"] == active else "loss")
        value = dict(atmosphere=atmosphere, notes=review["notes"], **{k: review[k] for k in checks})
        prepared.append((pair[active]["id"], value))
    # Validate the entire file before making any persistent review changes.
    for ident, value in prepared:
        request = urllib.request.Request(args.url + "/api/images/" + ident + "/review",
            data=json.dumps(value).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            json.load(response)
    print(json.dumps(dict(submitted=len(prepared),
        atmosphere_wins=sum(v["atmosphere"] == "win" for _, v in prepared),
        atmosphere_ties=sum(v["atmosphere"] == "tie" for _, v in prepared))))


if __name__ == "__main__":
    main()
