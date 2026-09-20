"""Queue a lighting-only definition candidate without changing active training inputs."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--round", choices=("minimal-pair-v8", "facial-shadows-v9", "hybrid-lighting-v10", "shadow-contrast-v11", "shadow-contrast-v12", "shadow-contrast-v13", "shadow-contrast-v14", "shadow-contrast-v15", "shadow-contrast-v16", "consistent-lighting-v17"), default="minimal-pair-v8")
    args = parser.parse_args()
    limit = 24 if args.split == "train" else 12
    if args.start < 0 or args.count < 1 or args.start + args.count > limit:
        raise ValueError(f"Candidate range must lie within the {limit} {args.split} rows")
    from lumen_studio.api import Generation, generation_payloads
    from lumen_studio.contracts import atomic_json, digest
    from lumen_studio.dataset import compile_manifest, load_characters, load_definitions
    from lumen_studio.provenance import model_identity
    from lumen_studio.store import Store
    root = ROOT / "artifacts/anima"
    clauses = ["neutral directional illumination, deep shadow contrast",
               "neutral high-angle illumination, deep shadow contrast",
               "neutral oblique illumination, deep shadow contrast",
               "neutral side illumination, deep shadow contrast",
               "neutral illumination from below, deep shadow contrast",
               "neutral narrow illumination, deep shadow contrast"]
    facial = ["deep facial shadows", "angular facial shadows", "shadowed eyes",
              "strong facial shadow contrast", "sculpted facial shadows", "low-key facial lighting"]
    def lighting(d):
        index = int(d.id.rsplit('-', 1)[1]) - 1
        if args.round == "consistent-lighting-v17":
            return d.neutral_lighting + ", lower illumination" if index == 4 else d.lighting
        if args.round == "shadow-contrast-v16" and index == 0:
            return d.neutral_lighting + ", subdued lighting"
        if args.round in ("shadow-contrast-v15", "shadow-contrast-v16") and index in (2, 4):
            return d.neutral_lighting + ", " + ("shadowed facial features" if index == 2 else "soft shadows")
        if args.round in ("shadow-contrast-v13", "shadow-contrast-v14") and index in (2, 4):
            clause = "subdued lighting" if index == 2 else (
                "low-key lighting" if args.round == "shadow-contrast-v13" else "subdued illumination")
            return d.neutral_lighting + ", " + clause
        if args.round in ("shadow-contrast-v11", "shadow-contrast-v12", "shadow-contrast-v13", "shadow-contrast-v14", "shadow-contrast-v15", "shadow-contrast-v16"):
            contrasts = (("deep", "strong", "high", "stark", "pronounced", "dramatic")
                         if args.round != "shadow-contrast-v12" else
                         ("deep", "strong", "deepened", "stark", "increased", "dramatic"))
            contrast = contrasts[index]
            return d.neutral_lighting.replace("moderate shadow contrast", contrast + " shadow contrast")
        directional = args.round == "minimal-pair-v8" or (
            args.round == "hybrid-lighting-v10" and index in (0, 4, 5))
        return clauses[index] if directional else d.neutral_lighting + ", " + facial[index]

    definitions = [replace(d, lighting=lighting(d))
                   if d.variation == "theatrical" and d.split == "train" else d
                   for d in load_definitions()]
    manifests = {split: compile_manifest(split, definitions=definitions) for split in ("train", "dev")}
    for split, candidate in manifests.items():
        before = compile_manifest(split)
        for original, changed in zip(before["rows"], candidate["rows"]):
            assert original["shared"] == changed["shared"] and original["neutral"] == changed["neutral"]
            if original["variation"] != "theatrical" or original["definition_split"] != "train":
                assert original == changed
    bundle = dict(schema=1, candidate="theatrical-" + args.round, active_training_unchanged=True,
                  definitions=[asdict(d) for d in definitions], characters=load_characters(), **manifests)
    bundle["sha256"] = digest(bundle)
    path = root / ("definition-candidates/theatrical-" + args.round) / "catalog.json"
    if path.exists() and json.loads(path.read_text()) != bundle:
        raise ValueError("Definition candidate is immutable")
    if not path.exists():
        atomic_json(path, bundle)
    store = Store(root / "studio/studio.sqlite3")
    identity = model_identity(root / "model")
    rows = [r for r in manifests[args.split]["rows"] if r["variation"] == "theatrical"]
    payloads = []
    for row in rows[args.start:args.start + args.count]:
        for side in ("neutral", "positive"):
            request = Generation(prompt=row[side], seed=row["seeds"][0], width=512, height=512,
                                 steps=10, energy=0., mix={})
            payload = generation_payloads(request, store, identity)[0]
            payload.update(purpose="definition_candidate", candidate=bundle["sha256"],
                           manifest_sha256=manifests[args.split]["sha256"], case=row["id"],
                           character=row["character"], definition=row["definition"], reference_side=side)
            payloads.append(payload)
    result = store.enqueue(payloads, group="definition-candidate-" + digest(payloads))
    print(json.dumps(dict(candidate=bundle["sha256"], manifest=manifests[args.split]["sha256"], **result)))


if __name__ == "__main__":
    main()
