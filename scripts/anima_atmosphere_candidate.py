"""Compile an isolated replacement candidate and optionally queue teacher pairs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima/remote")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--rows", nargs="+", type=int, default=[0, 3, 6, 9])
    args = parser.parse_args()
    from lumen_studio.candidates import compile_candidate
    from lumen_studio.contracts import atomic_json, digest
    bundle = compile_candidate(args.spec)
    destination = args.root / "definition-candidates" / bundle["candidate"] / bundle["sha256"]
    path = destination / "catalog.json"
    if path.exists() and json.loads(path.read_text()) != bundle:
        raise ValueError("Candidate evidence changed")
    atomic_json(path, bundle)
    result = dict(candidate=bundle["candidate"], sha256=bundle["sha256"], catalog=str(path))
    if args.queue:
        from lumen_studio.api import Generation, generation_payloads
        from lumen_studio.provenance import model_identity
        from lumen_studio.store import Store
        rows = bundle["train"]["rows"]
        if len(set(args.rows)) != len(args.rows) or any(i < 0 or i >= len(rows) for i in args.rows):
            raise ValueError("Select distinct valid training rows")
        store = Store(args.root / "studio/studio.sqlite3")
        identity = model_identity(args.root / "model")
        payloads = []
        for i in args.rows:
            row = rows[i]
            for side in ("neutral", "positive"):
                req = Generation(prompt=row[side], seed=row["seeds"][0], width=512, height=512,
                                 steps=10, energy=0., mix={})
                payload = generation_payloads(req, store, identity)[0]
                payload.update(purpose="atmosphere_candidate", candidate=bundle["sha256"],
                               candidate_label=bundle["label"], candidate_id=bundle["candidate"],
                               manifest_sha256=bundle["train"]["sha256"], case=row["id"],
                               character=row["character"], definition=row["definition"], reference_side=side)
                payloads.append(payload)
        result.update(store.enqueue(payloads, group="atmosphere-candidate-" + digest(payloads)))
        atomic_json(destination / "queued.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
