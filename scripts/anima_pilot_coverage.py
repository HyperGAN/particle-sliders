"""Complete the fixed four-character development grid at the pilot checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from lumen_studio.api import Generation, generation_payloads
    from lumen_studio.audits import active_checkpoint, all_images
    from lumen_studio.contracts import VARIATIONS, digest
    from lumen_studio.dataset import compile_manifest
    from lumen_studio.provenance import model_identity
    from lumen_studio.store import Store
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    parser.add_argument("--variation", choices=VARIATIONS, action="append")
    args = parser.parse_args()
    root = args.root.resolve()
    store = Store(root / "studio/studio.sqlite3")
    identity = model_identity(root / "model")
    manifest = compile_manifest("dev")
    images = all_images(store)
    with store.connect() as db:
        pending = [store.unpack(r)["payload"] for r in db.execute(
            "SELECT * FROM jobs WHERE status IN ('queued','running')")]
    existing = [i["metadata"] for i in images] + pending
    results = {}
    for variation in (args.variation or VARIATIONS):
        run = json.loads((root / "runs" / variation / "run.json").read_text())
        check_path = root / "runs" / variation / "resume-check.json"
        check = json.loads(check_path.read_text()) if check_path.exists() else {}
        if not (check.get("passed") is True and check.get("step") == 200 and check.get("run") == run):
            raise ValueError("Wait for the 200-update resume check before completing pilot coverage")
        checkpoints = [c for c in store.catalog() if c["variation"] == variation and c["metadata"]["step"] == 200
                       and active_checkpoint(c, variation)
                       and c["metadata"]["model_identity"] == identity
                       and c["metadata"]["normalization"]["cache_fingerprint"] == run["cache"]]
        if len(checkpoints) != 1:
            raise ValueError("Expected one matching immutable 200-update checkpoint per variation")
        checkpoint = checkpoints[0]["sha256"]
        rows = [r for r in manifest["rows"] if r["variation"] == variation][:4]
        if len({r["character"] for r in rows}) != 4:
            raise ValueError("Expected all four development characters")
        group = "pilot-coverage-" + digest(dict(variation=variation, checkpoint=checkpoint,
            model=identity, manifest=manifest["sha256"], cases=[r["id"] for r in rows],
            seed=29001, width=512, height=512, steps=10, energies=[0., .25, .5, 1.]))
        payloads = []
        for row in rows:
            request = Generation(prompt=row["neutral"], seed=29001, width=512, height=512,
                steps=10, mix={variation: 1.}, checkpoints={variation: checkpoint})
            for value in generation_payloads(request, store, identity, [0., .25, .5, 1.]):
                shared = ("prompt", "seed", "width", "height", "steps", "energy", "checkpoints", "model_identity")
                if any(m.get("purpose") == "development"
                       and m.get("case") == row["id"]
                       and m.get("sampling_step") == 200
                       and all(m.get(k) == value[k] for k in shared) for m in existing):
                    continue
                value.update(case=row["id"], character=row["character"], definition=row["definition"],
                    bare=row["bare"], purpose="development", sampling_step=200, pilot_coverage=True,
                    manifest_sha256=manifest["sha256"])
                payloads.append(value)
        if payloads:
            results[variation] = store.enqueue(payloads, group=group)
        else:
            results[variation] = dict(ids=[], complete=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
