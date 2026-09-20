"""Queue matched 8/12-step development renders from an immutable EMA export."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from lumen_studio.api import Generation, generation_payloads
    from lumen_studio.contracts import VARIATIONS, digest
    from lumen_studio.dataset import compile_manifest, compatible_manifest_hashes
    from lumen_studio.provenance import model_identity
    from lumen_studio.store import Store
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variation", choices=VARIATIONS)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--checkpoint", help="Exact immutable checkpoint hash")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima")
    args = parser.parse_args()
    root = args.root.resolve()
    identity = model_identity(root / "model")
    store = Store(root / "studio/studio.sqlite3")
    manifests = compatible_manifest_hashes(root, "train", args.variation)
    matches = [c for c in store.catalog() if c["variation"] == args.variation
               and c["metadata"]["step"] == args.step and c["metadata"]["model_identity"] == identity
               and c["metadata"]["manifest_sha256"] in manifests
               and (not args.checkpoint or c["sha256"] == args.checkpoint)]
    if len(matches) != 1:
        raise ValueError("Expected one immutable matching EMA checkpoint")
    checkpoint = matches[0]["sha256"]
    manifest = compile_manifest("dev")
    rows = [r for r in manifest["rows"] if r["variation"] == args.variation]
    payloads = []
    for row in (rows[0], rows[9]):
        for steps in (8, 12):
            request = Generation(prompt=row["neutral"], seed=29001, width=768, height=768,
                steps=steps, mix={args.variation: 1.}, checkpoints={args.variation: checkpoint})
            for value in generation_payloads(request, store, identity, [0., 1.]):
                value.update(case=row["id"], character=row["character"], definition=row["definition"],
                    bare=row["bare"], purpose="step_compatibility", sampling_step=args.step,
                    manifest_sha256=manifest["sha256"])
                payloads.append(value)
    # Stable group identity makes rerunning the command resume the same jobs.
    group = "step-compatibility-" + digest(payloads)
    print(json.dumps(store.enqueue(payloads, group=group)))


if __name__ == "__main__":
    main()
