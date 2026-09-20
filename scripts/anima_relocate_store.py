"""Validate a copied Studio and rebase only its host-specific file paths."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from lumen_studio.cache import TargetCache
    from lumen_studio.contracts import atomic_json, canonical, file_hash
    from lumen_studio.store import Store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    store = Store(root / "studio/studio.sqlite3")
    source = Path(args.source_root)
    changes, checkpoints, caches = [], [], {}

    def rebase(value):
        path = Path(value)
        if path.is_relative_to(source):
            path = root / path.relative_to(source)
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError(f"Missing or external migrated path: {path}")
        return str(path)

    for run in store.runs():
        config = dict(run["config"])
        for key in ("directory", "train_cache", "dev_cache"):
            config[key] = rebase(config[key])
        identity = json.loads((Path(config["directory"]) / "run.json").read_text())
        for key in ("train_cache", "dev_cache"):
            path = config[key]
            if path not in caches:
                cache = TargetCache(path)
                caches[path] = cache.index["fingerprint"]
        if (identity["cache"] != caches[config["train_cache"]]
                or identity["variation"] != config["variation"]
                or identity["seed"] != config.get("seed", 7)):
            raise ValueError(f"Run/target identity mismatch: {run['id']}")
        if config != run["config"]:
            changes.append((run["id"], run["config"], config))
    for checkpoint in store.catalog():
        path = rebase(checkpoint["path"])
        if file_hash(path) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint bytes changed: {path}")
        if path != checkpoint["path"]:
            checkpoints.append((checkpoint["sha256"], checkpoint["path"], path))
    with store.connect() as db:
        for row in db.execute("SELECT path FROM images"):
            path = store.path.parent / row[0]
            if not path.is_file() or not path.resolve().is_relative_to(store.path.parent):
                raise ValueError(f"Missing or external history image: {path}")
        image_count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    report = dict(source_root=str(source), destination_root=str(root),
                  caches=caches, image_count=image_count,
                  runs_rebased=len(changes), checkpoints_rebased=len(checkpoints),
                  applied=False, at=time.time())
    if args.apply and (changes or checkpoints):
        folder = root / "migrations" / f"relocation-{time.time_ns()}"
        folder.mkdir(parents=True)
        backup = folder / "before.sqlite3"
        with sqlite3.connect(store.path) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
        report["backup_sha256"] = file_hash(backup)
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for ident, old, new in changes:
                row = db.execute("SELECT config FROM training_runs WHERE id=?", (ident,)).fetchone()
                if json.loads(row[0]) != old:
                    raise ValueError("Run changed during relocation; retry validation")
                db.execute("UPDATE training_runs SET config=? WHERE id=?", (canonical(new), ident))
            for sha, old, new in checkpoints:
                row = db.execute("SELECT path FROM checkpoints WHERE sha256=?", (sha,)).fetchone()
                if row[0] != old:
                    raise ValueError("Catalog changed during relocation; retry validation")
                db.execute("UPDATE checkpoints SET path=? WHERE sha256=?", (new, sha))
            report["applied"] = True
            store.emit(db, "relocation", report)
        atomic_json(folder / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
