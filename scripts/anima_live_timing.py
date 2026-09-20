"""Measure operational costs from existing events without touching the trainer."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def summary(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values)) if values else dict(count=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/anima"))
    parser.add_argument("--since", type=float, required=True, help="Earliest UTC epoch time on this host")
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.root / 'studio/studio.sqlite3'}?mode=ro", uri=True)
    events = db.execute("SELECT kind,data,created FROM events WHERE created>=? ORDER BY id", (args.since,)).fetchall()
    images = {ident: json.loads(data) for ident, data in db.execute(
        "SELECT id,metadata FROM images WHERE created>=?", (args.since,))}
    db.close()
    owner = None
    last_image = last_training = None
    claims, load, release, render, completion = {}, [], [], defaultdict(list), []
    for kind, raw, at in events:
        data = json.loads(raw)
        if kind == "training" and data.get("status") == "running":
            last_training = (at, data.get("step", 0))
        elif kind == "job" and data.get("status") == "running":
            claims[data["id"]] = at
        elif kind == "image":
            last_image = at
            m = images.get(data["id"])
            if m:
                key = f"{m['width']}x{m['height']}/{m['steps']}/{'off' if m['energy'] == 0 else 'on'}"
                render[key].append(m["render_seconds"])
                if data["id"] in claims:
                    completion.append(at - claims[data["id"]] - m["render_seconds"])
        elif kind == "controls" and isinstance(data.get("owner"), dict):
            state = data["owner"]["state"]
            if state == "training" and owner == "rendering" and last_image is not None:
                load.append(at - last_image)
            if state == "rendering" and owner == "training" and last_training:
                timestamp, step = last_training
                if step > 20 and step % 100:
                    release.append(at - timestamp)
            owner = state
    report = dict(since=args.since, source="observed event timestamps; not a controlled benchmark",
        render_to_training_load_seconds=summary(load), training_save_release_seconds=summary(release),
        render_load_and_png_overhead_seconds=summary(completion),
        rendering_seconds={k: summary(v) for k, v in sorted(render.items())},
        notes=["Training load includes runtime reload, cache/optimizer restore, and any initial run setup.",
               "Save/release excludes update-20 and 100-update probe/export boundaries.",
               "Rendering durations exclude runtime/checkpoint loading and final PNG writes."])
    path = args.root / "benchmarks/operational-timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
