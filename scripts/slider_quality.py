#!/usr/bin/env python3
"""Prepare, measure, label, fit, validate and score the LM slider benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Optional audio model installed without modifying the studio environment.
DEPENDENCIES = ROOT.parent / ".cache/slider-quality/python"
if DEPENDENCIES.exists():
    sys.path.insert(0, str(DEPENDENCIES))

from slider_selection.dataset import prepare, build_session, load_manifest, read_labels
from slider_selection.features import AudioMeasurer, digest, file_hash, ladder_features, write_json
from slider_selection import model as M


def measure(manifest, out, cache, device):
    measurer = AudioMeasurer(cache, device)
    paths = sorted({path for record in manifest["records"]
                    for path in [record["baseline"]] + [clip["path"] for clip in record["clips"]]})
    audio = {}
    for i, path in enumerate(paths, 1):
        print(f"Measuring {i}/{len(paths)}: {Path(path).parent.name}/{Path(path).name}", flush=True)
        audio[path] = measurer.measure(Path(path))
    provenance = {digest(item["provenance"]) for item in audio.values()}
    if len(provenance) != 1:
        raise ValueError("Mixed measurement versions")
    protocol = {"audio_provenance": next(iter(provenance)),
                "reducer_sha256": file_hash(ROOT / "slider_selection/features.py"),
                "rungs": "exact unit endpoints on every declared pole"}
    result = {"schema": 1, "manifest_id": manifest["id"], "protocol": protocol,
              "measurement_id": digest(protocol), "records": []}
    for record in manifest["records"]:
        expected = set(record["audio_hashes"])
        actual = {audio[path]["sha256"] for path in [record["baseline"]] + [clip["path"] for clip in record["clips"]]}
        if actual != expected or file_hash(Path(record["score_source"])) != record["score_sha256"]:
            raise ValueError("Source audio or transcript measurements changed after benchmark freeze")
        clean = {key: value for key, value in record.items() if key not in {"lyrics", "clips"}}
        clean["features"] = ladder_features(record, audio)
        result["records"].append(clean)
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folders", type=Path, nargs="+")
    p = commands.add_parser("measure")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "eval/slider-quality/audio-cache")
    p.add_argument("--device", default="cuda:0", help="Visible device; set CUDA_VISIBLE_DEVICES=1")
    p = commands.add_parser("session")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-preferences", type=int, default=2)
    p = commands.add_parser("serve")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8902)
    p.add_argument("--features", type=Path, help="Fit an exploratory model automatically after all trials are answered")
    for command in ("fit", "validate"):
        p = commands.add_parser(command)
        p.add_argument("--features", type=Path, required=True)
        p.add_argument("--session", type=Path, required=True)
        p.add_argument("--responses", type=Path)
        p.add_argument("--out", type=Path, required=True)
        if command == "validate":
            p.add_argument("--model", type=Path, required=True)
    p = commands.add_parser("score")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out, args.folders)
        print(f"Prepared {len(result['records'])} ladders; skipped {len(result['skipped'])}")
    elif args.command == "measure":
        result = measure(load_manifest(args.manifest), args.out, args.cache, args.device)
        print(f"Saved measurements for {len(result['records'])} ladders")
    elif args.command == "session":
        result = build_session(load_manifest(args.manifest), args.out, max_preferences=args.max_preferences)
        print(f"Prepared {len(result['trials'])} blind trials in {args.out}")
    elif args.command == "serve":
        from slider_selection.server import make_server
        server = make_server(args.session, args.bind, args.port, args.features)
        print(f"Listening study: http://{args.bind}:{args.port}/", flush=True)
        server.serve_forever()
    elif args.command in {"fit", "validate"}:
        labels, audit = read_labels(args.session, args.responses or args.session / "responses.jsonl")
        data = json.loads(args.features.read_text())
        if args.command == "fit":
            result = M.fit(data, labels, audit)
        else:
            model = json.loads(args.model.read_text())
            report = M.validate(model, data, labels, audit)
            result = {**model, "status": "validated" if report["passed"] else "exploratory",
                      "validation": report}
        write_json(args.out, result)
        print(f"Saved {args.command} result: {result['status']}")
    else:
        result = M.score(json.loads(args.model.read_text()), json.loads(args.features.read_text()))
        write_json(args.out, result)
        for candidate in result["candidates"]:
            print(candidate["candidate"], candidate["status"], candidate["optimizer_score"], candidate["reasons"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
