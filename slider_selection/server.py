"""Serve only public study assets; persist validated responses outside that root."""
from __future__ import annotations

import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .dataset import validate_response, read_labels
from .features import file_hash, digest, write_json


def make_server(session_dir: Path, bind: str, port: int, features_path: Path | None = None):
    root = (session_dir / "public").resolve()
    private = json.loads((session_dir / "session-key.json").read_text())
    if file_hash(root / "session.json") != private["public_sha256"]:
        raise ValueError("Public session changed after creation")
    public = json.loads((root / "session.json").read_text())
    trials = {trial["id"]: trial for trial in public["trials"]}
    responses = session_dir / "responses.jsonl"
    lock = threading.Lock()
    last_fit = [None]

    def maybe_fit():
        if features_path is None:
            return
        latest = {item["trial_id"]: item for item in
                  (json.loads(line) for line in responses.read_text().splitlines() if line)}
        if set(latest) != set(private["trials"]) or digest(latest) == last_fit[0]:
            return
        last_fit[0] = digest(latest)
        try:
            from . import model as M
            labels, audit = read_labels(session_dir, responses)
            features = json.loads(features_path.read_text())
            model = M.fit(features, labels, audit)
            write_json(session_dir / "model.json", model)
            write_json(session_dir / "score-preview.json", M.score(model, features))
            status = {"state": "exploratory_fit_complete", "labeled_ladders": len(labels),
                      "message": "First model fitted. Independent listening validation is still required."}
        except (ValueError, FileNotFoundError) as exc:
            status = {"state": "needs_more_evidence", "message": str(exc)}
        write_json(session_dir / "calibration-status.json", status)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def respond(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/api/calibration":
                status_path = session_dir / "calibration-status.json"
                return self.respond(200, json.loads(status_path.read_text()) if status_path.exists()
                                    else {"state": "awaiting_listening", "message": "Listening answers are needed before fitting."})
            if self.path == "/api/responses":
                with lock:
                    values = [json.loads(line) for line in responses.read_text().splitlines() if line] if responses.exists() else []
                return self.respond(200, values)
            relative = unquote(urlparse(self.path).path).lstrip("/") or "index.html"
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                return self.respond(404, {"error": "Not found"})
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/response":
                return self.respond(404, {"error": "Not found"})
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond(403, {"error": "Cross-origin response rejected"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid response size")
                data = json.loads(self.rfile.read(size))
                validate_response(data, private)
                trial = trials[data["trial_id"]]
                required = {trial["original"]}
                required.update([trial["a"], trial["b"]] if trial["kind"] == "preference"
                                else [clip["url"] for clip in trial["settings"]])
                if not required.issubset(set(data["played"])):
                    raise ValueError("Listen to every clip before answering")
                with lock:
                    with responses.open("a") as stream:
                        stream.write(json.dumps(data, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    maybe_fit()
                return self.respond(200, {"saved": True})
            except (ValueError, TypeError, KeyError) as exc:
                return self.respond(400, {"error": str(exc)})

    return ThreadingHTTPServer((bind, port), Handler)
