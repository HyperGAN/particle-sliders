#!/usr/bin/env python3
"""Poll the :7860 studio until the queue is idle long enough to take a GPU.

Prints only DONE to stdout. Intended for the grok monitor. Status goes to
the log file next to this script, not stdout.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

STUDIO = "http://127.0.0.1:7860/api/studio"
STABLE_SEC = 90
POLL_SEC = 30
LOG = Path("/ml2/music/sliders-conceptmod/models/wait-studio-idle.log")


def snapshot() -> dict:
    with urllib.request.urlopen(STUDIO, timeout=10) as resp:
        return json.loads(resp.read().decode())


def idle(data: dict) -> bool:
    queued = data.get("queued") or []
    active = data.get("active") or []
    keep = data.get("keep") or {}
    return (not queued) and (not active) and (not keep.get("enabled"))


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    idle_since: float | None = None
    while True:
        try:
            data = snapshot()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            log(f"studio unreachable: {exc}")
            idle_since = None
            time.sleep(POLL_SEC)
            continue
        q = len(data.get("queued") or [])
        a = len(data.get("active") or [])
        keep_on = bool((data.get("keep") or {}).get("enabled"))
        if idle(data):
            if idle_since is None:
                idle_since = time.time()
                log(f"idle start (q={q} a={a} keep={keep_on})")
            elif time.time() - idle_since >= STABLE_SEC:
                log("idle stable — DONE")
                print("DONE", flush=True)
                return 0
        else:
            if idle_since is not None:
                log(f"idle broken (q={q} a={a} keep={keep_on})")
            idle_since = None
            log(f"busy q={q} a={a} keep={keep_on}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
