"""Serve the blind test and saved ratings; only opaque audio URLs are public."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import study

PROCESS = None
PROCESS_LOCK = threading.Lock()
STOP = threading.Event()


def kick():
    global PROCESS
    with PROCESS_LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            return
        state = study.load()
        if not any(not study.ready(c) for c in state["clips"]):
            return
        if study.read(study.DATA / "worker.json", {}).get("status") == "failed":
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1",
                   HF_HOME="/ml2/music/.cache/huggingface", PYTHONPATH=str(study.ROOT))
        with (study.DATA / "render.log").open("a") as log:
            PROCESS = subprocess.Popen([sys.executable, "-u", str(study.WORK / "worker.py")],
                cwd=study.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def supervise():
    while not STOP.wait(3):
        try:
            if PROCESS is not None and PROCESS.poll() not in (None, 0):
                status = study.read(study.DATA / "worker.json", {})
                if status.get("status") != "failed":
                    study.write(study.DATA / "worker.json", dict(status="failed",
                        error="Render process stopped; retry resumes the same fixtures and seeds.", updated=time.time()))
            kick()
        except Exception as exc:
            print(f"Render supervisor: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app):
    if os.environ.get("LOFI_TEST_NO_WORKER") != "1":
        STOP.clear()
        threading.Thread(target=supervise, daemon=True).start()
        kick()
    yield
    STOP.set()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_origin(request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail":"Use the listening page to save ratings"}, status_code=403)
        if int(request.headers.get("content-length", "0")) > 20000:
            return JSONResponse({"detail":"Request too large"}, status_code=413)
    response = await call_next(request)
    if not request.url.path.startswith("/audio/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail":str(exc)}, status_code=400)


@app.get("/")
@app.head("/")
def index():
    return FileResponse(study.WORK / "page.html")


@app.get("/api/study")
def state():
    return study.public()


@app.get("/audio/{name}")
@app.head("/audio/{name}")
def audio(name: str):
    if not re.fullmatch(r"[a-f0-9]{24}\.(mp3|wav)", name):
        raise HTTPException(404)
    clip_id = name.split(".")[0]
    if clip_id not in study.load()["clips"] or not study.ready(clip_id):
        raise HTTPException(404)
    return FileResponse(study.DATA / "audio" / name,
        media_type="audio/mpeg" if name.endswith("mp3") else "audio/wav",
        headers={"Cache-Control":"private, max-age=86400"})


@app.post("/api/rate/{take_id}")
def rate(take_id: str, values: dict):
    study.rate(take_id, values)
    return {"saved":True}


@app.post("/api/reveal/{stage}")
def reveal(stage: str):
    study.reveal(stage)
    return {"saved":True}


@app.post("/api/choose/{stage}")
def choose(stage: str, values: dict):
    study.choose(stage, values.get("choices"))
    if os.environ.get("LOFI_TEST_NO_WORKER") != "1":
        kick()
    return {"saved":True}


@app.post("/api/retry")
def retry():
    with PROCESS_LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            raise ValueError("Rendering is already running")
        study.write(study.DATA / "worker.json", dict(status="queued", updated=time.time()))
    kick()
    return {"saved":True}


@app.post("/api/notes/{stage}")
def notes(stage: str, values: dict):
    with study.LOCK:
        state = study.load()
        if stage not in state["stages"] or not state["stages"][stage]["revealed"]:
            raise ValueError("Finish this stage first")
        note = values.get("note")
        if not isinstance(note, str) or len(note) > 10000:
            raise ValueError("Note must be text under 10000 characters")
        study.validate_note(note)
        state.setdefault("notes", {})[stage] = note
        study.save(state, "stage_note", stage)
    return {"saved":True}


@app.get("/api/export")
def export():
    return JSONResponse(study.export(), headers={"Content-Disposition":"attachment; filename=lofi-listening-results.json"})


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", default=8890, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.bind, port=args.port, access_log=False)
