"""Populate exact official frozen feature cache using only an owned CPU server."""

# ruff: noqa: E402
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from conceptmod.textsliders.bonsai_gguf_backend import (
    BonsaiGGUFBackend,
    build_server_cmd,
)
from conceptmod.textsliders.caption_recovery import FrozenReadouts
from conceptmod.textsliders.train_lora_bonsai_gguf import (
    load_slider_rows,
    DEFAULT_PROMPTS,
)
from caption_quality import HELDOUT, CANARIES

ROOT = Path("/mnt/nvme/sliders/qualification")
p = argparse.ArgumentParser()
p.add_argument("--recipe", type=Path, required=True)
a = p.parse_args()
recipe = json.loads(a.recipe.read_text())
if recipe["model"] != "bonsai":
    raise ValueError("Only exact Bonsai PTQ1 frozen readouts can be cached")
assets = json.loads(
    (ROOT / "models/bonsai-b072e1d/qualification-manifest.json").read_text()
)
binary = Path(assets["binary"])
os.environ["LD_LIBRARY_PATH"] = str(binary.parent)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
cmd = build_server_cmd(
    str(binary), recipe["weights"], port=port, ctx=512, threads=4
) + ["--parallel", "1", "--embd-normalize", "2", "--threads-batch", "4"]
run = a.recipe.parent
log = (run / "cache-server.log").open("x")
server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
(run / "cache-server.json").write_text(
    json.dumps(dict(pid=server.pid, command=cmd), indent=2) + "\n"
)
try:
    start = time.monotonic()
    url = f"http://127.0.0.1:{port}"
    while True:
        if server.poll() is not None:
            raise RuntimeError("Own officialCPUserver exited before readiness")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    break
        except OSError:
            pass
        if time.monotonic() - start > 180:
            raise TimeoutError("OwnCPUserver readiness exceeded180seconds")
        time.sleep(1)
    backend = BonsaiGGUFBackend(
        device="cpu", weights=recipe["weights"], server_url=url, dummy=False
    )
    cache = FrozenReadouts(
        backend,
        ROOT / "models/bonsai-caption-readouts-20261017",
        recipe["base_identity"],
    )
    rows = load_slider_rows(str(DEFAULT_PROMPTS))
    texts = sorted(
        set(CANARIES) | {r[k] for r in rows + HELDOUT for k in ("positive", "neutral")}
    )
    for i, text in enumerate(texts):
        enc = cache.encode(text)
        cache.hidden(enc)
        print(json.dumps(dict(cached=i + 1, total=len(texts), text=text)), flush=True)
    (run / "cache-preparation.json").write_text(
        json.dumps(
            dict(
                status="verified_exact_frozen_readouts",
                prompts=len(texts),
                base_identity=recipe["base_identity"],
                cache_dir=str(cache.root),
                independent_requests_exact=True,
                elapsed_seconds=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )
finally:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    log.close()
