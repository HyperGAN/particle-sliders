"""Use GPU 0 for reference audio while GPU 1 samples fresh training histories."""
import json
import gc
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(WORK / "runtime"), str(ROOT.parent)]
from analysis.gan_bcap.render_v2 import render
import torch

protocol = json.loads((WORK / "protocol.json").read_text())
paths = [ROOT / "models/uni16-gan-v1/lofi-warmup600-a02/lofi-warmup600-a02_last.safetensors",
    ROOT / "models/uni16-gan-v1/lofi-bounded660-a01/lofi-bounded660-a01_step660.safetensors",
    WORK / "lofi-fixed-history/lofi-fixed-history_step660.safetensors"]
try:
    for row in protocol["evaluation"]["rows"]:
        out = ROOT / f"eval/listen/fresh-seed-lofi-20260910/prewarm-row-{row}"
        if all((out / f"{p.stem}-s{seed}/checkpoint.json").exists()
               for p in paths for seed in protocol["evaluation"]["seeds"]):
            continue
        gc.collect()
        torch.cuda.empty_cache()
        render(paths, out,
            Path(protocol["evaluation"]["prompts"]), row=row, seeds=protocol["evaluation"]["seeds"], duration=20.)
    (WORK / "prewarm.json").write_text(json.dumps({"status": "complete"}) + "\n")
except BaseException as error:
    (WORK / "prewarm.json").write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
    raise
