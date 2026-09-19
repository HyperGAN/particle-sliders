"""Immutable recipe and paths for the two-GPU 3400-update campaign."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
RUNTIME = WORK / "runtime"
MODELS = ROOT / "models/uni16-fresh3400-v1"
PAGE = ROOT / "eval/listen/uni16-fresh3400-20260912"
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
MILESTONES = (600, 1000, 2000, 3000, 3400)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_for(index, update):
    if not 0 <= index < 16 or not 601 <= update <= 3400:
        raise ValueError("Seed request outside campaign bounds")
    return 2_000_001 + index * 10_000 + update - 601


def warmup_args(item, run, resume=None):
    args = ["--name", run.name, "--prompts_file", item["train_prompts"], "--save_dir", str(run),
        "--lm_target", "faithful_plus_neu_lyric", "--pole_mode", "hidden", "--rank", "8", "--alpha", "8",
        "--lr", "5e-4", "--steps", "600", "--seed", "7", "--no-early_stop", "--endreg_weight", "1",
        "--save_every", "50", "--device", "0", "--adv_arch", "tx", "--adv_in", "scaled",
        "--adv_readout", "mean_last", "--adv_condition", "none", "--adv_weight", "1", "--fm_weight", "1",
        "--fm_mode", "batch", "--pole_weight", "0", "--lyrichold_weight", "0", "--adv_reg_coeff", "1",
        "--adv_reg_kappa", "1", "--adv_batch", "4", "--gan_beta1", "0", "--gan_lr_schedule", "constant",
        "--grad_account", "--parts", "0", "--save_training_state", "--endreg_cache", item["fixed_cache"]]
    if resume:
        args += ["--resume_state", str(resume)]
    return args


def verify():
    manifest = read(WORK / "manifest.json")
    if not manifest:
        raise ValueError("Campaign has not been prepared")
    for name, expected in manifest["files"].items():
        if sha(WORK / name) != expected:
            raise ValueError(f"Frozen campaign input changed: {name}")
    for name, expected in manifest["model_files"].items():
        p = Path(name)
        if not p.is_file() or (p.stat().st_size, p.stat().st_mtime_ns) != tuple(expected):
            raise ValueError(f"Base model file changed: {name}")
    return manifest


def env(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
        HF_HOME="/ml2/music/.cache/huggingface", PYTHONPATH=f"{RUNTIME}:{ROOT.parent}",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
