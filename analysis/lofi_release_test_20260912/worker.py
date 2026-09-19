"""Resumable first-draw rendering on GPU 1; originals and matched previews."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/ml2/music/.cache/huggingface"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import soundfile as sf

from study import DATA, ROOT, WORK, digest, load, read, ready, sha, write

RUNTIME = ROOT / "analysis/fresh_seed_20260910/runtime"


def preview(wav, mp3):
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "info", "-i", str(wav),
            "-af", "loudnorm=I=-18:TP=-2:LRA=11:print_format=json", "-f", "null", "-"]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not match:
        raise RuntimeError("Loudness analysis did not return measurements")
    measure = json.loads(match.group())
    # Silence is retained without adding synthetic noise or changing the seed.
    if all(np.isfinite(float(measure[k])) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
        filt = ("loudnorm=I=-18:TP=-2:LRA=11:linear=true:print_format=json:"
                f"measured_I={measure['input_i']}:measured_TP={measure['input_tp']}:"
                f"measured_LRA={measure['input_lra']}:measured_thresh={measure['input_thresh']}:"
                f"offset={measure['target_offset']}")
        audio_args = ["-af", filt]
        method = "two-pass EBU R128, -18 LUFS, -2 dBTP"
    else:
        audio_args = []
        method = "unchanged; loudness measurement non-finite"
    temp = mp3.with_name(mp3.stem + ".tmp.mp3")
    result2 = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(wav),
        *audio_args, "-map_metadata", "-1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "320k",
        "-threads", "1", str(temp)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    temp.replace(mp3)
    normalized = re.search(r'\{\s*"input_i".*?\}', result2.stderr, re.S)
    return dict(method=method, source=measure,
                output=json.loads(normalized.group()) if normalized else None)


def finish_clip(clip, wav, provenance):
    audio, sr = sf.read(wav, always_2d=True, dtype="float32")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Unreadable, empty or non-finite audio; retained for investigation")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    duration = len(audio) / sr
    clipped = float(np.mean(np.abs(audio) >= .999))
    flags = []
    if rms < .0001:
        flags.append("near_silent")
    if clipped >= .01:
        flags.append("clipping")
    if duration < clip["fixture"]["duration"] * .95:
        flags.append("ended_before_cap")
    matched = preview(wav, wav.with_suffix(".mp3"))
    meta = dict(clip=clip["id"], identity=clip["identity"], duration=duration, sample_rate=sr,
        rms=rms, peak=float(np.max(np.abs(audio))), clipped_fraction=clipped, flags=flags,
        wav_sha256=sha(wav), mp3_sha256=sha(wav.with_suffix(".mp3")),
        preview=matched, provenance=provenance, completed=time.time())
    write(wav.with_suffix(".json"), meta)
    return meta


class Renderer:
    def __init__(self, state):
        self.gpu_lease = (ROOT.parent / ".music-gpu-1.lock").open("a")
        try:
            fcntl.flock(self.gpu_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write(DATA / "worker.json", dict(status="waiting_gpu", updated=time.time()))
            fcntl.flock(self.gpu_lease, fcntl.LOCK_EX)
        sys.path[:0] = [str(RUNTIME), str(ROOT.parent)]
        import torch
        from conceptmod.textsliders import generate_listen as G
        from conceptmod.textsliders.lora import LoRANetwork
        self.torch, self.G = torch, G
        torch.set_num_threads(4)
        # The original inference implementation is a frozen source snapshot.
        for rel, expected in read(ROOT / "analysis/fresh_seed_20260910/runtime-snapshot.json").items():
            if sha(RUNTIME / rel) != expected:
                raise ValueError(f"Frozen inference source changed: {rel}")
        self.pipe = G._load_pipeline(G.DEFAULT_MODEL_DIR, "cuda:0")
        self.network = LoRANetwork(self.pipe.language_model, multiplier=0., rank=8, alpha=8.,
            delimiter="-", target_replace=["Qwen3Attention"], prefix="lora_te",
            train_method="full").to("cuda:0").eval()
        if len(self.network.unet_loras) != 144:
            raise ValueError("Unexpected LoRA module count")
        self.loaded = None
        self.configs = state["configs"]

    def render(self, clip, dest):
        from safetensors.torch import load_file
        torch, G = self.torch, self.G
        cfg = self.configs[clip["config"]]
        if cfg["kind"] == "adapter" and self.loaded != cfg["weights"]:
            if sha(cfg["weights"]) != cfg["sha256"]:
                raise ValueError("Frozen checkpoint changed")
            tensors = load_file(cfg["weights"])
            if not all(torch.isfinite(t).all() for t in tensors.values()):
                raise ValueError("Non-finite checkpoint")
            self.network.load_state_dict(tensors, strict=True)
            self.loaded = cfg["weights"]
        f = clip["fixture"]
        self.network.set_lora_slider(cfg["scale"])
        prompt = f["row"]["positive" if cfg["kind"] == "caption" else "neutral"]
        with torch.inference_mode(), self.network:
            audio = self.pipe(prompt=prompt, lyrics=f["row"]["lyrics"], audio_duration=f["duration"],
                generator=torch.Generator("cuda:0").manual_seed(f["seed"]), output="audios")[0]
        G._write_wav(dest, audio, int(self.pipe.sampling_rate), f["duration"],
                     accept_short=True, accept_silent=True)
        return dict(kind="generated", physical_gpu=1, seed=f["seed"], seed_retries=0,
            checkpoint_sha256=cfg.get("sha256"), scale=cfg["scale"],
            prompt_sha256=digest(prompt), lyrics_sha256=digest(f["row"]["lyrics"]),
            runtime_snapshot_sha256=sha(ROOT / "analysis/fresh_seed_20260910/runtime-snapshot.json"),
            worker_sha256=sha(__file__))


def run(smoke=False):
    state = load()
    if digest(read(DATA / "protocol.json")) != state["protocol_hash"]:
        raise ValueError("Frozen protocol changed")
    (DATA / "audio").mkdir(parents=True, exist_ok=True)
    pending = [c for c in state["clips"].values() if not ready(c["id"])]
    if smoke:
        clip = next(c for c in state["clips"].values() if c["config"] == "step600-s1")
        renderer = Renderer(state)
        output = WORK / "smoke-reproduction.wav"
        provenance = renderer.render(clip, output)
        write(WORK / "smoke-audit.json", dict(**provenance, generated_sha256=sha(output),
            original_sha256=clip["source"]["sha256"], exact_match=sha(output)==clip["source"]["sha256"]))
        print("GPU smoke render complete", flush=True)
        return
    renderer = None
    started = time.time()
    for index, clip in enumerate(pending):
        write(DATA / "worker.json", dict(status="rendering", stage=clip["stage"],
            done=index, total=len(pending), started=started, updated=time.time(), current=clip["id"]))
        wav = DATA / "audio" / f"{clip['id']}.wav"
        if "source" in clip:
            source = clip["source"]
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Source recording hash changed")
            if not wav.exists():
                os.link(source["path"], wav)
            if sha(wav) != source["sha256"]:
                raise ValueError("Imported audio hash mismatch")
            provenance = dict(kind="imported", **source)
        elif wav.exists() and read(wav.with_suffix(".provenance.json")):
            provenance = read(wav.with_suffix(".provenance.json"))
            if provenance["identity"] != clip["identity"] or provenance["wav_sha256"] != sha(wav):
                raise ValueError("Resume identity mismatch")
        else:
            if renderer is None:
                renderer = Renderer(state)
            provenance = renderer.render(clip, wav)
            provenance.update(identity=clip["identity"], wav_sha256=sha(wav))
            write(wav.with_suffix(".provenance.json"), provenance)
        finish_clip(clip, wav, provenance)
        print(f"Completed {index+1}/{len(pending)}: {clip['stage']}", flush=True)
    write(DATA / "worker.json", dict(status="complete", done=len(pending), total=len(pending),
        started=started, updated=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.smoke)
        except Exception as e:
            if not args.smoke:
                write(DATA / "worker.json", dict(status="failed", error=str(e), updated=time.time()))
            raise
