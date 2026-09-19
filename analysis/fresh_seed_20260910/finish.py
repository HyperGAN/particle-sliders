"""Monitor both training arms, render the locked comparison, and publish CE results."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
PAGE = ROOT / "eval/listen/fresh-seed-lofi-20260910"
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
LABELS = {"parent600": "Starting checkpoint · 600", "published660": "Published · 660",
          "fixed660": "One prompt · fixed seed", "fresh660": "One prompt · fresh seed each step"}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(phase, **details):
    arms = {a: read(WORK / f"lofi-{a}-history/status.json", {"status": "starting"}) for a in ("fixed", "fresh")}
    value = dict(phase=phase, arms=arms, updated=time.time(), **details)
    write(PAGE / "status.json", value)
    write(WORK / "status.json", value)


def publish(results=None):
    protocol = read(WORK / "protocol.json")
    sources = {}
    for row in protocol["evaluation"]["rows"]:
        spec = read(PAGE / f"row-{row}/render_spec.json")
        if spec:
            for checkpoint in spec["checkpoints"]:
                sources[checkpoint["path"]] = checkpoint
    checkpoint_paths = weights()
    lookup = {str(p): key for key, p in checkpoint_paths.items()}
    records = results["records"] if results else []
    scores = {(r["checkpoint"]["path"], r["seed"], Path(r["candidate"]["audio"]).parent.parent.name): r for r in records}
    pairs = []
    for row in protocol["evaluation"]["rows"]:
        for seed in protocol["evaluation"]["seeds"]:
            takes = []
            for key, path in checkpoint_paths.items():
                folder = PAGE / f"row-{row}" / f"{path.stem}-s{seed}"
                if not (folder / "checkpoint.json").exists():
                    folder = PAGE / f"prewarm-row-{row}" / f"{path.stem}-s{seed}"
                if not (folder / "checkpoint.json").exists():
                    continue
                candidates = list(folder.glob("02_slider_*_plus1.wav"))
                if not candidates:
                    continue
                wav = candidates[0]
                dest = PAGE / "audio" / f"r{row}-s{seed}-{key}.wav"
                mp3 = dest.with_suffix(".mp3")
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    os.link(wav, dest)
                if not mp3.exists():
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(dest), "-map_metadata", "-1",
                        "-c:a", "libmp3lame", "-b:a", "192k", "-threads", "1", str(mp3)], check=True)
                score = scores.get((str(path), seed, f"row-{row}"))
                takes.append(dict(key=key, label=LABELS[key], wav=f"audio/{dest.name}", mp3=f"audio/{mp3.name}",
                    sha256=sha(dest), mp3_sha256=sha(mp3), ce=score["candidate"]["enjoyment"] if score else None,
                    pq=score["candidate"]["production"] if score else None))
            pairs.append(dict(row=row, seed=seed, takes=takes))
    summary = {}
    if records:
        by_key = {key: [r for r in records if lookup[r["checkpoint"]["path"]] == key] for key in LABELS}
        for key, group in by_key.items():
            summary[key] = dict(mean_ce=statistics.mean(r["candidate"]["enjoyment"] for r in group),
                examples=len(group), mean_pq=statistics.mean(r["candidate"]["production"] for r in group))
        fixed = {(r["fixture"], r["seed"]): r for r in by_key["fixed660"]}
        differences = [r["candidate"]["enjoyment"]-fixed[(r["fixture"],r["seed"])]["candidate"]["enjoyment"] for r in by_key["fresh660"]]
        summary["fresh_minus_fixed"] = dict(mean_ce=statistics.mean(differences), wins=sum(x>0 for x in differences),
            total=len(differences), worst=min(differences), paired_deltas=differences)
        write(WORK / "summary.json", summary)
    write(PAGE / "manifest.json", dict(pairs=pairs, summary=summary,
        metric="Content Enjoyment; equal weight per heldout prompt and seed", original_levels=True))


def weights():
    return {
        "parent600": ROOT / "models/uni16-gan-v1/lofi-warmup600-a02/lofi-warmup600-a02_last.safetensors",
        "published660": ROOT / "models/uni16-gan-v1/lofi-bounded660-a01/lofi-bounded660-a01_step660.safetensors",
        "fixed660": WORK / "lofi-fixed-history/lofi-fixed-history_step660.safetensors",
        "fresh660": WORK / "lofi-fresh-history/lofi-fresh-history_step660.safetensors",
    }


def main():
    PAGE.mkdir(exist_ok=True)
    (PAGE / "index.html").write_text((WORK / "page.html").read_text())
    publish()
    protocol = read(WORK / "protocol.json")
    while True:
        publish()
        status("Training")
        arms = [read(WORK / f"lofi-{a}-history/status.json", {}) for a in ("fixed", "fresh")]
        if all(a.get("status") == "complete" for a in arms):
            break
        for arm in ("fixed", "fresh"):
            failure = read(WORK / f"lofi-{arm}-history/failure.json")
            if failure:
                status("Training stopped", error=failure)
                raise RuntimeError(f"{arm} failed: {failure}")
        time.sleep(10)
    while read(WORK / "prewarm.json", {}).get("status") != "complete":
        if read(WORK / "prewarm.json", {}).get("status") == "failed":
            raise RuntimeError("Reference pre-render failed; inspect the retained log")
        status("Preparing held-out reference audio")
        time.sleep(10)
    reuse = []
    for row in protocol["evaluation"]["rows"]:
        early = PAGE / f"prewarm-row-{row}"
        spec = read(early / "render_spec.json")
        if not spec:
            continue
        assert spec["row"] == row and spec["seeds"] == protocol["evaluation"]["seeds"]
        assert spec["prompts_sha256"] == sha(Path(protocol["evaluation"]["prompts"]))
        target = PAGE / f"row-{row}"
        target.mkdir(exist_ok=True)
        for folder in early.iterdir():
            if not folder.is_dir() or not (folder / "checkpoint.json").exists():
                continue
            destination = target / folder.name
            destination.mkdir(exist_ok=True)
            for original in folder.iterdir():
                if not original.is_file():
                    continue
                copy = destination / original.name
                if not copy.exists():
                    shutil.copyfile(original, copy)
                assert sha(copy) == sha(original)
                reuse.append(dict(source=str(original), target=str(copy), sha256=sha(copy)))
    write(WORK / "reference-reuse.json", reuse)
    # A renderer may have been restarted independently after a GPU handover.
    # Wait for it instead of launching a second pipeline onto its device.
    while subprocess.run(["systemctl", "--user", "is-active", "--quiet", "music-seed-render-row3-20260910"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        publish()
        status("Rendering held-out comparisons", rendered=sum(len(p["takes"]) for p in read(PAGE / "manifest.json")["pairs"]), total=16)
        time.sleep(10)
    processes = []
    for row, gpu in zip(protocol["evaluation"]["rows"], (0, 1)):
        if all((PAGE / f"row-{row}" / f"{p.stem}-s{seed}" / "checkpoint.json").exists()
               for p in weights().values() for seed in protocol["evaluation"]["seeds"]):
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
            PYTHONPATH=f"{WORK}/runtime:/ml2/music", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        log = (WORK / f"render-row-{row}.log").open("a")
        command = [PY, "-u", str(WORK / "runtime/analysis/gan_bcap/render_v2.py"), "--weights",
            *map(str, weights().values()), "--out", str(PAGE / f"row-{row}"), "--prompts", protocol["evaluation"]["prompts"],
            "--row", str(row), "--seeds", *map(str, protocol["evaluation"]["seeds"]), "--duration", "20"]
        processes.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    while any(p.poll() is None for p, _ in processes):
        publish()
        status("Rendering held-out comparisons", rendered=sum(len(p["takes"]) for p in read(PAGE / "manifest.json")["pairs"]), total=16)
        time.sleep(10)
    for process, log in processes:
        log.close()
        if process.returncode:
            status("Rendering stopped", error="See the preserved render log")
            raise RuntimeError("Render failed")
    publish()
    status("Measuring Content Enjoyment")
    while subprocess.run(["systemctl", "--user", "is-active", "--quiet", "music-seed-premeasure-20260910"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
        PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with (WORK / "score.log").open("a") as log:
        subprocess.run([PY, "-u", str(ROOT / "analysis/uni16_20260906/score.py"), "--id", "lofi", "--folders",
            *[str(PAGE / f"row-{r}") for r in protocol["evaluation"]["rows"]], "--output", str(WORK / "scores.json")],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = read(WORK / "scores.json")
    assert len(results["records"]) == 16
    publish(results)
    status("Complete", summary=read(WORK / "summary.json"))


if __name__ == "__main__":
    main()
