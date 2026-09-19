#!/usr/bin/env python3
"""Dual-GPU Music 3 particle-bridge catalog (Yue2 winning formulation, 1200 updates)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = ROOT / "analysis/uni16_fresh3400_20260912/prompts"
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
ORDER = [
    "female", "metal", "male", "pop", "hiphop", "rnb", "indie-rock", "pop-punk",
    "country", "acoustic-folk", "house", "disco-funk", "kpop", "reggaeton",
    "afrobeats", "lofi",
]


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slider_page_html(label: str) -> str:
    """Per-slider live board — Yue2 particle layout with Music 3 soft y-scale."""
    import html as html_lib

    label = html_lib.escape(label)
    body = (Path(__file__).parent / "assets/music3-training-dashboard.html").read_text()
    return (
        "<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Music 3 · {label} · particle bridge</title>"
        "<style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}"
        "a{color:#a9d7ff}</style>"
        f"<p><a href=\"../\">← catalog</a></p>"
        f"<h1>Music 3 · {label} · particle bridge</h1>"
        f"<p>0 = Off · 1 = {label}. Paired-error GAN with routed particles and particle VIC; EMA weights. Trained at +1.</p>"
        "<p id=\"status\">Waiting for training status…</p>"
        f"{body}"
    )


def campaign_argv(manifest, job):
    return [
        manifest["python"], "-u", str(ROOT / "conceptmod/textsliders/train_lora_music3_particle.py"),
        "--name", job["name"], "--save_dir", job["save_dir"], "--steps", str(manifest["steps"]),
        "--seed", "7", "--device", "cuda:0", "--prompts_file", job["train_prompts"],
        "--save_every", "100",
    ]


def prepare(args):
    sys.path.insert(0, str(args.registry.parent.parent))
    from conceptmod.textsliders.music3_particle_bridge import load_prompts
    from app.rewriter import _artist_name_hit

    target = args.queue_dir / "manifest.json"
    if target.exists():
        verify(read(target))
        print("Existing queue verified")
        return
    registry = read(args.registry)
    active = {item["id"]: item for item in registry["sliders"]}
    if set(ORDER) != set(active):
        raise ValueError(f"Expected the 16-slider MiniMax catalog, got {sorted(active)}")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = args.models_root.resolve()
    models.mkdir(parents=True, exist_ok=True)
    dashboard = (Path(__file__).parent / "assets/music3-particle-catalog-dashboard.html").read_text()
    (output / "index.html").write_text(dashboard)
    manifest = dict(
        source_root=str(ROOT),
        python=PY,
        steps=1200,
        seed=7,
        recipe="particle_bridge",
        formulation="anneal-routed-particle-error-music3-v1",
        learning_rates=dict(generator=0.0006, discriminator=0.0009, particles=0.006),
        noise_decay_steps=8000,
        merge_to_trainer=False,
        registry_sha256=sha(args.registry),
        gpus=[0, 1],
        authorization="User requested Music 3 particle-bridge 1200 on both GPUs with live progress page.",
        output_root=str(output),
        models_root=str(models),
        gpu_lock_root=str(args.registry.parent.parent),
        min_free_mib=10000,
        jobs=[],
        files={},
    )
    for index, key in enumerate(ORDER):
        item = active[key]
        train_prompts = PROMPTS / f"{key}-train.yaml"
        eval_prompts = PROMPTS / f"{key}-eval.yaml"
        if not train_prompts.is_file():
            raise FileNotFoundError(train_prompts)
        rows, meta = load_prompts(train_prompts)
        for row in rows:
            hit = _artist_name_hit("", row["positive"] + "\n" + row["neutral"], row["lyrics"])
            if hit:
                raise ValueError(f"{key} prompt names an artist: {hit}")
        job = dict(
            id=key,
            label=item["label_plus"],
            gpu=index % 2,
            name=f"{key}-music3-particle-1200-s7",
            save_dir=str(models / key),
            output_dir=str(output / key),
            train_prompts=str(train_prompts),
            eval_prompts=str(eval_prompts),
            train_rows=len(rows),
            plus_label=meta["plus_label"],
        )
        Path(job["save_dir"]).mkdir(parents=True, exist_ok=True)
        Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(job["output_dir"]) / "index.html").write_text(slider_page_html(job["label"]))
        job["argv"] = campaign_argv(manifest, job)
        manifest["jobs"].append(job)
        manifest["files"][str(train_prompts.relative_to(ROOT))] = sha(train_prompts)
    for name in [
        "scripts/queue_music3_particle_catalog.py",
        "conceptmod/textsliders/train_lora_music3_particle.py",
        "conceptmod/textsliders/music3_particle_bridge.py",
        "conceptmod/textsliders/particle_bridge_gan.py",
        "scripts/assets/music3-particle-catalog-dashboard.html",
        "scripts/assets/music3-training-dashboard.html",
    ]:
        manifest["files"][name] = sha(ROOT / name)
    write(target, manifest)
    write(output / "commands.json", manifest)
    write(output / "report.md", "\n".join([
        "# Music 3 · particle bridge · 1,200",
        "",
        "Yue2 winning formulation (`anneal-routed`) on MiniMax Music 3 LM attention.",
        "Both GPUs. Live page on the listen server.",
        "",
        f"<http://100.90.104.57:8888/{output.name}/>",
        "",
    ]))
    print(f"Prepared {len(manifest['jobs'])} jobs under {args.queue_dir}")


def verify(manifest):
    if manifest["source_root"] != str(ROOT):
        raise ValueError("Queue belongs to a different source directory")
    if manifest["steps"] != 1200 or len(manifest["jobs"]) != 16:
        raise ValueError("Expected the approved 16-slider, 1200-update queue")
    for name, expected in manifest["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen queue input changed: {name}")
    for job in manifest["jobs"]:
        if job["gpu"] not in (0, 1) or job["argv"] != campaign_argv(manifest, job):
            raise ValueError("Queue command or GPU assignment changed")


def completed(manifest, job):
    training = read(Path(job["save_dir"]) / "status.json", {})
    return training.get("completed") == manifest["steps"] and training.get("status") == "complete"


def publish(queue_dir, manifest):
    from scripts.yue2_training_dashboard import publish_metrics

    jobs = []
    for job in manifest["jobs"]:
        state = read(queue_dir / "jobs" / f"{job['id']}.json", dict(status="queued"))
        train = read(Path(job["save_dir"]) / "status.json", {})
        progress = read(Path(job["save_dir"]) / "progress.json", {})
        output = Path(job["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if train:
            # Slider pages poll ./status.json the same way Yue2 campaign pages do.
            write(output / "status.json", dict(
                status=train.get("status"),
                stage=(
                    f"Training {train.get('completed', 0)}/{train.get('total', manifest['steps'])}"
                    if train.get("status") == "training"
                    else train.get("status")
                ),
                completed=train.get("completed", 0),
                total=train.get("total", manifest["steps"]),
            ))
        try:
            if list(Path(job["save_dir"]).glob("updates-from-*.jsonl")):
                publish_metrics(Path(job["save_dir"]), output)
        except (OSError, ValueError) as exc:
            print(f"metrics {job['id']}: {exc}", file=sys.stderr, flush=True)
        jobs.append(dict(
            id=job["id"], label=job["label"], gpu=job["gpu"], status=state.get("status", "queued"),
            completed=train.get("completed", 0), total=manifest["steps"],
            stage=train.get("status", "Queued"),
            metrics={key: progress.get(key) for key in ("g_adv", "particle_vic", "grad_norm", "cos_pos", "step_seconds")},
        ))
    write(Path(manifest["output_root"]) / "queue-state.json", dict(
        updated=time.time(), jobs=jobs,
        workers=[read(queue_dir / f"worker-{gpu}.json", dict(gpu=gpu, status="not_started")) for gpu in (0, 1)],
    ))


def worker(args, manifest):
    stop = False
    child = None

    def interrupted(*_):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            child.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    worker_file = args.queue_dir / f"worker-{args.gpu}.json"

    def status(phase, **extra):
        write(worker_file, dict(gpu=args.gpu, status=phase, pid=os.getpid(), updated=time.time(), **extra))

    with (args.queue_dir / f"worker-{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in manifest["jobs"]:
            if stop:
                break
            if job["gpu"] != args.gpu:
                continue
            record_file = args.queue_dir / "jobs" / f"{job['id']}.json"
            prior = read(record_file, {})
            if prior.get("status") == "complete" and completed(manifest, job):
                continue
            if prior.get("status") == "failed":
                continue
            verify(manifest)
            lease_path = Path(manifest["gpu_lock_root"]) / f".music-gpu-{args.gpu}.lock"
            with lease_path.open("a") as lease:
                while not stop:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        status("waiting_gpu_lock", job=job["id"])
                        time.sleep(2)
                while not stop:
                    free = int(subprocess.check_output(
                        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                        text=True,
                    ).strip())
                    if free >= manifest["min_free_mib"]:
                        break
                    status("waiting_memory", job=job["id"], free_mib=free)
                    time.sleep(5)
                if stop:
                    break
                log_path = args.queue_dir / "logs" / f"{job['id']}.log"
                log_path.parent.mkdir(exist_ok=True)
                status("running", job=job["id"])
                write(record_file, dict(status="running", gpu=args.gpu, id=job["id"], started=time.time(), argv=job["argv"]))
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(args.gpu),
                    HF_HUB_OFFLINE="1",
                    HF_HOME=os.environ.get("HF_HOME", "/ml2/music/.cache/huggingface"),
                    PYTHONPATH=str(ROOT),
                    OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                )
                env.pop("TRANSFORMERS_CACHE", None)
                with log_path.open("a") as log:
                    child = subprocess.Popen(
                        job["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        pass_fds=(lock.fileno(), lease.fileno()),
                    )
                    while child.poll() is None:
                        time.sleep(1)
                phase = (
                    "paused" if stop
                    else "complete" if child.returncode == 0 and completed(manifest, job) else "failed"
                )
                write(record_file, dict(
                    status=phase, gpu=args.gpu, id=job["id"], finished=time.time(),
                    returncode=child.returncode, log=str(log_path),
                ))
                child = None
        status("paused" if stop else "finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "worker", "publish"])
    parser.add_argument("--queue_dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0, 1])
    parser.add_argument("--registry", type=Path, default=Path("/ml2/music/app/sliders.json"))
    parser.add_argument("--models_root", type=Path)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.queue_dir = args.queue_dir.resolve()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        if not args.models_root or not args.output_root:
            parser.error("Preparation needs --models_root and --output_root")
        prepare(args)
        return
    manifest = read(args.queue_dir / "manifest.json")
    verify(manifest)
    if args.mode == "worker":
        if args.gpu is None:
            parser.error("Worker needs --gpu")
        worker(args, manifest)
    else:
        with (args.queue_dir / "publisher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                publish(args.queue_dir, manifest)
                if not args.watch:
                    break
                time.sleep(2)


if __name__ == "__main__":
    main()
