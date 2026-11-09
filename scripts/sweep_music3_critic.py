#!/usr/bin/env python3
"""Sweep Music 3 particle-bridge critic configs on metal; keep the best.

Scores late-window cos lock. Mid-run U-dips are expected on metal — give
runs budget to climb the right arm, and reward recovery instead of
punishing the valley. Writes a leaderboard under analysis/music3_critic_sweep_*/.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import statistics as stats
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# This retained checkout script uses the helper shipped with the same source.
CORE_SOURCE = ROOT / "packages/particle-sliders-core/src"
if str(CORE_SOURCE) not in sys.path:
    sys.path.insert(0, str(CORE_SOURCE))

from particle_sliders.execution.process_identity import (
    capture_process_identity, process_liveness, valid_identity,
)
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
PROMPTS = ROOT / "analysis/uni16_fresh3400_20260912/prompts/metal-train.yaml"

# Capacity-matched transformers first. Fat mix kept as a negative control.
SCREEN = [
    dict(name="bn_t8_w48_l1", critic="bottleneck", critic_tokens=8, critic_width=48, critic_layers=1, critic_heads=4, critic_score_bound=8.0),
    dict(name="bn_t16_w48_l1", critic="bottleneck", critic_tokens=16, critic_width=48, critic_layers=1, critic_heads=4, critic_score_bound=8.0),
    dict(name="hybrid_t8_w48", critic="hybrid", critic_tokens=8, critic_width=48, critic_layers=1, critic_heads=4, critic_score_bound=8.0),
    dict(name="lowrank_r64_t8", critic="lowrank", critic_tokens=8, critic_width=48, critic_layers=1, critic_heads=4, critic_rank=64, critic_score_bound=8.0),
    dict(name="bquery_t8_q4", critic="bquery", critic_tokens=8, critic_queries=4, critic_width=48, critic_layers=1, critic_heads=4, critic_score_bound=8.0),
    dict(name="bn_t8_w48_l2", critic="bottleneck", critic_tokens=8, critic_width=48, critic_layers=2, critic_heads=4, critic_score_bound=8.0),
    dict(name="mix_t8_w48_l1", critic="mix", critic_tokens=8, critic_width=48, critic_layers=1, critic_heads=4, critic_score_bound=8.0),
    dict(name="mlp", critic="mlp"),
]


def run_dir(root: Path, cfg: dict) -> Path:
    return root / "runs" / cfg["name"]


def log_path(root: Path, cfg: dict) -> Path:
    return root / "logs" / f"{cfg['name']}.log"


def load_rows(run: Path):
    rows = []
    for path in sorted(run.glob("updates-from-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def load_status(run: Path):
    marker = run / "status.json"
    if not marker.exists():
        return {}
    return json.loads(marker.read_text())


def score(rows, *, late_frac=0.3):
    """Late-window lock + U-recovery. Do not punish the valley of a U-curve."""
    if len(rows) < 10:
        return dict(ok=False, reason="too_few_steps", n=len(rows))
    late = rows[int(len(rows) * (1 - late_frac)) :]
    cos = [r["cos_pos"] for r in rows]
    cos_late = [r["cos_pos"] for r in late]
    d_adv = [r["d_adv"] for r in late]
    g_adv = [r["g_adv"] for r in late]
    gn = [r["grad_norm"] for r in late]
    gan = [r["particle_gan_grad_norm"] for r in rows]
    # Only count late-window drops — early U valleys are expected.
    drops = 0
    late_start = max(20, int(len(rows) * (1 - late_frac)))
    for i in range(late_start, len(rows)):
        med = stats.median(cos[i - 20 : i])
        if med - cos[i] > 0.15:
            drops += 1
    zero_frac = sum(1 for x in d_adv if x <= 1e-8) / len(d_adv)
    lock = stats.median(cos_late)
    floor = min(cos_late)
    volatility = stats.pstdev(cos_late) if len(cos_late) > 1 else 0.0
    g_med = stats.median(g_adv)
    gn_med = stats.median(gn)
    cos_min = min(cos)
    # Reward climbing the right arm of a U (late lock above the global valley).
    recovery = max(0.0, lock - cos_min)
    fitness = (
        2.0 * lock
        + 0.5 * floor
        + 0.75 * recovery
        - 1.0 * volatility
        - 1.5 * zero_frac
        - 0.05 * max(0.0, g_med - 8.0)
        - 0.002 * max(0.0, gn_med - 80.0)
        - 0.03 * drops
    )
    return dict(
        ok=True,
        fitness=fitness,
        n=len(rows),
        cos_late_med=lock,
        cos_late_min=floor,
        cos_late_std=volatility,
        cos_min=cos_min,
        recovery=recovery,
        drops=drops,
        d_adv_zero_frac=zero_frac,
        d_adv_med=stats.median(d_adv),
        g_adv_med=g_med,
        grad_med=gn_med,
        gan_frac=sum(1 for x in gan if x and x > 0) / len(gan),
    )


def argv_for(cfg, *, steps, save_dir, gpu):
    cmd = [
        PY, "-u", str(ROOT / "packages/particle-sliders-core/src/particle_sliders/textsliders/train_lora_music3_particle.py"),
        "--name", f"metal-{cfg['name']}",
        "--prompts_file", str(PROMPTS),
        "--save_dir", str(save_dir),
        "--steps", str(steps),
        "--until", str(steps),
        "--save_every", str(min(100, steps)),
        "--seed", "7",
        "--device", "cuda:0",
        "--critic", cfg["critic"],
    ]
    for key in (
        "critic_patch", "critic_width", "critic_layers", "critic_heads",
        "critic_tokens", "critic_queries", "critic_rank",
    ):
        if key in cfg:
            cmd += [f"--{key}", str(cfg[key])]
    if "critic_score_bound" in cfg:
        cmd += ["--critic_score_bound", str(cfg["critic_score_bound"])]
    if "d_lr" in cfg:
        cmd += ["--d_lr", str(cfg["d_lr"])]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HUB_OFFLINE="1",
        HF_HOME=os.environ.get("HF_HOME", "/ml2/music/.cache/huggingface"),
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    env.pop("TRANSFORMERS_CACHE", None)
    return cmd, env


def migrate_legacy_dirs(root: Path, cfg: dict):
    """Prefer stable runs/<name>/; adopt older runs/<name>-sN if present."""
    target = run_dir(root, cfg)
    if target.exists():
        return target
    legacy = sorted(root.glob(f"runs/{cfg['name']}-s*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if legacy:
        # Never relocate a live/unverified writer or an incomplete capture.
        prepare_attempt(legacy[0], 0)
        legacy[0].rename(target)
        old_log = root / "logs" / f"{legacy[0].name}.log"
        new_log = log_path(root, cfg)
        if old_log.exists() and not new_log.exists():
            old_log.rename(new_log)
    return target


def completed_steps(run: Path) -> int:
    status = load_status(run)
    if status.get("completed"):
        return int(status["completed"])
    rows = load_rows(run)
    return rows[-1]["step"] if rows else 0


OWNER_SCHEMA = "particle-sliders-sweep-owner-v1"


def owner_path(run):
    return run / "sweep-owner.json"


def read_owner(run):
    try:
        record = json.loads(owner_path(run).read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot verify attempt owner in {run}; retain it and use a new immutable output directory") from error
    if (not isinstance(record, dict)
            or set(record) != {"schema", "process", "controller", "exit_code"}
            or record["schema"] != OWNER_SCHEMA
            or not valid_identity(record["process"])
            or not valid_identity(record["controller"])
            or (record["exit_code"] is not None and type(record["exit_code"]) is not int)):
        raise RuntimeError(f"Invalid attempt owner in {run}; retain it and use a new immutable output directory")
    return record


def write_owner(run, record):
    # Readers never observe a truncated owner or a half-written exit receipt.
    fd, temporary = tempfile.mkstemp(prefix=".sweep-owner-", dir=run)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, owner_path(run))
    finally:
        Path(temporary).unlink(missing_ok=True)


def wait_for_owner(run, *, poll_seconds=5):
    """Wait only for proven-live local work, then require a reaped exit receipt."""
    original = read_owner(run)
    if process_liveness(original["controller"]) == "unknown":
        raise RuntimeError(f"Unverifiable controller in {run}; retained without automatic reroll")
    while True:
        record = read_owner(run)
        if any(record[key] != original[key] for key in ("process", "controller")):
            raise RuntimeError(f"Attempt owner changed in {run}; retained without automatic reroll")
        alive = process_liveness(record["process"])
        if alive == "unknown":
            raise RuntimeError(f"Unverifiable attempt owner in {run}; retained without automatic reroll")
        if alive == "live":
            if record["exit_code"] is not None:
                raise RuntimeError(f"Live attempt has an invalid exit receipt in {run}")
            time.sleep(poll_seconds)
            continue
        if record["exit_code"] is not None:
            if record["exit_code"] != 0:
                raise RuntimeError(f"Attempt owner exited {record['exit_code']} in {run}; retained without automatic reroll")
            return
        # A training marker is never evidence that a dead process will progress.
        if load_status(run).get("status") == "training":
            raise RuntimeError(f"Training owner exited in {run}; retained without automatic reroll")
        if process_liveness(record["controller"]) != "live":
            raise RuntimeError(f"No live controller can certify the exit in {run}; retained without automatic reroll")
        time.sleep(poll_seconds)


def prepare_attempt(run, steps):
    if not run.exists():
        return 0, False
    if owner_path(run).exists():
        wait_for_owner(run)
    status = load_status(run)
    done = completed_steps(run)
    if status.get("status") == "complete" and done >= steps:
        # Historical terminal directories predate owner receipts; keep their
        # existing reuse behavior, without claiming new process provenance.
        return done, True
    checkpoint = run / "state.pt"
    if (status.get("status") in {"complete", "checkpoint_ready", "paused"}
            and checkpoint.is_file() and checkpoint.stat().st_size > 0):
        # Actual full-state/schema admission stays with the trainer's loader.
        return done, False
    raise RuntimeError(f"Attempt {run} has no verified terminal completion or resumable full checkpoint; retain it and use a new immutable output directory")


@contextmanager
def launch_lock(root, cfg):
    locks = root / "owner-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{cfg['name']}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller is preparing this attempt; retain it and retry after ownership is recorded") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def start_attempt(cfg, *, steps, root, gpu):
    with launch_lock(root, cfg):
        save_dir = migrate_legacy_dirs(root, cfg)
        done, reused = prepare_attempt(save_dir, steps)
        if reused:
            result = score(load_rows(save_dir))
            result.update(cfg=cfg, steps=steps, save_dir=str(save_dir), reused=True)
            return dict(result=result)
        save_dir.mkdir(parents=True, exist_ok=True)
        log = log_path(root, cfg)
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd, env = argv_for(cfg, steps=steps, save_dir=save_dir, gpu=gpu)
        mode = "a" if log.exists() else "w"
        handle = log.open(mode)
        proc = None
        try:
            if mode == "a":
                handle.write(f"\n# resume → {steps} from {done}\n")
                handle.flush()
            controller = capture_process_identity()
            if controller is None:
                raise RuntimeError("Cannot establish this controller's local process identity")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            process = capture_process_identity(proc.pid)
            if process is None:
                raise RuntimeError("Cannot establish the started trainer's process identity; attempt retained")
            owner = dict(schema=OWNER_SCHEMA, process=process, controller=controller, exit_code=None)
            write_owner(save_dir, owner)
        except BaseException:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()  # Only the child created by this call.
                proc.wait()
            handle.close()
            raise
        return dict(proc=proc, handle=handle, owner=owner, cfg=cfg,
                    steps=steps, save_dir=save_dir, log=log, done=done)


def finish_attempt(attempt):
    if "result" in attempt:
        return attempt["result"]
    try:
        code = attempt["proc"].wait()
        if read_owner(attempt["save_dir"]) != attempt["owner"]:
            raise RuntimeError("Attempt ownership changed before child exit; output retained")
        write_owner(attempt["save_dir"], dict(attempt["owner"], exit_code=code))
    finally:
        attempt["handle"].close()
    result = score(load_rows(attempt["save_dir"]))
    result.update(cfg=attempt["cfg"], steps=attempt["steps"], save_dir=str(attempt["save_dir"]),
                  log=str(attempt["log"]), returncode=code, reused=False, resumed_from=attempt["done"])
    if code != 0:
        result.update(ok=False, reason=f"train_exit_{code}", fitness=-1e9)
    return result


def run_one(cfg, *, steps, root, gpu):
    return finish_attempt(start_attempt(cfg, steps=steps, root=root, gpu=gpu))


def write_board(root, board):
    board = sorted(board, key=lambda r: r.get("fitness", -1e9), reverse=True)
    (root / "leaderboard.json").write_text(json.dumps(board, indent=2) + "\n")
    lines = ["# Music 3 metal critic sweep", ""]
    for row in board:
        cfg = row.get("cfg", {})
        lines.append(
            f"- **{cfg.get('name')}** fitness={row.get('fitness', float('nan')):.3f} "
            f"late_cos={row.get('cos_late_med', float('nan')):.3f} "
            f"min={row.get('cos_late_min', float('nan')):.3f} "
            f"rec={row.get('recovery', float('nan')):.3f} "
            f"std={row.get('cos_late_std', float('nan')):.3f} "
            f"d0={row.get('d_adv_zero_frac', float('nan')):.2f} "
            f"drops={row.get('drops')} steps={row.get('n', row.get('steps'))}"
        )
    (root / "leaderboard.md").write_text("\n".join(lines) + "\n")
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "analysis/music3_critic_sweep_20260917")
    parser.add_argument("--screen_steps", type=int, default=900)
    parser.add_argument("--final_steps", type=int, default=1800)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "runs").mkdir(exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    board = []

    # Phase 1: screen on both GPUs. Resume short runs up to screen_steps.
    pending = list(SCREEN)
    while pending:
        batch = []
        for gpu in args.gpus:
            if not pending:
                break
            cfg = pending.pop(0)
            print(json.dumps(dict(phase="screen", cfg=cfg["name"], gpu=gpu)), flush=True)
            batch.append((cfg, gpu))
        attempts = []
        error = None
        try:
            for cfg, gpu in batch:
                attempts.append(start_attempt(cfg, steps=args.screen_steps, root=args.root, gpu=gpu))
        except BaseException as failure:
            error = failure
        # Reap every child we started even if another admission/spawn/receipt
        # failed. No foreign process is terminated or otherwise altered.
        for attempt in attempts:
            try:
                result = finish_attempt(attempt)
                board.append(result)
                write_board(args.root, board)
                print(json.dumps(dict(done=result["cfg"]["name"], reused=result["reused"],
                                      fitness=result.get("fitness"), resumed_from=result.get("resumed_from"))), flush=True)
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error

    board = write_board(args.root, board)
    finalists = [r for r in board if r.get("ok") and r.get("steps") == args.screen_steps][: args.top_k]
    print(json.dumps(dict(phase="promote", finalists=[r["cfg"]["name"] for r in finalists])), flush=True)

    # Phase 2: longer runs for top_k (resume from screen checkpoint).
    for i, row in enumerate(finalists):
        cfg = row["cfg"]
        gpu = args.gpus[i % len(args.gpus)]
        print(json.dumps(dict(phase="final", cfg=cfg["name"], gpu=gpu, steps=args.final_steps)), flush=True)
        result = run_one(cfg, steps=args.final_steps, root=args.root, gpu=gpu)
        board.append(result)
        write_board(args.root, board)
        print(json.dumps(dict(
            final_done=cfg["name"], fitness=result.get("fitness"),
            cos_late_med=result.get("cos_late_med"), recovery=result.get("recovery"),
            drops=result.get("drops"),
        )), flush=True)

    board = write_board(args.root, board)
    best = next((r for r in board if r.get("ok")), None)
    if best is None:
        raise SystemExit("No successful critic configs")
    (args.root / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(dict(best=best["cfg"]["name"], fitness=best["fitness"], steps=best.get("n", best["steps"]))), flush=True)


if __name__ == "__main__":
    main()
