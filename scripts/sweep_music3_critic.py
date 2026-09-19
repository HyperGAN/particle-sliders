#!/usr/bin/env python3
"""Sweep Music 3 particle-bridge critic configs on metal; keep the best.

Scores late-window cos lock. Mid-run U-dips are expected on metal — give
runs budget to climb the right arm, and reward recovery instead of
punishing the valley. Writes a leaderboard under analysis/music3_critic_sweep_*/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics as stats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
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
        PY, "-u", str(ROOT / "conceptmod/textsliders/train_lora_music3_particle.py"),
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


def run_one(cfg, *, steps, root, gpu):
    save_dir = migrate_legacy_dirs(root, cfg)
    log = log_path(root, cfg)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = completed_steps(save_dir) if save_dir.exists() else 0
    status = load_status(save_dir) if save_dir.exists() else {}
    if status.get("status") == "complete" and done >= steps:
        result = score(load_rows(save_dir))
        result.update(cfg=cfg, steps=steps, save_dir=str(save_dir), reused=True)
        return result
    if save_dir.exists() and done == 0 and not (save_dir / "state.pt").exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd, env = argv_for(cfg, steps=steps, save_dir=save_dir, gpu=gpu)
    mode = "a" if log.exists() and done > 0 else "w"
    with log.open(mode) as handle:
        if mode == "a":
            handle.write(f"\n# resume → {steps} from {done}\n")
            handle.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    rows = load_rows(save_dir)
    result = score(rows)
    result.update(
        cfg=cfg, steps=steps, save_dir=str(save_dir), log=str(log),
        returncode=proc.returncode, reused=False, resumed_from=done,
    )
    if proc.returncode != 0:
        result["ok"] = False
        result["reason"] = f"train_exit_{proc.returncode}"
        result["fitness"] = -1e9
    return result


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
        procs = []
        for cfg, gpu in batch:
            save_dir = migrate_legacy_dirs(args.root, cfg)
            log = log_path(args.root, cfg)
            log.parent.mkdir(parents=True, exist_ok=True)
            done = completed_steps(save_dir) if save_dir.exists() else 0
            status = load_status(save_dir) if save_dir.exists() else {}
            if status.get("status") == "complete" and done >= args.screen_steps:
                result = score(load_rows(save_dir))
                result.update(cfg=cfg, steps=args.screen_steps, save_dir=str(save_dir), reused=True)
                board.append(result)
                write_board(args.root, board)
                print(json.dumps(dict(
                    done=cfg["name"], reused=True,
                    **{k: result.get(k) for k in ("fitness", "cos_late_med", "recovery", "drops")},
                )), flush=True)
                continue
            if save_dir.exists() and done == 0 and not (save_dir / "state.pt").exists():
                shutil.rmtree(save_dir)
            # If a live trainer still owns this dir, wait for it instead of wiping.
            if status.get("status") == "training" and done < args.screen_steps:
                print(json.dumps(dict(wait=cfg["name"], completed=done, until=status.get("until"))), flush=True)
                while True:
                    time.sleep(5)
                    status = load_status(save_dir)
                    done = completed_steps(save_dir)
                    if status.get("status") in {"complete", "failed", "checkpoint_ready"}:
                        break
                    if status.get("status") != "training":
                        break
                if done >= args.screen_steps and status.get("status") == "complete":
                    result = score(load_rows(save_dir))
                    result.update(cfg=cfg, steps=args.screen_steps, save_dir=str(save_dir), reused=True)
                    board.append(result)
                    write_board(args.root, board)
                    print(json.dumps(dict(
                        done=cfg["name"], waited=True,
                        **{k: result.get(k) for k in ("fitness", "cos_late_med", "recovery", "drops")},
                    )), flush=True)
                    continue
            save_dir.mkdir(parents=True, exist_ok=True)
            cmd, env = argv_for(cfg, steps=args.screen_steps, save_dir=save_dir, gpu=gpu)
            mode = "a" if log.exists() and done > 0 else "w"
            handle = log.open(mode)
            if mode == "a":
                handle.write(f"\n# resume → {args.screen_steps} from {done}\n")
                handle.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            procs.append((cfg, gpu, proc, handle, save_dir, log, done))
        for cfg, gpu, proc, handle, save_dir, log, done in procs:
            code = proc.wait()
            handle.close()
            result = score(load_rows(save_dir))
            result.update(
                cfg=cfg, steps=args.screen_steps, save_dir=str(save_dir), log=str(log),
                returncode=code, reused=False, resumed_from=done,
            )
            if code != 0:
                result["ok"] = False
                result["reason"] = f"train_exit_{code}"
                result["fitness"] = -1e9
            board.append(result)
            write_board(args.root, board)
            print(json.dumps(dict(
                done=cfg["name"], fitness=result.get("fitness"),
                cos_late_med=result.get("cos_late_med"), recovery=result.get("recovery"),
                drops=result.get("drops"), d0=result.get("d_adv_zero_frac"),
                resumed_from=done,
            )), flush=True)

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
