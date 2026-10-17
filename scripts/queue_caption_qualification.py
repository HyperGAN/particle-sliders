"""Closed low-priority CPU training campaigns; no GPU, downloads, or retries."""

import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/nvme/sliders/qualification")
SOURCE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("tiny", "bonsai"), required=True)
    args = parser.parse_args()
    run = ROOT / ("runs/" + args.model + "-caption-native-1200-20261017")
    recipe = json.loads((run / "recipe.json").read_text())
    lock = (run / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "queue-launch.json").exists() or (run / "progress.json").exists():
        raise ValueError(
            "Inspect existing native CPU receipt before any continuation; no retries"
        )
    stop = False
    active = None

    def stopping(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stopping)

    def event(kind, **values):
        value = dict(
            event=kind,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(),
            **values,
        )
        pending = run / "queue-status.pending"
        pending.write_text(json.dumps(value, indent=2) + "\n")
        pending.replace(run / "queue-status.json")
        with (run / "queue-events.jsonl").open("a") as f:
            f.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    (run / "queue-launch.json").write_text(
        json.dumps(
            dict(
                pid=os.getpid(),
                code_commit=recipe["code_commit"],
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                gpu_used=False,
            ),
            indent=2,
        )
        + "\n"
    )
    if args.model == "bonsai":
        event("waiting_for_owned_uncached20_smoke", status="waiting")
        start = time.monotonic()
        while True:
            path = ROOT / "tests/bonsai-cpu20-authorized-20261017/20-receipt.json"
            value = json.loads(path.read_text()) if path.exists() else {}
            if value.get("status") == "passed":
                break
            if value.get("status") not in (None, "running"):
                event(
                    "dependency_failed",
                    status="blocked",
                    reason="Official uncached20-update Bonsai smoke did not pass",
                )
                return
            if stop or time.monotonic() - start > 7200:
                event("wait_stopped", status="stopped" if stop else "blocked")
                return
            time.sleep(5)
    stages = (
        ["prepare", "canary", "full"] if args.model == "bonsai" else ["canary", "full"]
    )
    for stage in stages:
        if stop:
            event("stopped", status="stopped")
            return
        if (
            subprocess.check_output(
                ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
            ).strip()
            != recipe["code_commit"]
        ):
            raise ValueError("Caption source pin changed")
        subprocess.run(
            ["git", "-C", str(SOURCE), "diff", "--quiet", "HEAD"], check=True
        )
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TMPDIR="/mnt/nvme/sliders/tmp",
            XDG_CACHE_HOME="/mnt/nvme/sliders/cache/xdg",
        )
        worker = (
            ["scripts/prepare_bonsai_readouts.py", "--recipe", str(run / "recipe.json")]
            if stage == "prepare"
            else [
                "scripts/qualify_captions.py",
                "--model",
                args.model,
                "--until",
                "100" if stage == "canary" else "1200",
            ]
        )
        cmd = [
            "nice",
            "-n",
            "15",
            "ionice",
            "-c",
            "3",
            str(ROOT / "envs/cpu/bin/python"),
            "-u",
            *worker,
        ]
        cap = recipe["limits"][stage + "_seconds"]
        receipt = dict(
            command=cmd,
            stage=stage,
            status="running",
            timeout_seconds=cap,
            rss_cap_gib=8,
            host_reserve_gib=8,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        log = (run / (stage + ".log")).open("x")
        start = time.monotonic()
        active = subprocess.Popen(
            cmd,
            cwd=SOURCE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        receipt["pid"] = active.pid
        path = run / (stage + "-receipt.json")
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        event(
            "cpu_worker_started", status="running", stage=stage, worker_pid=active.pid
        )
        peak = 0
        reason = None
        while active.poll() is None:
            available = next(
                int(v.split()[1])
                for v in Path("/proc/meminfo").read_text().splitlines()
                if v.startswith("MemAvailable:")
            )
            rss = 0
            for proc in Path("/proc").glob("[0-9]*"):
                try:
                    if os.getpgid(int(proc.name)) == active.pid:
                        rss += next(
                            (
                                int(v.split()[1])
                                for v in (proc / "status").read_text().splitlines()
                                if v.startswith("VmRSS:")
                            ),
                            0,
                        )
                except (ProcessLookupError, PermissionError, FileNotFoundError):
                    pass
            peak = max(peak, rss)
            if available < 8 * 1024**2:
                reason = "Host available below8GiB"
            if rss > 8 * 1024**2:
                reason = "AggregateCPUworkerRSSabove8GiB"
            if shutil.disk_usage("/mnt/nvme").free < 100 * 1024**3:
                reason = "NVMe reserve below100GiB"
            if time.monotonic() - start > cap:
                reason = "Closed native CPU stage deadline"
            if stop:
                reason = "Own coordinator stop requested"
            if reason:
                os.killpg(active.pid, signal.SIGTERM)
                try:
                    active.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait()
                break
            time.sleep(2)
        receipt.update(
            status="passed" if active.returncode == 0 else "failed_or_stopped",
            exit_code=active.returncode,
            wall_seconds=time.monotonic() - start,
            peak_rss_kib=peak,
            stop_reason=reason,
        )
        log.close()
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        active = None
        if receipt["exit_code"] != 0:
            event("native_stage_failed", status="blocked", stage=stage, reason=reason)
            return
        if stage == "prepare":
            value = json.loads((run / "cache-preparation.json").read_text())
            if value.get("status") != "verified_exact_frozen_readouts":
                raise ValueError("Exact frozen feature gate missing")
        else:
            value = json.loads((run / "progress.json").read_text())
            expected = 100 if stage == "canary" else 1200
            if (
                value.get("latest_step") != expected
                or value.get("committed_step") != expected
                or not value.get("zero_exact")
                or not value.get("ema_export_reload_exact")
            ):
                event(
                    "native_recovery_or_replay_gate_failed",
                    status="blocked",
                    stage=stage,
                )
                return
            if (
                stage == "canary"
                and value["projected_1200_update_seconds"]
                > recipe["limits"]["full_seconds"]
            ):
                event(
                    "measured_full_budget_failed",
                    status="blocked",
                    projected_1200_update_seconds=value[
                        "projected_1200_update_seconds"
                    ],
                )
                return
        event("native_stage_complete", status="running", stage=stage)
    event(
        "native_1200_update_budget_complete",
        status="awaiting_quality_review"
        if value.get("readout_transfer_gate_passed")
        else "native_budget_complete_quality_gate_failed",
        latest_step=1200,
        quality_qualified=False,
        readout_transfer_gate_passed=value.get("readout_transfer_gate_passed"),
    )


if __name__ == "__main__":
    main()
