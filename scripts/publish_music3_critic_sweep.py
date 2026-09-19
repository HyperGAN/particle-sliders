#!/usr/bin/env python3
"""Publish Music 3 critic-sweep runs to the live listen page (:8888)."""
from __future__ import annotations

import argparse
import fcntl
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.yue2_training_dashboard import publish_metrics

DASH = (Path(__file__).parent / "assets/music3-training-dashboard.html").read_text()


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def run_page(name: str) -> str:
    label = html.escape(name)
    return (
        "<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Music 3 · critic sweep · {label}</title>"
        "<style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}"
        "a{color:#a9d7ff}</style>"
        f"<p><a href=\"../\">← critic sweep</a></p>"
        f"<h1>Music 3 · metal · {label}</h1>"
        "<p>Transformer critic screen on metal. Expect a U-shaped cos curve — not the MLP lock shape.</p>"
        "<p id=\"status\">Waiting…</p>"
        f"{DASH}"
        "<script>setInterval(async()=>{try{const r=await fetch('status.json',{cache:'no-store'});"
        "const d=await r.json();document.getElementById('status').textContent="
        "`${d.status||'?'}: ${d.completed||0}/${d.total||0}`;}catch(e){}},1000);</script>"
    )


def index_html(jobs: list[dict]) -> str:
    rows = []
    for job in jobs:
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(job['id'])}/\">{html.escape(job['id'])}</a></td>"
            f"<td>{html.escape(str(job.get('critic','')))}</td>"
            f"<td>{job.get('status','?')}</td>"
            f"<td>{job.get('completed',0)}/{job.get('total',0)}</td>"
            f"<td>{job.get('cos_pos','—')}</td>"
            f"<td>{job.get('fitness','—')}</td>"
            "</tr>"
        )
    board = ""
    board_path = Path(jobs[0]["sweep_root"]) / "leaderboard.md" if jobs else None
    if board_path and board_path.exists():
        board = f"<pre style=\"background:#1b222b;padding:16px;border-radius:10px;overflow:auto\">{html.escape(board_path.read_text())}</pre>"
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Music 3 · critic sweep</title>
<style>
body{{max-width:1100px;margin:36px auto;padding:0 22px;background:#151a20;color:#e3edf8;font:16px/1.5 system-ui}}
a{{color:#9fd6ff}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #394957}}
th{{color:#a8bacb;font-size:13px}}.muted{{color:#a8bacb}}
</style>
<h1>Music 3 · metal critic sweep</h1>
<p class="muted">Live curves for each critic config. Auto-refreshes every 2s via the listen server.</p>
<p class="muted" id="updated">Connecting…</p>
{board}
<table><thead><tr><th>Config</th><th>Critic</th><th>Status</th><th>Steps</th><th>cos_pos</th><th>fitness</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=6>No runs yet</td></tr>'}</tbody></table>
<script>
async function tick(){{
  try{{
    const r=await fetch('queue-state.json',{{cache:'no-store'}});
    const d=await r.json();
    document.getElementById('updated').textContent='Updated '+new Date(d.updated*1000).toLocaleTimeString();
    // soft reload table via location when count changes
    if(d.reload) location.reload();
  }}catch(e){{}}
}}
setInterval(tick,2000);tick();
</script>
"""


def publish(sweep_root: Path, output: Path):
    runs_root = sweep_root / "runs"
    runs = []
    if runs_root.exists():
        runs = sorted(
            [p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    board = {}
    board_path = sweep_root / "leaderboard.json"
    if board_path.exists():
        for row in json.loads(board_path.read_text()):
            name = row.get("cfg", {}).get("name")
            if name:
                board[name] = row
                steps = row.get("steps") or row.get("n")
                if steps:
                    board[f"{name}-s{steps}"] = row
    jobs = []
    for run in runs:
        out = output / run.name
        out.mkdir(parents=True, exist_ok=True)
        write(out / "index.html", run_page(run.name))
        train = {}
        status = run / "status.json"
        if status.exists():
            train = json.loads(status.read_text())
            write(out / "status.json", json.dumps(dict(
                status=train.get("status"),
                completed=train.get("completed", 0),
                total=train.get("total", 0),
            ), indent=2) + "\n")
        try:
            if list(run.glob("updates-from-*.jsonl")):
                publish_metrics(run, out)
        except (OSError, ValueError) as exc:
            print(f"metrics {run.name}: {exc}", flush=True)
        progress = {}
        prog = run / "progress.json"
        if prog.exists():
            progress = json.loads(prog.read_text())
        scored = board.get(run.name, {}) or board.get(run.name.split("-s")[0], {})
        if run.name.startswith("mlp"):
            critic = "mlp"
        elif run.name.startswith("query"):
            critic = "query"
        elif run.name.startswith("mix"):
            critic = "mix"
        else:
            critic = "other"
        jobs.append(dict(
            id=run.name,
            critic=critic,
            status=train.get("status", "queued"),
            completed=train.get("completed", 0),
            total=train.get("total", 0),
            cos_pos=None if progress.get("cos_pos") is None else round(float(progress["cos_pos"]), 4),
            fitness=None if scored.get("fitness") is None else round(float(scored["fitness"]), 3),
            sweep_root=str(sweep_root),
        ))
    write(output / "index.html", index_html(jobs))
    write(output / "queue-state.json", json.dumps(dict(updated=time.time(), jobs=jobs, reload=False), indent=2) + "\n")
    if (sweep_root / "leaderboard.md").exists():
        write(output / "leaderboard.md", (sweep_root / "leaderboard.md").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_root", type=Path, default=ROOT / "analysis/music3_critic_sweep_20260917")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/listen/music3-critic-sweep-20260917")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            publish(args.sweep_root, args.output)
            if not args.watch:
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
