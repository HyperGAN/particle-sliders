#!/usr/bin/env python3
"""Publish the YuE2 dual-GPU critic search to the live listen page (:8888)."""
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

DASH = (Path(__file__).parent / "assets/yue2-training-dashboard.html").read_text()


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def critic_label(name: str, search: dict) -> str:
    for candidate in search.get("candidates") or []:
        if candidate.get("name") == name:
            return str(candidate.get("critic") or "?")
    if name.startswith("mlp"):
        return "mlp"
    if "mix" in name:
        return "mix"
    if name.startswith("bottleneck"):
        return "bottleneck"
    if name.startswith("hybrid"):
        return "hybrid"
    if name.startswith("lowrank"):
        return "lowrank"
    if name.startswith("bquery"):
        return "bquery"
    return "other"


def training_fitness(run: Path) -> float | None:
    try:
        from scripts.music_architecture_scoreboard import training_rows, training_score
        score = training_score(training_rows(run)).get("training_fitness")
        return None if score is None else float(score)
    except Exception:
        return None


def run_page(name: str) -> str:
    label = html.escape(name)
    return (
        "<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>YuE2 · critic search · {label}</title>"
        "<style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}"
        "a{color:#a9d7ff}</style>"
        f"<p><a href=\"../\">← critic search</a></p>"
        f"<h1>YuE2 · metal · {label}</h1>"
        "<p>Transferred Music 3 particle/critic formulation on YuE2. Formulation and architecture only — not Music 3 weights.</p>"
        "<p id=\"status\">Waiting…</p>"
        f"{DASH}"
        "<script>setInterval(async()=>{try{const r=await fetch('status.json',{cache:'no-store'});"
        "const d=await r.json();document.getElementById('status').textContent="
        "`${d.status||'?'}: ${d.completed||0}/${d.total||0}`;}catch(e){}},1000);</script>"
    )


def index_html(jobs: list[dict], board_md: str | None) -> str:
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
    if board_md:
        board = (
            "<pre style=\"background:#1b222b;padding:16px;border-radius:10px;overflow:auto\">"
            f"{html.escape(board_md)}</pre>"
        )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YuE2 · critic search</title>
<style>
body{{max-width:1100px;margin:36px auto;padding:0 22px;background:#151a20;color:#e3edf8;font:16px/1.5 system-ui}}
a{{color:#9fd6ff}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #394957}}
th{{color:#a8bacb;font-size:13px}}.muted{{color:#a8bacb}}
</style>
<h1>YuE2 · metal critic search</h1>
<p class="muted">Dual-GPU architecture screen transferred from Music 3. Promotion still needs matched held-out enjoyment/production.</p>
<p class="muted" id="updated">Connecting…</p>
{board}
<table><thead><tr><th>Config</th><th>Critic</th><th>Status</th><th>Steps</th><th>cos_pos</th><th>train fitness</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=6>No runs yet</td></tr>'}</tbody></table>
<script>
async function tick(){{
  try{{
    const r=await fetch('queue-state.json',{{cache:'no-store'}});
    const d=await r.json();
    document.getElementById('updated').textContent='Updated '+new Date(d.updated*1000).toLocaleTimeString();
  }}catch(e){{}}
}}
setInterval(tick,2000);tick();
</script>
"""


_LAST_BOARD_REFRESH = 0.0


def refresh_training_board(search_root: Path, search: dict, min_interval_sec: float = 30.0):
    """Keep a training-only scoreboard fresh between controller batches."""
    global _LAST_BOARD_REFRESH
    now = time.time()
    if now - _LAST_BOARD_REFRESH < min_interval_sec:
        return
    runs_root = search_root / "runs"
    if not runs_root.exists():
        return
    have = []
    for candidate in search.get("candidates") or []:
        name = candidate.get("name")
        if name and (runs_root / name / "status.json").exists():
            have.append(candidate)
    if not have:
        return
    try:
        from scripts.search_yue2_critic import provisional_board
        provisional_board(search_root, have)
        _LAST_BOARD_REFRESH = now
    except Exception as exc:
        print(f"scoreboard refresh: {exc}", flush=True)


def publish(search_root: Path, output: Path):
    search = {}
    search_path = search_root / "search.json"
    if search_path.exists():
        search = json.loads(search_path.read_text())
    refresh_training_board(search_root, search)
    runs_root = search_root / "runs"
    runs = []
    if runs_root.exists():
        runs = sorted(
            [p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=lambda p: p.name,
        )
    board_path = search_root / "scoreboard.json"
    board_rows = {}
    if board_path.exists():
        payload = json.loads(board_path.read_text())
        for row in payload.get("rows") or []:
            if row.get("model") == "yue2" and row.get("candidate"):
                board_rows[row["candidate"]] = row
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
        scored = board_rows.get(run.name, {})
        fitness = scored.get("training_fitness")
        if fitness is None:
            fitness = training_fitness(run)
        jobs.append(dict(
            id=run.name,
            critic=critic_label(run.name, search),
            status=train.get("status", "queued"),
            completed=train.get("completed", 0),
            total=train.get("total", 0),
            cos_pos=None if progress.get("cos_pos") is None else round(float(progress["cos_pos"]), 4),
            fitness=None if fitness is None else round(float(fitness), 3),
        ))
    board_md = None
    if (search_root / "scoreboard.md").exists():
        board_md = (search_root / "scoreboard.md").read_text()
        write(output / "scoreboard.md", board_md)
    write(output / "index.html", index_html(jobs, board_md))
    write(output / "queue-state.json", json.dumps(dict(updated=time.time(), jobs=jobs), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_root", type=Path, default=ROOT / "analysis/yue2_critic_search_20260917")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/listen/yue2-critic-search-20260917")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            publish(args.search_root, args.output)
            if not args.watch:
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
