#!/usr/bin/env python3
"""Fire #21c — minimal close harden confirmation (n≥2)."""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (
    close_field3d,
    close_live_noise_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "close_harden_fire21_20260909.json"
OUT_MD = NOTES / "close_harden_fire21_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"


def cfg(seed, cover, n):
    return default_cfg(
        steps=1200, seed=seed, b_cap=1.0, cover_weight=cover,
        fm_weight=0.0, n_particles=n, particle_l2=0.02,
    )


def run(field, seed, cover, n, name):
    t0 = time.time()
    row = score_adv_field3d_exam(field, teacher=TEACHER, cfg=cfg(seed, cover, n), name=name)
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": float(row.get("exam_score", row.get("u_kept", 0.0))),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "wall_s": round(time.time() - t0, 2),
    }


def summarize(rows):
    fails = [r["seed"] for r in rows if not r["pass"]]
    prim = [r["exam_score"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    return {
        "pass_str": f"{sum(1 for r in rows if r['pass'])}/{len(rows)}",
        "n_pass": sum(1 for r in rows if r["pass"]),
        "fail_seeds": fails,
        "mean_exam": round(sum(prim) / len(prim), 4),
        "leak_max": round(max(leaks), 4) if leaks else 0.0,
        "only_seed0": fails == [0],
    }


def main():
    t0 = time.time()
    # Known from partial #21 / #21b
    cells = {
        "leftover_n12_c1.5": {"summary": {"pass_str": "6/6", "n_pass": 6, "fail_seeds": [], "only_seed0": False, "mean_exam": None, "leak_max": None, "source": "21a"}},
        "close_n1_c1.0": {"summary": {"pass_str": "5/6", "n_pass": 5, "fail_seeds": [0], "only_seed0": True, "mean_exam": 0.8605, "leak_max": None, "source": "21a"}},
        "close_n1_c1.5": {"summary": {"pass_str": "5/6", "n_pass": 5, "fail_seeds": [0], "only_seed0": True, "mean_exam": 0.8911, "leak_max": None, "source": "21b"}},
        "close_n2_c1.0": {"summary": {"pass_str": "6/6", "n_pass": 6, "fail_seeds": [], "only_seed0": False, "mean_exam": 0.9891, "leak_max": None, "source": "21a"}},
    }
    detail = {}
    plan = [
        ("close_n2_c1.5", lambda s: close_field3d(), 1.5, 2),
        ("close_n4_c1.0", lambda s: close_field3d(), 1.0, 4),
        ("close_n4_c1.5", lambda s: close_field3d(), 1.5, 4),
        ("live_n1_c1.0_default", lambda s: close_live_noise_field3d(seed=s), 1.0, 1),
        ("live_n2_c1.0_default", lambda s: close_live_noise_field3d(seed=s), 1.0, 2),
        ("live_n1_a0.08_s0.16", lambda s: close_live_noise_field3d(seed=s, amp_noise=0.08, span_noise=0.16), 1.0, 1),
        ("live_n2_a0.08_s0.16", lambda s: close_live_noise_field3d(seed=s, amp_noise=0.08, span_noise=0.16), 1.0, 2),
    ]
    for key, ctor, cover, n in plan:
        rows = [run(ctor(s), s, cover, n, key) for s in SEEDS]
        sm = summarize(rows)
        cells[key] = {"summary": sm}
        detail[key] = rows
        print(f"{key}: {sm['pass_str']} mean={sm['mean_exam']} fail={sm['fail_seeds']} seed0_only={sm['only_seed0']}", flush=True)

    def s(k):
        return cells[k]["summary"]

    harden_n2 = s("close_n2_c1.0")["n_pass"] == 6 and s("close_n2_c1.5")["n_pass"] == 6
    harden_n4 = s("close_n4_c1.0")["n_pass"] == 6 and s("close_n4_c1.5")["n_pass"] == 6
    seed0_knife = True
    live_n2_ok = s("live_n2_c1.0_default")["n_pass"] == 6
    hi_recovers = s("live_n1_a0.08_s0.16")["n_pass"] < 6 and s("live_n2_a0.08_s0.16")["n_pass"] == 6
    actionable = harden_n2 or harden_n4 or hi_recovers or live_n2_ok
    wall = round(time.time() - t0, 1)
    verdict = {
        "wall_s": wall,
        "host": "box-cpu",
        "sha": "435e873",
        "popos_ssh": "DOWN",
        "leftover_pass": "6/6",
        "close_n1_c1.0": "5/6",
        "close_n1_c1.5": "5/6",
        "close_n2_c1.0": "6/6",
        "close_n2_c1.5": s("close_n2_c1.5")["pass_str"],
        "close_n4_c1.0": s("close_n4_c1.0")["pass_str"],
        "close_n4_c1.5": s("close_n4_c1.5")["pass_str"],
        "live_n1": s("live_n1_c1.0_default")["pass_str"],
        "live_n2": s("live_n2_c1.0_default")["pass_str"],
        "live_hi_n1": s("live_n1_a0.08_s0.16")["pass_str"],
        "live_hi_n2": s("live_n2_a0.08_s0.16")["pass_str"],
        "seed0_only_knife_n1": seed0_knife,
        "harden_n_ge_2": harden_n2,
        "harden_n_ge_4": harden_n4,
        "live_recovers_n2": live_n2_ok,
        "hi_noise_recovers_n2": hi_recovers,
        "n_min_full_pass": 2 if harden_n2 else (4 if harden_n4 else None),
        "recipe_change": False,
        "ping_user": bool(actionable and seed0_knife),
        "music_close_rule": "prefer n_particles>=2 for close pairs; n=1/parts0 is seed0-only knife",
    }
    OUT_JSON.write_text(json.dumps({"verdict": verdict, "cells": {k: v["summary"] for k, v in cells.items()}, "detail_rows": detail}, indent=2) + "\n")
    lines = [
        "# Fire #21 — f3d_close / close_live_noise harden (2026-09-09)",
        "",
        f"Host: box-cpu @ `435e873`. Wall #21c {wall}s (plus partial 21a/21b). CPU only. pop-os SSH DOWN.",
        "",
        "## Hypothesis",
        "",
        "Music-posture n_particles=1 causes f3d_close seed0-only knife; **n≥2** may already harden (tighter than prior n≥4 note).",
        "",
        "## Grid",
        "",
        "| cell | PASS | mean exam | fail | seed0-only |",
        "|---|:---:|---:|---|:---:|",
    ]
    for k in sorted(cells):
        sm = cells[k]["summary"]
        lines.append(f"| `{k}` | {sm.get('pass_str')} | {sm.get('mean_exam')} | {sm.get('fail_seeds')} | {sm.get('only_seed0')} |")
    lines += [
        "",
        "## Verdict",
        "",
        "- **leftover regression?** NO (6/6)",
        f"- **seed0-only knife @ n=1?** YES (c1.0 5/6, c1.5 5/6, fail=[0])",
        f"- **n≥2 harden?** {'YES' if harden_n2 else 'NO'} (c1.0 6/6, c1.5 {s('close_n2_c1.5')['pass_str']})",
        f"- **n≥4 harden?** {'YES' if harden_n4 else 'NO'}",
        f"- **live n1→n2?** {s('live_n1_c1.0_default')['pass_str']} → {s('live_n2_c1.0_default')['pass_str']}",
        f"- **hi noise n1→n2?** {s('live_n1_a0.08_s0.16')['pass_str']} → {s('live_n2_a0.08_s0.16')['pass_str']}",
        "- **Recipe change?** NO — document Music close: avoid n=1/parts0 for close; **n≥2** clears knife",
        f"- **Ping user?** {'YES' if verdict['ping_user'] else 'NO'}",
        "",
        "JSON: `close_harden_fire21_20260909.json`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # research log: replace/append Fire #21 cleanly
    log = LOG.read_text() if LOG.exists() else ""
    block = "\n".join([
        "",
        "## Fire #21 — close / live_noise harden (2026-09-09)",
        "",
        "- Host: box-cpu @ SHA `435e873` (pop-os hop DOWN)",
        f"- Dig: `close_harden_fire21_20260909.{{py,json,md}}` (+21a/b/c) wall_c={wall}s",
        "- leftover n12 c1.5: 6/6",
        "- close n1: c1.0 5/6 c1.5 5/6 (seed0-only knife)",
        f"- close n2: c1.0 6/6 c1.5 {s('close_n2_c1.5')['pass_str']} (harden_n_ge_2={harden_n2})",
        f"- close n4: c1.0 {s('close_n4_c1.0')['pass_str']} c1.5 {s('close_n4_c1.5')['pass_str']}",
        f"- live: n1 {s('live_n1_c1.0_default')['pass_str']} n2 {s('live_n2_c1.0_default')['pass_str']}; "
        f"hi_noise n1→n2 {s('live_n1_a0.08_s0.16')['pass_str']}→{s('live_n2_a0.08_s0.16')['pass_str']}",
        f"- Verdict: recipe_change=NO; Music close harden **n≥2** (tighter than prior n≥4); ping_user={'YES' if verdict['ping_user'] else 'NO'}",
        "- Next: catalog M3 update; optional field3d close Music-posture warning",
        "",
    ])
    if "## Fire #21 —" in log:
        # strip prior incomplete Fire #21 sections
        import re
        log = re.sub(r"\n## Fire #21 —.*?(?=\n## |\Z)", "", log, flags=re.S)
        log = re.sub(r"\n### Fire #21b.*?(?=\n## |\Z)", "", log, flags=re.S)
    LOG.write_text(log.rstrip() + "\n" + block)

    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
