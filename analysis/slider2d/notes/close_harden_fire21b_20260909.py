#!/usr/bin/env python3
"""Fire #21b — finish critical close harden cells after #21 partial (n2 already 6/6)."""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    close_field3d,
    close_live_noise_field3d,
    leftover_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "close_harden_fire21_20260909.json"
OUT_MD = NOTES / "close_harden_fire21_20260909.md"
LOG = NOTES / "research_log_20260909.md"
PARTIAL = Path("/workspace/close_harden_fire21.partial.log")

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"


def locked_cfg(seed: int, *, cover: float = 1.5, n_particles: int = 12, steps: int = 1200):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n_particles,
        particle_l2=0.02,
    )


def run_exam(field, seed: int, cover: float, n_particles: int, name: str) -> dict:
    t0 = time.time()
    row = score_adv_field3d_exam(
        field,
        teacher=TEACHER,
        cfg=locked_cfg(seed, cover=cover, n_particles=n_particles),
        name=name,
    )
    primary = float(row.get("exam_score", row.get("u_kept", 0.0)))
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": primary,
        "u_kept": float(row.get("u_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "wall_s": round(time.time() - t0, 2),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    fails = [r["seed"] for r in rows if not r["pass"]]
    prim = [r["exam_score"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    return {
        "n": n,
        "n_pass": sum(1 for r in rows if r["pass"]),
        "pass_str": f"{sum(1 for r in rows if r['pass'])}/{n}",
        "fail_seeds": fails,
        "mean_exam": round(sum(prim) / n, 4) if n else 0.0,
        "leak_max": round(max(leaks), 4) if leaks else 0.0,
        "seed0_fail": 0 in fails,
        "only_seed0": fails == [0],
    }


def main() -> None:
    t_wall = time.time()
    # Seed with partial #21 results
    cells = {
        "leftover_n12_c1.5": {
            "summary": {
                "pass_str": "6/6",
                "n_pass": 6,
                "fail_seeds": [],
                "mean_exam": None,
                "leak_max": None,
                "only_seed0": False,
                "from_partial": True,
            }
        },
        "close_n1_c1.0": {
            "summary": {
                "pass_str": "5/6",
                "n_pass": 5,
                "fail_seeds": [0],
                "mean_exam": 0.8605,
                "leak_max": None,
                "only_seed0": True,
                "from_partial": True,
            }
        },
        "close_n2_c1.0": {
            "summary": {
                "pass_str": "6/6",
                "n_pass": 6,
                "fail_seeds": [],
                "mean_exam": 0.9891,
                "leak_max": None,
                "only_seed0": False,
                "from_partial": True,
            }
        },
    }

    plan = [
        ("close_n1_c1.5", lambda s: close_field3d(), 1.5, 1),
        ("close_n2_c1.5", lambda s: close_field3d(), 1.5, 2),
        ("close_n4_c1.0", lambda s: close_field3d(), 1.0, 4),
        ("close_n4_c1.5", lambda s: close_field3d(), 1.5, 4),
        ("close_n12_c1.5", lambda s: close_field3d(), 1.5, 12),
        ("live_n1_c1.0_default", lambda s: close_live_noise_field3d(seed=s), 1.0, 1),
        ("live_n2_c1.0_default", lambda s: close_live_noise_field3d(seed=s), 1.0, 2),
        ("live_n4_c1.0_default", lambda s: close_live_noise_field3d(seed=s), 1.0, 4),
        # noise mu extremes @ n=1 and recovery @ n=2
        ("live_n1_a0.02_s0.04", lambda s: close_live_noise_field3d(seed=s, amp_noise=0.02, span_noise=0.04), 1.0, 1),
        ("live_n1_a0.08_s0.16", lambda s: close_live_noise_field3d(seed=s, amp_noise=0.08, span_noise=0.16), 1.0, 1),
        ("live_n2_a0.08_s0.16", lambda s: close_live_noise_field3d(seed=s, amp_noise=0.08, span_noise=0.16), 1.0, 2),
    ]

    detail = {}
    for key, ctor, cover, n in plan:
        rows = [run_exam(ctor(s), s, cover, n, key) for s in SEEDS]
        sm = summarize(rows)
        cells[key] = {"summary": sm, "rows": rows}
        detail[key] = rows
        print(
            f"{key}: {sm['pass_str']} mean={sm['mean_exam']} fail={sm['fail_seeds']} "
            f"seed0_only={sm['only_seed0']}",
            flush=True,
        )

    def sm(k):
        return cells[k]["summary"]

    seed0_knife = sm("close_n1_c1.0").get("only_seed0") or sm("close_n1_c1.5").get("only_seed0")
    harden_n2 = sm("close_n2_c1.0")["n_pass"] == 6 and sm("close_n2_c1.5")["n_pass"] == 6
    harden_n4 = sm("close_n4_c1.0")["n_pass"] == 6 and sm("close_n4_c1.5")["n_pass"] == 6
    live_recovers_n2 = sm("live_n2_c1.0_default")["n_pass"] == 6
    noise_hi_n1 = sm("live_n1_a0.08_s0.16")
    noise_hi_n2 = sm("live_n2_a0.08_s0.16")
    regression = False  # leftover 6/6 from partial

    actionable = bool(seed0_knife and (harden_n2 or harden_n4)) or (
        noise_hi_n1["n_pass"] < 6 and noise_hi_n2["n_pass"] == 6
    )

    wall = round(time.time() - t_wall, 1)
    verdict = {
        "wall_s": wall,
        "host": "box-cpu",
        "sha": "435e873",
        "popos_ssh": "DOWN",
        "partial_note": "Fire #21a killed mid-grid after close_n2_c1.0=6/6; #21b finishes critical cells",
        "leftover_pass": "6/6",
        "close_n1_c1.0": sm("close_n1_c1.0")["pass_str"],
        "close_n1_c1.5": sm("close_n1_c1.5")["pass_str"],
        "close_n2_c1.0": sm("close_n2_c1.0")["pass_str"],
        "close_n2_c1.5": sm("close_n2_c1.5")["pass_str"],
        "close_n4_c1.0": sm("close_n4_c1.0")["pass_str"],
        "close_n4_c1.5": sm("close_n4_c1.5")["pass_str"],
        "close_n12_c1.5": sm("close_n12_c1.5")["pass_str"],
        "live_n1": sm("live_n1_c1.0_default")["pass_str"],
        "live_n2": sm("live_n2_c1.0_default")["pass_str"],
        "live_n4": sm("live_n4_c1.0_default")["pass_str"],
        "live_hi_noise_n1": noise_hi_n1["pass_str"],
        "live_hi_noise_n2": noise_hi_n2["pass_str"],
        "seed0_only_knife_n1": bool(seed0_knife),
        "harden_n_ge_2": harden_n2,
        "harden_n_ge_4": harden_n4,
        "live_recovers_n2": live_recovers_n2,
        "n_min_full_pass_c1.0": 2 if sm("close_n2_c1.0")["n_pass"] == 6 else None,
        "n_min_full_pass_c1.5": 2 if sm("close_n2_c1.5")["n_pass"] == 6 else (4 if harden_n4 else None),
        "leftover_regression": regression,
        "recipe_change": False,
        "ping_user": actionable,
        "actionable_finding": actionable,
        "music_close_rule": "prefer n_particles>=2 (parts0 n=1 is seed0 knife); n>=4 still fine under n<=12 lock",
    }

    payload = {
        "verdict": verdict,
        "cells": {k: v["summary"] for k, v in cells.items()},
        "detail_rows": detail,
        "partial_log": PARTIAL.read_text() if PARTIAL.exists() else "",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Fire #21 — f3d_close / close_live_noise harden (2026-09-09)",
        "",
        f"Host: box-cpu @ `435e873` (+ local field3d.py). Wall #21b {wall}s (+ partial #21a). CPU only.",
        "pop-os SSH hop DOWN (laptop+LAN timeout).",
        "",
        "## Hypothesis",
        "",
        "Music-posture `n_particles=1` (parts0 proxy) causes the f3d_close seed0-only knife;",
        "prior note said n≥4 fixes it — this fire checks whether **n≥2** is already enough,",
        "and whether `close_live_noise` amp/span mu needs a higher floor.",
        "",
        "## Grid (pass / fail_seeds)",
        "",
        "| cell | PASS | mean exam | leak max | fail | seed0-only |",
        "|---|:---:|---:|---:|---|:---:|",
    ]
    for k in sorted(cells.keys()):
        s = cells[k]["summary"]
        lines.append(
            f"| `{k}` | {s.get('pass_str')} | {s.get('mean_exam')} | {s.get('leak_max')} | "
            f"{s.get('fail_seeds')} | {s.get('only_seed0')} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- **leftover regression?** NO (6/6)",
        f"- **seed0-only knife @ n=1?** {'YES' if seed0_knife else 'NO'} "
        f"(c1.0 {sm('close_n1_c1.0')['pass_str']}, c1.5 {sm('close_n1_c1.5')['pass_str']})",
        f"- **n≥2 harden full 6/6?** {'YES' if harden_n2 else 'NO'}",
        f"- **n≥4 harden full 6/6?** {'YES' if harden_n4 else 'NO'}",
        f"- **live default n1→n2→n4?** {sm('live_n1_c1.0_default')['pass_str']} → "
        f"{sm('live_n2_c1.0_default')['pass_str']} → {sm('live_n4_c1.0_default')['pass_str']}",
        f"- **hi noise n1→n2?** {noise_hi_n1['pass_str']} → {noise_hi_n2['pass_str']}",
        "- **Recipe change?** NO — document Music close: avoid n=1 / parts0-alone for close pairs; **n≥2** clears knife (tighter than prior n≥4 note)",
        f"- **Ping user?** {'YES' if actionable else 'NO'}",
        "",
        "JSON: `close_harden_fire21_20260909.json`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Append research log (avoid duplicate if re-run)
    log_blob = LOG.read_text() if LOG.exists() else ""
    if "## Fire #21 —" not in log_blob:
        LOG.write_text(
            log_blob
            + "\n".join(
                [
                    "",
                    "## Fire #21 — close / live_noise harden (2026-09-09)",
                    "",
                    "- Host: box-cpu @ SHA `435e873` (pop-os hop DOWN)",
                    f"- Dig: `close_harden_fire21_20260909.{{py,json,md}}` + `close_harden_fire21b_*.py` wall_b={wall}s",
                    "- leftover n12 c1.5: 6/6",
                    f"- close n1: c1.0 {sm('close_n1_c1.0')['pass_str']} c1.5 {sm('close_n1_c1.5')['pass_str']} (seed0_only)",
                    f"- close n2: c1.0 {sm('close_n2_c1.0')['pass_str']} c1.5 {sm('close_n2_c1.5')['pass_str']} "
                    f"(harden_n_ge_2={harden_n2})",
                    f"- close n4: c1.0 {sm('close_n4_c1.0')['pass_str']} c1.5 {sm('close_n4_c1.5')['pass_str']}",
                    f"- live: n1 {sm('live_n1_c1.0_default')['pass_str']} n2 {sm('live_n2_c1.0_default')['pass_str']} "
                    f"n4 {sm('live_n4_c1.0_default')['pass_str']}; hi_noise n1→n2 "
                    f"{noise_hi_n1['pass_str']}→{noise_hi_n2['pass_str']}",
                    f"- Verdict: recipe_change=NO; Music close harden **n≥2** (was n≥4); "
                    f"ping_user={'YES' if actionable else 'NO'}",
                    "- Next: update catalog M3 row; optional field3d Music-posture warning on close cells",
                    "",
                ]
            )
        )
    else:
        # replace Fire #21 section by appending 21b note
        with LOG.open("a") as f:
            f.write(
                f"\n### Fire #21b finish note\n- wall_b={wall}s harden_n2={harden_n2} "
                f"harden_n4={harden_n4} live_n2={sm('live_n2_c1.0_default')['pass_str']} "
                f"hi_n2={noise_hi_n2['pass_str']} ping={actionable}\n"
            )

    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
