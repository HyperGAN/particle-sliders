#!/usr/bin/env python3
"""b_cap / FM false-path probes on Field3D — 2026-09-09.

Document that b_cap=1 and FM=0 remain necessary under Music posture.
Reject enabling FM or dropping b_cap. CPU. No Music train. No lyric_span chase.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
from analysis.slider2d.field3d import (
    close_field3d, leftover_field3d, unused_e_field3d, score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "bcaps_fm_falsepath_20260909.json"
OUT_MD = NOTES / "bcaps_fm_falsepath_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"




def make_cfg(seed, **kw):
    base = dict(steps=1200, seed=seed, b_cap=1.0, cover_weight=1.5,
                fm_weight=0.0, fm_normalize=True, n_particles=12, particle_l2=0.02)
    base.update(kw)
    return default_cfg(**base)


def grid(name, field_fn, seeds, **kw):
    runs = []
    for s in seeds:
        t0 = time.time()
        row = score_adv_field3d_exam(
            field_fn(seed=s), teacher=TEACHER, cfg=make_cfg(s, **kw), name=f"{name}_s{s}"
        )
        runs.append({
            "seed": s, "pass": bool(row.get("exam_pass")),
            "exam_score": float(row["exam_score"]), "u_kept": float(row["u_kept"]),
            "leak_ratio": float(row["leak_ratio"]), "wall_s": round(time.time()-t0, 2),
        })
    npass = sum(1 for r in runs if r["pass"])
    out = {
        "name": name, "pass": f"{npass}/{len(runs)}", "n_pass": npass,
        "mean_exam": round(sum(r["exam_score"] for r in runs)/len(runs), 4),
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "kw": kw, "runs": runs,
    }
    print(f"  {name}: {out['pass']} exam={out['mean_exam']} leak={out['leak_max']} fail={out['fail_seeds']}", flush=True)
    return out


def main():
    t0 = time.time()
    cells = []
    # locked controls
    cells.append(grid("leftover_locked", leftover_field3d, [0,1,2], b_cap=1.0, fm_weight=0.0))
    cells.append(grid("unused_e_locked", unused_e_field3d, [0,1,2], b_cap=1.0, fm_weight=0.0))
    # b_cap ablation
    for bc in (0.0, 0.25, 0.5, 1.0):
        cells.append(grid(f"leftover_bcap{bc}", leftover_field3d, SEEDS[:3], b_cap=bc))
        cells.append(grid(f"close_n1_bcap{bc}", close_field3d, SEEDS, b_cap=bc, n_particles=1, cover_weight=1.5))
    # FM probes (normalized) — expect flat or worse; do not enable
    for fw in (0.0, 0.1, 0.5, 1.0):
        cells.append(grid(f"leftover_fm{fw}", leftover_field3d, SEEDS[:3], fm_weight=fw, fm_normalize=True))
        cells.append(grid(f"close_n12_fm{fw}", close_field3d, SEEDS[:3], fm_weight=fw, fm_normalize=True, n_particles=12))
    # raw FM control (known-bad path)
    cells.append(grid("leftover_fm0.5_raw", leftover_field3d, SEEDS[:3], fm_weight=0.5, fm_normalize=False))
    # Music leftover n=1 seed matrix under locked else
    cells.append(grid("leftover_music_n1_c1.0", leftover_field3d, SEEDS, n_particles=1, cover_weight=1.0))
    wall = round(time.time()-t0, 1)
    payload = {"wall_s": wall, "host": "box-cpu", "recipe_change": False, "cells": cells,
               "summary": {c["name"]: {k:c[k] for k in ("pass","mean_exam","leak_max","fail_seeds","kw")} for c in cells}}
    OUT_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    lines = ["# b_cap / FM false-path Field3D — 2026-09-09", "", f"Wall {wall}s. Expect b_cap=1 + FM=0 remain best.", "",
             "| cell | PASS | exam | leak | fail |", "|---|:---:|---:|---:|---|"]
    for c in cells:
        lines.append(f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | {c['leak_max']} | {c['fail_seeds']} |")
    lines += ["", "Verdict: recipe_change=NO unless a probe clearly wins without false lock.", "",
              f"JSON: `{OUT_JSON.name}`", ""]
    OUT_MD.write_text("\n".join(lines)+"\n")
    with LOG.open("a") as fh:
        fh.write(f"\n## Fire — b_cap/FM false-path (2026-09-09)\n\n- Notes: `bcaps_fm_falsepath_20260909.{{py,json,md}}` wall={wall}s\n- Recipe change=NO (confirm b_cap=1 FM=0).\n")
    print(json.dumps({"wall_s": wall, "n": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
