#!/usr/bin/env python3
"""M20–M22 sheet/2D Music proxies + tiny_slider n≥2 confirm — 2026-09-09.

Bridge Field3D bites back to 2D sheet leftover where possible.
Also confirm M13 tiny_slider recovers at n=2 (Fire #21 rule transfer).
CPU. No Music train. No lyric_span recipe chase.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (
    tiny_slider_dom_field3d, close_with_leak_field3d, grit_content_dom_field3d,
    e_on_u_declare_lie_field3d, leftover_field3d, score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "music_toy_m20_sheet_span_20260909.json"
OUT_MD = NOTES / "music_toy_m20_sheet_span_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"


def make_cfg(seed, **kw):
    base = dict(steps=1200, seed=seed, b_cap=1.0, cover_weight=1.5,
                fm_weight=0.0, n_particles=12, particle_l2=0.02, vicreg_weight=0.05)
    base.update(kw)
    return default_cfg(**base)


def grid(name, field_fn, seeds=SEEDS, **kw):
    runs = []
    for s in seeds:
        t0 = time.time()
        row = score_adv_field3d_exam(
            field_fn(seed=s), teacher=TEACHER, cfg=make_cfg(s, **kw), name=f"{name}_s{s}"
        )
        runs.append({
            "seed": s, "pass": bool(row.get("exam_pass")),
            "exam_score": float(row["exam_score"]), "u_kept": float(row["u_kept"]),
            "leak_ratio": float(row["leak_ratio"]),
            "rows_covered": int(row.get("rows_covered", 0)),
            "pass_multi_row": bool(row.get("pass_multi_row", False)),
            "wall_s": round(time.time() - t0, 2),
        })
    npass = sum(1 for r in runs if r["pass"])
    out = {
        "name": name, "pass": f"{npass}/{len(runs)}", "n_pass": npass,
        "mean_exam": round(sum(r["exam_score"] for r in runs)/len(runs), 4),
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4),
        "mean_rows": round(sum(r["rows_covered"] for r in runs)/len(runs), 2),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "knife": 0 < npass < len(runs), "bites": npass == 0,
        "kw": kw, "runs": runs,
    }
    flag = "BITES" if out["bites"] else ("KNIFE" if out["knife"] else "PASS")
    print(f"  [{flag}] {name}: {out['pass']} exam={out['mean_exam']} fail={out['fail_seeds']}", flush=True)
    return out


def main():
    t0 = time.time()
    cells = []
    cells.append(grid("leftover_n12", leftover_field3d, seeds=[0,1,2]))
    # M13 Fire #21 transfer: n=2 should clear knife
    cells.append(grid("M13_tiny_n1_c1.0", tiny_slider_dom_field3d, n_particles=1, cover_weight=1.0))
    cells.append(grid("M13_tiny_n2_c1.0", tiny_slider_dom_field3d, n_particles=2, cover_weight=1.0))
    cells.append(grid("M13_tiny_n2_c1.5", tiny_slider_dom_field3d, n_particles=2, cover_weight=1.5))
    cells.append(grid("M13_tiny_n1_vic0", tiny_slider_dom_field3d, n_particles=1, cover_weight=1.0, vicreg_weight=0.0))
    # M18 close_with_leak n floor
    cells.append(grid("M18_n1_c1.0", close_with_leak_field3d, n_particles=1, cover_weight=1.0))
    cells.append(grid("M18_n2_c1.0", close_with_leak_field3d, n_particles=2, cover_weight=1.0))
    # M14 declare lie — confirm still bites at n=12 and n=2
    cells.append(grid("M14_n12", e_on_u_declare_lie_field3d, n_particles=12))
    cells.append(grid("M14_n2", e_on_u_declare_lie_field3d, n_particles=2))
    # M19 grit
    cells.append(grid("M19_n12", grit_content_dom_field3d, n_particles=12))
    cells.append(grid("M19_n1_c1.0", grit_content_dom_field3d, n_particles=1, cover_weight=1.0))
    wall = round(time.time() - t0, 1)
    wins = [f"{c['name']}: {c['pass']}" for c in cells if c["bites"] or c["knife"]]
    payload = {"wall_s": wall, "host": "box-cpu", "fire21_n_ge_2": True,
               "recipe_change": False, "wins": wins, "cells": cells,
               "summary": {c["name"]: {k: c[k] for k in ("pass","mean_exam","leak_max","fail_seeds","knife","bites","kw")} for c in cells}}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# M20 thread — tiny/closeleak n≥2 transfer + M14/M19 (2026-09-09)", "",
             f"Wall {wall}s. Fire #21 n≥2 rule transfer to M13/M18.", "",
             "| cell | PASS | exam | leak | fail | flag |",
             "|---|:---:|---:|---:|---|---|"]
    for c in cells:
        flag = "BITES" if c["bites"] else ("KNIFE" if c["knife"] else "ok")
        lines.append(f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | {c['leak_max']} | {c['fail_seeds']} | {flag} |")
    lines += ["", f"Wins: {wins}", "", "Recipe change? NO", "", f"JSON: `{OUT_JSON.name}`", ""]
    OUT_MD.write_text("\n".join(lines) + "\n")
    with LOG.open("a") as fh:
        fh.write(f"\n## Fire — M20 tiny/closeleak n≥2 transfer (2026-09-09)\n\n- Notes: `music_toy_m20_sheet_span_20260909.{{py,json,md}}` wall={wall}s\n- Wins: {wins}\n- Fire #21 n≥2 transfer; recipe_change=NO.\n")
    print(json.dumps({"wall_s": wall, "wins": wins}, indent=2))


if __name__ == "__main__":
    main()
