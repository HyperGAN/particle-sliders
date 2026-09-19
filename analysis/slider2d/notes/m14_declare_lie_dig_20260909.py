#!/usr/bin/env python3
"""M14 e_on_u_declare_lie fail-mode dig — 2026-09-09.

Why exam≈0.18 with u_kept≈0.98? Isolate leftover gate vs swing vs multi-row.
Also: does faithful (no guard) vs faithful_guard_e change the bite?
CPU. No recipe change. No Music train.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
from analysis.slider2d.field3d import e_on_u_declare_lie_field3d, leftover_field3d, score_adv_field3d_exam
from analysis.slider2d.gan import default_cfg

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "m14_declare_lie_dig_20260909.json"
OUT_MD = NOTES / "m14_declare_lie_dig_20260909.md"
SEEDS = [0, 1, 2, 3, 7, 42]


def run(name, field_fn, teacher, seeds=SEEDS, **kw):
    runs = []
    for s in seeds:
        cfg = default_cfg(steps=1200, seed=s, b_cap=1.0, cover_weight=kw.get("cover", 1.5),
                          fm_weight=0.0, n_particles=kw.get("n", 12), particle_l2=0.02)
        t0 = time.time()
        row = score_adv_field3d_exam(field_fn(seed=s), teacher=teacher, cfg=cfg, name=f"{name}_s{s}")
        runs.append({
            "seed": s,
            "pass": bool(row.get("exam_pass")),
            "exam_score": float(row["exam_score"]),
            "u_kept": float(row["u_kept"]),
            "content_kept": float(row["content_kept"]),
            "leak_ratio": float(row["leak_ratio"]),
            "exam_cont": float(row.get("exam_cont", 0)),
            "exam_swing": float(row.get("exam_swing", 0)),
            "pass_cont": bool(row.get("pass_cont")),
            "pass_swing": bool(row.get("pass_swing")),
            "pass_leftover_gate": bool(row.get("pass_leftover_gate")),
            "pass_multi_row": bool(row.get("pass_multi_row")),
            "pass_u": bool(row.get("pass_u")),
            "pass_content": bool(row.get("pass_content")),
            "rows_covered": int(row.get("rows_covered", 0)),
            "wall_s": round(time.time()-t0, 2),
        })
    n = len(runs)
    out = {
        "name": name, "teacher": teacher,
        "pass": f"{sum(1 for r in runs if r['pass'])}/{n}",
        "mean_exam": round(sum(r["exam_score"] for r in runs)/n, 4),
        "mean_cont": round(sum(r["exam_cont"] for r in runs)/n, 4),
        "mean_swing": round(sum(r["exam_swing"] for r in runs)/n, 4),
        "frac_leftover": round(sum(1 for r in runs if r["pass_leftover_gate"])/n, 2),
        "frac_multi": round(sum(1 for r in runs if r["pass_multi_row"])/n, 2),
        "frac_cont": round(sum(1 for r in runs if r["pass_cont"])/n, 2),
        "frac_swing": round(sum(1 for r in runs if r["pass_swing"])/n, 2),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "runs": runs,
    }
    print(f"  {name}: {out['pass']} exam={out['mean_exam']} cont={out['mean_cont']} "
          f"swing={out['mean_swing']} left={out['frac_leftover']} multi={out['frac_multi']} "
          f"fail={out['fail_seeds']}", flush=True)
    return out


def main():
    t0 = time.time()
    cells = []
    cells.append(run("leftover_guard", leftover_field3d, "faithful_guard_e", seeds=[0,1,2]))
    cells.append(run("m14_guard_n12", e_on_u_declare_lie_field3d, "faithful_guard_e"))
    cells.append(run("m14_faithful_n12", e_on_u_declare_lie_field3d, "faithful"))
    cells.append(run("m14_guard_n1", e_on_u_declare_lie_field3d, "faithful_guard_e", n=1, cover=1.0))
    # soft e_on_u ladder
    for eou in (0.0, 0.3, 0.75, 1.5):
        def fn(seed=0, e=eou):
            return e_on_u_declare_lie_field3d(e_on_u=e, e_on_content=0.1 if e > 0 else 0.0,
                                              e_unused=0.1 if e > 0 else 1.0)
        cells.append(run(f"m14_eou{eou}_guard", fn, "faithful_guard_e", seeds=[0,1,2]))
    wall = round(time.time()-t0, 1)
    payload = {"wall_s": wall, "cells": cells, "recipe_change": False}
    OUT_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    lines = ["# M14 declare-lie fail-mode dig — 2026-09-09", "", f"Wall {wall}s.", "",
             "| cell | PASS | exam | cont | swing | leftover | multi | fail |",
             "|---|:---:|---:|---:|---:|---:|---:|---|"]
    for c in cells:
        lines.append(f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | {c['mean_cont']} | "
                     f"{c['mean_swing']} | {c['frac_leftover']} | {c['frac_multi']} | {c['fail_seeds']} |")
    lines += ["", "Recipe change? NO", "", f"JSON: `{OUT_JSON.name}`", ""]
    OUT_MD.write_text("\n".join(lines)+"\n")
    print(json.dumps({"wall_s": wall}, indent=2))


if __name__ == "__main__":
    main()
