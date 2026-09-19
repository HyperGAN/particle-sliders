#!/usr/bin/env python3
"""Isolate M16 scale_stagger vs M17 roles_split vs M2 cross_axis — 2026-09-09.

How mild can multi-row axis heterogeneity get before shared AdvResidual bites?
Locked recipe. CPU. No lyric_span recipe chase. No Music train.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
from analysis.slider2d.field3d import (
    Field3D, cross_axis_rows_field3d, roles_split_proxy_field3d,
    scale_stagger_homo_field3d, leftover_field3d, score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "roles_scale_isolation_20260909.json"
OUT_MD = NOTES / "roles_scale_isolation_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"


def cfg(seed, n=12, cover=1.5):
    return default_cfg(steps=1200, seed=seed, b_cap=1.0, cover_weight=cover,
                       fm_weight=0.0, n_particles=n, particle_l2=0.02)


def make_blend(mix: float) -> Field3D:
    """mix=0 → all u-primary; mix=1 → full roles_split amps."""
    u = (1.1, 0.25, 0.1)
    c = (0.35, 1.3, 0.1)
    amps = []
    for i in range(4):
        # rows 0,1 stay u-ish; 2,3 lerp toward content as mix↑
        if i < 2:
            amps.append(u)
        else:
            amps.append(tuple((1 - mix) * a + mix * b for a, b in zip(u, c)))
    return Field3D(
        kind=f"roles_blend_{mix}",
        rows=4, row_scales=(1.0,) * 4, row_amps=tuple(amps),
        slider=1.0, content=0.55, leak=0.2,
        e_on_u=0.0, e_on_content=0.0, e_unused=1.0,
    )


def run_grid(name, field_fn, seeds=SEEDS, n=12, cover=1.5):
    runs = []
    for s in seeds:
        t0 = time.time()
        row = score_adv_field3d_exam(
            field_fn(seed=s) if "seed" in field_fn.__code__.co_varnames else field_fn(),
            teacher=TEACHER, cfg=cfg(s, n=n, cover=cover), name=f"{name}_s{s}",
        )
        runs.append({
            "seed": s,
            "pass": bool(row.get("exam_pass")),
            "exam_score": float(row["exam_score"]),
            "u_kept": float(row["u_kept"]),
            "leak_ratio": float(row["leak_ratio"]),
            "rows_covered": int(row["rows_covered"]),
            "rows_total": int(row["rows_total"]),
            "pass_multi_row": bool(row.get("pass_multi_row")),
            "wall_s": round(time.time() - t0, 2),
        })
    npass = sum(1 for r in runs if r["pass"])
    multi = sum(1 for r in runs if r["pass_multi_row"])
    out = {
        "name": name, "pass": f"{npass}/{len(runs)}", "n_pass": npass,
        "multi": f"{multi}/{len(runs)}",
        "mean_exam": round(sum(r["exam_score"] for r in runs) / len(runs), 4),
        "mean_rows": round(sum(r["rows_covered"] for r in runs) / len(runs), 2),
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "runs": runs, "n": n, "cover": cover,
    }
    print(f"  {name}: {out['pass']} multi={out['multi']} exam={out['mean_exam']} "
          f"rows≈{out['mean_rows']} fail={out['fail_seeds']}", flush=True)
    return out


def main():
    t0 = time.time()
    cells = []
    cells.append(run_grid("leftover_n12", leftover_field3d, seeds=[0, 1, 2]))
    cells.append(run_grid("scale_stagger_homo_n12", scale_stagger_homo_field3d))
    cells.append(run_grid("roles_split_n12", roles_split_proxy_field3d))
    cells.append(run_grid("cross_axis_rows_n12", cross_axis_rows_field3d))
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
        cells.append(run_grid(f"blend_mix{mix}_n12", lambda seed=None, m=mix: make_blend(m), seeds=[0, 1, 2]))
    # music posture on roles
    cells.append(run_grid("roles_split_n1_c1.0", roles_split_proxy_field3d, n=1, cover=1.0))
    cells.append(run_grid("scale_stagger_n1_c1.0", scale_stagger_homo_field3d, n=1, cover=1.0))
    wall = round(time.time() - t0, 1)
    payload = {"wall_s": wall, "host": "box-cpu", "cells": cells,
               "summary": {c["name"]: {k: c[k] for k in ("pass","multi","mean_exam","mean_rows","leak_max","fail_seeds")} for c in cells}}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# roles/scale isolation vs cross_axis — 2026-09-09", "",
             f"Wall {wall}s. Soften ladder: homo stagger → roles split → full cross_axis.", "",
             "| cell | PASS | multi | exam | rows | leak | fail |",
             "|---|:---:|:---:|---:|---:|---:|---|"]
    for c in cells:
        lines.append(f"| `{c['name']}` | {c['pass']} | {c['multi']} | {c['mean_exam']} | {c['mean_rows']} | {c['leak_max']} | {c['fail_seeds']} |")
    lines += ["", f"JSON: `{OUT_JSON.name}`", ""]
    OUT_MD.write_text("\n".join(lines) + "\n")
    with LOG.open("a") as fh:
        fh.write(f"\n## Fire — roles/scale isolation (2026-09-09)\n\n- Notes: `roles_scale_isolation_20260909.{{py,json,md}}` wall={wall}s\n- Soften ladder vs M2 hard boundary; recipe_change=NO.\n")
    print(json.dumps({"wall_s": wall, "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
