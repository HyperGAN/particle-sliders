#!/usr/bin/env python3
"""Music→toy new stressors M13–M19 + recipe probes — 2026-09-09.

After harden bites: lyric_span HARD BOUNDARY; close seed0 knife documented
(multi-seed / n≥2 / vic=0@n=1 candidate). Invent more portable Music proxies.

Locked: 1200 / c1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / vic=0.05 / b_cap=1.
Music posture: n=1 cover=1.0. Reject 800×cover3. CPU only. No Music GPU train.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    CELLS_3D,
    close_field3d,
    close_live_noise_field3d,
    close_with_leak_field3d,
    e_on_u_declare_lie_field3d,
    grit_content_dom_field3d,
    leftover_field3d,
    prefix_shared_proxy_field3d,
    roles_split_proxy_field3d,
    scale_stagger_homo_field3d,
    score_adv_field3d_exam,
    tiny_slider_dom_field3d,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "music_toy_new_stressors_m13_20260909.json"
OUT_MD = NOTES / "music_toy_new_stressors_m13_20260909.md"
LOG = NOTES / "research_log_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"


def cfg(seed, *, cover=1.5, n=12, steps=1200, vic=0.05, b_cap=1.0,
        end_margin=0.60, span_frac=0.40, **kw):
    return default_cfg(
        steps=steps, seed=seed, b_cap=b_cap, cover_weight=cover,
        fm_weight=0.0, n_particles=n, particle_l2=0.02,
        vicreg_weight=vic, end_margin=end_margin, span_frac=span_frac, **kw,
    )


def run_one(field, *, seed, name, **kw):
    t0 = time.time()
    row = score_adv_field3d_exam(
        field, teacher=TEACHER, cfg=cfg(seed, **kw), name=name,
    )
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": float(row.get("exam_score", 0.0)),
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "pass_multi_row": bool(row.get("pass_multi_row", False)),
        "pass_u": bool(row.get("pass_u", False)),
        "pass_leftover_gate": bool(row.get("pass_leftover_gate", False)),
        "wall_s": round(time.time() - t0, 2),
    }


def summarize(runs):
    n = len(runs)
    npass = sum(1 for r in runs if r["pass"])
    prim = [r["exam_score"] for r in runs]
    return {
        "n": n,
        "pass": f"{npass}/{n}",
        "n_pass": npass,
        "mean_exam": round(sum(prim) / n, 4) if n else 0.0,
        "mean_u": round(sum(r["u_kept"] for r in runs) / n, 4) if n else 0.0,
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4) if n else 0.0,
        "mean_rows_cov": round(sum(r["rows_covered"] for r in runs) / n, 2) if n else 0.0,
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "knife": bool(0 < npass < n),
        "bites": bool(npass == 0),
        "runs": runs,
    }


def grid(name, field_fn, seeds, **kw):
    runs = [run_one(field_fn(seed=s), seed=s, name=f"{name}_s{s}", **kw) for s in seeds]
    out = summarize(runs)
    out["name"] = name
    out["kw"] = kw
    flag = "BITES" if out["bites"] else ("KNIFE" if out["knife"] else ("PASS" if out["n_pass"] == out["n"] else "MIX"))
    print(
        f"  [{flag}] {name}: {out['pass']} exam={out['mean_exam']} "
        f"u={out['mean_u']} leak={out['leak_max']} rows≈{out['mean_rows_cov']} "
        f"fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    t_wall = time.time()
    cells = []

    # A) Positive controls
    cells.append(grid("leftover_locked_n12", leftover_field3d, SEEDS_SMOKE, cover=1.5, n=12))
    cells.append(grid("close_locked_n12", close_field3d, SEEDS_SMOKE, cover=1.5, n=12))

    # B) New stressors — locked + music posture
    new_cells = [
        ("M13_tiny_slider_dom", tiny_slider_dom_field3d),
        ("M14_e_on_u_declare_lie", e_on_u_declare_lie_field3d),
        ("M15_prefix_shared_proxy", prefix_shared_proxy_field3d),
        ("M16_scale_stagger_homo", scale_stagger_homo_field3d),
        ("M17_roles_split_proxy", roles_split_proxy_field3d),
        ("M18_close_with_leak", close_with_leak_field3d),
        ("M19_grit_content_dom", grit_content_dom_field3d),
    ]
    for label, fn in new_cells:
        cells.append(grid(f"{label}_locked_n12_c1.5", fn, SEEDS_FULL, cover=1.5, n=12))
        cells.append(grid(f"{label}_music_n1_c1.0", fn, SEEDS_FULL, cover=1.0, n=1))

    # C) Close-family knives under parts0 + vic0 candidate (confirm transfer)
    cells.append(grid("M13_tiny_music_n1_vic0", tiny_slider_dom_field3d, SEEDS_FULL, cover=1.0, n=1, vic=0.0))
    cells.append(grid("M18_closeleak_music_n1_vic0", close_with_leak_field3d, SEEDS_FULL, cover=1.0, n=1, vic=0.0))
    cells.append(grid("M13_tiny_music_n2", tiny_slider_dom_field3d, SEEDS_FULL, cover=1.0, n=2))

    # D) Recipe probes (not cell soften) — b_cap=0 / end_margin=0 on leftover+close
    cells.append(grid("probe_leftover_bcap0_n12", leftover_field3d, SEEDS_SMOKE, cover=1.5, n=12, b_cap=0.0))
    cells.append(grid("probe_close_n1_end0", close_field3d, SEEDS_FULL, cover=1.5, n=1, end_margin=0.0))
    cells.append(grid("probe_close_n1_span1", close_field3d, [0, 1, 2], cover=1.5, n=1, span_frac=1.0))
    cells.append(grid("probe_live_n1_vic0", close_live_noise_field3d, SEEDS_FULL, cover=1.5, n=1, vic=0.0))

    # E) roles_split vs cross_axis mildness check at n=12 already in B

    wall = round(time.time() - t_wall, 1)
    wins = []
    for c in cells:
        if c["bites"]:
            wins.append(f"{c['name']}: 0/{c['n']} BITES")
        elif c["knife"]:
            wins.append(f"{c['name']}: {c['pass']} KNIFE fail={c['fail_seeds']}")

    payload = {
        "fire": "music_toy_new_stressors_m13",
        "host": "box-cpu",
        "sha": git_sha(),
        "wall_s": wall,
        "locked_recipe": {
            "steps": 1200, "cover": 1.5, "teacher": TEACHER,
            "fm": 0.0, "n_max": 12, "l2": 0.02, "vic": 0.05, "b_cap": 1.0,
        },
        "wins": wins,
        "cells": cells,
        "summary": {
            c["name"]: {
                "pass": c["pass"], "mean_exam": c["mean_exam"], "mean_u": c["mean_u"],
                "leak_max": c["leak_max"], "mean_rows_cov": c["mean_rows_cov"],
                "fail_seeds": c["fail_seeds"], "bites": c["bites"], "knife": c["knife"],
                "kw": c["kw"],
            }
            for c in cells
        },
        "recipe_change": False,
        "registry_new": [
            "tiny_slider_dom", "e_on_u_declare_lie", "prefix_shared_proxy",
            "scale_stagger_homo", "roles_split_proxy", "close_with_leak", "grit_content_dom",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Music→toy new stressors M13–M19 — 2026-09-09",
        "",
        f"Host: box-cpu @ `{payload['sha']}`. Wall {wall}s. CPU only. No Music train.",
        "",
        "## New cells (CELLS_3D)",
        "",
        "| ID | cell | Music proxy |",
        "|---|---|---|",
        "| M13 | `tiny_slider_dom` | harder close (û=0.05, content=1.2) |",
        "| M14 | `e_on_u_declare_lie` | YAML ê restates û |",
        "| M15 | `prefix_shared_proxy` | whole-prefix hold / shared neu |",
        "| M16 | `scale_stagger_homo` | mild multi-row (homo amps + scale stagger) |",
        "| M17 | `roles_split_proxy` | role-split UNI (û vs content rows) |",
        "| M18 | `close_with_leak` | close + small unused ê |",
        "| M19 | `grit_content_dom` | grit/distortion content-dom |",
        "",
        "## Results",
        "",
        "| cell | PASS | exam | u | leak | rows | fail | flag |",
        "|---|:---:|---:|---:|---:|---:|---|---|",
    ]
    for c in cells:
        flag = "BITES" if c["bites"] else ("KNIFE" if c["knife"] else "ok")
        lines.append(
            f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | {c['mean_u']} | "
            f"{c['leak_max']} | {c['mean_rows_cov']} | {c['fail_seeds']} | {flag} |"
        )
    lines += [
        "",
        "## Wins (biting / knife)",
        "",
    ]
    for w in wins:
        lines.append(f"- {w}")
    if not wins:
        lines.append("- (none — all green or mixed)")
    lines += [
        "",
        "## Verdict",
        "",
        "- Recipe change? **NO**",
        "- lyric_span still HARD BOUNDARY (not re-chased)",
        "- close parts0 rule from harden bites still stands (multi-seed / n≥2 / vic0@n1 candidate)",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    log = f"""
## Fire — Music→toy M13–M19 new stressors (2026-09-09)

- Host: box-cpu @ SHA `{payload['sha']}`
- Notes: `music_toy_new_stressors_m13_20260909.{{py,json,md}}` wall={wall}s
- Registered CELLS_3D: {payload['registry_new']}
- Wins: {wins}
- Recipe change: NO. No Music GPU train.
"""
    with LOG.open("a") as fh:
        fh.write(log)

    # Catalog append
    cat_extra = """

## M13–M19 new stressors (2026-09-09)

| ID | Cell | Expected under locked | Notes |
|---|---|---|---|
| M13 | `tiny_slider_dom` | knife @ n=1 like close | harder û |
| M14 | `e_on_u_declare_lie` | FAIL or content/u survival stress | YAML lie on û |
| M15 | `prefix_shared_proxy` | TBD | prefix-hold proxy |
| M16 | `scale_stagger_homo` | soft multi-row bite? | milder than M1/M2 |
| M17 | `roles_split_proxy` | likely BITES (mild cross-axis) | role-split UNI |
| M18 | `close_with_leak` | knife @ n=1 | close + ê |
| M19 | `grit_content_dom` | content survival | grit proxy |

See `music_toy_new_stressors_m13_20260909.md` for numbers.
"""
    if "M13–M19 new stressors" not in CATALOG.read_text():
        with CATALOG.open("a") as fh:
            fh.write(cat_extra)

    print(json.dumps({"wall_s": wall, "wins": wins, "n_cells": len(cells)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
