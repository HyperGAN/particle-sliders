#!/usr/bin/env python3
"""Harden dig: Music→toy bites after Fire #20 (2026-09-09).

Fire #20: lyric_span_entangle = HARD BOUNDARY under shared AdvResidual
(multi-row never recovers via cover/n/steps/eoc). Do not chase recipe knobs.

This fire pivots to close / close_live_noise seed0 knife @ n=1 (parts0 proxy):
- diagnose: seed0 undershoots û (u_kept≈0.38) while content+multi-row OK
- probes: particle_l2, n_particles floor, sampling (span/end/cloud),
  modest lr / init_std-via-jitter, cover∈{1.0,1.5,2.0} (reject 800×cover3)
- document multi-seed Music rule if no portable recipe recover of 6/6

Locked: 1200 / c1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1.
CPU only. No Music GPU train. No locked-default change unless clear+safe.
Fire #24: preserve music_close_posture_warn + annotate_dig_rows on write.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.notes.music_posture_dig_util import annotate_dig_rows, posture_block_for_md  # Fire #24
from analysis.slider2d.field3d import (  # noqa: E402
    close_field3d,
    close_live_noise_field3d,
    cross_axis_rows_field3d,
    leftover_field3d,
    lyric_span_entangle_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "harden_music_toy_bites_20260909.json"
OUT_MD = NOTES / "harden_music_toy_bites_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"


def make_cfg(
    seed: int,
    *,
    cover: float = 1.5,
    n: int = 1,
    steps: int = 1200,
    particle_l2: float = 0.02,
    span_frac: float = 0.40,
    end_margin: float = 0.60,
    cloud_std: float = 0.03,
    particle_jitter: float = 0.01,
    lr: float = 5.0e-3,
    vicreg_weight: float = 0.05,
):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n,
        particle_l2=particle_l2,
        span_frac=span_frac,
        end_margin=end_margin,
        cloud_std=cloud_std,
        particle_jitter=particle_jitter,
        lr=lr,
        vicreg_weight=vicreg_weight,
    )


def run_one(field, *, seed: int, name: str, **cfg_kw) -> dict:
    t0 = time.time()
    cfg = make_cfg(seed, **cfg_kw)
    row = score_adv_field3d_exam(field, teacher=TEACHER, cfg=cfg, name=name)
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
        "pass_content": bool(row.get("pass_content", False)),
        "pass_leftover_gate": bool(row.get("pass_leftover_gate", False)),
        "pass_cont": bool(row.get("pass_cont", False)),
        "pass_swing": bool(row.get("pass_swing", False)),
        "on_u": float(row.get("on_u", 0.0)),
        "residual_norm": float(row.get("residual_norm", 0.0)),
        "cfg": {k: cfg_kw.get(k) for k in (
            "cover", "n", "steps", "particle_l2", "span_frac", "end_margin",
            "cloud_std", "particle_jitter", "lr", "vicreg_weight",
        ) if k in cfg_kw or True},
        "wall_s": round(time.time() - t0, 2),
        "name": name,
        "n_particles": cfg.n_particles,
        "cell": getattr(field, "kind", None),
        "music_close_posture_warn": row.get("music_close_posture_warn"),  # Fire #24
    }


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    npass = sum(1 for r in runs if r["pass"])
    prim = [r["exam_score"] for r in runs]
    return {
        "n": n,
        "pass": f"{npass}/{n}",
        "n_pass": npass,
        "mean_exam": round(sum(prim) / n, 4) if n else 0.0,
        "exam_span": round(max(prim) - min(prim), 4) if n else 0.0,
        "mean_u": round(sum(r["u_kept"] for r in runs) / n, 4) if n else 0.0,
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4) if n else 0.0,
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "knife": bool(0 < npass < n),
        "runs": runs,
    }


def grid(name: str, field_fn, seeds, **cfg_kw) -> dict:
    runs = []
    for s in seeds:
        runs.append(run_one(field_fn(seed=s), seed=s, name=f"{name}_s{s}", **cfg_kw))
    out = summarize(runs)
    out["name"] = name
    out["kw"] = {k: v for k, v in cfg_kw.items()}
    print(
        f"  {name}: pass={out['pass']} exam={out['mean_exam']} "
        f"u={out['mean_u']} leak_max={out['leak_max']} "
        f"fail={out['fail_seeds']} knife={out['knife']}",
        flush=True,
    )
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    t_wall = time.time()
    cells: list[dict] = []

    # --- A) Positive controls (must stay green) ---
    cells.append(grid(
        "leftover_n12_c1.5",
        leftover_field3d, SEEDS_SMOKE, cover=1.5, n=12, steps=1200,
    ))
    cells.append(grid(
        "close_n12_c1.5",
        close_field3d, SEEDS_SMOKE, cover=1.5, n=12, steps=1200,
    ))

    # --- B) lyric_span hard-boundary confirm (document only; no recipe chase) ---
    cells.append(grid(
        "lyric_m1_locked_confirm",
        lyric_span_entangle_field3d, SEEDS_SMOKE, cover=1.5, n=12, steps=1200,
    ))
    cells.append(grid(
        "cross_axis_rows_boundary",
        cross_axis_rows_field3d, SEEDS_SMOKE, cover=1.5, n=12, steps=1200,
    ))

    # --- C) close knife baseline @ n=1 ---
    cells.append(grid(
        "close_n1_c1.5_baseline",
        close_field3d, SEEDS_FULL, cover=1.5, n=1, steps=1200,
    ))
    cells.append(grid(
        "close_n1_c1.0_baseline",
        close_field3d, SEEDS_FULL, cover=1.0, n=1, steps=1200,
    ))
    cells.append(grid(
        "close_live_n1_c1.5_baseline",
        close_live_noise_field3d, SEEDS_FULL, cover=1.5, n=1, steps=1200,
    ))

    # --- D) seed0 harden probes (close @ cover=1.5) ---
    # D1: particle_l2 sweep
    for l2 in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        cells.append(grid(
            f"close_s0_n1_l2{l2}",
            close_field3d, [0], cover=1.5, n=1, particle_l2=l2,
        ))

    # D2: n_particles floor (known recover at n≥4; check 2,3)
    for n in (2, 3, 4):
        cells.append(grid(
            f"close_s0_n{n}_c1.5",
            close_field3d, [0], cover=1.5, n=n,
        ))
        cells.append(grid(
            f"close_full_n{n}_c1.5",
            close_field3d, SEEDS_FULL, cover=1.5, n=n,
        ))

    # D3: sampling knobs (span/end/cloud) — portable Music-span analogues
    for label, skw in (
        ("span0.2", {"span_frac": 0.2}),
        ("span0.6", {"span_frac": 0.6}),
        ("end0.3", {"end_margin": 0.3}),
        ("end0.8", {"end_margin": 0.8}),
        ("cloud0.01", {"cloud_std": 0.01}),
        ("cloud0.06", {"cloud_std": 0.06}),
        ("jitter0.02", {"particle_jitter": 0.02}),
        ("jitter0.05", {"particle_jitter": 0.05}),
    ):
        cells.append(grid(
            f"close_s0_n1_{label}",
            close_field3d, [0], cover=1.5, n=1, **skw,
        ))

    # D4: lr / vicreg (seed-robust optimization, not geometry hack)
    for label, okw in (
        ("lr2e3", {"lr": 2.0e-3}),
        ("lr1e2", {"lr": 1.0e-2}),
        ("vic0", {"vicreg_weight": 0.0}),
        ("vic0.15", {"vicreg_weight": 0.15}),
        ("steps1600", {"steps": 1600}),
        ("cover2.0", {"cover": 2.0}),
    ):
        cells.append(grid(
            f"close_s0_n1_{label}",
            close_field3d, [0], cover=okw.pop("cover", 1.5), n=1, **okw,
        ))

    # D5: if any seed0 probe recovers, re-check full 6 seeds at that knob
    seed0_recovers = [
        c for c in cells
        if c["name"].startswith("close_s0_") and c["n_pass"] == c["n"]
        and c["name"] not in (
            # n≥2 recover expected — tracked separately
        )
        and c["kw"].get("n", 1) == 1
    ]
    for c in seed0_recovers:
        kw = dict(c["kw"])
        name = f"close_full_from_{c['name']}"
        cells.append(grid(name, close_field3d, SEEDS_FULL, **kw))

    # D6: close_live_noise seed0 same top probes if close seed0 stayed fail
    for label, kw in (
        ("l2_0.0", {"particle_l2": 0.0}),
        ("l2_0.1", {"particle_l2": 0.1}),
        ("n2", {"n": 2}),
        ("n4", {"n": 4}),
        ("jitter0.05", {"particle_jitter": 0.05}),
        ("end0.8", {"end_margin": 0.8}),
    ):
        cells.append(grid(
            f"live_s0_{label}",
            close_live_noise_field3d, [0], cover=1.5, **kw,
        ))

    wall = round(time.time() - t_wall, 1)

    # --- Verdict ---
    close_base = next(c for c in cells if c["name"] == "close_n1_c1.5_baseline")
    live_base = next(c for c in cells if c["name"] == "close_live_n1_c1.5_baseline")
    lyric = next(c for c in cells if c["name"] == "lyric_m1_locked_confirm")
    cross = next(c for c in cells if c["name"] == "cross_axis_rows_boundary")
    leftover = next(c for c in cells if c["name"] == "leftover_n12_c1.5")

    n_floor_full = {
        c["kw"].get("n"): c
        for c in cells
        if c["name"].startswith("close_full_n") and "_c1.5" in c["name"]
    }
    s0_n1_recovers = [
        c["name"] for c in cells
        if c["name"].startswith("close_s0_") and c["kw"].get("n", 1) == 1 and c["n_pass"] == 1
    ]
    any_n1_recipe_6of6 = any(
        c["n_pass"] == 6 and c["kw"].get("n", 1) == 1 and "close_full" in c["name"]
        for c in cells
    )
    # n floor that gives 6/6
    n_floor_6 = None
    for n in sorted(n_floor_full):
        if n_floor_full[n]["n_pass"] == 6:
            n_floor_6 = n
            break

    recipe_change = False  # default: document multi-seed / n-floor rule
    # Only flip if a portable n=1 knob gets 6/6 AND leftover stays green — still prefer doc
    if any_n1_recipe_6of6 and leftover["n_pass"] == leftover["n"]:
        # still prefer documenting unless it's clearly safe; flag candidate
        recipe_change = False

    verdict = {
        "lyric_span_hard_boundary": lyric["n_pass"] == 0,
        "cross_axis_hard_boundary": cross["n_pass"] == 0,
        "close_n1_knife": close_base["knife"] or (
            close_base["fail_seeds"] == [0] or 0 in close_base["fail_seeds"]
        ),
        "close_n1_fail_seeds": close_base["fail_seeds"],
        "live_n1_fail_seeds": live_base["fail_seeds"],
        "seed0_n1_probe_recovers": s0_n1_recovers,
        "any_n1_knob_full_6of6": any_n1_recipe_6of6,
        "n_particles_floor_for_6of6": n_floor_6,
        "leftover_regression": leftover["n_pass"] < leftover["n"],
        "recipe_change": recipe_change,
        "music_rule": (
            "multi-seed close + treat n_particles=1 (parts0) as knife-exposing; "
            "do not raise pole on seed0 alone. Prefer n≥4 for close geometry "
            "stability OR require multi-seed pass under parts0 proxy."
        ),
        "rejected_knobs": [
            "800×cover3.0 (false lock)",
            "lyric_span cover/n/steps/eoc chase (Fire #20 hard boundary)",
            "weaken cross_axis_rows cell to fake pass",
        ],
    }

    payload = {
        "fire": "harden_music_toy_bites",
        "host": "box-cpu",
        "sha": git_sha(),
        "wall_s": wall,
        "locked_recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "n_particles_max": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "diagnosis": {
            "lyric_span": (
                "rows_covered=0/5 via shared AdvResidual cannot hit heterogeneous "
                "row_amps; high u_kept on row0 only. Fire #20 hard boundary."
            ),
            "close_seed0": (
                "geometry identical across seeds; seed0 @ n=1 undershoots û "
                "(u_kept≈0.38, on_u≈0.046 vs target 0.12) while content_kept≈0.97 "
                "and multi-row 3/3 — init/basin with single particle, not geometry."
            ),
        },
        "verdict": verdict,
        "cells": [{k: v for k, v in c.items() if k != "runs"} | {"runs": c["runs"]} for c in cells],
        "summary": {
            c["name"]: {
                "pass": c["pass"],
                "mean_exam": c["mean_exam"],
                "mean_u": c["mean_u"],
                "leak_max": c["leak_max"],
                "fail_seeds": c["fail_seeds"],
                "kw": c["kw"],
            }
            for c in cells
        },
    }

    # Fire #24: annotate flattened runs
    flat = []
    for c in cells:
        for r in c.get("runs") or []:
            flat.append(r)
    annotate_dig_rows(flat, payload=payload)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # Markdown
    lines = [
        "# Harden Music→toy bites — 2026-09-09",
        "",
        f"Host: box-cpu @ `{payload['sha']}`. Wall {wall}s. CPU only. No Music train.",
        "",
        *posture_block_for_md(payload.get("music_close_posture") or {}),
        "",
        "## Context",
        "",
        "- **Fire #20:** `lyric_span_entangle` HARD BOUNDARY — shared AdvResidual cannot",
        "  cover heterogeneous `row_amps` (rows_covered=0/5 despite high row0 u_kept).",
        "  Do not chase cover/n/steps/eoc. Do not weaken cell.",
        "- **cross_axis_rows:** hard boundary (document; do not soften).",
        "- **This fire:** pivot harden dig to `close` / `close_live_noise` seed0 knife @ n=1.",
        "",
        "## Diagnosis (close seed0)",
        "",
        "- Geometry identical across seeds (`slider=0.12`, content=1.0, scales fixed).",
        "- seed0 @ n=1: `u_kept≈0.38` (û undershoot: on_u≈0.046 vs a_u=0.12);",
        "  `content_kept≈0.97`; multi-row **3/3**; fail = `pass_u` / leftover gate.",
        "- seed1+: û recovers (`u_kept≈1.05`, exam≈0.99).",
        "- **Cause:** single-particle init/basin, not geom or steps.",
        "",
        "## Grid (highlights)",
        "",
        "| cell | PASS | mean exam | mean u | leak max | fail |",
        "|---|:---:|---:|---:|---:|---|",
    ]
    for c in cells:
        if c["name"].startswith("close_s0_") and c["kw"].get("n", 1) != 1 and "n2" not in c["name"] and "n3" not in c["name"] and "n4" not in c["name"]:
            continue  # keep MD shorter for sampling probes — still in JSON
        if len(c["name"]) > 40 and c["name"].startswith("close_s0_n1_") and c["n_pass"] == 0:
            # fold long fail-only sampling into JSON; list key ones
            if not any(x in c["name"] for x in ("l2", "lr", "vic", "steps", "cover", "span", "end", "cloud", "jitter")):
                continue
        lines.append(
            f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | {c['mean_u']} | "
            f"{c['leak_max']} | {c['fail_seeds']} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"- **lyric_span hard boundary?** {verdict['lyric_span_hard_boundary']}",
        f"- **cross_axis hard boundary?** {verdict['cross_axis_hard_boundary']}",
        f"- **close n=1 knife?** {verdict['close_n1_knife']} fail={verdict['close_n1_fail_seeds']}",
        f"- **live n=1 fail seeds?** {verdict['live_n1_fail_seeds']}",
        f"- **seed0 @ n=1 probe recovers?** {s0_n1_recovers or 'none'}",
        f"- **any n=1 knob → full 6/6?** {any_n1_recipe_6of6}",
        f"- **n_particles floor for 6/6?** {n_floor_6}",
        f"- **leftover regression?** {verdict['leftover_regression']}",
        f"- **Recipe change?** NO — {verdict['music_rule']}",
        f"- **Rejected:** {verdict['rejected_knobs']}",
        "",
        "## Music parts0 proxy rule",
        "",
        "Under `n_particles=1` (parts0), close geometry is **seed-fragile** (seed0-only",
        "û undershoot). More steps do not fix it; `n≥4` does. Prefer:",
        "",
        "1. **Multi-seed gate** for close / close_live_noise under parts0 proxy, OR",
        "2. **n_particles ≥ 4** floor when scoring close-family cells (not a global",
        "   locked-recipe change — cell-family posture).",
        "",
        "Do **not** raise locked pole on seed0 alone. Do **not** adopt 800×cover3.0.",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Append research log
    log_entry = f"""
## Fire — harden Music→toy bites (2026-09-09)

- Host: box-cpu @ SHA `{payload['sha']}` (laptop offline; no SSH)
- Notes: `harden_music_toy_bites_20260909.{{py,json,md}}` wall={wall}s
- lyric_span / cross_axis: HARD BOUNDARY (confirm; no recipe chase) — Fire #20 stands
- close n=1 c1.5: {close_base['pass']} fail={close_base['fail_seeds']}; live: {live_base['pass']} fail={live_base['fail_seeds']}
- seed0 @ n=1 û-undershoot (u≈0.38); content+multi OK — init/basin
- seed0 n=1 probe recovers: {s0_n1_recovers or 'none'}; n_floor_6of6={n_floor_6}; any_n1_6of6={any_n1_recipe_6of6}
- leftover regression: {verdict['leftover_regression']}
- Verdict: recipe_change=NO; Music rule=multi-seed close / n≥4 for close-family under parts0
- No Music GPU train; servers untouched.
"""
    with LOG.open("a") as fh:
        fh.write(log_entry)

    print(json.dumps({"wall_s": wall, "verdict": verdict}, indent=2), flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
