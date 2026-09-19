#!/usr/bin/env python3
"""Declare-lie family unified theory + negative-control suite — 2026-09-09.

Family: YAML / declared-ê lies that delete content under faithful_guard_e.
Members: M14 e_on_u_declare_lie, M20 amp_lie_leftover_declare, M26 declare_split_three.
Contrast: M6 cross_axis_mismatch_declare (existing neg control); M16 DoF (per-row clears).

Negative controls (must PASS / stay distinct):
  - leftover CTRL locked
  - M14 e_on_u=0 (no û-lie)
  - M14/M20 teacher=faithful (no leftover gate) — bite needs guard
  - M16 locked still BITES shared; per_row clears (DoF ≠ declare-lie)

CPU only. No Music GPU. Locked recipe UNCHANGED.
Outputs: declare_lie_family_20260909.{md,json} + research_log/scoreboard append.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
NOTES = _REPO / "analysis/slider2d/notes"

_spec = importlib.util.spec_from_file_location(
    "per_row_explore", NOTES / "per_row_residual_explore_20260909.py"
)
ex = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["per_row_explore"] = ex
_spec.loader.exec_module(ex)

_dspec = importlib.util.spec_from_file_location(
    "per_row_deepen", NOTES / "per_row_residual_deepen_20260909.py"
)
dep = importlib.util.module_from_spec(_dspec)
assert _dspec.loader is not None
sys.modules["per_row_deepen"] = dep
_dspec.loader.exec_module(dep)

from analysis.slider2d.field3d import (  # noqa: E402
    amp_lie_leftover_declare_field3d,
    cross_axis_mismatch_declare_field3d,
    declare_split_three_field3d,
    e_on_u_declare_lie_field3d,
    leftover_field3d,
    scale_stagger_homo_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

OUT_JSON = NOTES / "declare_lie_family_20260909.json"
OUT_MD = NOTES / "declare_lie_family_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_shared(name, field_fn, teacher, seeds, *, n=12, cover=1.5, vic=None):
    runs = []
    for s in seeds:
        cfg = default_cfg(
            steps=1200, seed=s, b_cap=1.0, cover_weight=cover,
            fm_weight=0.0, n_particles=n, particle_l2=0.02,
        )
        if vic is not None:
            cfg = replace(cfg, vicreg_weight=float(vic))
        t0 = time.time()
        try:
            field = field_fn(seed=s)
        except TypeError:
            field = field_fn()
        row = score_adv_field3d_exam(field, teacher=teacher, cfg=cfg, name=f"{name}_s{s}")
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
            "pass_leak": bool(row.get("pass_leak")),
            "rows_covered": int(row.get("rows_covered", 0)),
            "wall_s": round(time.time() - t0, 2),
        })
        print(
            f"    {name} s={s} pass={runs[-1]['pass']} exam={runs[-1]['exam_score']:.4f} "
            f"cont={runs[-1]['content_kept']:.4f} leak={runs[-1]['leak_ratio']:.4f} "
            f"({runs[-1]['wall_s']}s)",
            flush=True,
        )
    return pack(name, "shared", teacher, n, cover, vic, runs)


def run_per_row(name, field_fn, seeds, *, coupling_weight=0.0):
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s} w={coupling_weight}", flush=True)
        try:
            f = field_fn(seed=s)
        except TypeError:
            f = field_fn()
        out = dep._recompute_gates(
            ex.score_scaffold(
                f, mode="per_row", seed=s, name=f"{name}_s{s}",
                coupling_weight=float(coupling_weight),
            )
        )
        runs.append({
            "seed": s,
            "pass": bool(out.get("exam_pass") or out.get("pass")),
            "exam_score": float(out.get("exam_score") or out.get("mean_exam") or 0),
            "u_kept": float(out.get("u_kept", 0)),
            "content_kept": float(out.get("content_kept", 0)),
            "leak_ratio": float(out.get("leak_ratio", 0)),
            "exam_cont": float(out.get("exam_cont", 0)),
            "exam_swing": float(out.get("exam_swing", 0)),
            "pass_cont": bool(out.get("pass_cont")),
            "pass_swing": bool(out.get("pass_swing")),
            "pass_leftover_gate": bool(out.get("pass_leftover_gate")),
            "pass_multi_row": bool(out.get("pass_multi_row")),
            "pass_u": bool(out.get("pass_u")),
            "pass_content": bool(out.get("pass_content")),
            "pass_leak": bool(out.get("pass_leak")),
            "rows_covered": int(out.get("rows_covered", 0)),
            "bite_cleared": bool(out.get("bite_cleared")),
        })
    return pack(name, "per_row", "faithful_guard_e", 12, 1.5, None, runs, coupling=coupling_weight)


def pack(name, mode, teacher, n, cover, vic, runs, coupling=None):
    nn = len(runs)

    def mean(k):
        vals = [float(r[k]) for r in runs if r.get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def frac(k):
        return round(sum(1 for r in runs if r.get(k)) / nn, 2) if nn else 0.0

    fail_runs = [r for r in runs if not r["pass"]]
    gate_counts = {
        "cont": sum(1 for r in fail_runs if not r.get("pass_cont")),
        "swing": sum(1 for r in fail_runs if not r.get("pass_swing")),
        "leftover": sum(1 for r in fail_runs if not r.get("pass_leftover_gate")),
        "multi": sum(1 for r in fail_runs if not r.get("pass_multi_row")),
    }
    out = {
        "name": name, "mode": mode, "teacher": teacher,
        "n_particles": n, "cover_weight": cover, "vicreg_weight": vic,
        "coupling_weight": coupling,
        "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
        "n_pass": sum(1 for r in runs if r["pass"]), "n": nn,
        "mean_exam": mean("exam_score"), "mean_u": mean("u_kept"),
        "mean_content": mean("content_kept"), "mean_leak": mean("leak_ratio"),
        "mean_cont": mean("exam_cont"), "mean_swing": mean("exam_swing"),
        "mean_rows": mean("rows_covered"),
        "frac_cont": frac("pass_cont"), "frac_swing": frac("pass_swing"),
        "frac_leftover": frac("pass_leftover_gate"),
        "frac_multi": frac("pass_multi_row"),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "fail_gate_counts": gate_counts, "runs": runs,
    }
    print(
        f"  {name}: {out['pass']} exam={out['mean_exam']} u={out['mean_u']} "
        f"content={out['mean_content']} leak={out['mean_leak']} "
        f"left={out['frac_leftover']} multi={out['frac_multi']} gates={gate_counts}",
        flush=True,
    )
    return out


def nature_of(cell: dict) -> str:
    if cell["n_pass"] == cell["n"]:
        return "PASS"
    gc = cell["fail_gate_counts"]
    nfail = max(len(cell["fail_seeds"]), 1)
    if gc.get("leftover", 0) >= nfail and cell["mean_content"] < 0.75 and cell["mean_u"] >= 0.85:
        return "content_deleted_under_declare_lie"
    if gc.get("multi", 0) >= nfail and cell["mean_leak"] < 0.15:
        return "multi_row_dof"
    if gc.get("leftover", 0) >= nfail and cell["mean_leak"] >= 0.5:
        return "declare_lie_raw_leak_blowup"
    if gc.get("leftover", 0) >= nfail and cell["mean_leak"] >= 0.18:
        return "leak_or_content_under_declare"
    top = max(gc, key=gc.get) if gc else "unknown"
    return f"gate_{top}"


def cleared(c):
    return c["n_pass"] == c["n"]


def main():
    t0 = time.time()
    sha = git_sha()
    print(f"=== declare-lie family suite @ {sha} ===", flush=True)
    cells = []

    print("\n[CTRL] leftover", flush=True)
    cells.append(run_shared("CTRL_leftover", leftover_field3d, "faithful_guard_e", SEEDS_SMOKE))

    print("\n[A] family locked characterization", flush=True)
    for name, fn in (
        ("M14_locked", e_on_u_declare_lie_field3d),
        ("M20_locked", amp_lie_leftover_declare_field3d),
        ("M26_locked", declare_split_three_field3d),
        ("M6_mismatch_declare", cross_axis_mismatch_declare_field3d),
    ):
        cells.append(run_shared(name, fn, "faithful_guard_e", SEEDS_FULL))

    print("\n[B] negative controls (must PASS or stay non-declare)", flush=True)
    # eou=0 — no û-lie
    cells.append(run_shared(
        "NEG_M14_eou0",
        lambda seed=None: e_on_u_declare_lie_field3d(e_on_u=0.0, e_on_content=0.0, e_unused=1.0),
        "faithful_guard_e", SEEDS_SMOKE,
    ))
    # faithful no-guard — should PASS M14/M20 (bite needs guard)
    cells.append(run_shared("NEG_M14_faithful", e_on_u_declare_lie_field3d, "faithful", SEEDS_SMOKE))
    cells.append(run_shared("NEG_M20_faithful", amp_lie_leftover_declare_field3d, "faithful", SEEDS_SMOKE))
    cells.append(run_shared("NEG_M26_faithful", declare_split_three_field3d, "faithful", SEEDS_SMOKE))
    # DoF contrast: M16 shared BITES
    cells.append(run_shared("CONTRAST_M16_shared", scale_stagger_homo_field3d, "faithful_guard_e", SEEDS_SMOKE))

    print("\n[C] clearance smoke (n2 / vic0 / per_row) — expect NO for declare-lie", flush=True)
    for mid, fn, seeds in (
        ("M14", e_on_u_declare_lie_field3d, SEEDS_SMOKE),
        ("M20", amp_lie_leftover_declare_field3d, SEEDS_SMOKE),
        ("M26", declare_split_three_field3d, SEEDS_FULL),
    ):
        cells.append(run_shared(f"{mid}_n2", fn, "faithful_guard_e", seeds, n=2, cover=1.5))
        cells.append(run_shared(f"{mid}_n1_vic0", fn, "faithful_guard_e", seeds, n=1, cover=1.0, vic=0.0))
        cells.append(run_per_row(f"{mid}_per_row", fn, seeds, coupling_weight=0.0))

    print("\n[D] DoF contrast per_row — M16 must CLEAR", flush=True)
    cells.append(run_per_row("CONTRAST_M16_per_row", scale_stagger_homo_field3d, SEEDS_SMOKE, coupling_weight=0.0))

    print("\n[E] soft eou ladder reconfirm", flush=True)
    for eou in (0.0, 0.15, 0.3, 1.5):
        cells.append(run_shared(
            f"M14_eou{eou}",
            lambda seed=None, e=eou: e_on_u_declare_lie_field3d(
                e_on_u=e,
                e_on_content=0.1 if e > 0 else 0.0,
                e_unused=0.1 if e > 0 else 1.0,
            ),
            "faithful_guard_e", SEEDS_SMOKE,
        ))

    wall = round(time.time() - t0, 1)
    by = {c["name"]: c for c in cells}

    natures = {k: nature_of(by[k]) for k in (
        "M14_locked", "M20_locked", "M26_locked", "M6_mismatch_declare", "CONTRAST_M16_shared"
    )}

    neg_ok = {
        "leftover": cleared(by["CTRL_leftover"]),
        "m14_eou0": cleared(by["NEG_M14_eou0"]),
        "m14_faithful": cleared(by["NEG_M14_faithful"]),
        "m20_faithful": cleared(by["NEG_M20_faithful"]),
        "m26_faithful": cleared(by["NEG_M26_faithful"]),
        "m16_shared_bites": by["CONTRAST_M16_shared"]["n_pass"] == 0,
        "m16_per_row_clears": cleared(by["CONTRAST_M16_per_row"]),
    }

    clearance = {}
    for mid in ("M14", "M20", "M26"):
        clearance[mid] = {
            "n2": cleared(by[f"{mid}_n2"]),
            "vic0": cleared(by[f"{mid}_n1_vic0"]),
            "per_row": cleared(by[f"{mid}_per_row"]),
        }

    family_same = all(
        natures[k] == "content_deleted_under_declare_lie"
        for k in ("M14_locked", "M20_locked")
    )
    # M26 may be same or leak_or_content
    any_declare_cleared = any(
        any(clearance[m].values()) for m in ("M14", "M20", "M26")
    )
    suite_pass = (
        neg_ok["leftover"] and neg_ok["m14_eou0"]
        and neg_ok["m14_faithful"] and neg_ok["m20_faithful"]
        and neg_ok["m16_shared_bites"] and neg_ok["m16_per_row_clears"]
        and not any_declare_cleared
        and by["M14_locked"]["n_pass"] == 0
        and by["M20_locked"]["n_pass"] == 0
        and by["M26_locked"]["n_pass"] == 0
    )

    verdict = (
        f"family M14/M20 natures={natures['M14_locked']}/{natures['M20_locked']} "
        f"same_content_delete={family_same}; M26={natures['M26_locked']}; "
        f"M6={natures['M6_mismatch_declare']}; "
        f"clearance n/vic/per_row any_declare={any_declare_cleared} ({clearance}); "
        f"neg_ok={neg_ok}; suite_pass={suite_pass}; keep HARD_BITEs; recipe_change=NO"
    )

    payload = {
        "wall_s": wall, "sha": sha, "recipe_change": False, "merge_to_trainer": False,
        "natures": natures, "clearance": clearance, "neg_ok": neg_ok,
        "suite_pass": suite_pass, "family_same_content_delete": family_same,
        "verdict": verdict, "cells": cells,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    theory = f"""# Declare-lie family — unified theory + negative-control suite (2026-09-09)

Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. No Music GPU.
Locked shared AdvResidual **unchanged**. per-row = analysis-only.

## Theory

**Declare-lie family:** the YAML / declared-ê vector is *not* the exam leak axis
the leftover gate scores. Under `faithful_guard_e`, the teacher subtracts along
the *declared* direction. When that direction restates **û** (M14), points at
**content** (M20), or splits ambiguously (M26), the gate deletes or fails content
while û may still look healthy.

| member | lie geometry | locked nature | exam≈ | content | leak |
|---|---|---|---:|---:|---:|
| **M14** `e_on_u_declare_lie` | declared ê ∥ û (`e_on_u` hot) | {natures['M14_locked']} | {by['M14_locked']['mean_exam']} | {by['M14_locked']['mean_content']} | {by['M14_locked']['mean_leak']} |
| **M20** `amp_lie_leftover_declare` | declared ê ∥ content on leftover geom | {natures['M20_locked']} | {by['M20_locked']['mean_exam']} | {by['M20_locked']['mean_content']} | {by['M20_locked']['mean_leak']} |
| **M26** `declare_split_three` | ê split û/content/unused | {natures['M26_locked']} | {by['M26_locked']['mean_exam']} | {by['M26_locked']['mean_content']} | {by['M26_locked']['mean_leak']} |
| **M6** `cross_axis_mismatch_declare` | hetero mismatch declare (legacy neg) | {natures['M6_mismatch_declare']} | {by['M6_mismatch_declare']['mean_exam']} | {by['M6_mismatch_declare']['mean_content']} | {by['M6_mismatch_declare']['mean_leak']} |

**Not in family (DoF contrast):** M16 `scale_stagger_homo` — shared BITES multi_row,
per-row **clears**. Fix = more heads / per-row residual (analysis-only), not YAML.

**Not in family (ê-floor):** M21/M30/M31 — teacher declared_e tilt vs exam pure ê;
analytic floor; per-row cannot clear (see `m21_eoc_leak_irreducible`).

**Not in family (guard refuse):** M28 — blend guard refuses → raw-pole leak≈0.66.

### Why n / vic / per-row cannot clear

Declare-lie is a **target/YAML bug**, not a particle / VICReg / DoF bug.
More particles or per-row heads still train toward the *wrong declared axis*;
the leftover content gate still fails (or content stays deleted).

### Cliff (M14)

`e_on_u=0` → PASS; `e_on_u≥0.3` → BITES (any û-restatement on).

## Verdict

{verdict}

## Negative-control suite

| control | expected | result | PASS |
|---|---|---|:---:|
| leftover CTRL | PASS | {by['CTRL_leftover']['pass']} | {'YES' if neg_ok['leftover'] else 'no'} |
| M14 eou=0 | PASS | {by['NEG_M14_eou0']['pass']} | {'YES' if neg_ok['m14_eou0'] else 'no'} |
| M14 faithful (no guard) | PASS | {by['NEG_M14_faithful']['pass']} | {'YES' if neg_ok['m14_faithful'] else 'no'} |
| M20 faithful | PASS | {by['NEG_M20_faithful']['pass']} | {'YES' if neg_ok['m20_faithful'] else 'no'} |
| M26 faithful | PASS? | {by['NEG_M26_faithful']['pass']} | {'YES' if neg_ok['m26_faithful'] else 'no'} |
| M16 shared | BITES | {by['CONTRAST_M16_shared']['pass']} | {'YES' if neg_ok['m16_shared_bites'] else 'no'} |
| M16 per_row | CLEARS | {by['CONTRAST_M16_per_row']['pass']} | {'YES' if neg_ok['m16_per_row_clears'] else 'no'} |

**suite_pass = {suite_pass}** (declare-lies stay biting; negs + DoF contrast hold).

## Clearance (expect all NO)

| mid | n2 | vic0@n1 | per_row |
|---|:---:|:---:|:---:|
| M14 | {by['M14_n2']['pass']} | {by['M14_n1_vic0']['pass']} | {by['M14_per_row']['pass']} |
| M20 | {by['M20_n2']['pass']} | {by['M20_n1_vic0']['pass']} | {by['M20_per_row']['pass']} |
| M26 | {by['M26_n2']['pass']} | {by['M26_n1_vic0']['pass']} | {by['M26_per_row']['pass']} |

## eou ladder

| eou | PASS | exam | content |
|---|:---:|---:|---:|
"""
    for eou in (0.0, 0.15, 0.3, 1.5):
        c = by[f"M14_eou{eou}"]
        theory += f"| {eou} | {c['pass']} | {c['mean_exam']} | {c['mean_content']} |\n"

    theory += """
## Full cells

| cell | mode | PASS | exam | u | content | leak | fail |
|---|---|:---:|---:|---:|---:|---:|---|
"""
    for c in cells:
        theory += (
            f"| `{c['name']}` | {c['mode']} | {c['pass']} | {c['mean_exam']} | "
            f"{c['mean_u']} | {c['mean_content']} | {c['mean_leak']} | {c['fail_seeds']} |\n"
        )

    theory += f"""
## Recipe / ADOPT

- Recipe change: **NO**
- merge_to_trainer: **NO**
- Keep M14 / M20 / M26 as HARD_BITEs (overpowered-head / YAML-lie falsifiers)
- Music-posture n≥2 / vic0@n1 do **not** apply
- Per-row w≤0.3 ADOPT does **not** clear this family (clears DoF only)
- Fix path: correct declared ê / YAML targets — not recipe knobs

JSON: `{OUT_JSON.name}`
"""
    OUT_MD.write_text(theory)

    log_block = (
        f"\n## Fire — declare-lie family unified theory + neg-control suite (2026-09-09)\n\n"
        f"- Host: box-cpu @ `{sha}`\n"
        f"- Dig: `declare_lie_family_20260909.{{py,json,md}}` wall={wall}s\n"
        f"- Natures: M14={natures['M14_locked']}; M20={natures['M20_locked']}; "
        f"M26={natures['M26_locked']}; M6={natures['M6_mismatch_declare']}\n"
        f"- Clearance any_declare={any_declare_cleared}: {clearance}\n"
        f"- neg_ok={neg_ok}; suite_pass={suite_pass}\n"
        f"- Verdict: {verdict}\n"
        f"- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.\n"
    )
    with LOG.open("a") as f:
        f.write(log_block)

    fold = (
        f"\n\n## Declare-lie family theory + neg-control suite (2026-09-09)\n\n"
        f"Source: `declare_lie_family_20260909` wall={wall}s. **suite_pass={suite_pass}**.\n\n"
        f"| member | nature |\n|---|---|\n"
        f"| M14 | {natures['M14_locked']} |\n"
        f"| M20 | {natures['M20_locked']} |\n"
        f"| M26 | {natures['M26_locked']} |\n"
        f"| M6 | {natures['M6_mismatch_declare']} |\n\n"
        f"n/vic/per_row clear any? **{any_declare_cleared}**. "
        f"Neg controls (eou0 / faithful / leftover / M16 DoF contrast): **{neg_ok}**.\n"
        f"Keep HARD_BITEs. Recipe change=NO.\n"
    )
    with SCOREBOARD.open("a") as f:
        f.write(fold)

    with CATALOG.open("a") as f:
        f.write(
            f"\n\n## Declare-lie family (2026-09-09)\n\n"
            f"Unified note: `declare_lie_family_20260909`. "
            f"M14/M20 = content_deleted_under_declare_lie; M26={natures['M26_locked']}; "
            f"suite_pass={suite_pass}. n/vic/per_row NO. Fix=YAML/target not recipe.\n"
        )

    print(json.dumps({
        "wall_s": wall, "suite_pass": suite_pass, "verdict": verdict, "natures": natures,
    }, indent=2))


if __name__ == "__main__":
    main()
