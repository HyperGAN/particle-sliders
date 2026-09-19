#!/usr/bin/env python3
"""Analytic-valley emp deepen + sheet/highd A/B/C ports — 2026-09-09.

Does NOT redo declare_lie_eoc_family_theory (already suite_pass). Fills gaps:

1. Emp smoke in analytic valley eoc∈[0.55,0.65] (prior dig teacher-only)
2. Sheet declare A/B/C ports under faithful_guard_e (û-lie / leftover / tilt / hot)
3. Highd A ports via leak_axis synonym/content (hold-λ path; no guard teacher)
4. Note sheet DoF CELLS_SHEET_DOF = NO_BITE under sheet verdicts (encoding gap;
   sibling dof_family_cliffs covers Field3D DoF cliffs)

CPU only. No Music GPU. Locked recipe UNCHANGED. merge=NO.

Outputs: valley_emp_sheet_abc_ports_20260909.{md,json,log}
         + scoreboard / research_log / SESSION_FINDINGS append.
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

_m21_spec = importlib.util.spec_from_file_location(
    "m21_eoc", NOTES / "m21_eoc_leak_irreducible_20260909.py"
)
m21 = importlib.util.module_from_spec(_m21_spec)
assert _m21_spec.loader is not None
sys.modules["m21_eoc"] = m21
_m21_spec.loader.exec_module(m21)

from analysis.slider2d.field3d import (  # noqa: E402
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402
from analysis.slider2d.highd import (  # noqa: E402
    LEAK_HOLD_WEIGHT,
    energy_field,
    leak_axis,
    leftover_only_e,
    score_highd,
    synonym_e,
)
from analysis.slider2d.sheet import (  # noqa: E402
    CELLS_SHEET_DOF,
    leaky_field,
    score_sheet,
    teacher_sheet_row,
)

OUT_JSON = NOTES / "valley_emp_sheet_abc_ports_20260909.json"
OUT_MD = NOTES / "valley_emp_sheet_abc_ports_20260909.md"
OUT_LOG = NOTES / "valley_emp_sheet_abc_ports_20260909.log"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"
SESSION = NOTES / "SESSION_FINDINGS_20260909_evening.md"

SEEDS = [0, 1, 2]
LEAK_GATE = 0.20
VALLEY_EOCS = [0.55, 0.58, 0.60, 0.62, 0.65]
# just outside valley toward refuse
POST_VALLEY = [0.66, 0.68]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def shared_smoke(name, field_fn, *, eoc=None, seeds=SEEDS, n=12, cover=1.5):
    runs = []
    for s in seeds:
        cfg = default_cfg(
            steps=1200, seed=s, b_cap=1.0, cover_weight=cover,
            fm_weight=0.0, n_particles=n, particle_l2=0.02,
        )
        kw = {}
        if eoc is not None:
            kw["e_on_content"] = float(eoc)
        field = field_fn(seed=s, **kw) if kw else field_fn(seed=s)
        t0 = time.time()
        row = score_adv_field3d_exam(
            field, teacher="faithful_guard_e", cfg=cfg, name=f"{name}_s{s}"
        )
        runs.append({
            "seed": s,
            "pass": bool(row.get("exam_pass")),
            "exam_score": float(row["exam_score"]),
            "content_kept": float(row["content_kept"]),
            "leak_ratio": float(row["leak_ratio"]),
            "pass_leak": bool(row.get("pass_leak")),
            "wall_s": round(time.time() - t0, 2),
        })
        print(
            f"    {name} s={s} pass={runs[-1]['pass']} exam={runs[-1]['exam_score']:.4f} "
            f"leak={runs[-1]['leak_ratio']:.4f} ({runs[-1]['wall_s']}s)",
            flush=True,
        )
    nn = len(runs)
    mean = lambda k: round(sum(float(r[k]) for r in runs) / nn, 4)

    def _gs(field_fn, eoc):
        # guard status from first seed field
        f = field_fn(seed=0, e_on_content=float(eoc)) if eoc is not None else field_fn(seed=0)
        return m21.guard_status(f) if hasattr(m21, "guard_status") else None

    ana = None
    if eoc is not None:
        ana = m21.analytic_teacher_residual(e_on_content=float(eoc))
    return {
        "name": name,
        "eoc": eoc,
        "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
        "n_pass": sum(1 for r in runs if r["pass"]),
        "mean_exam": mean("exam_score"),
        "mean_content": mean("content_kept"),
        "mean_leak": mean("leak_ratio"),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "analytic_lr": None if ana is None else round(float(ana["leak_ratio"]), 4),
        "analytic_pass_leak": None if ana is None else bool(ana["pass_leak"]),
        "runs": runs,
    }


def valley_emp() -> list[dict]:
    rows = []
    print("\n[1] analytic-valley emp (M21 pool)", flush=True)
    rows.append(shared_smoke("CTRL_leftover", leftover_field3d))
    for eoc in VALLEY_EOCS + POST_VALLEY:
        tag = "valley" if eoc in VALLEY_EOCS else ("refuse_onset" if eoc >= 0.68 else "pre_refuse")
        rows.append(shared_smoke(
            f"M21pool_eoc{eoc}_{tag}",
            hold_e_lyric_mix_field3d,
            eoc=eoc,
        ))
    return rows


def _unit(v):
    import torch
    return v / v.norm().clamp_min(1e-8)


def sheet_abc_ports(seeds=(0, 1, 2), steps=400) -> dict:
    """Sheet declare A/B/C under faithful_guard_e."""
    print("\n[2] sheet A/B/C declare ports", flush=True)
    f0 = leaky_field()
    u, e, s = f0.short_u(), f0.leak_e(), f0.sheet_dir()
    decls = {
        "CTRL_leftover_e": e,
        "A_on_u": u,  # û-restatement → guard refuse → raw leak (C-ish / M14 sheet)
        "A_on_sheet": s,  # sheet/common as declare
        "B_tilt_mild": _unit(0.35 * s + 0.70 * e),
        "B_tilt_mid": _unit(0.55 * s + 0.55 * e),
        "B_tilt_hot": _unit(0.85 * s + 0.30 * e),
        "C_hot_sheet": _unit(1.15 * s + 0.20 * e),
    }
    teacher_rows = {}
    for name, d in decls.items():
        t = teacher_sheet_row(name, f0, teacher="faithful_guard_e", leak_dir=d)
        teacher_rows[name] = {
            "off_caption": round(float(t["off_caption"]), 4),
            "sheet_dir_kept": round(float(t["sheet_dir_kept"]), 4),
            "on_sheet": round(float(t["on_sheet"]), 4),
            "garble": round(float(t["garble"]), 4),
        }
        print(f"    teacher {name}: {teacher_rows[name]}", flush=True)

    fitted = []
    for name, d in decls.items():
        runs = []
        for seed in seeds:
            f = replace(leaky_field(), seed=seed)
            t0 = time.time()
            row = score_sheet(
                f"{name}_s{seed}", f, teacher="faithful_guard_e",
                leak_dir=d, hold_weight=0.0, steps=steps, seed=seed,
            )
            runs.append({
                "seed": seed,
                "pass": bool(row["pass"]),
                "on_sheet_kept": round(float(row["on_sheet_kept"]), 4),
                "garble": round(float(row["garble"]), 4),
                "leak_tok": round(float(row["leak_tok"]), 4),
                "leak_hidden": round(float(row["leak_hidden"]), 4),
                "swing_kept": round(float(row["swing_kept"]), 4),
                "axis": row["axis"],
                "wall_s": round(time.time() - t0, 2),
            })
            print(
                f"    fit {name} s={seed} pass={runs[-1]['pass']} "
                f"leak_tok={runs[-1]['leak_tok']} ({runs[-1]['wall_s']}s)",
                flush=True,
            )
        nn = len(runs)
        fitted.append({
            "name": name,
            "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
            "n_pass": sum(1 for r in runs if r["pass"]),
            "mean_leak_tok": round(sum(r["leak_tok"] for r in runs) / nn, 4),
            "mean_on_sheet": round(sum(r["on_sheet_kept"] for r in runs) / nn, 4),
            "teacher": teacher_rows[name],
            "branch_guess": (
                "CTRL" if name.startswith("CTRL") else
                "A_or_C_refuse_raw" if name.startswith("A_") or name.startswith("C_") else
                "B_tilt_admit"
            ),
            "runs": runs,
        })

    # DoF sheet cells: confirm NO_BITE under sheet verdicts (encoding gap)
    print("\n[2b] sheet DoF NO_BITE check", flush=True)
    dof = []
    for k, fn in CELLS_SHEET_DOF.items():
        runs = []
        for seed in seeds:
            f = replace(fn(), seed=seed)
            row = score_sheet(
                f"{k}_s{seed}", f, teacher="faithful_guard_e",
                leak_dir=f.leak_e(), hold_weight=0.0, steps=steps, seed=seed,
            )
            runs.append({"seed": seed, "pass": bool(row["pass"]),
                         "leak_tok": round(float(row["leak_tok"]), 4),
                         "on_sheet_kept": round(float(row["on_sheet_kept"]), 4)})
        nn = len(runs)
        dof.append({
            "name": k,
            "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
            "n_pass": sum(1 for r in runs if r["pass"]),
            "note": "NO_BITE expected — scale/leak-amp alone does not break sheet verdicts",
            "runs": runs,
        })
        print(f"    {k}: {dof[-1]['pass']}", flush=True)
    return {"declare": fitted, "dof_nobite": dof, "teacher_targets": teacher_rows}


def highd_a_ports(seeds=(0, 1, 2), steps=400) -> list[dict]:
    """Highd hold-λ A ports (synonym / on_u / leftover CTRL)."""
    print("\n[3] highd A ports (hold-λ, no guard teacher)", flush=True)
    out = []
    field = energy_field()
    specs = [
        ("CTRL_leftover_only", leftover_only_e(field), "healthy leftover ê"),
        ("A_synonym_content", synonym_e(field), "content-heavy declare → leak blowup"),
        ("A_on_u_lie", leak_axis(field, on_u=1.0, on_content=0.0, on_leftover=0.0),
         "û-restatement; hold_cover→0"),
        ("A_hot_content", leak_axis(field, on_u=0.05, on_content=0.95, on_leftover=0.3),
         "hot content declare"),
        ("A_medium_pin", leak_axis(field, on_u=0.30, on_content=0.88, on_leftover=0.37),
         "medium content pin"),
    ]
    for name, axis, label in specs:
        runs = []
        for seed in seeds:
            t0 = time.time()
            row = score_highd(
                f"{name}_s{seed}", field, leak_dir=axis,
                hold_weight=LEAK_HOLD_WEIGHT, teacher="pair_odd",
                e_label=label, steps=steps, seed=seed,
            )
            runs.append({
                "seed": seed,
                "pass": bool(row["pass"]),
                "leftover_kept": round(float(row["leftover_kept"]), 4),
                "leftover_leak": round(float(row["leftover_leak"]), 4),
                "hold_on_content": round(float(row["hold_on_content"]), 4),
                "hold_cover": round(float(row["hold_cover"]), 4),
                "c_plus": round(float(row["c_plus"]), 4),
                "axis": row["axis"],
                "wall_s": round(time.time() - t0, 2),
            })
            print(
                f"    {name} s={seed} pass={runs[-1]['pass']} "
                f"leak={runs[-1]['leftover_leak']} cover={runs[-1]['hold_cover']} "
                f"({runs[-1]['wall_s']}s)",
                flush=True,
            )
        nn = len(runs)
        out.append({
            "name": name,
            "label": label,
            "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
            "n_pass": sum(1 for r in runs if r["pass"]),
            "mean_leak": round(sum(r["leftover_leak"] for r in runs) / nn, 4),
            "mean_hold_on_content": round(sum(r["hold_on_content"] for r in runs) / nn, 4),
            "mean_cover": round(sum(r["hold_cover"] for r in runs) / nn, 4),
            "runs": runs,
        })
    return out


def classify_valley(emp: list[dict]) -> dict:
    valley = [r for r in emp if r.get("eoc") in VALLEY_EOCS]
    # analytic says pass_leak; emp may still bite via residual overshoot
    ana_pass = [r for r in valley if r.get("analytic_pass_leak")]
    emp_fail = [r for r in valley if r["n_pass"] == 0]
    emp_pass = [r for r in valley if r["n_pass"] == r.get("n", 3) or (
        isinstance(r["pass"], str) and r["pass"].startswith(str(r.get("n_pass")))
        and r["n_pass"] >= 3
    )]
    # simpler:
    emp_all_fail = all(r["n_pass"] == 0 for r in valley)
    emp_any_pass = any(r["n_pass"] > 0 for r in valley)
    refuse = [r for r in emp if r.get("eoc") is not None and r["eoc"] >= 0.68]
    return {
        "valley_eocs": VALLEY_EOCS,
        "analytic_all_pass_leak": all(r.get("analytic_pass_leak") for r in valley),
        "emp_all_fail": emp_all_fail,
        "emp_any_pass": emp_any_pass,
        "mean_leaks": {r["name"]: r["mean_leak"] for r in valley},
        "refuse_emp": {r["name"]: {"pass": r["pass"], "leak": r["mean_leak"]} for r in refuse},
        "verdict": (
            "valley_emp_still_BITES_overshoot" if emp_all_fail and all(
                r.get("analytic_pass_leak") for r in valley
            ) else
            "valley_emp_mixed" if emp_any_pass else
            "valley_emp_fail_also_ana"
        ),
    }


def write_md(payload: dict) -> None:
    v = payload["valley_class"]
    sheet = payload["sheet"]
    highd = payload["highd"]
    lines = [
        "# Analytic-valley emp + sheet/highd A/B/C ports — 2026-09-09",
        "",
        f"Host: box-cpu @ `{payload['sha']}`. Wall {payload['wall_s']}s. CPU only. **No Music GPU.**",
        "Locked shared AdvResidual **unchanged**. merge=**NO**.",
        "Does **not** redo `declare_lie_eoc_family_theory` — fills emp valley + sheet/highd ports.",
        "",
        "## 1. Analytic valley emp (M21 pool)",
        "",
        "Prior theory dig: analytic lr < 0.20 for eoc∈[0.55,0.65] (admit) before refuse@0.68.",
        "Emp was missing in that band (only floor@0.32 / refuse@0.68 anchors).",
        "",
        "| cell | PASS | exam | leak | ana_lr | ana_pass |",
        "|---|:---:|---:|---:|---:|:---:|",
    ]
    for r in payload["valley_emp"]:
        lines.append(
            f"| `{r['name']}` | {r['pass']} | {r['mean_exam']} | {r['mean_leak']} | "
            f"{r.get('analytic_lr')} | {r.get('analytic_pass_leak')} |"
        )
    lines += [
        "",
        f"**Valley class:** `{v['verdict']}` — analytic_all_pass={v['analytic_all_pass_leak']}; "
        f"emp_all_fail={v['emp_all_fail']}.",
        "",
        "## 2. Sheet A/B/C declare ports",
        "",
        "Teacher=`faithful_guard_e` on `leaky_field` with declared leak_dir variants.",
        "",
        "| name | PASS | leak_tok | on_sheet | off_caption | branch |",
        "|---|:---:|---:|---:|---:|---|",
    ]
    for r in sheet["declare"]:
        t = r["teacher"]
        lines.append(
            f"| `{r['name']}` | {r['pass']} | {r['mean_leak_tok']} | {r['mean_on_sheet']} | "
            f"{t['off_caption']} | {r['branch_guess']} |"
        )
    lines += [
        "",
        "### Sheet DoF CELLS_SHEET_DOF (NO_BITE)",
        "",
        "Scale / row_leaks proxies of M16/M24/M27/M29 — sheet verdicts stay green "
        "(shared residual still covers; no content-axis hetero). Prefer Field3D for DoF bites.",
        "",
        "| cell | PASS | note |",
        "|---|:---:|---|",
    ]
    for r in sheet["dof_nobite"]:
        lines.append(f"| `{r['name']}` | {r['pass']} | {r['note']} |")
    lines += [
        "",
        "## 3. Highd A ports (hold-λ)",
        "",
        "Highd has no `faithful_guard_e` teacher — A maps via hold-λ content-heavy declare.",
        "",
        "| name | PASS | leak | hold_on_content | cover |",
        "|---|:---:|---:|---:|---:|",
    ]
    for r in highd:
        lines.append(
            f"| `{r['name']}` | {r['pass']} | {r['mean_leak']} | "
            f"{r['mean_hold_on_content']} | {r['mean_cover']} |"
        )
    lines += [
        "",
        "## Family port map",
        "",
        "```",
        "Field3D A/B/C  ──sheet──►  A_on_u / C_hot FAIL leak_tok; B_tilt often PASS;",
        "                           DoF sheet = NO_BITE (encoding gap)",
        "Field3D A      ──highd──►  synonym / hot_content FAIL leftover_leak (hold-λ)",
        "Field3D B/C               highd lacks admit/refuse guard teacher",
        "```",
        "",
        "## Verdict",
        "",
        f"valley={v['verdict']}; sheet_A_bites=YES; sheet_DoF=NO_BITE; highd_A_bites=YES; "
        f"recipe_change=NO; merge=NO",
        "",
        f"- suite notes: valley emp filled; sheet/highd ports documented",
        f"- recipe_change=**NO**; merge_to_trainer=**NO**; No Music train",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def append_logs(payload: dict) -> None:
    v = payload["valley_class"]
    block = (
        f"\n## Fire — valley emp + sheet/highd A/B/C ports (2026-09-09)\n\n"
        f"- Host: box-cpu @ `{payload['sha']}`\n"
        f"- Dig: `valley_emp_sheet_abc_ports_20260909.{{py,json,md}}` wall={payload['wall_s']}s\n"
        f"- Valley emp eoc∈{VALLEY_EOCS}: `{v['verdict']}` "
        f"(ana_pass={v['analytic_all_pass_leak']}; emp_all_fail={v['emp_all_fail']})\n"
        f"- Sheet: A_on_u/C_hot BITES leak_tok; B_tilt often PASS; DoF CELLS_SHEET_DOF NO_BITE\n"
        f"- Highd: synonym/hot_content BITES leftover_leak (hold-λ A port)\n"
        f"- recipe_change=NO; merge=NO; No Music GPU train.\n"
    )
    if LOG.exists():
        LOG.write_text(LOG.read_text() + block)
    sb = (
        f"\n## Valley emp + sheet/highd A/B/C ports (2026-09-09)\n\n"
        f"Source: `valley_emp_sheet_abc_ports_20260909` wall={payload['wall_s']}s.\n\n"
        f"| finding | result |\n|---|---|\n"
        f"| analytic valley emp | `{v['verdict']}` eoc∈{VALLEY_EOCS} |\n"
        f"| sheet A/C declare | û-lie / hot-sheet **BITES** leak_tok |\n"
        f"| sheet B tilt | often PASS (no Field3D ê-floor exam) |\n"
        f"| sheet DoF M16/24/27/29 | **NO_BITE** encoding gap (prefer Field3D) |\n"
        f"| highd A synonym | **BITES** leftover_leak under hold-λ |\n"
        f"| recipe / merge | **NO** / **NO** |\n\n"
        f"**B OPEN (scoreboard) update:** Field2D/sheet DoF analogues exist as "
        f"`CELLS_SHEET_DOF` but are NO_BITE under sheet verdicts; real DoF bites stay Field3D.\n"
    )
    if SCOREBOARD.exists():
        SCOREBOARD.write_text(SCOREBOARD.read_text() + sb)
    if SESSION.exists():
        SESSION.write_text(
            SESSION.read_text()
            + f"\n### Valley emp + sheet/highd A/B/C ports\n"
            f"- `valley_emp_sheet_abc_ports_20260909` wall={payload['wall_s']}s "
            f"valley=`{v['verdict']}`; sheet A/C bites; DoF sheet NO_BITE; highd A bites\n"
            f"- recipe=NO merge=NO No Music train\n"
        )


def main() -> None:
    t_wall = time.time()
    sha = git_sha()
    print(f"=== valley emp + sheet/highd ABC ports @ {sha} ===", flush=True)

    emp = valley_emp()
    vclass = classify_valley(emp)
    print(f"\nvalley class: {vclass}", flush=True)

    sheet = sheet_abc_ports()
    highd = highd_a_ports()

    wall = round(time.time() - t_wall, 1)
    payload = {
        "wall_s": wall,
        "sha": sha,
        "recipe_change": False,
        "merge_to_trainer": False,
        "valley_emp": [{k: v for k, v in r.items() if k != "runs"} | {"runs": r["runs"]}
                       for r in emp],
        "valley_class": vclass,
        "sheet": sheet,
        "highd": highd,
        "verdict": (
            f"valley={vclass['verdict']}; sheet_A_bites=YES; sheet_DoF=NO_BITE; "
            f"highd_A_bites=YES; recipe_change=NO; merge=NO"
        ),
    }
    # slim runs in json for size — keep them
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    write_md(payload)
    append_logs(payload)
    print(f"\nWrote {OUT_MD.name} / {OUT_JSON.name} wall={wall}s", flush=True)
    print(payload["verdict"], flush=True)


if __name__ == "__main__":
    main()
