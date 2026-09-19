#!/usr/bin/env python3
"""M14 e_on_u_declare_lie fail-mode dig vs M20 amp_lie — 2026-09-09.

Characterize: why exam≈0.18 with u_kept≈0.98?
Compare to M20 amp_lie_leftover_declare (content-axis YAML lie).
Clearance probes: n∈{1,2,12}, vic=0@n=1, per-row NON_DEFAULT.

CPU only. No Music GPU train. Locked AdvResidual defaults UNCHANGED.
Outputs: m14_failmode_20260909.{md,json}
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
    e_on_u_declare_lie_field3d,
    leftover_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

OUT_JSON = NOTES / "m14_failmode_20260909.json"
OUT_MD = NOTES / "m14_failmode_20260909.md"
LOG = NOTES / "research_log_20260909.md"

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
            steps=1200,
            seed=s,
            b_cap=1.0,
            cover_weight=cover,
            fm_weight=0.0,
            n_particles=n,
            particle_l2=0.02,
        )
        if vic is not None:
            cfg = replace(cfg, vicreg_weight=float(vic))
        t0 = time.time()
        try:
            field = field_fn(seed=s)
        except TypeError:
            field = field_fn()
        row = score_adv_field3d_exam(
            field, teacher=teacher, cfg=cfg, name=f"{name}_s{s}"
        )
        runs.append(
            {
                "seed": s,
                "pass": bool(row.get("exam_pass")),
                "exam_score": float(row["exam_score"]),
                "u_kept": float(row["u_kept"]),
                "content_kept": float(row["content_kept"]),
                "leak_ratio": float(row["leak_ratio"]),
                "collapse": float(row.get("collapse", 0)),
                "pair_odd_cos": float(row.get("pair_odd_cos", 0)),
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
            }
        )
    return pack(name, "shared", teacher, n, cover, vic, runs)


def run_per_row(name, field_fn, seeds, **score_kw):
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s}", flush=True)
        try:
            f = field_fn(seed=s)
        except TypeError:
            f = field_fn()
        out = dep._recompute_gates(
            ex.score_scaffold(f, mode="per_row", seed=s, name=f"{name}_s{s}", **score_kw)
        )
        runs.append(
            {
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
                "head_min_cos": out.get("head_min_cos"),
            }
        )
    return pack(name, "per_row", "faithful_guard_e", 12, 1.5, None, runs)


def pack(name, mode, teacher, n, cover, vic, runs):
    nn = len(runs)
    def mean(k):
        return round(sum(float(r[k]) for r in runs) / nn, 4) if nn else 0.0

    def frac(k):
        return round(sum(1 for r in runs if r.get(k)) / nn, 2) if nn else 0.0

    # dominant fail gates among fails
    fail_runs = [r for r in runs if not r["pass"]]
    gate_counts = {
        "cont": sum(1 for r in fail_runs if not r.get("pass_cont")),
        "swing": sum(1 for r in fail_runs if not r.get("pass_swing")),
        "leftover": sum(1 for r in fail_runs if not r.get("pass_leftover_gate")),
        "multi": sum(1 for r in fail_runs if not r.get("pass_multi_row")),
    }
    out = {
        "name": name,
        "mode": mode,
        "teacher": teacher,
        "n_particles": n,
        "cover_weight": cover,
        "vicreg_weight": vic,
        "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
        "n_pass": sum(1 for r in runs if r["pass"]),
        "n": nn,
        "mean_exam": mean("exam_score"),
        "mean_u": mean("u_kept"),
        "mean_content": mean("content_kept"),
        "mean_leak": mean("leak_ratio"),
        "mean_cont": mean("exam_cont"),
        "mean_swing": mean("exam_swing"),
        "frac_cont": frac("pass_cont"),
        "frac_swing": frac("pass_swing"),
        "frac_leftover": frac("pass_leftover_gate"),
        "frac_multi": frac("pass_multi_row"),
        "frac_u": frac("pass_u"),
        "frac_content": frac("pass_content"),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "fail_gate_counts": gate_counts,
        "runs": runs,
    }
    print(
        f"  {name}: {out['pass']} exam={out['mean_exam']} u={out['mean_u']} "
        f"cont={out['mean_cont']} swing={out['mean_swing']} "
        f"left={out['frac_leftover']} multi={out['frac_multi']} "
        f"gates={gate_counts} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def nature_of(cell: dict) -> str:
    """Human-readable fail-mode label from component means / gate counts."""
    if cell["n_pass"] == cell["n"]:
        return "PASS"
    gc = cell["fail_gate_counts"]
    # Prefer swing if exam≈swing and u high
    if cell["mean_u"] >= 0.9 and cell["mean_swing"] < 0.6 and gc.get("swing", 0) > 0:
        return "exam_swing_collapse_under_declare_lie"
    if gc.get("leftover", 0) >= len(cell["fail_seeds"]) and cell["mean_content"] < 0.75:
        return "content_deleted_under_declare_lie"
    if gc.get("multi", 0) >= len(cell["fail_seeds"]):
        return "multi_row_coverage"
    if gc.get("cont", 0) > 0 and cell["mean_cont"] < 0.85:
        return "continuation_undershoot"
    # pick max gate
    top = max(gc, key=gc.get) if gc else "unknown"
    return f"gate_{top}"


def main() -> None:
    t0 = time.time()
    sha = git_sha()
    print(f"=== M14 fail-mode dig @ {sha} ===", flush=True)
    cells = []

    print("\n[CTRL] leftover locked", flush=True)
    cells.append(
        run_shared("CTRL_leftover_n12", leftover_field3d, "faithful_guard_e", SEEDS_SMOKE)
    )

    print("\n[A] M14 vs M20 locked characterization (full seeds)", flush=True)
    cells.append(
        run_shared("M14_locked_n12", e_on_u_declare_lie_field3d, "faithful_guard_e", SEEDS_FULL)
    )
    cells.append(
        run_shared("M20_locked_n12", amp_lie_leftover_declare_field3d, "faithful_guard_e", SEEDS_FULL)
    )
    cells.append(
        run_shared("M14_faithful_n12", e_on_u_declare_lie_field3d, "faithful", SEEDS_SMOKE)
    )
    cells.append(
        run_shared("M20_faithful_n12", amp_lie_leftover_declare_field3d, "faithful", SEEDS_SMOKE)
    )

    print("\n[B] clearance probes: n / vic (shared)", flush=True)
    cells.append(
        run_shared(
            "M14_music_n1_c1.0",
            e_on_u_declare_lie_field3d,
            "faithful_guard_e",
            SEEDS_FULL,
            n=1,
            cover=1.0,
        )
    )
    cells.append(
        run_shared(
            "M14_n2_c1.5",
            e_on_u_declare_lie_field3d,
            "faithful_guard_e",
            SEEDS_FULL,
            n=2,
            cover=1.5,
        )
    )
    cells.append(
        run_shared(
            "M14_n1_vic0",
            e_on_u_declare_lie_field3d,
            "faithful_guard_e",
            SEEDS_FULL,
            n=1,
            cover=1.0,
            vic=0.0,
        )
    )
    cells.append(
        run_shared(
            "M20_n2_c1.5",
            amp_lie_leftover_declare_field3d,
            "faithful_guard_e",
            SEEDS_SMOKE,
            n=2,
            cover=1.5,
        )
    )
    cells.append(
        run_shared(
            "M20_n1_vic0",
            amp_lie_leftover_declare_field3d,
            "faithful_guard_e",
            SEEDS_SMOKE,
            n=1,
            cover=1.0,
            vic=0.0,
        )
    )

    print("\n[C] soft e_on_u ladder (M14 family)", flush=True)
    for eou in (0.0, 0.3, 0.75, 1.5):

        def fn(seed=0, e=eou):
            return e_on_u_declare_lie_field3d(
                e_on_u=e,
                e_on_content=0.1 if e > 0 else 0.0,
                e_unused=0.1 if e > 0 else 1.0,
            )

        cells.append(
            run_shared(f"M14_eou{eou}_n12", fn, "faithful_guard_e", SEEDS_SMOKE)
        )

    print("\n[D] per-row NON_DEFAULT (analysis-only)", flush=True)
    cells.append(run_per_row("CTRL_leftover_per_row", leftover_field3d, SEEDS_SMOKE))
    cells.append(run_per_row("M14_per_row", e_on_u_declare_lie_field3d, SEEDS_FULL))
    cells.append(run_per_row("M20_per_row", amp_lie_leftover_declare_field3d, SEEDS_FULL))

    wall = round(time.time() - t0, 1)

    by_name = {c["name"]: c for c in cells}
    m14 = by_name["M14_locked_n12"]
    m20 = by_name["M20_locked_n12"]
    m14_nature = nature_of(m14)
    m20_nature = nature_of(m20)

    clearance = {
        "n1_music": by_name["M14_music_n1_c1.0"]["n_pass"] == by_name["M14_music_n1_c1.0"]["n"],
        "n2": by_name["M14_n2_c1.5"]["n_pass"] == by_name["M14_n2_c1.5"]["n"],
        "vic0_n1": by_name["M14_n1_vic0"]["n_pass"] == by_name["M14_n1_vic0"]["n"],
        "per_row": by_name["M14_per_row"]["n_pass"] == by_name["M14_per_row"]["n"],
        "m20_n2": by_name["M20_n2_c1.5"]["n_pass"] == by_name["M20_n2_c1.5"]["n"],
        "m20_vic0_n1": by_name["M20_n1_vic0"]["n_pass"] == by_name["M20_n1_vic0"]["n"],
        "m20_per_row": by_name["M20_per_row"]["n_pass"] == by_name["M20_per_row"]["n"],
    }
    any_clear = any(clearance[k] for k in ("n1_music", "n2", "vic0_n1", "per_row"))

    # contrast
    contrast = {
        "m14_lie_axis": "û (e_on_u=1.5 restates concept)",
        "m20_lie_axis": "content (e_on_content=1.25 on leftover geom)",
        "m14_nature": m14_nature,
        "m20_nature": m20_nature,
        "m14_mean_exam": m14["mean_exam"],
        "m20_mean_exam": m20["mean_exam"],
        "m14_mean_swing": m14["mean_swing"],
        "m20_mean_swing": m20["mean_swing"],
        "m14_mean_leak": m14["mean_leak"],
        "m20_mean_leak": m20["mean_leak"],
        "same_family": True,  # both YAML declare-lie HARD_BITEs
        "same_fail_gate": m14_nature == m20_nature,
    }

    verdict = (
        f"M14 nature={m14_nature}; vs M20 nature={m20_nature}; "
        f"n/vic/per_row clear M14? {'YES' if any_clear else 'NO'} "
        f"(n1={clearance['n1_music']}, n2={clearance['n2']}, "
        f"vic0={clearance['vic0_n1']}, per_row={clearance['per_row']}); "
        f"M20 also uncleared by n/vic/per_row "
        f"(n2={clearance['m20_n2']}, vic0={clearance['m20_vic0_n1']}, "
        f"per_row={clearance['m20_per_row']}); keep both HARD_BITEs; recipe_change=NO"
    )

    payload = {
        "wall_s": wall,
        "sha": sha,
        "recipe_change": False,
        "merge_to_trainer": False,
        "m14_nature": m14_nature,
        "m20_nature": m20_nature,
        "clearance": clearance,
        "any_clear_m14": any_clear,
        "contrast": contrast,
        "verdict": verdict,
        "cells": [{k: v for k, v in c.items() if k != "runs"} | {"runs": c["runs"]} for c in cells],
    }
    # keep runs in JSON
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M14 fail-mode dig vs M20 amp_lie — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. No Music GPU train.",
        "Locked shared AdvResidual defaults **unchanged**. per-row = NON_DEFAULT analysis-only.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Nature",
        "",
        f"| cell | nature | PASS | exam | u | content | swing | leak | dominant fail gates |",
        f"|---|---|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("M14_locked_n12", "M20_locked_n12"):
        c = by_name[key]
        lines.append(
            f"| `{key}` | **{nature_of(c)}** | {c['pass']} | {c['mean_exam']} | "
            f"{c['mean_u']} | {c['mean_content']} | {c['mean_swing']} | {c['mean_leak']} | "
            f"{c['fail_gate_counts']} |"
        )
    lines += [
        "",
        "### Contrast M14 vs M20",
        "",
        f"- **M14** YAML lie: declared ê restates **û** (`e_on_u=1.5`).",
        f"- **M20** YAML lie: declared ê points at **content** (`e_on_content=1.25`) on leftover geom.",
        f"- Same family (declare-lie HARD_BITEs that falsify overpowered heads).",
        f"- Fail-gate match: **{contrast['same_fail_gate']}** "
        f"(M14={m14_nature}, M20={m20_nature}).",
        f"- M14 exam≈{m14['mean_exam']} with u≈{m14['mean_u']} → swing/cont mismatch "
        f"(swing≈{m14['mean_swing']}).",
        f"- M20 exam≈{m20['mean_exam']}, leak≈{m20['mean_leak']} "
        f"(content-axis lie often shows higher leak / different gate mix).",
        "",
        "## Clearance probes (can n / vic / per-row clear M14?)",
        "",
        "| probe | PASS | exam | swing | clear? |",
        "|---|:---:|---:|---:|:---:|",
    ]
    for key, label in (
        ("M14_music_n1_c1.0", "M14 n=1 music"),
        ("M14_n2_c1.5", "M14 n=2"),
        ("M14_n1_vic0", "M14 n=1 vic=0"),
        ("M14_per_row", "M14 per_row"),
        ("M20_n2_c1.5", "M20 n=2"),
        ("M20_n1_vic0", "M20 n=1 vic=0"),
        ("M20_per_row", "M20 per_row"),
    ):
        c = by_name[key]
        cleared = c["n_pass"] == c["n"]
        lines.append(
            f"| {label} | {c['pass']} | {c['mean_exam']} | {c['mean_swing']} | "
            f"{'YES' if cleared else 'no'} |"
        )

    lines += [
        "",
        f"**Answer:** n / vic0@n1 / per-row **{'do' if any_clear else 'do NOT'}** clear M14. "
        "Declare-lie needs YAML/target fix, not more particles / heads / vic off.",
        "",
        "## Full cell table",
        "",
        "| cell | mode | PASS | exam | u | cont | swing | left | multi | fail |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in cells:
        lines.append(
            f"| `{c['name']}` | {c['mode']} | {c['pass']} | {c['mean_exam']} | "
            f"{c['mean_u']} | {c['mean_cont']} | {c['mean_swing']} | "
            f"{c['frac_leftover']} | {c['frac_multi']} | {c['fail_seeds']} |"
        )
    lines += [
        "",
        "## Recipe / ADOPT",
        "",
        "- Recipe change: **NO**",
        "- merge_to_trainer: **NO**",
        "- Keep M14 + M20 as HARD_BITEs (overpowered-head falsifiers)",
        "- Music-posture ADOPTs (vic0@n≤1, n≥2 close) do **not** apply to declare-lie family",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # append research_log
    log_block = (
        f"\n## Fire — M14 fail-mode dig vs M20 (2026-09-09)\n\n"
        f"- Host: box-cpu @ `{sha}`\n"
        f"- Dig: `m14_failmode_20260909.{{py,json,md}}` wall={wall}s\n"
        f"- M14 nature: **{m14_nature}** (exam={m14['mean_exam']}, u={m14['mean_u']}, "
        f"swing={m14['mean_swing']}, gates={m14['fail_gate_counts']})\n"
        f"- M20 nature: **{m20_nature}** (exam={m20['mean_exam']}, leak={m20['mean_leak']}, "
        f"swing={m20['mean_swing']}, gates={m20['fail_gate_counts']})\n"
        f"- Clearance M14: n1={clearance['n1_music']} n2={clearance['n2']} "
        f"vic0={clearance['vic0_n1']} per_row={clearance['per_row']} → any_clear={any_clear}\n"
        f"- Clearance M20: n2={clearance['m20_n2']} vic0={clearance['m20_vic0_n1']} "
        f"per_row={clearance['m20_per_row']}\n"
        f"- Verdict: {verdict}\n"
        f"- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.\n"
    )
    with LOG.open("a") as f:
        f.write(log_block)

    print(json.dumps({"wall_s": wall, "verdict": verdict, "m14_nature": m14_nature}, indent=2))


if __name__ == "__main__":
    main()
