#!/usr/bin/env python3
"""M16 scale_stagger + M17 roles_split mechanistic dig — 2026-09-09.

Like M14 fail-mode dig:
  - Characterize fail nature (gates / exam / rows) vs M2 cross_axis + M27 scale_descent
  - Clearance: n∈{1,2,12}, vic=0@n=1, per-row w∈{0.0,0.3} NON_DEFAULT
  - Ladders: scale-span for M16; blend_mix cliff for M17 (refine 0.25→0.5)

CPU only. No Music GPU. Locked AdvResidual defaults UNCHANGED.
Outputs: m16_m17_mech_20260909.{md,json} + research_log append.
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
    Field3D,
    cross_axis_rows_field3d,
    leftover_field3d,
    roles_split_proxy_field3d,
    scale_descent_homo_field3d,
    scale_stagger_homo_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

OUT_JSON = NOTES / "m16_m17_mech_20260909.json"
OUT_MD = NOTES / "m16_m17_mech_20260909.md"
OUT_LOG = NOTES / "m16_m17_mech_20260909.log"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"

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
                "rows_total": int(row.get("rows_total", 0)),
                "wall_s": round(time.time() - t0, 2),
            }
        )
        print(
            f"    {name} seed={s} pass={runs[-1]['pass']} exam={runs[-1]['exam_score']:.4f} "
            f"rows={runs[-1]['rows_covered']}/{runs[-1]['rows_total']} "
            f"multi={runs[-1]['pass_multi_row']} leak={runs[-1]['leak_ratio']:.4f} "
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
                f,
                mode="per_row",
                seed=s,
                name=f"{name}_s{s}",
                coupling_weight=float(coupling_weight),
            )
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
                "rows_total": int(out.get("rows_total", 0) or 0),
                "bite_cleared": bool(out.get("bite_cleared")),
                "head_min_cos": out.get("head_min_cos"),
            }
        )
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
        "name": name,
        "mode": mode,
        "teacher": teacher,
        "n_particles": n,
        "cover_weight": cover,
        "vicreg_weight": vic,
        "coupling_weight": coupling,
        "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
        "n_pass": sum(1 for r in runs if r["pass"]),
        "n": nn,
        "mean_exam": mean("exam_score"),
        "mean_u": mean("u_kept"),
        "mean_content": mean("content_kept"),
        "mean_leak": mean("leak_ratio"),
        "mean_cont": mean("exam_cont"),
        "mean_swing": mean("exam_swing"),
        "mean_rows": mean("rows_covered"),
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
        f"rows≈{out['mean_rows']} left={out['frac_leftover']} multi={out['frac_multi']} "
        f"leak={out['mean_leak']} gates={gate_counts} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def nature_of(cell: dict) -> str:
    if cell["n_pass"] == cell["n"]:
        return "PASS"
    gc = cell["fail_gate_counts"]
    nfail = max(len(cell["fail_seeds"]), 1)
    # multi-row DoF (rows partial / zero) — dominant for scale/roles family
    if gc.get("multi", 0) >= nfail and cell["mean_leak"] < 0.15:
        if cell["mean_rows"] >= 2.5:
            return "multi_row_scale_dof_partial"
        if cell["mean_rows"] < 0.5:
            return "multi_row_axis_roles_zero"
        return "multi_row_coverage"
    if gc.get("multi", 0) >= nfail and cell["mean_leak"] >= 0.15:
        return "multi_row_plus_leak_cross_axis"
    if cell["mean_u"] >= 0.9 and cell["mean_content"] < 0.75 and gc.get("leftover", 0) >= nfail:
        return "content_deleted_under_declare_lie"
    if gc.get("leftover", 0) >= nfail and cell["mean_leak"] >= 0.18:
        return "leak_gate_eoc_floor"
    top = max(gc, key=gc.get) if gc else "unknown"
    return f"gate_{top}"


def make_scale_span(span: float) -> Field3D:
    """Homo amps; scales centered at 1.0 with ±span/2 range across 5 rows."""
    # span=0 → flat; span≈0.75 → default M16 (0.7..1.45)
    mid = 1.0
    half = span / 2.0
    scales = tuple(mid - half + i * (span / 4.0 if span > 0 else 0.0) for i in range(5))
    if span <= 0:
        scales = (1.0,) * 5
    return scale_stagger_homo_field3d(row_scales=scales)


def make_blend(mix: float) -> Field3D:
    """mix=0 → all u-primary; mix=1 → full roles_split amps (same as isolation dig)."""
    u = (1.1, 0.25, 0.1)
    c = (0.35, 1.3, 0.1)
    amps = []
    for i in range(4):
        if i < 2:
            amps.append(u)
        else:
            amps.append(tuple((1 - mix) * a + mix * b for a, b in zip(u, c)))
    return Field3D(
        kind=f"roles_blend_{mix}",
        rows=4,
        row_scales=(1.0,) * 4,
        row_amps=tuple(amps),
        slider=1.0,
        content=0.55,
        leak=0.2,
        e_on_u=0.0,
        e_on_content=0.0,
        e_unused=1.0,
    )


def cleared(cell: dict) -> bool:
    return cell["n_pass"] == cell["n"]


def main() -> None:
    t0 = time.time()
    sha = git_sha()
    print(f"=== M16/M17 mechanistic dig @ {sha} ===", flush=True)
    cells = []

    print("\n[CTRL] leftover locked", flush=True)
    cells.append(
        run_shared("CTRL_leftover_n12", leftover_field3d, "faithful_guard_e", SEEDS_SMOKE)
    )

    print("\n[A] locked characterization: M16 / M17 / M2 / M27", flush=True)
    cells.append(
        run_shared("M16_locked_n12", scale_stagger_homo_field3d, "faithful_guard_e", SEEDS_FULL)
    )
    cells.append(
        run_shared("M17_locked_n12", roles_split_proxy_field3d, "faithful_guard_e", SEEDS_FULL)
    )
    cells.append(
        run_shared("M2_cross_axis_n12", cross_axis_rows_field3d, "faithful_guard_e", SEEDS_FULL)
    )
    cells.append(
        run_shared("M27_scale_descent_n12", scale_descent_homo_field3d, "faithful_guard_e", SEEDS_FULL)
    )
    # faithful (no guard) smoke — is bite guard-gated?
    cells.append(
        run_shared("M16_faithful_n12", scale_stagger_homo_field3d, "faithful", SEEDS_SMOKE)
    )
    cells.append(
        run_shared("M17_faithful_n12", roles_split_proxy_field3d, "faithful", SEEDS_SMOKE)
    )

    print("\n[B] clearance: n / vic shared", flush=True)
    for label, fn, seeds in (
        ("M16", scale_stagger_homo_field3d, SEEDS_FULL),
        ("M17", roles_split_proxy_field3d, SEEDS_FULL),
    ):
        cells.append(
            run_shared(
                f"{label}_music_n1_c1.0",
                fn,
                "faithful_guard_e",
                seeds,
                n=1,
                cover=1.0,
            )
        )
        cells.append(
            run_shared(
                f"{label}_n2_c1.5",
                fn,
                "faithful_guard_e",
                seeds,
                n=2,
                cover=1.5,
            )
        )
        cells.append(
            run_shared(
                f"{label}_n1_vic0",
                fn,
                "faithful_guard_e",
                seeds,
                n=1,
                cover=1.0,
                vic=0.0,
            )
        )

    print("\n[C] M16 scale-span ladder (homo amps)", flush=True)
    for span in (0.0, 0.2, 0.4, 0.6, 0.75, 1.0):
        cells.append(
            run_shared(
                f"M16_span{span}_n12",
                lambda seed=None, sp=span: make_scale_span(sp),
                "faithful_guard_e",
                SEEDS_SMOKE,
            )
        )

    print("\n[D] M17 blend_mix cliff refine", flush=True)
    for mix in (0.0, 0.25, 0.35, 0.4, 0.5, 0.75, 1.0):
        cells.append(
            run_shared(
                f"M17_blend{mix}_n12",
                lambda seed=None, m=mix: make_blend(m),
                "faithful_guard_e",
                SEEDS_SMOKE,
            )
        )

    print("\n[E] per-row NON_DEFAULT (w=0 and w=0.3)", flush=True)
    cells.append(run_per_row("CTRL_leftover_per_row", leftover_field3d, SEEDS_SMOKE, coupling_weight=0.0))
    for label, fn in (
        ("M16", scale_stagger_homo_field3d),
        ("M17", roles_split_proxy_field3d),
        ("M2", cross_axis_rows_field3d),
        ("M27", scale_descent_homo_field3d),
    ):
        cells.append(run_per_row(f"{label}_per_row_w0", fn, SEEDS_FULL, coupling_weight=0.0))
        cells.append(
            run_per_row(f"{label}_per_row_w0.3", fn, SEEDS_SMOKE, coupling_weight=0.3)
        )

    wall = round(time.time() - t0, 1)
    by_name = {c["name"]: c for c in cells}

    natures = {
        "M16": nature_of(by_name["M16_locked_n12"]),
        "M17": nature_of(by_name["M17_locked_n12"]),
        "M2": nature_of(by_name["M2_cross_axis_n12"]),
        "M27": nature_of(by_name["M27_scale_descent_n12"]),
    }

    clearance = {}
    for mid in ("M16", "M17"):
        clearance[mid] = {
            "n1_music": cleared(by_name[f"{mid}_music_n1_c1.0"]),
            "n2": cleared(by_name[f"{mid}_n2_c1.5"]),
            "vic0_n1": cleared(by_name[f"{mid}_n1_vic0"]),
            "per_row_w0": cleared(by_name[f"{mid}_per_row_w0"]),
            "per_row_w0.3": cleared(by_name[f"{mid}_per_row_w0.3"]),
        }
    clearance["M2_per_row_w0"] = cleared(by_name["M2_per_row_w0"])
    clearance["M27_per_row_w0"] = cleared(by_name["M27_per_row_w0"])

    # scale cliff: first span that fails all smoke seeds
    scale_cliff = None
    for span in (0.0, 0.2, 0.4, 0.6, 0.75, 1.0):
        c = by_name[f"M16_span{span}_n12"]
        if c["n_pass"] == 0:
            scale_cliff = span
            break
    # blend cliff: last PASS mix / first FAIL mix
    blend_last_pass = None
    blend_first_fail = None
    for mix in (0.0, 0.25, 0.35, 0.4, 0.5, 0.75, 1.0):
        c = by_name[f"M17_blend{mix}_n12"]
        if cleared(c):
            blend_last_pass = mix
        elif blend_first_fail is None:
            blend_first_fail = mix

    m16_any = any(clearance["M16"].values())
    m17_any = any(clearance["M17"].values())

    verdict = (
        f"M16 nature={natures['M16']}; M17 nature={natures['M17']}; "
        f"M2={natures['M2']}; M27={natures['M27']}; "
        f"M16 cleared by n/vic/per_row? {'YES' if m16_any else 'NO'} {clearance['M16']}; "
        f"M17 cleared? {'YES' if m17_any else 'NO'} {clearance['M17']}; "
        f"scale_cliff_span={scale_cliff}; blend_cliff lastPASS={blend_last_pass} "
        f"firstFAIL={blend_first_fail}; "
        f"M27_per_row={clearance['M27_per_row_w0']}; M2_per_row={clearance['M2_per_row_w0']}; "
        f"recipe_change=NO"
    )

    payload = {
        "wall_s": wall,
        "sha": sha,
        "recipe_change": False,
        "merge_to_trainer": False,
        "natures": natures,
        "clearance": clearance,
        "scale_cliff_span": scale_cliff,
        "blend_last_pass": blend_last_pass,
        "blend_first_fail": blend_first_fail,
        "verdict": verdict,
        "cells": cells,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M16/M17 mechanistic dig — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. No Music GPU train.",
        "Locked shared AdvResidual defaults **unchanged**. per-row = NON_DEFAULT analysis-only (w≤0.3).",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Nature (locked n12)",
        "",
        "| cell | nature | PASS | exam | u | rows | leak | fail gates |",
        "|---|---|:---:|---:|---:|---:|---:|---|",
    ]
    for key, mid in (
        ("M16_locked_n12", "M16"),
        ("M17_locked_n12", "M17"),
        ("M2_cross_axis_n12", "M2"),
        ("M27_scale_descent_n12", "M27"),
    ):
        c = by_name[key]
        lines.append(
            f"| `{key}` | **{natures[mid]}** | {c['pass']} | {c['mean_exam']} | "
            f"{c['mean_u']} | {c['mean_rows']} | {c['mean_leak']} | {c['fail_gate_counts']} |"
        )

    lines += [
        "",
        "### Family read",
        "",
        f"- **M16** = homo amps + ascending scale stagger — expected **scale DoF / multi_row** "
        f"(rows≈{by_name['M16_locked_n12']['mean_rows']}); mirror of M27 descending.",
        f"- **M17** = role-split û vs content primary — soft→hard into M2; rows≈"
        f"{by_name['M17_locked_n12']['mean_rows']}.",
        f"- **M2** hard cross-axis: rows≈{by_name['M2_cross_axis_n12']['mean_rows']}, "
        f"leak≈{by_name['M2_cross_axis_n12']['mean_leak']}.",
        f"- **M27** descending scales: nature={natures['M27']}.",
        f"- faithful (no guard): M16={by_name['M16_faithful_n12']['pass']}, "
        f"M17={by_name['M17_faithful_n12']['pass']}.",
        "",
        "## Clearance probes",
        "",
        "| probe | PASS | exam | rows | multi | clear? |",
        "|---|:---:|---:|---:|:---:|:---:|",
    ]
    for mid in ("M16", "M17"):
        for key, label in (
            (f"{mid}_music_n1_c1.0", f"{mid} n=1 music"),
            (f"{mid}_n2_c1.5", f"{mid} n=2"),
            (f"{mid}_n1_vic0", f"{mid} n=1 vic=0"),
            (f"{mid}_per_row_w0", f"{mid} per_row w=0"),
            (f"{mid}_per_row_w0.3", f"{mid} per_row w=0.3"),
        ):
            c = by_name[key]
            lines.append(
                f"| {label} | {c['pass']} | {c['mean_exam']} | {c['mean_rows']} | "
                f"{c['frac_multi']} | {'YES' if cleared(c) else 'no'} |"
            )
    for key, label in (
        ("M2_per_row_w0", "M2 per_row w=0"),
        ("M2_per_row_w0.3", "M2 per_row w=0.3"),
        ("M27_per_row_w0", "M27 per_row w=0"),
        ("M27_per_row_w0.3", "M27 per_row w=0.3"),
        ("CTRL_leftover_per_row", "CTRL leftover per_row"),
    ):
        c = by_name[key]
        lines.append(
            f"| {label} | {c['pass']} | {c['mean_exam']} | {c['mean_rows']} | "
            f"{c['frac_multi']} | {'YES' if cleared(c) else 'no'} |"
        )

    lines += [
        "",
        f"**M16 any clear:** {'YES' if m16_any else 'NO'} — {clearance['M16']}",
        f"**M17 any clear:** {'YES' if m17_any else 'NO'} — {clearance['M17']}",
        "",
        "## Ladders",
        "",
        f"- **M16 scale-span cliff:** first all-fail span = **{scale_cliff}** "
        f"(default M16 span≈0.75).",
        f"- **M17 blend cliff:** last PASS mix=**{blend_last_pass}**, "
        f"first FAIL mix=**{blend_first_fail}**.",
        "",
        "| ladder cell | PASS | exam | rows | multi | leak |",
        "|---|:---:|---:|---:|:---:|---:|",
    ]
    for span in (0.0, 0.2, 0.4, 0.6, 0.75, 1.0):
        c = by_name[f"M16_span{span}_n12"]
        lines.append(
            f"| `span={span}` | {c['pass']} | {c['mean_exam']} | {c['mean_rows']} | "
            f"{c['frac_multi']} | {c['mean_leak']} |"
        )
    for mix in (0.0, 0.25, 0.35, 0.4, 0.5, 0.75, 1.0):
        c = by_name[f"M17_blend{mix}_n12"]
        lines.append(
            f"| `blend={mix}` | {c['pass']} | {c['mean_exam']} | {c['mean_rows']} | "
            f"{c['frac_multi']} | {c['mean_leak']} |"
        )

    lines += [
        "",
        "## Full cell table",
        "",
        "| cell | mode | PASS | exam | u | rows | left | multi | leak | fail |",
        "|---|---|:---:|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for c in cells:
        lines.append(
            f"| `{c['name']}` | {c['mode']} | {c['pass']} | {c['mean_exam']} | "
            f"{c['mean_u']} | {c['mean_rows']} | {c['frac_leftover']} | "
            f"{c['frac_multi']} | {c['mean_leak']} | {c['fail_seeds']} |"
        )

    lines += [
        "",
        "## Recipe / ADOPT",
        "",
        "- Recipe change: **NO**",
        "- merge_to_trainer: **NO**",
        "- Keep M16 + M17 as HARD_BITEs under locked shared",
        "- If per-row clears M16/M27: confirms **scale DoF** family (analysis-only w≤0.3 ADOPT)",
        "- If per-row clears M17/M2: confirms **axis-mix DoF** family (same ADOPT; not declare-lie)",
        "- Music-posture n/vic ADOPTs do **not** replace documenting these hard bites",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    log_block = (
        f"\n## Fire — M16/M17 mechanistic dig (2026-09-09)\n\n"
        f"- Host: box-cpu @ `{sha}`\n"
        f"- Dig: `m16_m17_mech_20260909.{{py,json,md}}` wall={wall}s\n"
        f"- Natures: M16={natures['M16']}; M17={natures['M17']}; "
        f"M2={natures['M2']}; M27={natures['M27']}\n"
        f"- Clearance M16: {clearance['M16']} → any={m16_any}\n"
        f"- Clearance M17: {clearance['M17']} → any={m17_any}\n"
        f"- Scale cliff span={scale_cliff}; blend lastPASS={blend_last_pass} "
        f"firstFAIL={blend_first_fail}\n"
        f"- M27_per_row={clearance['M27_per_row_w0']}; M2_per_row={clearance['M2_per_row_w0']}\n"
        f"- Verdict: {verdict}\n"
        f"- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.\n"
    )
    with LOG.open("a") as f:
        f.write(log_block)

    # Fold into scoreboard
    fold = (
        f"\n\n## Dig: M16/M17 mechanistic (2026-09-09)\n\n"
        f"Source: `m16_m17_mech_20260909.{{md,json}}` wall={wall}s.\n\n"
        f"| finding | result |\n"
        f"|---|---|\n"
        f"| M16 nature | **{natures['M16']}** |\n"
        f"| M17 nature | **{natures['M17']}** |\n"
        f"| M2 / M27 | {natures['M2']} / {natures['M27']} |\n"
        f"| M16 clearance | {clearance['M16']} |\n"
        f"| M17 clearance | {clearance['M17']} |\n"
        f"| scale-span cliff | first all-fail = **{scale_cliff}** |\n"
        f"| blend cliff | lastPASS=**{blend_last_pass}** → firstFAIL=**{blend_first_fail}** |\n"
        f"| Keep | both HARD_BITEs; recipe_change=NO |\n"
    )
    with SCOREBOARD.open("a") as f:
        f.write(fold)

    print(json.dumps({"wall_s": wall, "verdict": verdict, "natures": natures}, indent=2))


if __name__ == "__main__":
    main()
