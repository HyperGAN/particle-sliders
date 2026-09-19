#!/usr/bin/env python3
"""Fire #13: HARDER multi-pair / cross-axis stress (transfer focus).

Locked: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles=12, particle_l2=0.02, b_cap=1. Do NOT adopt 800×cover3.0.

Ask: does the locked recipe survive (a) more pairs than Fire #8/#12,
(b) mismatched leftover↔content amplitudes / declared_e composition,
(c) wider row_scales + 5 rows, across ≥6 seeds — or do we find knife-edge
false locks?

Cells mix Sheet/PairField 2D + Field3D R³ so transfer claims are not
geometry-local.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.exam import close_field, divergent_field, unused_e_field
from analysis.slider2d.field3d import (
    Field3D,
    close_field3d,
    divergent_field3d,
    leftover_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
    unused_e_field3d,
)
from analysis.slider2d.gan import default_cfg, score_adv_exam, score_adv_sheet
from analysis.slider2d.sheet import gender_like_field, leaky_field

OUT = Path(__file__).resolve().parent / "field3d_hard_multipair_20260909.json"
MD = Path(__file__).resolve().parent / "field3d_hard_multipair_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"

# Harder Field3D geometries: extreme mismatch + wide multi-row.
def mismatch_content_heavy(**kw):
    """Content dominates a; declared ê wrongly points at û (teacher must refuse)."""
    base = dict(
        kind="mismatch_content_heavy",
        rows=5,
        row_scales=(0.70, 0.85, 1.0, 1.15, 1.30),
        slider=0.80,
        content=2.40,
        leak=0.05,
        e_on_u=1.50,
        e_on_content=0.20,
        e_unused=0.0,
    )
    base.update(kw)
    return Field3D(**base)


def mismatch_leak_heavy(**kw):
    """Unused ê dominates a; content tiny; declared ê correctly names unused."""
    base = dict(
        kind="mismatch_leak_heavy",
        rows=5,
        row_scales=(0.70, 0.85, 1.0, 1.15, 1.30),
        slider=1.00,
        content=0.08,
        leak=1.20,
        e_on_u=0.0,
        e_on_content=0.0,
        e_unused=1.0,
    )
    base.update(kw)
    return Field3D(**base)


def mismatch_cross_declare(**kw):
    """a has unused ê, but declared_e points at content (guard should refuse subtract)."""
    base = dict(
        kind="mismatch_cross_declare",
        rows=4,
        row_scales=(0.80, 0.95, 1.05, 1.20),
        slider=1.00,
        content=0.55,
        leak=0.70,
        e_on_u=0.10,
        e_on_content=1.80,
        e_unused=0.05,
    )
    base.update(kw)
    return Field3D(**base)


def entangled_wide_rows(**kw):
    """Equal content+leak amplitudes, 5 mismatched row scales."""
    base = dict(
        kind="entangled_wide_rows",
        rows=5,
        row_scales=(0.55, 0.80, 1.0, 1.25, 1.50),
        slider=1.00,
        content=0.85,
        leak=0.85,
        e_on_u=0.0,
        e_on_content=0.0,
        e_unused=1.0,
    )
    base.update(kw)
    return Field3D(**base)


CELLS = {
    # 2D baselines (transfer control)
    "sheet_leftover": ("sheet", leaky_field),
    "sheet_gender": ("sheet", gender_like_field),
    "exam_divergent": ("exam", divergent_field),
    "exam_close": ("exam", close_field),
    "exam_unused_e": ("exam", unused_e_field),
    # Field3D PairField ports
    "f3d_divergent": ("f3d_exam", divergent_field3d),
    "f3d_close": ("f3d_exam", close_field3d),
    "f3d_unused_e": ("f3d_exam", unused_e_field3d),
    # Harder mismatch suite
    "f3d_mismatch_content_heavy": ("f3d_exam", mismatch_content_heavy),
    "f3d_mismatch_leak_heavy": ("f3d_exam", mismatch_leak_heavy),
    "f3d_mismatch_cross_declare": ("f3d_exam", mismatch_cross_declare),
    "f3d_entangled_wide_rows": ("f3d_exam", entangled_wide_rows),
    # Single-row leftover sanity
    "f3d_leftover": ("f3d", leftover_field3d),
}


def locked_cfg(seed: int):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        fm_weight=0.0,
        n_particles=12,
        particle_l2=0.02,
    )


def _sheet_pass(row: dict) -> bool:
    return bool(
        float(row.get("on_sheet_kept") or 0.0) >= 0.90
        and abs(float(row.get("leak_tok", 1))) <= 0.20
    )


def _exam_pass(row: dict) -> bool:
    return bool(
        float(row.get("roll_overlap") or 0.0) >= 0.85
        and float(row.get("roll_swing_kept") or 0.0) >= 0.60
    )


def run_one(kind: str, ctor, seed: int) -> dict:
    cfg = locked_cfg(seed)
    st = time.time()
    if kind == "sheet":
        field = ctor()
        row = score_adv_sheet(field, teacher=TEACHER, cfg=cfg)
        row["pass"] = _sheet_pass(row)
        row["metric_primary"] = float(row.get("on_sheet_kept") or 0.0)
        row["metric_leak"] = float(row.get("leak_tok") or 0.0)
    elif kind == "exam":
        field = ctor()
        row = score_adv_exam(field, teacher=TEACHER, cfg=cfg)
        row["pass"] = _exam_pass(row)
        row["metric_primary"] = float(row.get("roll_overlap") or 0.0)
        row["metric_leak"] = float(row.get("leak_tok") or 0.0)
    elif kind == "f3d_exam":
        field = ctor(seed=seed)
        row = score_adv_field3d_exam(field, teacher=TEACHER, cfg=cfg)
        row["metric_primary"] = float(row.get("exam_score") or 0.0)
        row["metric_leak"] = float(row.get("leak_ratio") or 0.0)
        row["pass"] = bool(row.get("exam_pass"))
    else:  # f3d leftover smoke
        field = ctor(seed=seed)
        row = score_adv_field3d(field, teacher=TEACHER, cfg=cfg)
        row["metric_primary"] = float(row.get("u_kept") or 0.0)
        row["metric_leak"] = float(row.get("leak_ratio") or 0.0)
        row["pass"] = bool(row.get("pass"))
    row["wall_s"] = round(time.time() - st, 2)
    row["seed"] = seed
    return row


def summarize(rows: list[dict]) -> dict:
    n_pass = sum(1 for r in rows if r["pass"])
    prim = [r["metric_primary"] for r in rows]
    leaks = [abs(r["metric_leak"]) for r in rows]
    return {
        "n_pass": n_pass,
        "n_total": len(rows),
        "pass_rate": "%d/%d" % (n_pass, len(rows)),
        "primary_mean": sum(prim) / len(prim),
        "primary_min": min(prim),
        "primary_max": max(prim),
        "primary_span": max(prim) - min(prim),
        "leak_abs_max": max(leaks),
        "knife_edge": bool(n_pass not in (0, len(rows)) or (max(prim) - min(prim)) > 0.08),
    }


def write_md(blob: dict) -> None:
    lines = [
        "# Field3D HARD multi-pair / cross-axis — 2026-09-09 (Fire #13)",
        "",
        "Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,",
        "fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.",
        "",
        "Ask: more pairs + mismatched leftover/content + wide rows + ≥6 seeds.",
        "",
        "Seeds: %s" % (blob["seeds"],),
        "",
        "## Per-cell summary",
        "",
        "| cell | PASS | primary mean | primary span | leak abs max | knife_edge |",
        "|:---|:---:|---:|---:|---:|:---:|",
    ]
    for name, summ in blob["by_cell"].items():
        lines.append(
            "| %s | %s | %.4f | %.4f | %.4f | %s |"
            % (
                name,
                summ["pass_rate"],
                summ["primary_mean"],
                summ["primary_span"],
                summ["leak_abs_max"],
                "YES" if summ["knife_edge"] else "no",
            )
        )
    lines += [
        "",
        "### Finding",
        "",
        "- %s" % blob["finding"],
        "- Verdict: `%s`" % blob["verdict"],
        "- Wall: %.1fs" % blob["wall_s"],
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")


def append_log(blob: dict) -> None:
    block = (
        "\n## Fire #13 — HARD multi-pair / cross-axis (2026-09-09)\n\n"
        "- Host: pop-os-cpu @ SHA `435e873`\n"
        "- Tests: pytest 2d_adv+highd already green this sprint (48 passed)\n"
        "- Cells: %d × seeds %s @ locked 1200+c1.5 / faithful_guard_e\n"
        % (len(blob["by_cell"]), blob["seeds"])
    )
    for name, summ in blob["by_cell"].items():
        block += (
            "  - `%s`: **%s** primary mean=%.4f span=%.4f leak_max=%.4f knife=%s\n"
            % (
                name,
                summ["pass_rate"],
                summ["primary_mean"],
                summ["primary_span"],
                summ["leak_abs_max"],
                summ["knife_edge"],
            )
        )
    block += (
        "- Verdict: **%s** — %s\n"
        "- Notes: `field3d_hard_multipair_20260909.{py,json,md}`\n"
        % (blob["verdict"], blob["finding"])
    )
    status = (
        "\n### STATUS (~Fire #13)\n"
        "- Best transferable: locked recipe survives harder mismatch suite "
        "iff all cells 6/6 and no knife_edge; else document the failing cell.\n"
        "- Current verdict: `%s`\n"
        % blob["verdict"]
    )
    for path in (LOG, LOG_ROOT):
        if path.exists():
            path.write_text(path.read_text() + block + status)


def main() -> None:
    t0 = time.time()
    print("Fire #13 HARD multipair cells=%s seeds=%s" % (list(CELLS), SEEDS), flush=True)
    by_cell = {}
    detail = {}
    n_knife = 0
    all_full_pass = True
    for name, (kind, ctor) in CELLS.items():
        print("--- cell=%s kind=%s" % (name, kind), flush=True)
        rows = []
        for seed in SEEDS:
            row = run_one(kind, ctor, seed)
            rows.append(row)
            print(
                "  seed=%2d pass=%s primary=%.4f leak=%.4f (%.1fs)"
                % (
                    seed,
                    row["pass"],
                    row["metric_primary"],
                    row["metric_leak"],
                    row["wall_s"],
                ),
                flush=True,
            )
        summ = summarize(rows)
        by_cell[name] = summ
        # Slim detail for JSON
        detail[name] = [
            {
                "seed": r["seed"],
                "pass": r["pass"],
                "metric_primary": r["metric_primary"],
                "metric_leak": r["metric_leak"],
                "wall_s": r["wall_s"],
                "u_kept": r.get("u_kept"),
                "content_kept": r.get("content_kept"),
                "leak_ratio": r.get("leak_ratio"),
                "exam_score": r.get("exam_score"),
                "on_sheet_kept": r.get("on_sheet_kept"),
                "roll_overlap": r.get("roll_overlap"),
                "rows_covered": r.get("rows_covered"),
                "rows_total": r.get("rows_total"),
            }
            for r in rows
        ]
        if summ["knife_edge"]:
            n_knife += 1
        if summ["n_pass"] != summ["n_total"]:
            all_full_pass = False
        print(
            "  >> %s primary_mean=%.4f span=%.4f knife=%s"
            % (summ["pass_rate"], summ["primary_mean"], summ["primary_span"], summ["knife_edge"]),
            flush=True,
        )

    if all_full_pass and n_knife == 0:
        verdict = "hard_multipair_solid"
        finding = (
            "Locked recipe seed-stable across 2D sheet/exam + Field3D PairField "
            "+ harder mismatch/wide-row cells. No knife-edge false locks."
        )
    elif all_full_pass:
        verdict = "hard_multipair_pass_with_span_warn"
        finding = (
            "All cells full PASS but %d cell(s) flagged knife_edge (span>0.08 or mixed). "
            "Inspect spans before claiming Music transfer."
            % n_knife
        )
    else:
        fails = [n for n, s in by_cell.items() if s["n_pass"] != s["n_total"]]
        verdict = "hard_multipair_partial"
        finding = (
            "Locked recipe fails on: %s. Document as transfer risk; do not "
            "weaken recipe without multi-seed margin."
            % ", ".join(fails)
        )

    blob = {
        "fire": 13,
        "date": "2026-09-09",
        "sha": "435e873",
        "recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "seeds": SEEDS,
        "by_cell": by_cell,
        "detail": detail,
        "n_knife_cells": n_knife,
        "verdict": verdict,
        "finding": finding,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    write_md(blob)
    append_log(blob)
    print("VERDICT=%s wall=%.1fs -> %s" % (verdict, blob["wall_s"], OUT), flush=True)


if __name__ == "__main__":
    main()
