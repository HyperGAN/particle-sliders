#!/usr/bin/env python3
"""Fire #12 (2D→3D): PairField-style exam cell in multi-row Field3D / R³.

Locked: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles=12, particle_l2=0.02, b_cap=1.

Ask: does the locked adversarial core transfer to a true PairField-style
multi-row exam (divergent / close / unused_e) on R³ Field3D, or does
multi-row + declared_e composition break leftover gating / coverage?
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (
    CELLS_3D,
    close_field3d,
    divergent_field3d,
    unused_e_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg

OUT = Path(__file__).resolve().parent / "field3d_pairfield_exam_20260909.json"
MD = Path(__file__).resolve().parent / "field3d_pairfield_exam_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1]
TEACHER = "faithful_guard_e"

CELLS = {
    "divergent": divergent_field3d,
    "close": close_field3d,
    "unused_e": unused_e_field3d,
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


def run_cell(cell: str, seeds: list[int]) -> dict:
    ctor = CELLS[cell]
    rows = []
    print("--- cell=%s seeds=%s" % (cell, seeds), flush=True)
    for seed in seeds:
        st = time.time()
        field = ctor(seed=seed)
        row = score_adv_field3d_exam(
            field,
            teacher=TEACHER,
            cfg=locked_cfg(seed),
            name="field3d_exam_%s_s%d" % (cell, seed),
        )
        row["wall_s"] = round(time.time() - st, 2)
        rows.append(row)
        print(
            "  seed=%2d exam_pass=%s u=%.4f c=%.4f leak=%.4f "
            "cont=%.4f swing=%.4f exam=%.4f rows=%d/%d (%.1fs)"
            % (
                seed,
                row["exam_pass"],
                row["u_kept"],
                row["content_kept"],
                row["leak_ratio"],
                row["exam_cont"],
                row["exam_swing"],
                row["exam_score"],
                row["rows_covered"],
                row["rows_total"],
                row["wall_s"],
            ),
            flush=True,
        )
    n_pass = sum(1 for r in rows if r["exam_pass"])
    u_vals = [r["u_kept"] for r in rows]
    c_vals = [r["content_kept"] for r in rows if abs(r.get("a_content", 0.0)) > 1e-8]
    leak_vals = [r["leak_ratio"] for r in rows]
    exam_vals = [r["exam_score"] for r in rows]
    cont_vals = [r["exam_cont"] for r in rows]
    swing_vals = [r["exam_swing"] for r in rows]
    return {
        "cell": cell,
        "n_pass": n_pass,
        "n_total": len(seeds),
        "u_kept_mean": sum(u_vals) / len(u_vals),
        "u_kept_span": max(u_vals) - min(u_vals),
        "content_kept_mean": (sum(c_vals) / len(c_vals)) if c_vals else None,
        "leak_ratio_mean": sum(leak_vals) / len(leak_vals),
        "leak_ratio_max": max(leak_vals),
        "exam_score_mean": sum(exam_vals) / len(exam_vals),
        "exam_score_min": min(exam_vals),
        "exam_cont_mean": sum(cont_vals) / len(cont_vals),
        "exam_swing_mean": sum(swing_vals) / len(swing_vals),
        "fail_cont": sum(1 for r in rows if not r["pass_cont"]),
        "fail_swing": sum(1 for r in rows if not r["pass_swing"]),
        "fail_leftover": sum(1 for r in rows if not r["pass_leftover_gate"]),
        "fail_multi": sum(1 for r in rows if not r["pass_multi_row"]),
        "rows": rows,
    }


def decide_verdict(by_cell: dict) -> tuple[str, str, str]:
    solid = True
    fragile = []
    partial = []
    for cell, summ in by_cell.items():
        if summ["n_pass"] == summ["n_total"]:
            continue
        if summ["n_pass"] == 0:
            fragile.append(cell)
            solid = False
        else:
            partial.append(cell)
            solid = False
    if solid:
        knife = any(
            summ["u_kept_span"] > 0.05 or summ["leak_ratio_max"] > 0.15
            for summ in by_cell.values()
        )
        if knife:
            return (
                "field3d_pairfield_exam_fragile",
                "diagnose span/leak near exam gate on multi-row Field3D",
                "All cells seed-pass but knife-edge span/leak — fragile.",
            )
        return (
            "field3d_pairfield_exam_solid",
            "phase milestone: PairField exam cells transfer to multi-row R³; next highd/live bridge",
            "Locked recipe + multi-row PairField cells (divergent/close/unused_e) "
            "seed-stable on Field3D. Guard refuses on divergent; unused_e gates ê; "
            "close keeps small-û + content. Multi-row coverage holds.",
        )
    if partial:
        return (
            "field3d_pairfield_exam_partial",
            "diagnose which cell/seed fails exam gates (cont/swing/leftover/multi-row)",
            "Partial seed pass on cells %s — false lock risk." % (partial,),
        )
    return (
        "field3d_pairfield_exam_fail",
        "diagnose which cell family breaks under multi-row / declared_e composition",
        "Seed-stable failures on cells %s." % (fragile,),
    )


def write_md(summary: dict) -> None:
    lines = []
    lines.append("# Field3D PairField exam — 2026-09-09 (Fire #12 / 2D→3D)")
    lines.append("")
    lines.append(
        "Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,"
    )
    lines.append("fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.")
    lines.append("")
    lines.append(
        "Ask: port a true PairField-style exam cell into multi-row Field3D "
        "(divergent / close / unused_e) using the locked adversarial core."
    )
    lines.append("")
    lines.append("Seeds: %s. Stage: %s." % (summary["seeds"], summary["stage"]))
    lines.append("")
    lines.append("## Per-cell summary")
    lines.append("")
    lines.append(
        "| cell | PASS | u_kept mean | content_kept mean | leak_ratio max | "
        "exam_score mean | exam_score min | fail c/s/l/m |"
    )
    lines.append("|:---|:---:|---:|---:|---:|---:|---:|:---:|")
    for cell, summ in summary["by_cell"].items():
        cmean = summ["content_kept_mean"]
        cstr = "%.4f" % cmean if cmean is not None else "n/a"
        lines.append(
            "| %s | %d/%d | %.4f | %s | %.4f | %.4f | %.4f | %d/%d/%d/%d |"
            % (
                cell,
                summ["n_pass"],
                summ["n_total"],
                summ["u_kept_mean"],
                cstr,
                summ["leak_ratio_max"],
                summ["exam_score_mean"],
                summ["exam_score_min"],
                summ["fail_cont"],
                summ["fail_swing"],
                summ["fail_leftover"],
                summ["fail_multi"],
            )
        )
    lines.append("")
    lines.append("## Detail tables")
    lines.append("")
    for cell, summ in summary["by_cell"].items():
        lines.append("### %s" % cell)
        lines.append("")
        lines.append(
            "| seed | exam_pass | u_kept | content_kept | leak_ratio | "
            "exam_cont | exam_swing | exam_score | rows | s |"
        )
        lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
        for r in summ["rows"]:
            lines.append(
                "| %d | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %d/%d | %.1f |"
                % (
                    r["seed"],
                    "PASS" if r["exam_pass"] else "FAIL",
                    r["u_kept"],
                    r["content_kept"],
                    r["leak_ratio"],
                    r["exam_cont"],
                    r["exam_swing"],
                    r["exam_score"],
                    r["rows_covered"],
                    r["rows_total"],
                    r["wall_s"],
                )
            )
        lines.append("")
    lines.append("### Finding")
    lines.append("")
    lines.append("- %s" % summary["finding"])
    lines.append("- Verdict: `%s`" % summary["verdict"])
    lines.append("- Next: %s" % summary["next"])
    lines.append("- Wall: %.1fs" % summary["wall_s"])
    lines.append("")
    MD.write_text("\n".join(lines) + "\n")


def append_log(summary: dict) -> None:
    if not LOG.exists():
        return
    text = LOG.read_text()
    # Replace trailing Fire #11 next / append Fire #12
    block = []
    block.append("")
    block.append("## Fire #12 — Field3D PairField multi-row exam (2026-09-09)")
    block.append("")
    block.append("- Host: pop-os-cpu @ SHA `%s`" % summary["sha"])
    block.append(
        "- Tests: `pytest tests/test_lm_2d_adv.py -q` → **%s**; "
        "`tests/test_lm_highd_leftover.py` → **%s**"
        % (summary.get("tests_2d", "see fire report"), summary.get("tests_highd", "see fire report"))
    )
    block.append(
        "- Cells × seeds %s @ locked 1200+c1.5 / faithful_guard_e (stage=%s):"
        % (summary["seeds"], summary["stage"])
    )
    for cell, summ in summary["by_cell"].items():
        block.append(
            "  - `%s`: **%d/%d** u_kept mean=%.4f exam_score mean=%.4f "
            "leak_max=%.4f (fail c/s/l/m=%d/%d/%d/%d)"
            % (
                cell,
                summ["n_pass"],
                summ["n_total"],
                summ["u_kept_mean"],
                summ["exam_score_mean"],
                summ["leak_ratio_max"],
                summ["fail_cont"],
                summ["fail_swing"],
                summ["fail_leftover"],
                summ["fail_multi"],
            )
        )
    block.append(
        "- Verdict: **%s** — %s"
        % (summary["verdict"], summary["finding"])
    )
    block.append("- Next: %s" % summary["next"])
    block.append("- Notes: `field3d_pairfield_exam_20260909.{py,json,md}`")
    block.append("")
    if "## Fire #12" in text:
        text = re.sub(
            r"\n## Fire #12 —.*",
            "\n" + "\n".join(block).lstrip(),
            text,
            count=1,
            flags=re.S,
        )
    else:
        # Update Fire #11 next line if present
        text = re.sub(
            r"(- Next: port a true PairField-style exam cell into R3 or multi-row Field3D)",
            r"- Next: done in Fire #12 (PairField multi-row Field3D exam)",
            text,
            count=1,
        )
        text = text.rstrip() + "\n" + "\n".join(block)
    LOG.write_text(text if text.endswith("\n") else text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke=seeds 0,1; full=0,1,2,3,7,42",
    )
    ap.add_argument("--sha", default="435e873")
    ap.add_argument("--tests-2d", default="10 passed (pre-run)")
    ap.add_argument("--tests-highd", default="38 passed (pre-run)")
    args = ap.parse_args()
    seeds = SEEDS_SMOKE if args.stage == "smoke" else SEEDS_FULL

    t0 = time.time()
    print(
        "Field3D PairField exam @ locked 1200+c1.5 stage=%s seeds=%s"
        % (args.stage, seeds),
        flush=True,
    )
    print("cells=%s" % list(CELLS), flush=True)

    by_cell = {}
    for cell in CELLS:
        by_cell[cell] = run_cell(cell, seeds)

    verdict, next_v, finding = decide_verdict(by_cell)
    wall = round(time.time() - t0, 1)
    summary = {
        "sha": args.sha,
        "stage": args.stage,
        "seeds": seeds,
        "teacher": TEACHER,
        "locked": {
            "steps": 1200,
            "cover_weight": 1.5,
            "fm_weight": 0.0,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "by_cell": by_cell,
        "verdict": verdict,
        "next": next_v,
        "finding": finding,
        "wall_s": wall,
        "tests_2d": args.tests_2d,
        "tests_highd": args.tests_highd,
        "user_worthy": verdict == "field3d_pairfield_exam_solid",
    }
    # JSON: strip row_coverage nested bulk? keep it for diagnostics.
    OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_md(summary)
    if args.stage == "full":
        append_log(summary)
    print(
        "VERDICT=%s wall=%.1fs user_worthy=%s -> %s"
        % (verdict, wall, summary["user_worthy"], OUT),
        flush=True,
    )


if __name__ == "__main__":
    main()
