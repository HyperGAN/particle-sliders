#!/usr/bin/env python3
"""Fire #11 (2D->3D): Field3D content<->leftover entanglement / exam-port analogues.

Locked: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles=12, particle_l2=0.02, b_cap=1.

Ask: does locked leftover gating on Field3D stay seed-stable when content vs
leftover amplitudes vary (exam-port analogues), or does content<->e-hat
entanglement create false locks / leak failures?
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import leftover_field3d, score_adv_field3d
from analysis.slider2d.gan import default_cfg

OUT = Path(__file__).resolve().parent / "field3d_exam_entangle_20260909.json"
MD = Path(__file__).resolve().parent / "field3d_exam_entangle_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"

# Geometry families (slider=1.0, lyric=1.0 unless noted).
# content_zero: content gate intentionally N/A (a_c~0 -> pass_content True).
# leak_zero: declared_e -> None (no e-hat to gate).
FAMILIES = {
    "baseline": {"content": 0.55, "leak": 0.45},
    "close_like": {"content": 0.90, "leak": 0.10},
    "unused_e_like": {"content": 0.10, "leak": 0.90},
    "entangled": {"content": 0.70, "leak": 0.70},
    "content_zero": {"content": 0.0, "leak": 0.45},
    "leak_zero": {"content": 0.55, "leak": 0.0},
}

CONTENT_FAMILIES = [k for k, v in FAMILIES.items() if float(v["content"]) > 0.0]


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


def main() -> None:
    t0 = time.time()
    print("Field3D exam-entangle @ locked 1200+c1.5", flush=True)
    print("families=%s seeds=%s" % (list(FAMILIES), SEEDS), flush=True)

    by_family = {}
    all_rows = []
    for fam, kwargs in FAMILIES.items():
        rows = []
        print(
            "--- family=%s content=%.2f leak=%.2f"
            % (fam, kwargs["content"], kwargs["leak"]),
            flush=True,
        )
        for seed in SEEDS:
            st = time.time()
            field = leftover_field3d(**kwargs)
            row = score_adv_field3d(
                field,
                teacher=TEACHER,
                cfg=locked_cfg(seed),
                name="field3d_%s_s%d" % (fam, seed),
            )
            row["family"] = fam
            row["wall_s"] = round(time.time() - st, 2)
            row["geom_content"] = float(kwargs["content"])
            row["geom_leak"] = float(kwargs["leak"])
            rows.append(row)
            all_rows.append(row)
            print(
                "  seed=%2d pass=%s u=%.4f c=%.4f leak=%.4f resid=%.3f collapse=%.3f (%.1fs)"
                % (
                    seed,
                    row["pass"],
                    row["u_kept"],
                    row["content_kept"],
                    row["leak_ratio"],
                    row["residual_norm"],
                    row["collapse"],
                    row["wall_s"],
                ),
                flush=True,
            )
        n_pass = sum(1 for r in rows if r["pass"])
        u_vals = [r["u_kept"] for r in rows]
        c_vals = [
            r["content_kept"]
            for r in rows
            if abs(r.get("a_content", 0.0)) > 1e-8
        ]
        leak_vals = [r["leak_ratio"] for r in rows]
        by_family[fam] = {
            "content": float(kwargs["content"]),
            "leak": float(kwargs["leak"]),
            "n_pass": n_pass,
            "n_total": len(SEEDS),
            "u_kept_mean": sum(u_vals) / len(u_vals),
            "u_kept_span": max(u_vals) - min(u_vals),
            "content_kept_mean": (sum(c_vals) / len(c_vals)) if c_vals else None,
            "content_kept_span": (max(c_vals) - min(c_vals)) if c_vals else None,
            "leak_ratio_mean": sum(leak_vals) / len(leak_vals),
            "leak_ratio_max": max(leak_vals),
            "fail_u": sum(1 for r in rows if not r["pass_u"]),
            "fail_content": sum(1 for r in rows if not r["pass_content"]),
            "fail_leak": sum(1 for r in rows if not r["pass_leak"]),
            "rows": rows,
        }

    exam_proxy = []
    for seed in SEEDS:
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        u_content = [
            r["u_kept"] for r in seed_rows if r["family"] in CONTENT_FAMILIES
        ]
        exam_proxy.append(
            {
                "seed": seed,
                "u_kept_mean_content_fams": (
                    sum(u_content) / len(u_content) if u_content else None
                ),
                "leak_ratio_max_all_fams": max(r["leak_ratio"] for r in seed_rows),
                "n_pass_all_fams": sum(1 for r in seed_rows if r["pass"]),
                "n_total_fams": len(seed_rows),
            }
        )

    solid_ok = True
    fragile_fams = []
    false_lock_fams = []
    for fam, summ in by_family.items():
        n_pass = summ["n_pass"]
        n_tot = summ["n_total"]
        if n_pass == n_tot:
            continue
        if n_pass == 0:
            fragile_fams.append(fam)
            solid_ok = False
        else:
            false_lock_fams.append(fam)
            solid_ok = False

    if solid_ok and not fragile_fams and not false_lock_fams:
        knife = False
        for fam, summ in by_family.items():
            if summ["u_kept_span"] > 0.05:
                knife = True
            if summ["leak_ratio_max"] > 0.15:
                knife = True
        if knife:
            verdict = "field3d_entangle_fragile"
            next_v = (
                "diagnose amplitude ratio near knife-edge; which family spans/leaks"
            )
            finding = (
                "All families 6/6 but knife-edge (span or leak near gate) "
                "— treat as fragile."
            )
        else:
            verdict = "field3d_exam_entangle_solid"
            next_v = (
                "port a true PairField-style exam cell into R3 or multi-row Field3D"
            )
            finding = (
                "Locked leftover gating stays seed-stable across exam-port "
                "amplitude mixes (baseline/close/unused_e/entangled + "
                "content_zero/leak_zero). No false locks."
            )
    elif false_lock_fams:
        verdict = "field3d_false_lock"
        next_v = "diagnose which amplitude ratio / seed breaks leftover gating"
        finding = (
            "Partial seed pass on families %s — false lock risk."
            % (false_lock_fams,)
        )
    else:
        verdict = "field3d_entangle_fragile"
        next_v = "diagnose which amplitude ratio breaks leftover gating"
        finding = "Seed-stable failures on families %s." % (fragile_fams,)

    wall = round(time.time() - t0, 1)
    summary = {
        "recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "families": {k: dict(v) for k, v in FAMILIES.items()},
        "seeds": SEEDS,
        "by_family": {
            k: {kk: vv for kk, vv in v.items() if kk != "rows"}
            for k, v in by_family.items()
        },
        "by_family_rows": {k: v["rows"] for k, v in by_family.items()},
        "exam_proxy": exam_proxy,
        "verdict": verdict,
        "next": next_v,
        "finding": finding,
        "fragile_fams": fragile_fams,
        "false_lock_fams": false_lock_fams,
        "wall_s": wall,
        "notes": {
            "content_zero": "content gate skipped when a_c~0 (pass_content=True)",
            "leak_zero": "declared_e returns None when leak~0; leak_dir=None path",
            "exam_proxy": (
                "per-seed mean u_kept over content>0 families; "
                "max leak_ratio across all families"
            ),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2, default=float) + "\n")

    lines = [
        "# Field3D exam-entangle — 2026-09-09 (Fire #11 / 2D→3D)",
        "",
        "Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,",
        "fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.",
        "",
        "Ask: does leftover gating stay seed-stable when content vs leftover",
        "amplitudes vary (exam-port analogues), or does content↔ê entanglement",
        "create false locks / leak failures?",
        "",
        "Seeds: %s." % (SEEDS,),
        "",
        "## Per-family summary",
        "",
        "| family | content | leak | PASS | u_kept mean | content_kept mean | leak_ratio max | fail u/c/leak |",
        "|:---|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for fam in FAMILIES:
        s = by_family[fam]
        ck = (
            "n/a"
            if s["content_kept_mean"] is None
            else "%.4f" % s["content_kept_mean"]
        )
        lines.append(
            "| %s | %.2f | %.2f | %d/%d | %.4f | %s | %.4f | %d/%d/%d |"
            % (
                fam,
                s["content"],
                s["leak"],
                s["n_pass"],
                s["n_total"],
                s["u_kept_mean"],
                ck,
                s["leak_ratio_max"],
                s["fail_u"],
                s["fail_content"],
                s["fail_leak"],
            )
        )

    lines += ["", "## Detail tables", ""]
    for fam in FAMILIES:
        lines += [
            "### %s (content=%.2f leak=%.2f)"
            % (fam, FAMILIES[fam]["content"], FAMILIES[fam]["leak"]),
            "",
            "| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |",
            "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in by_family[fam]["rows"]:
            lines.append(
                "| %d | %s | %.4f | %.4f | %.4f | %.3f | %.3f | %.4f | %.1f |"
                % (
                    r["seed"],
                    "PASS" if r["pass"] else "FAIL",
                    r["u_kept"],
                    r["content_kept"],
                    r["leak_ratio"],
                    r["residual_norm"],
                    r["collapse"],
                    r["pair_odd_cos"],
                    r["wall_s"],
                )
            )
        lines.append("")

    lines += [
        "## exam_proxy (per seed)",
        "",
        "Mean u_kept over families with content>0; max leak_ratio across all families.",
        "Not a transferred scoreboard gate — diagnostic only.",
        "",
        "| seed | u_kept_mean (content fams) | leak_ratio max | PASS fams |",
        "|---:|---:|---:|:---:|",
    ]
    for p in exam_proxy:
        lines.append(
            "| %d | %.4f | %.4f | %d/%d |"
            % (
                p["seed"],
                p["u_kept_mean_content_fams"],
                p["leak_ratio_max_all_fams"],
                p["n_pass_all_fams"],
                p["n_total_fams"],
            )
        )

    lines += [
        "",
        "### Finding",
        "",
        "- %s" % finding,
        "- content_zero: content gate intentionally skipped (a_c≈0).",
        "- leak_zero: declared_e → None; leak_dir=None path.",
        "- Verdict: `%s`" % verdict,
        "- Next: %s" % next_v,
        "- Wall: %.1fs" % wall,
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")

    fam_lines = []
    for fam in FAMILIES:
        s = by_family[fam]
        ck = (
            "n/a"
            if s["content_kept_mean"] is None
            else "%.4f" % s["content_kept_mean"]
        )
        fam_lines.append(
            "  - `%s` (c=%.2f e=%.2f): **%d/%d** u_kept mean=%.4f "
            "content_kept=%s leak_max=%.4f"
            % (
                fam,
                s["content"],
                s["leak"],
                s["n_pass"],
                s["n_total"],
                s["u_kept_mean"],
                ck,
                s["leak_ratio_max"],
            )
        )

    fire_section = "\n".join(
        [
            "",
            "## Fire #11 — Field3D content↔leftover entanglement / exam-port (2026-09-09)",
            "",
            "- Host: pop-os-cpu @ SHA `435e873`",
            "- Tests: `pytest tests/test_lm_2d_adv.py -q` → **10 passed in 46.63s**; "
            "`tests/test_lm_highd_leftover.py` → **38 passed in 62.96s`",
            "- Families × seeds {0,1,2,3,7,42} @ locked 1200+c1.5 / faithful_guard_e:",
        ]
        + fam_lines
        + [
            "- Verdict: **%s** — %s" % (verdict, finding),
            "- Next: %s" % next_v,
            "- Notes: `field3d_exam_entangle_20260909.{py,json,md}`",
            "",
        ]
    )

    for log_path in (LOG, LOG_ROOT):
        if log_path.exists():
            prev = log_path.read_text()
            if "## Fire #11" not in prev:
                log_path.write_text(prev.rstrip() + "\n" + fire_section)
            else:
                new_prev = re.sub(
                    r"\n## Fire #11.*?(?=\n## Fire |\Z)",
                    fire_section,
                    prev,
                    count=1,
                    flags=re.S,
                )
                if new_prev == prev:
                    log_path.write_text(prev.rstrip() + "\n" + fire_section)
                else:
                    log_path.write_text(new_prev)
        else:
            log_path.write_text("# research log\n" + fire_section)

    print("DONE verdict=%s wall=%.1fs -> %s" % (verdict, wall, OUT), flush=True)
    for fam, s in by_family.items():
        print(
            "  %s: %d/%d u=%.4f leak_max=%.4f"
            % (fam, s["n_pass"], s["n_total"], s["u_kept_mean"], s["leak_ratio_max"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
