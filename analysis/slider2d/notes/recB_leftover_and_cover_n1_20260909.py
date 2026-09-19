#!/usr/bin/env python3
"""Fire #16: Recommendation B — leftover ∧ cover under n_particles=1.

Music #94 true target (locked strategy B):
  faithful_guard_e + mlp + pole_weight/cover=1.0 + FM0 + b_cap1 + parts0

Arm T (lyric+tx) is diagnostic only — cover_only alone leaks (0/3 single-arm
vs 3/3 locked in parallel ablation).

This harness stresses what transfers without ParticlePrior:
  - teacher = faithful_guard_e (leftover gate ON) × cover_weight sweep @ n=1
  - negative control: teacher = faithful (NO gate) × cover=1.5 @ n=1  → expect leak
  - b_cap held at 1; FM=0; seeds≥6

cover_weight is the toy stand-in for Music modest --pole_weight (lyrichold
is UNI/tx-arm; not this target).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (
    leftover_field3d,
    unused_e_field3d,
    divergent_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg, score_adv_sheet, score_adv_exam
from analysis.slider2d.sheet import leaky_field
from analysis.slider2d.exam import unused_e_field, divergent_field

OUT = Path(__file__).resolve().parent / "recB_leftover_and_cover_n1_20260909.json"
MD = Path(__file__).resolve().parent / "recB_leftover_and_cover_n1_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
N_PARTICLES = 1  # Music-transferable (--parts 0; no ParticlePrior)
B_CAP = 1.0
FM = 0.0

# Joint leftover∧cover grid (Rec B)
COVERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

CELLS = {
    "sheet_leftover": ("sheet", leaky_field),
    "f3d_leftover": ("f3d", leftover_field3d),
    "f3d_unused_e": ("f3d_exam", unused_e_field3d),
    "f3d_divergent": ("f3d_exam", divergent_field3d),
    "exam_unused_e": ("exam", unused_e_field),
}

# Negative: cover without leftover gate
NEG_TEACHER = "faithful"
NEG_COVER = 1.5


def cfg(seed, cover, teacher_unused=None):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=B_CAP,
        cover_weight=float(cover),
        fm_weight=FM,
        n_particles=N_PARTICLES,
        particle_l2=0.02,
    )


def run_one(kind, ctor, seed, cover, teacher):
    c = cfg(seed, cover)
    st = time.time()
    if kind == "sheet":
        row = score_adv_sheet(ctor(), teacher=teacher, cfg=c)
        primary = float(row.get("on_sheet_kept") or 0.0)
        leak = float(row.get("leak_tok") or 0.0)
        ok = primary >= 0.90 and abs(leak) <= 0.20
    elif kind == "exam":
        row = score_adv_exam(ctor(), teacher=teacher, cfg=c)
        primary = float(row.get("roll_overlap") or 0.0)
        leak = float(row.get("leak_tok") or 0.0)
        ok = primary >= 0.85 and float(row.get("roll_swing_kept") or 0.0) >= 0.60
    elif kind == "f3d_exam":
        row = score_adv_field3d_exam(ctor(seed=seed), teacher=teacher, cfg=c)
        primary = float(row.get("exam_score") or 0.0)
        leak = float(row.get("leak_ratio") or 0.0)
        ok = bool(row.get("exam_pass"))
    else:
        row = score_adv_field3d(ctor(seed=seed), teacher=teacher, cfg=c)
        primary = float(row.get("u_kept") or 0.0)
        leak = float(row.get("leak_ratio") or 0.0)
        ok = bool(row.get("pass"))
    return {
        "pass": bool(ok),
        "primary": primary,
        "leak": leak,
        "wall_s": round(time.time() - st, 2),
        "seed": seed,
    }


def summarize(rows):
    prim = [r["primary"] for r in rows]
    leaks = [abs(r["leak"]) for r in rows]
    n_pass = sum(1 for r in rows if r["pass"])
    return {
        "n_pass": n_pass,
        "n_total": len(rows),
        "pass_rate": "%d/%d" % (n_pass, len(rows)),
        "primary_mean": sum(prim) / len(prim),
        "primary_span": max(prim) - min(prim),
        "leak_abs_max": max(leaks),
        "leak_abs_mean": sum(leaks) / len(leaks),
        "knife_edge": bool(n_pass not in (0, len(rows)) or (max(prim) - min(prim)) > 0.05),
        "full_pass": n_pass == len(rows),
    }


def main():
    t0 = time.time()
    print(
        "Fire #16 RecB leftover∧cover n=%d covers=%s seeds=%s"
        % (N_PARTICLES, COVERS, SEEDS),
        flush=True,
    )
    print(
        "Music target: faithful_guard_e + mlp + modest pole_weight + FM0 + b_cap1 + parts0",
        flush=True,
    )
    grid = []

    # Priority: locked cover=1.5 first, then 0 (single-arm leak risk), then rest
    cover_order = [1.0, 1.5, 0.0, 0.5, 2.0, 3.0]  # Music-locked 1.0 first
    for cover in cover_order:
        teacher = "faithful_guard_e"
        for cell, (kind, ctor) in CELLS.items():
            tag = "gate+cover%.1f/%s" % (cover, cell)
            print("--- %s" % tag, flush=True)
            rows = []
            for seed in SEEDS:
                row = run_one(kind, ctor, seed, cover, teacher)
                rows.append(row)
                print(
                    "  seed=%2d pass=%s prim=%.4f leak=%.4f (%.1fs)"
                    % (seed, row["pass"], row["primary"], row["leak"], row["wall_s"]),
                    flush=True,
                )
            summ = summarize(rows)
            grid.append(
                {
                    "mode": "leftover_and_cover",
                    "teacher": teacher,
                    "cover_weight": cover,
                    "n_particles": N_PARTICLES,
                    "b_cap": B_CAP,
                    "cell": cell,
                    "summary": summ,
                    "detail": rows,
                }
            )
            print(
                "  >> %s mean=%.4f leak_max=%.4f knife=%s"
                % (summ["pass_rate"], summ["primary_mean"], summ["leak_abs_max"], summ["knife_edge"]),
                flush=True,
            )

    # Negative control: cover_only (no leftover gate) on sheet + f3d leftover
    print("==== NEGATIVE cover_only (teacher=faithful) ====", flush=True)
    for cell, (kind, ctor) in {
        "sheet_leftover": CELLS["sheet_leftover"],
        "f3d_leftover": CELLS["f3d_leftover"],
        "f3d_unused_e": CELLS["f3d_unused_e"],
    }.items():
        tag = "cover_only/%.1f/%s" % (NEG_COVER, cell)
        print("--- %s" % tag, flush=True)
        rows = []
        for seed in SEEDS:
            row = run_one(kind, ctor, seed, NEG_COVER, NEG_TEACHER)
            rows.append(row)
            print(
                "  seed=%2d pass=%s prim=%.4f leak=%.4f (%.1fs)"
                % (seed, row["pass"], row["primary"], row["leak"], row["wall_s"]),
                flush=True,
            )
        summ = summarize(rows)
        grid.append(
            {
                "mode": "cover_only_neg",
                "teacher": NEG_TEACHER,
                "cover_weight": NEG_COVER,
                "n_particles": N_PARTICLES,
                "b_cap": B_CAP,
                "cell": cell,
                "summary": summ,
                "detail": rows,
            }
        )
        print(
            "  >> %s mean=%.4f leak_max=%.4f knife=%s"
            % (summ["pass_rate"], summ["primary_mean"], summ["leak_abs_max"], summ["knife_edge"]),
            flush=True,
        )

    joint = [g for g in grid if g["mode"] == "leftover_and_cover"]
    locked = [g for g in joint if g["cover_weight"] == 1.0]
    cover0 = [g for g in joint if g["cover_weight"] == 0.0]
    neg = [g for g in grid if g["mode"] == "cover_only_neg"]

    locked_ok = all(g["summary"]["full_pass"] and not g["summary"]["knife_edge"] for g in locked)
    cover0_leak = any(g["summary"]["leak_abs_max"] > 0.20 or not g["summary"]["full_pass"] for g in cover0)
    neg_leaks = all(
        (g["summary"]["leak_abs_max"] > 0.20) or (not g["summary"]["full_pass"]) for g in neg
    )

    # Best cover among joint with full pass
    by_cover = {}
    for g in joint:
        by_cover.setdefault(g["cover_weight"], []).append(g)
    cover_solid = [
        c
        for c, gs in by_cover.items()
        if all(x["summary"]["full_pass"] and not x["summary"]["knife_edge"] for x in gs)
    ]

    if locked_ok and neg_leaks:
        verdict = "recB_confirmed_gate_and_cover"
        finding = (
            "Under n_particles=1: leftover∧cover@1.0 solid; cover_only (faithful) leaks/fails as predicted. "
            "Solid covers=%s. Music #94 = guard_e+mlp+modest pole_weight+FM0+b_cap1+parts0."
            % sorted(cover_solid)
        )
    elif locked_ok:
        verdict = "recB_locked_ok_neg_mixed"
        finding = (
            "leftover∧cover@1.0 @ n=1 solid; negative cover_only not uniformly leaking — inspect. "
            "Solid covers=%s" % sorted(cover_solid)
        )
    else:
        verdict = "recB_needs_cover_or_gate_tune"
        finding = "leftover∧cover@1.0 @ n=1 not fully solid — see grid before Music smoke."

    music_hypothesis = (
        "Music smoke (hypothesis): --lm_target faithful_guard_e --adv_arch mlp "
        "--adv_weight 1 --fm_weight 0 --adv_reg_coeff 1 --adv_reg_kappa 1 --parts 0 "
        "--pole_weight 1.0 (toy cover=1.0). Arm T lyric+tx diagnostic only."
    )

    blob = {
        "fire": 16,
        "strategy": "Recommendation_B",
        "date": "2026-09-09",
        "sha": "435e873",
        "n_particles": N_PARTICLES,
        "b_cap": B_CAP,
        "fm_weight": FM,
        "seeds": SEEDS,
        "covers": COVERS,
        "grid": [
            {
                **{k: v for k, v in g.items() if k != "detail"},
                "detail": [
                    {
                        "seed": r["seed"],
                        "pass": r["pass"],
                        "primary": r["primary"],
                        "leak": r["leak"],
                        "wall_s": r["wall_s"],
                    }
                    for r in g["detail"]
                ],
            }
            for g in grid
        ],
        "locked_ok": locked_ok,
        "cover0_fail_or_leak": cover0_leak,
        "neg_cover_only_leaks": neg_leaks,
        "solid_covers": sorted(cover_solid),
        "verdict": verdict,
        "finding": finding,
        "music_hypothesis": music_hypothesis,
        "wall_s": round(time.time() - t0, 1),
    }

    lines = [
        "# Rec B: leftover ∧ cover @ n_particles=1 — 2026-09-09 (Fire #16)",
        "",
        "Music #94 true target: `faithful_guard_e` + mlp + modest pole_weight + FM0 + b_cap1 + parts0.",
        "Arm T (lyric+tx) diagnostic only. cover_only alone leaks.",
        "",
        "Toy: n_particles=1, b_cap=1, FM=0, teacher=faithful_guard_e × cover∈%s; neg teacher=faithful."
        % (COVERS,),
        "",
        "| mode | teacher | cover | cell | PASS | prim mean | leak max | knife |",
        "|:---|:---|---:|:---|:---:|---:|---:|:---:|",
    ]
    for g in grid:
        s = g["summary"]
        lines.append(
            "| %s | %s | %.1f | %s | %s | %.4f | %.4f | %s |"
            % (
                g["mode"],
                g["teacher"],
                g["cover_weight"],
                g["cell"],
                s["pass_rate"],
                s["primary_mean"],
                s["leak_abs_max"],
                "YES" if s["knife_edge"] else "no",
            )
        )
    lines += [
        "",
        "### Finding",
        "",
        "- %s" % finding,
        "- Verdict: `%s`" % verdict,
        "- Music hypothesis: %s" % music_hypothesis,
        "- Wall: %.1fs" % blob["wall_s"],
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    block = (
        "\n## Fire #16 — Rec B leftover∧cover @ n=1 (2026-09-09)\n\n"
        "- Strategy: Recommendation B (true #94 = guard_e+mlp+pole_weight+FM0+b_cap1+parts0)\n"
        "- n_particles=1 throughout; cover sweep + cover_only negative\n"
        "- Verdict: **%s** — %s\n"
        "- Solid covers: %s\n"
        "- Music hypothesis: %s\n"
        "- Notes: `recB_leftover_and_cover_n1_20260909.{py,json,md}`\n"
        % (verdict, finding, sorted(cover_solid), music_hypothesis)
    )
    status = (
        "\n### STATUS (~Fire #16)\n"
        "- Rec B Field3D/2D: leftover∧cover under n=1; Arm T not the transfer target.\n"
        "- Current: `%s`\n" % verdict
    )
    for path in (LOG, LOG_ROOT):
        if path.exists():
            path.write_text(path.read_text() + block + status)
    print("VERDICT=%s wall=%.1fs" % (verdict, blob["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
