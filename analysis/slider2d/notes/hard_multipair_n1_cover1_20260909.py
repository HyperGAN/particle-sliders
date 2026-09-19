#!/usr/bin/env python3
"""Fire #17: hard multi-pair / Field3D under Music-transferable recipe.

Locked Music-transfer posture (Rec B, cover updated):
  n_particles=1, cover_weight=1.0, teacher=faithful_guard_e, FM=0, b_cap=1
  (maps to: guard_e + mlp + pole_weight=1 + FM0 + b_cap1 + parts0)

Ask: do PairField + leftover cells stay ≥6-seed solid at cover=1.0 / n=1?
Do Fire #13 mismatch cells still solid-fail (YAML hygiene)?
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

OUT = Path(__file__).resolve().parent / "hard_multipair_n1_cover1_20260909.json"
MD = Path(__file__).resolve().parent / "hard_multipair_n1_cover1_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"
N_PARTICLES = 1
COVER = 1.0
B_CAP = 1.0


def mismatch_cross_declare(**kw):
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


CELLS = {
    "sheet_leftover": ("sheet", leaky_field),
    "sheet_gender": ("sheet", gender_like_field),
    "exam_divergent": ("exam", divergent_field),
    "exam_close": ("exam", close_field),
    "exam_unused_e": ("exam", unused_e_field),
    "f3d_divergent": ("f3d_exam", divergent_field3d),
    "f3d_close": ("f3d_exam", close_field3d),
    "f3d_unused_e": ("f3d_exam", unused_e_field3d),
    "f3d_leftover": ("f3d", leftover_field3d),
    "f3d_mismatch_cross_declare": ("f3d_exam", mismatch_cross_declare),
}


def locked_cfg(seed):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=B_CAP,
        cover_weight=COVER,
        fm_weight=0.0,
        n_particles=N_PARTICLES,
        particle_l2=0.02,
    )


def run_one(kind, ctor, seed):
    cfg = locked_cfg(seed)
    st = time.time()
    if kind == "sheet":
        row = score_adv_sheet(ctor(), teacher=TEACHER, cfg=cfg)
        primary = float(row.get("on_sheet_kept") or 0.0)
        leak = float(row.get("leak_tok") or 0.0)
        ok = primary >= 0.90 and abs(leak) <= 0.20
    elif kind == "exam":
        row = score_adv_exam(ctor(), teacher=TEACHER, cfg=cfg)
        primary = float(row.get("roll_overlap") or 0.0)
        leak = float(row.get("leak_tok") or 0.0)
        ok = primary >= 0.85 and float(row.get("roll_swing_kept") or 0.0) >= 0.60
    elif kind == "f3d_exam":
        row = score_adv_field3d_exam(ctor(seed=seed), teacher=TEACHER, cfg=cfg)
        primary = float(row.get("exam_score") or 0.0)
        leak = float(row.get("leak_ratio") or 0.0)
        ok = bool(row.get("exam_pass"))
    else:
        row = score_adv_field3d(ctor(seed=seed), teacher=TEACHER, cfg=cfg)
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
        "knife_edge": bool(n_pass not in (0, len(rows)) or (max(prim) - min(prim)) > 0.05),
        "full_pass": n_pass == len(rows),
    }


def main():
    t0 = time.time()
    print(
        "Fire #17 hard multipair n=%d cover=%.1f b_cap=%.1f FM=0 seeds=%s"
        % (N_PARTICLES, COVER, B_CAP, SEEDS),
        flush=True,
    )
    by_cell = {}
    detail = {}
    for name, (kind, ctor) in CELLS.items():
        print("--- %s" % name, flush=True)
        rows = []
        for seed in SEEDS:
            row = run_one(kind, ctor, seed)
            rows.append(row)
            print(
                "  seed=%2d pass=%s prim=%.4f leak=%.4f (%.1fs)"
                % (seed, row["pass"], row["primary"], row["leak"], row["wall_s"]),
                flush=True,
            )
        summ = summarize(rows)
        by_cell[name] = summ
        detail[name] = rows
        print(
            "  >> %s mean=%.4f span=%.4f knife=%s"
            % (summ["pass_rate"], summ["primary_mean"], summ["primary_span"], summ["knife_edge"]),
            flush=True,
        )

    expect_pass = [k for k in by_cell if "mismatch" not in k]
    expect_fail = [k for k in by_cell if "mismatch" in k]
    pass_ok = all(by_cell[k]["full_pass"] and not by_cell[k]["knife_edge"] for k in expect_pass)
    fail_ok = all(by_cell[k]["n_pass"] == 0 for k in expect_fail)

    if pass_ok and fail_ok:
        verdict = "n1_cover1_multipair_solid"
        finding = (
            "Music-transfer posture n=1 cover=1.0: all PairField/sheet/leftover cells 6/6; "
            "mismatch_cross_declare solid-FAIL (YAML hygiene). Recipe transfers."
        )
    elif pass_ok:
        verdict = "n1_cover1_pass_cells_ok"
        finding = "Transfer cells solid; mismatch behavior unexpected — inspect."
    else:
        fails = [k for k in expect_pass if not by_cell[k]["full_pass"]]
        verdict = "n1_cover1_partial"
        finding = "Fails under cover=1.0/n=1: %s — may need cover=1.5 on those cells." % fails

    blob = {
        "fire": 17,
        "date": "2026-09-09",
        "recipe": {
            "n_particles": N_PARTICLES,
            "cover_weight": COVER,
            "b_cap": B_CAP,
            "fm_weight": 0.0,
            "teacher": TEACHER,
            "music_map": "faithful_guard_e+mlp+pole_weight=1+FM0+b_cap1+parts0",
        },
        "seeds": SEEDS,
        "by_cell": by_cell,
        "detail": {
            k: [
                {"seed": r["seed"], "pass": r["pass"], "primary": r["primary"], "leak": r["leak"], "wall_s": r["wall_s"]}
                for r in rows
            ]
            for k, rows in detail.items()
        },
        "verdict": verdict,
        "finding": finding,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    lines = [
        "# Hard multipair @ n=1 cover=1.0 — 2026-09-09 (Fire #17)",
        "",
        "Music-transfer posture: n_particles=1, cover=1.0, faithful_guard_e, FM0, b_cap=1.",
        "Maps to Arm B smoke: `music_arm_b_locked_smoke_20260909.sh` (POLE_WEIGHT=1).",
        "",
        "| cell | PASS | prim mean | span | leak max | knife |",
        "|:---|:---:|---:|---:|---:|:---:|",
    ]
    for name, summ in by_cell.items():
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
        "- %s" % finding,
        "- Verdict: `%s`" % verdict,
        "- Wall: %.1fs" % blob["wall_s"],
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    block = (
        "\n## Fire #17 — hard multipair n=1 cover=1.0 (2026-09-09)\n\n"
        "- Music-transfer: cover/pole **1.0**, n_particles=1, FM0, b_cap1, guard_e\n"
        "- Verdict: **%s** — %s\n"
        "- Notes: `hard_multipair_n1_cover1_20260909.{py,json,md}`\n"
        % (verdict, finding)
    )
    status = (
        "\n### STATUS (~Fire #17)\n"
        "- Locked Music cover=1.0 under n=1 multipair: `%s`\n" % verdict
    )
    for path in (LOG, LOG_ROOT):
        if path.exists():
            path.write_text(path.read_text() + block + status)
    print("VERDICT=%s wall=%.1fs" % (verdict, blob["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
