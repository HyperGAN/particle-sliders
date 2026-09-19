#!/usr/bin/env python3
"""Fire #15: dual-arm transfer under Music constraints (post CLI gaps).

Music cannot combine leftover-gated teacher with TX listen critic in one argv
(transfer_gaps_post_cli_20260909.md). Particles do NOT port: Music has no
ParticlePrior; keep --parts 0. Cover analogue is lyrichold/pole_weight
(unvalidated on listen).

This 2D/Field3D harness asks what still transfers as **sequential arms**:

  Arm A — leftover-gate alone (Music: faithful_guard_e + mlp, parts=0)
  Arm B — listen/cover alone (Music: faithful_plus_neu_lyric + tx; here:
           cover_weight / pole-pin without relying on ParticlePrior)

Fixed portable core where applicable:
  steps=1200, cover_weight=1.5 (Arm B), teacher=faithful_guard_e (Arm A),
  FM off, b_cap=1. n_particles swept only to document toy-only sensitivity
  (NOT a Music --parts recommendation).

Seeds ≥6. CPU only.
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
    score_adv_field3d,
    unused_e_field3d,
    divergent_field3d,
    close_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg, score_adv_sheet, score_adv_exam
from analysis.slider2d.sheet import leaky_field
from analysis.slider2d.exam import unused_e_field, divergent_field

OUT = Path(__file__).resolve().parent / "dual_arm_transfer_20260909.json"
MD = Path(__file__).resolve().parent / "dual_arm_transfer_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER_A = "faithful_guard_e"

# Arm A: leftover gate — particles fixed at Music-honest 0-analogue:
# use n_particles=1 (minimal jitter; closest to "no ParticlePrior") vs locked 12
# to show what survives when particles cannot port.
ARM_A = {
    "tag": "arm_A_leftover_gate",
    "music": "--lm_target faithful_guard_e --adv_arch mlp --parts 0 --fm_weight 0 --adv_weight 1 --adv_reg_coeff 1 --adv_reg_kappa 1",
    "variants": [
        {"label": "A_n1_minimal_jitter", "n_particles": 1, "cover_weight": 1.5, "b_cap": 1.0},
        {"label": "A_n12_locked_toy", "n_particles": 12, "cover_weight": 1.5, "b_cap": 1.0},
        {"label": "A_n1_cover0", "n_particles": 1, "cover_weight": 0.0, "b_cap": 1.0},
        {"label": "A_n1_cover1.5_b0.5", "n_particles": 1, "cover_weight": 1.5, "b_cap": 0.5},
        {"label": "A_n1_cover1.5_b2", "n_particles": 1, "cover_weight": 1.5, "b_cap": 2.0},
    ],
    "cells": {
        "sheet_leftover": ("sheet", leaky_field),
        "f3d_leftover": ("f3d", leftover_field3d),
        "f3d_unused_e": ("f3d_exam", unused_e_field3d),
        "f3d_divergent": ("f3d_exam", divergent_field3d),
        "exam_unused_e": ("exam", unused_e_field),
        "exam_divergent": ("exam", divergent_field),
    },
}

# Arm B: listen/cover alone — teacher still guard for geometry, but we stress
# cover_weight as lyrichold/pole analogue with n_particles=1 (no prior port).
ARM_B = {
    "tag": "arm_B_listen_cover",
    "music": "--lm_target faithful_plus_neu_lyric --adv_arch tx --parts 0 --fm_weight 0 --lyrichold_weight 1 --pole_weight 0 (cover unvalidated)",
    "variants": [
        {"label": "B_cover0_n1", "n_particles": 1, "cover_weight": 0.0, "b_cap": 1.0},
        {"label": "B_cover1.0_n1", "n_particles": 1, "cover_weight": 1.0, "b_cap": 1.0},
        {"label": "B_cover1.5_n1", "n_particles": 1, "cover_weight": 1.5, "b_cap": 1.0},
        {"label": "B_cover3.0_n1", "n_particles": 1, "cover_weight": 3.0, "b_cap": 1.0},
        {"label": "B_cover1.5_n12_control", "n_particles": 12, "cover_weight": 1.5, "b_cap": 1.0},
    ],
    "cells": {
        "sheet_leftover": ("sheet", leaky_field),
        "f3d_close": ("f3d_exam", close_field3d),
        "f3d_leftover": ("f3d", leftover_field3d),
    },
}


def make_cfg(seed, n_particles, cover_weight, b_cap):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=float(b_cap),
        cover_weight=float(cover_weight),
        fm_weight=0.0,
        n_particles=int(n_particles),
        particle_l2=0.02,
    )


def run_one(kind, ctor, seed, cfg):
    st = time.time()
    if kind == "sheet":
        row = score_adv_sheet(ctor(), teacher=TEACHER_A, cfg=cfg)
        primary = float(row.get("on_sheet_kept") or 0.0)
        leak = float(row.get("leak_tok") or 0.0)
        ok = primary >= 0.90 and abs(leak) <= 0.20
    elif kind == "exam":
        row = score_adv_exam(ctor(), teacher=TEACHER_A, cfg=cfg)
        primary = float(row.get("roll_overlap") or 0.0)
        leak = float(row.get("leak_tok") or 0.0)
        ok = primary >= 0.85 and float(row.get("roll_swing_kept") or 0.0) >= 0.60
    elif kind == "f3d_exam":
        row = score_adv_field3d_exam(ctor(seed=seed), teacher=TEACHER_A, cfg=cfg)
        primary = float(row.get("exam_score") or 0.0)
        leak = float(row.get("leak_ratio") or 0.0)
        ok = bool(row.get("exam_pass"))
    else:
        row = score_adv_field3d(ctor(seed=seed), teacher=TEACHER_A, cfg=cfg)
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


def run_arm(arm):
    out = []
    print("==== %s ====" % arm["tag"], flush=True)
    print("Music intent: %s" % arm["music"], flush=True)
    for var in arm["variants"]:
        for cell, (kind, ctor) in arm["cells"].items():
            label = "%s/%s" % (var["label"], cell)
            print("--- %s" % label, flush=True)
            rows = []
            for seed in SEEDS:
                cfg = make_cfg(seed, var["n_particles"], var["cover_weight"], var["b_cap"])
                row = run_one(kind, ctor, seed, cfg)
                rows.append(row)
                print(
                    "  seed=%2d pass=%s prim=%.4f leak=%.4f (%.1fs)"
                    % (seed, row["pass"], row["primary"], row["leak"], row["wall_s"]),
                    flush=True,
                )
            summ = summarize(rows)
            out.append(
                {
                    "arm": arm["tag"],
                    "variant": var["label"],
                    "cell": cell,
                    "n_particles": var["n_particles"],
                    "cover_weight": var["cover_weight"],
                    "b_cap": var["b_cap"],
                    "music_note": (
                        "n_particles is TOY-ONLY (no Music ParticlePrior); "
                        "n=1 ≈ honest --parts 0 transfer posture"
                    ),
                    "summary": summ,
                    "detail": rows,
                }
            )
            print(
                "  >> %s mean=%.4f span=%.4f knife=%s"
                % (summ["pass_rate"], summ["primary_mean"], summ["primary_span"], summ["knife_edge"]),
                flush=True,
            )
    return out


def write_md(blob):
    lines = [
        "# Dual-arm transfer (leftover-gate vs listen/cover) — 2026-09-09 (Fire #15)",
        "",
        "Constraints from `transfer_gaps_post_cli_20260909.md`:",
        "1. TX critic cannot combine with `faithful_guard_e` in one Music argv.",
        "2. Keep `--parts 0`; do **not** map toy `n_particles` → Music ParticlePrior (none).",
        "3. Cover analogue = lyrichold/pole_weight (unvalidated on listen).",
        "",
        "Seeds: %s" % (blob["seeds"],),
        "",
        "| arm | variant | cell | n_part | cover | b_cap | PASS | prim mean | span | knife |",
        "|:---|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for g in blob["grid"]:
        s = g["summary"]
        lines.append(
            "| %s | %s | %s | %d | %.1f | %.1f | %s | %.4f | %.4f | %s |"
            % (
                g["arm"].replace("arm_", ""),
                g["variant"],
                g["cell"],
                g["n_particles"],
                g["cover_weight"],
                g["b_cap"],
                s["pass_rate"],
                s["primary_mean"],
                s["primary_span"],
                "YES" if s["knife_edge"] else "no",
            )
        )
    lines += [
        "",
        "### Finding",
        "",
        "- %s" % blob["finding"],
        "- Verdict: `%s`" % blob["verdict"],
        "- What transfers under dual-arm: %s" % blob["transfers"],
        "- What does NOT transfer: %s" % blob["does_not_transfer"],
        "- Music sequential recipe hypothesis: %s" % blob["music_hypothesis"],
        "- Wall: %.1fs" % blob["wall_s"],
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")


def append_log(blob):
    block = (
        "\n## Fire #15 — dual-arm transfer under Music gaps (2026-09-09)\n\n"
        "- Host: pop-os-cpu @ `435e873` + adv CLI restore; gaps: `transfer_gaps_post_cli_20260909.md`\n"
        "- Arm A leftover-gate (mlp / faithful_guard_e / parts=0 analogue n=1)\n"
        "- Arm B listen/cover (cover sweep @ n=1; TX arm is Music-side only)\n"
        "- Verdict: **%s** — %s\n"
        "- Transfers: %s\n"
        "- Does NOT transfer: %s\n"
        "- Music hypothesis: %s\n"
        "- Notes: `dual_arm_transfer_20260909.{py,json,md}`\n"
        % (
            blob["verdict"],
            blob["finding"],
            blob["transfers"],
            blob["does_not_transfer"],
            blob["music_hypothesis"],
        )
    )
    status = (
        "\n### STATUS (~Fire #15 dual-arm)\n"
        "- Best transferable under dual-arm: leftover-gate arm with n=1 + cover1.5 + b_cap=1 if solid.\n"
        "- Current: `%s`\n" % blob["verdict"]
    )
    for path in (LOG, LOG_ROOT):
        if path.exists():
            path.write_text(path.read_text() + block + status)


def main():
    t0 = time.time()
    grid = []
    grid.extend(run_arm(ARM_A))
    grid.extend(run_arm(ARM_B))

    # Analyze: does Arm A at n=1 match n=12?
    def rows_for(arm_tag, variant_prefix, cell=None):
        return [
            g
            for g in grid
            if g["arm"] == arm_tag
            and g["variant"].startswith(variant_prefix)
            and (cell is None or g["cell"] == cell)
        ]

    a_n1 = [g for g in grid if g["arm"] == ARM_A["tag"] and g["variant"] == "A_n1_minimal_jitter"]
    a_n12 = [g for g in grid if g["arm"] == ARM_A["tag"] and g["variant"] == "A_n12_locked_toy"]
    a_n1_ok = all(g["summary"]["full_pass"] for g in a_n1)
    a_n12_ok = all(g["summary"]["full_pass"] for g in a_n12)
    # Compare means on shared cells
    deltas = []
    for g1 in a_n1:
        g12 = next((x for x in a_n12 if x["cell"] == g1["cell"]), None)
        if g12:
            deltas.append(g1["summary"]["primary_mean"] - g12["summary"]["primary_mean"])
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    a_cover0 = [g for g in grid if g["variant"] == "A_n1_cover0"]
    cover0_fail = [g for g in a_cover0 if not g["summary"]["full_pass"]]

    b_cover = [g for g in grid if g["arm"] == ARM_B["tag"]]
    b_fragile = [g for g in b_cover if g["summary"]["knife_edge"] or not g["summary"]["full_pass"]]

    if a_n1_ok and abs(mean_delta) < 0.01:
        particles_port = False
        particles_note = (
            "n_particles=1 (Music-honest, no ParticlePrior) ≈ n=12 on Arm A (Δprim=%.4f). "
            "Toy ParticlePrior is non-essential for leftover-gate transfer; keep Music --parts 0."
            % mean_delta
        )
    elif a_n1_ok:
        particles_port = False
        particles_note = (
            "n=1 still full PASS but Δprim vs n=12 = %.4f. Prefer documenting sensitivity; "
            "still do NOT map to --parts." % mean_delta
        )
    else:
        particles_port = False
        particles_note = (
            "n=1 fails some Arm A cells — leftover-gate may lean on toy ParticlePrior jitter. "
            "Music transfer risk: no ParticlePrior exists; need cover/b_cap or mlp capacity check."
        )

    if cover0_fail:
        cover_note = "cover_weight=0 fails under n=1 on: %s — cover/pole pin still needed without particles." % [
            g["cell"] for g in cover0_fail
        ]
    else:
        cover_note = "cover_weight=0 still passes under n=1 (surprising) — recheck sheet lock margin."

    transfers = (
        "Arm A leftover-gate (faithful_guard_e) with b_cap=1 + cover≈1.5; "
        "sequential Music mlp leftover smoke; FM=0; --parts 0."
    )
    does_not = (
        "Single-argv TX+guard_e; mapping n_particles→--parts/ParticlePrior; "
        "assuming lyrichold=cover without listen validation; extreme Field3D mismatch cells (Fire #13)."
    )

    if a_n1_ok and not b_fragile:
        verdict = "dual_arm_both_solid_parts0"
    elif a_n1_ok:
        verdict = "arm_A_solid_arm_B_mixed"
    else:
        verdict = "arm_A_needs_particles_or_cover"

    finding = particles_note + " " + cover_note
    music_hypothesis = (
        "Run sequential smokes (not one argv): "
        "(1) leftover mlp: faithful_guard_e --adv_arch mlp --parts 0 --fm_weight 0 --adv_reg_*=1 "
        "+ modest pole_weight; "
        "(2) listen tx: faithful_plus_neu_lyric --adv_arch tx --parts 0 --fm_weight 0 --lyrichold_weight 1. "
        "Do not raise --parts to chase toy n_particles."
    )

    blob = {
        "fire": 15,
        "date": "2026-09-09",
        "sha": "435e873",
        "constraints": [
            "dual-arm TX vs guard_e",
            "parts=0 no ParticlePrior",
            "cover=lyrichold/pole unvalidated",
        ],
        "seeds": SEEDS,
        "grid": [
            {
                **{k: v for k, v in g.items() if k != "detail"},
                "detail": [
                    {"seed": r["seed"], "pass": r["pass"], "primary": r["primary"], "leak": r["leak"], "wall_s": r["wall_s"]}
                    for r in g["detail"]
                ],
            }
            for g in grid
        ],
        "verdict": verdict,
        "finding": finding,
        "transfers": transfers,
        "does_not_transfer": does_not,
        "music_hypothesis": music_hypothesis,
        "particles_port_to_music": particles_port,
        "n1_vs_n12_delta_prim": mean_delta,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    write_md(blob)
    append_log(blob)
    print("VERDICT=%s wall=%.1fs" % (verdict, blob["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
