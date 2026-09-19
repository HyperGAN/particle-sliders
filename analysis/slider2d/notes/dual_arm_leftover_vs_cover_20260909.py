#!/usr/bin/env python3
"""Dual-arm transfer ablation: leftover-only vs cover-only under FM=0 b_cap=1.

Music-transferable subset: n_particles=1 (toy ParticlePrior cannot be 0 —
randint(0,0) fails; Music --parts 0 has no latent ParticlePrior at all).
Locked geometry otherwise: steps=1200, particle_l2=0.02, FM off, b_cap=1.

Cells (sheet leftover + Field3D leftover):
  leftover_only : teacher=faithful_guard_e, cover_weight=0
  cover_only    : teacher=faithful,         cover_weight=1.5
  locked        : teacher=faithful_guard_e, cover_weight=1.5
  neither       : teacher=faithful,         cover_weight=0

Ask: under the Music-transferable no-particle proxy, does leftover gate
still dominate leak and cover still dominate kept — i.e. can either arm
of the Music dual-arm stand alone as a #94 transfer?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import leftover_field3d, score_adv_field3d
from analysis.slider2d.gan import default_cfg, score_adv_sheet
from analysis.slider2d.sheet import leaky_field

OUT = Path(__file__).resolve().parent / "dual_arm_leftover_vs_cover_20260909.json"
MD = Path(__file__).resolve().parent / "dual_arm_leftover_vs_cover_20260909.md"

# Fewer seeds than full 6/6 lock — qualitative dual-arm map, CPU budget shared
# with in-flight field3d_hard_multipair.
SEEDS = [0, 1, 2]
STEPS = 1200
B_CAP = 1.0
FM = 0.0
N_PART = 1  # Music --parts 0 proxy (see module docstring)
PART_L2 = 0.02

CELLS = {
    "leftover_only": {"teacher": "faithful_guard_e", "cover_weight": 0.0},
    "cover_only": {"teacher": "faithful", "cover_weight": 1.5},
    "locked": {"teacher": "faithful_guard_e", "cover_weight": 1.5},
    "neither": {"teacher": "faithful", "cover_weight": 0.0},
}


def cfg(seed: int, cover: float) -> object:
    return default_cfg(
        steps=STEPS,
        seed=int(seed),
        b_cap=B_CAP,
        cover_weight=float(cover),
        fm_weight=FM,
        n_particles=N_PART,
        particle_l2=PART_L2,
    )


def run_sheet(seed: int, teacher: str, cover: float) -> dict:
    st = time.time()
    row = score_adv_sheet(leaky_field(), teacher=teacher, cfg=cfg(seed, cover))
    kept = float(row.get("on_sheet_kept") or 0.0)
    leak = float(row.get("leak_tok") or 0.0)
    ok = kept >= 0.90 and abs(leak) <= 0.20
    return {
        "geom": "sheet",
        "seed": seed,
        "teacher": teacher,
        "cover_weight": cover,
        "pass": bool(ok),
        "primary": kept,
        "leak": leak,
        "swing_kept": float(row.get("swing_kept") or 0.0),
        "residual_norm": float(row.get("residual_norm") or 0.0)
        if row.get("residual_norm") is not None
        else None,
        "wall_s": round(time.time() - st, 2),
    }


def run_f3d(seed: int, teacher: str, cover: float) -> dict:
    st = time.time()
    row = score_adv_field3d(
        leftover_field3d(seed=seed),
        teacher=teacher,
        cfg=cfg(seed, cover),
        name="f3d_s%d" % seed,
    )
    primary = float(row.get("u_kept") or 0.0)
    leak = float(row.get("leak_ratio") or 0.0)
    return {
        "geom": "field3d",
        "seed": seed,
        "teacher": teacher,
        "cover_weight": cover,
        "pass": bool(row.get("pass")),
        "primary": primary,
        "leak": leak,
        "content_kept": float(row.get("content_kept") or 0.0)
        if row.get("content_kept") is not None
        else None,
        "residual_norm": float(row.get("residual_norm") or 0.0)
        if row.get("residual_norm") is not None
        else None,
        "wall_s": round(time.time() - st, 2),
    }


def summarize(rows: list) -> dict:
    prim = [r["primary"] for r in rows]
    leaks = [abs(r["leak"]) for r in rows]
    n_pass = sum(1 for r in rows if r["pass"])
    return {
        "n_pass": n_pass,
        "n_total": len(rows),
        "pass_rate": "%d/%d" % (n_pass, len(rows)),
        "primary_mean": sum(prim) / len(prim),
        "primary_min": min(prim),
        "primary_span": max(prim) - min(prim),
        "leak_abs_max": max(leaks),
        "leak_abs_mean": sum(leaks) / len(leaks),
    }


def main() -> None:
    rows = []
    t0 = time.time()
    print(
        "dual-arm leftover vs cover | FM=0 b_cap=1 n_part=%d steps=%d seeds=%s"
        % (N_PART, STEPS, SEEDS),
        flush=True,
    )
    for cell_name, knobs in CELLS.items():
        teacher = knobs["teacher"]
        cover = knobs["cover_weight"]
        for seed in SEEDS:
            sheet = run_sheet(seed, teacher, cover)
            sheet["cell"] = cell_name
            rows.append(sheet)
            print(
                "sheet %-14s seed=%d pass=%s kept=%.4f leak=%.4f (%.1fs)"
                % (
                    cell_name,
                    seed,
                    sheet["pass"],
                    sheet["primary"],
                    sheet["leak"],
                    sheet["wall_s"],
                ),
                flush=True,
            )
            f3d = run_f3d(seed, teacher, cover)
            f3d["cell"] = cell_name
            rows.append(f3d)
            print(
                "f3d   %-14s seed=%d pass=%s u_kept=%.4f leak=%.4f (%.1fs)"
                % (
                    cell_name,
                    seed,
                    f3d["pass"],
                    f3d["primary"],
                    f3d["leak"],
                    f3d["wall_s"],
                ),
                flush=True,
            )

    summary = {}
    for cell_name in CELLS:
        for geom in ("sheet", "field3d"):
            sub = [r for r in rows if r["cell"] == cell_name and r["geom"] == geom]
            summary["%s__%s" % (cell_name, geom)] = summarize(sub)

    blob = {
        "sha_hint": "435e873",
        "recipe": {
            "steps": STEPS,
            "b_cap": B_CAP,
            "fm_weight": FM,
            "n_particles": N_PART,
            "particle_l2": PART_L2,
            "music_parts_proxy": 0,
            "note": "n_particles=1 proxies Music --parts 0; toy n=0 breaks ParticlePrior.sample",
        },
        "cells": CELLS,
        "seeds": SEEDS,
        "rows": rows,
        "summary": summary,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2))

    lines = [
        "# Dual-arm leftover-only vs cover-only (2026-09-09)",
        "",
        "FM=0, b_cap=1, steps=1200, n_particles=1 (Music `--parts 0` proxy).",
        "Seeds: %s. Wall: %.1fs." % (SEEDS, blob["wall_s"]),
        "",
        "| cell | geom | pass | primary_mean | primary_min | leak_abs_max |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, s in summary.items():
        cell, geom = key.split("__")
        lines.append(
            "| %s | %s | %s | %.4f | %.4f | %.4f |"
            % (
                cell,
                geom,
                s["pass_rate"],
                s["primary_mean"],
                s["primary_min"],
                s["leak_abs_max"],
            )
        )
    lines += [
        "",
        "## Read for Music dual-arm",
        "",
        "- **leftover_only** (guard, cover=0): tests gate without cover pin — Music",
        "  guard+mlp arm with pole_weight≈0.",
        "- **cover_only** (faithful, cover=1.5): cover without leftover gate — maps to",
        "  lyric/pole pin without `faithful_guard_e` (listen arm risk).",
        "- **locked** (guard+cover1.5): true #94 cell.",
        "- **neither**: baseline collapse.",
        "",
        "JSON: `dual_arm_leftover_vs_cover_20260909.json`",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, MD, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
