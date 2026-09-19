#!/usr/bin/env python3
"""Falsify attempt: is b_cap=1 still best under Arm B locked posture?

Axis: toy b_cap ∈ {0.5, 1.0, 2.0}  (Music maps to --adv_reg_coeff;
toy margin is hard-coded ≈1 in cap_penalty — Music --adv_reg_kappa).

Fixed (Recommendation B / Arm B transfer posture):
  teacher=faithful_guard_e (leftover ON), cover_weight=1.0, FM=0,
  n_particles=1 (Music --parts 0 proxy), particle_l2=0.02, steps=1200.
  Geoms: sheet leftover + Field3D leftover. Seeds ≥2 ({0,1,2}).

Ask: does b_cap=1 remain best (full pass, comfortable kept, low leak,
no knife), or does 0.5 / 2.0 beat it?

CPU only. No Music LM GPU train.
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

OUT = Path(__file__).resolve().parent / "arm_b_bcap_or_kappa_check_20260909.json"
MD = Path(__file__).resolve().parent / "arm_b_bcap_or_kappa_check_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"

SEEDS = [0, 1, 2]
STEPS = 1200
COVER = 1.0  # Arm B recommended pole/cover
FM = 0.0
N_PART = 1
PART_L2 = 0.02
TEACHER = "faithful_guard_e"
B_CAPS = [0.5, 1.0, 2.0]
AXIS = "b_cap"  # not kappa — toy has single coeff; margin hard-coded 1

SHEET_KEPT_MIN = 0.90
SHEET_LEAK_MAX = 0.20
SHEET_COMFORT = 0.92


def cfg(seed: int, b_cap: float):
    return default_cfg(
        steps=STEPS,
        seed=int(seed),
        b_cap=float(b_cap),
        cover_weight=COVER,
        fm_weight=FM,
        n_particles=N_PART,
        particle_l2=PART_L2,
    )


def run_sheet(seed: int, b_cap: float) -> dict:
    st = time.time()
    row = score_adv_sheet(leaky_field(), teacher=TEACHER, cfg=cfg(seed, b_cap))
    kept = float(row.get("on_sheet_kept") or 0.0)
    leak = float(row.get("leak_tok") or 0.0)
    ok = kept >= SHEET_KEPT_MIN and abs(leak) <= SHEET_LEAK_MAX
    return {
        "geom": "sheet",
        "seed": seed,
        "b_cap": b_cap,
        "pass": bool(ok),
        "primary": kept,
        "leak": leak,
        "swing_kept": float(row.get("swing_kept") or 0.0),
        "residual_norm": float(row.get("residual_norm") or 0.0)
        if row.get("residual_norm") is not None
        else None,
        "wall_s": round(time.time() - st, 2),
    }


def run_f3d(seed: int, b_cap: float) -> dict:
    st = time.time()
    row = score_adv_field3d(
        leftover_field3d(seed=seed),
        teacher=TEACHER,
        cfg=cfg(seed, b_cap),
        name="f3d_s%d_b%g" % (seed, b_cap),
    )
    primary = float(row.get("u_kept") or 0.0)
    leak = float(row.get("leak_ratio") or 0.0)
    return {
        "geom": "field3d",
        "seed": seed,
        "b_cap": b_cap,
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
    span = max(prim) - min(prim) if prim else 0.0
    return {
        "n_pass": n_pass,
        "n_total": len(rows),
        "pass_rate": "%d/%d" % (n_pass, len(rows)),
        "primary_mean": sum(prim) / len(prim) if prim else None,
        "primary_min": min(prim) if prim else None,
        "primary_span": span,
        "leak_abs_max": max(leaks) if leaks else None,
        "leak_abs_mean": sum(leaks) / len(leaks) if leaks else None,
        "knife_edge": bool(n_pass not in (0, len(rows)) or span > 0.05),
        "full_pass": n_pass == len(rows) and len(rows) > 0,
    }


def recommend(summary: dict) -> dict:
    """Prefer b_cap=1 if comfortable dual-geom; else lowest comfortable; flag revise."""
    candidates = []
    for bc in B_CAPS:
        sh = summary["b_cap_%.1f__sheet" % bc]
        f3 = summary["b_cap_%.1f__field3d" % bc]
        comfortable = (
            sh["full_pass"]
            and f3["full_pass"]
            and not sh["knife_edge"]
            and not f3["knife_edge"]
            and (sh["primary_min"] or 0.0) >= SHEET_COMFORT
        )
        candidates.append(
            {
                "b_cap": bc,
                "music_adv_reg_coeff": bc,
                "music_adv_reg_kappa": 1.0,  # toy margin fixed; Music kappa default
                "comfortable": comfortable,
                "sheet": sh,
                "field3d": f3,
                "sheet_margin": (sh["primary_min"] or 0.0) - 0.90,
                "f3d_leak_max": f3["leak_abs_max"],
                "score": (
                    1 if comfortable else 0,
                    sh["n_pass"] + f3["n_pass"],
                    -(sh["leak_abs_max"] or 99),
                    -(f3["leak_abs_max"] or 99),
                    # prefer canonical 1.0 when tied; else closer to 1.0
                    -abs(bc - 1.0),
                    -(sh["primary_mean"] or 0),
                ),
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    locked_ok = any(c["b_cap"] == 1.0 and c["comfortable"] for c in candidates)
    best_is_1 = best["b_cap"] == 1.0 and best["comfortable"]
    if best_is_1:
        verdict = "CONFIRMED"
        note = "b_cap=1 remains best under Arm B posture (cover=1.0, leftover ON, FM0, n=1)."
    elif locked_ok:
        # 1.0 comfortable but another scored higher — still confirm if 1 is fine
        one = next(c for c in candidates if c["b_cap"] == 1.0)
        if one["comfortable"]:
            verdict = "CONFIRMED"
            note = (
                "b_cap=1 still comfortable full-pass; alternate b_cap=%.1f scored "
                "higher on leak/kept but recipe stays at 1 (do not chase)."
                % best["b_cap"]
            )
            best = one
            best["note_override"] = note
        else:
            verdict = "REVISED"
            note = "b_cap=1 not comfortable; prefer b_cap=%.1f." % best["b_cap"]
    else:
        verdict = "REVISED"
        note = (
            "b_cap=1 failed comfort/full-pass; soft/best pick b_cap=%.1f."
            % best["b_cap"]
        )
    out = {k: v for k, v in best.items() if k != "score"}
    out["verdict"] = verdict
    out["note"] = best.get("note_override") or note
    out["all_comfortable"] = [c["b_cap"] for c in candidates if c["comfortable"]]
    return out


def main() -> None:
    rows = []
    t0 = time.time()
    print(
        "arm_b b_cap falsify | teacher=%s cover=%.1f FM=0 n_part=%d steps=%d "
        "b_caps=%s seeds=%s"
        % (TEACHER, COVER, N_PART, STEPS, B_CAPS, SEEDS),
        flush=True,
    )
    for bc in B_CAPS:
        for seed in SEEDS:
            sheet = run_sheet(seed, bc)
            rows.append(sheet)
            print(
                "sheet b_cap=%.1f seed=%d pass=%s kept=%.4f leak=%.4f (%.1fs)"
                % (
                    bc,
                    seed,
                    sheet["pass"],
                    sheet["primary"],
                    sheet["leak"],
                    sheet["wall_s"],
                ),
                flush=True,
            )
            f3d = run_f3d(seed, bc)
            rows.append(f3d)
            print(
                "f3d   b_cap=%.1f seed=%d pass=%s u_kept=%.4f leak=%.4f (%.1fs)"
                % (
                    bc,
                    seed,
                    f3d["pass"],
                    f3d["primary"],
                    f3d["leak"],
                    f3d["wall_s"],
                ),
                flush=True,
            )

    summary = {}
    for bc in B_CAPS:
        for geom in ("sheet", "field3d"):
            sub = [r for r in rows if r["b_cap"] == bc and r["geom"] == geom]
            summary["b_cap_%.1f__%s" % (bc, geom)] = summarize(sub)

    rec = recommend(summary)
    blob = {
        "sha_hint": "435e873",
        "arm": "B_falsify_bcap",
        "axis": AXIS,
        "axis_values": B_CAPS,
        "why_not_kappa": (
            "toy AdvConfig has single b_cap (coeff); cap_penalty margin "
            "hard-coded at 1.0. Music --adv_reg_kappa maps to that margin; "
            "sweeping toy b_cap stresses the transferable coeff axis."
        ),
        "music_map": {
            "teacher": "faithful_guard_e",
            "adv_arch": "mlp",
            "fm_weight": 0,
            "adv_weight": 1,
            "pole_weight": COVER,
            "adv_reg_coeff": "← swept b_cap",
            "adv_reg_kappa": 1,
            "parts": 0,
        },
        "recipe_fixed": {
            "steps": STEPS,
            "cover_weight": COVER,
            "fm_weight": FM,
            "n_particles": N_PART,
            "particle_l2": PART_L2,
            "teacher": TEACHER,
            "music_parts_proxy": 0,
        },
        "seeds": SEEDS,
        "rows": rows,
        "summary": summary,
        "recommended": rec,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2))

    lines = [
        "# Arm B b_cap falsify check (2026-09-09)",
        "",
        "Falsify attempt under Recommendation **B** posture: leftover ON",
        "(`faithful_guard_e`), `cover_weight=1.0`, FM=0, `n_particles=1`",
        "(Music `--parts 0` proxy), steps=1200. Axis: toy **`b_cap`** ∈ %s"
        % B_CAPS,
        "(Music `--adv_reg_coeff`; κ left at toy hard-coded margin=1 /",
        "Music `--adv_reg_kappa=1`). Seeds: %s. Wall: %.1fs."
        % (SEEDS, blob["wall_s"]),
        "",
        "| b_cap | geom | pass | primary_mean | primary_min | leak_abs_max | knife |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for bc in B_CAPS:
        for geom in ("sheet", "field3d"):
            s = summary["b_cap_%.1f__%s" % (bc, geom)]
            lines.append(
                "| %.1f | %s | %s | %.4f | %.4f | %.4f | %s |"
                % (
                    bc,
                    geom,
                    s["pass_rate"],
                    s["primary_mean"],
                    s["primary_min"],
                    s["leak_abs_max"],
                    s["knife_edge"],
                )
            )

    lines += [
        "",
        "## Verdict",
        "",
        "- **%s** — %s" % (rec["verdict"], rec["note"]),
        "- Best / locked pick: **b_cap = %.1f** (Music `--adv_reg_coeff=%.1f`,"
        % (rec["b_cap"], rec["music_adv_reg_coeff"]),
        "  `--adv_reg_kappa=%.1f`)" % rec["music_adv_reg_kappa"],
        "- Comfortable b_caps: %s" % (rec.get("all_comfortable") or []),
        "",
        "## Read",
        "",
        "- If CONFIRMED: recipe card keeps c=κ=1; no revise.",
        "- If REVISED: update `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md` + Arm B smoke.",
        "- κ not swept on toy (single coeff API); Music split remains c=κ paired at 1",
        "  unless a later Music-only kappa smoke says otherwise.",
        "",
        "## Paths",
        "",
        "- JSON: `arm_b_bcap_or_kappa_check_20260909.json`",
        "- Script: `arm_b_bcap_or_kappa_check_20260909.py`",
        "- Recipe: `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md`",
        "- Arm B smoke: `music_arm_b_locked_smoke_20260909.sh`",
        "- Prior cover sweep: `arm_b_pole_cover_sweep_20260909.md`",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        "verdict=%s b_cap=%.1f note=%s" % (rec["verdict"], rec["b_cap"], rec["note"]),
        flush=True,
    )
    print("wrote", OUT, MD, flush=True)


if __name__ == "__main__":
    main()
