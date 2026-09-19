#!/usr/bin/env python3
"""Arm B (#94 / Recommendation B) modest pole/cover analogue sweep.

Locked leftover ON (teacher=faithful_guard_e), FM=0, b_cap=1, steps=1200,
n_particles=1 (Music --parts 0 proxy). Sweep cover_weight in
{0.5, 1.0, 1.5, 2.0} — Music maps this to modest --pole_weight.

Reject high-cover false locks (Fire #4 pattern: short×high-pin / cover3.0
looking "kept" while leak geometry is wrong). Prefer lowest modest cover
that keeps sheet+Field3D pass with low leak and no knife_edge.

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

OUT = Path(__file__).resolve().parent / "arm_b_pole_cover_sweep_20260909.json"
MD = Path(__file__).resolve().parent / "arm_b_pole_cover_sweep_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"

SEEDS = [0, 1, 2]
STEPS = 1200
B_CAP = 1.0
FM = 0.0
N_PART = 1
PART_L2 = 0.02
TEACHER = "faithful_guard_e"
COVERS = [0.5, 1.0, 1.5, 2.0]

# Pass gates mirror dual_arm_leftover_vs_cover / sheet score_adv_sheet
SHEET_KEPT_MIN = 0.90
SHEET_LEAK_MAX = 0.20


def cfg(seed: int, cover: float):
    return default_cfg(
        steps=STEPS,
        seed=int(seed),
        b_cap=B_CAP,
        cover_weight=float(cover),
        fm_weight=FM,
        n_particles=N_PART,
        particle_l2=PART_L2,
    )


def run_sheet(seed: int, cover: float) -> dict:
    st = time.time()
    row = score_adv_sheet(leaky_field(), teacher=TEACHER, cfg=cfg(seed, cover))
    kept = float(row.get("on_sheet_kept") or 0.0)
    leak = float(row.get("leak_tok") or 0.0)
    ok = kept >= SHEET_KEPT_MIN and abs(leak) <= SHEET_LEAK_MAX
    return {
        "geom": "sheet",
        "seed": seed,
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


def run_f3d(seed: int, cover: float) -> dict:
    st = time.time()
    row = score_adv_field3d(
        leftover_field3d(seed=seed),
        teacher=TEACHER,
        cfg=cfg(seed, cover),
        name="f3d_s%d_c%g" % (seed, cover),
    )
    primary = float(row.get("u_kept") or 0.0)
    leak = float(row.get("leak_ratio") or 0.0)
    return {
        "geom": "field3d",
        "seed": seed,
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
    """Pick modest cover: full sheet+f3d pass, comfortable kept margin, prefer lower.

    Comfort: sheet primary_min >= 0.92 (reject knife-adjacent 0.90+eps).
    Reject cover>=2.0 when a lower comfortable cover exists (Fire #4 caution).
    """
    SHEET_COMFORT = 0.92
    candidates = []
    for cover in COVERS:
        sh = summary["cover_%.1f__sheet" % cover]
        f3 = summary["cover_%.1f__field3d" % cover]
        if not (sh["full_pass"] and f3["full_pass"]):
            continue
        if sh["knife_edge"] or f3["knife_edge"]:
            continue
        if (sh["primary_min"] or 0.0) < SHEET_COMFORT:
            continue
        candidates.append(
            {
                "cover_weight": cover,
                "music_pole_weight": cover,  # name map: cover → pole on Arm B
                "sheet": sh,
                "field3d": f3,
                "sheet_margin": (sh["primary_min"] or 0.0) - 0.90,
                "f3d_leak_max": f3["leak_abs_max"],
            }
        )
    candidates.sort(
        key=lambda c: (
            c["cover_weight"],
            c["f3d_leak_max"] or 99,
            -(c["sheet"]["primary_mean"] or 0),
        )
    )
    comfortable_lt2 = [c for c in candidates if c["cover_weight"] < 2.0]
    if comfortable_lt2:
        best = comfortable_lt2[0]
        best["note"] = (
            "lowest modest cover with full sheet+Field3D pass and sheet "
            "kept_min>=%.2f; reject cover>=2.0 while lower comfortable exists"
            % SHEET_COMFORT
        )
        return best
    if candidates:
        best = candidates[0]
        best["caution"] = (
            "only cover>=2.0 comfortable-passed; watch Fire #4 high-pin false lock"
        )
        return best
    # soft fallback
    soft = []
    for cover in COVERS:
        sh = summary["cover_%.1f__sheet" % cover]
        f3 = summary["cover_%.1f__field3d" % cover]
        soft.append(
            {
                "cover_weight": cover,
                "music_pole_weight": cover,
                "sheet": sh,
                "field3d": f3,
                "soft": True,
                "score_tuple": (
                    sh["n_pass"] + f3["n_pass"],
                    -(sh["leak_abs_max"] or 99),
                    -(f3["leak_abs_max"] or 99),
                    -cover,
                ),
            }
        )
    soft.sort(key=lambda c: c["score_tuple"], reverse=True)
    best = soft[0]
    best["rejected_reason"] = "no comfortable dual-geom pass; soft pick"
    return best


def main() -> None:
    rows = []
    t0 = time.time()
    print(
        "arm_b pole/cover sweep | teacher=%s FM=0 b_cap=1 n_part=%d steps=%d covers=%s seeds=%s"
        % (TEACHER, N_PART, STEPS, COVERS, SEEDS),
        flush=True,
    )
    for cover in COVERS:
        for seed in SEEDS:
            sheet = run_sheet(seed, cover)
            rows.append(sheet)
            print(
                "sheet cover=%.1f seed=%d pass=%s kept=%.4f leak=%.4f (%.1fs)"
                % (
                    cover,
                    seed,
                    sheet["pass"],
                    sheet["primary"],
                    sheet["leak"],
                    sheet["wall_s"],
                ),
                flush=True,
            )
            f3d = run_f3d(seed, cover)
            rows.append(f3d)
            print(
                "f3d   cover=%.1f seed=%d pass=%s u_kept=%.4f leak=%.4f (%.1fs)"
                % (
                    cover,
                    seed,
                    f3d["pass"],
                    f3d["primary"],
                    f3d["leak"],
                    f3d["wall_s"],
                ),
                flush=True,
            )

    summary = {}
    for cover in COVERS:
        for geom in ("sheet", "field3d"):
            sub = [r for r in rows if r["cover_weight"] == cover and r["geom"] == geom]
            summary["cover_%.1f__%s" % (cover, geom)] = summarize(sub)

    rec = recommend(summary)
    # strip non-json score_tuple if present
    rec_out = {k: v for k, v in rec.items() if k != "score_tuple"}
    # also strip nested if any
    for key in ("sheet", "field3d"):
        if key in rec_out and isinstance(rec_out[key], dict):
            pass

    blob = {
        "sha_hint": "435e873",
        "arm": "B_recommendation_true_94",
        "music_map": {
            "teacher": "faithful_guard_e",
            "adv_arch": "mlp",
            "fm_weight": 0,
            "adv_weight": 1,
            "adv_reg_coeff": 1,
            "adv_reg_kappa": 1,
            "gan_beta1": 0,
            "parts": 0,
            "cover_analogue": "pole_weight",
            "lyrichold": "as needed (0 on leftover arm default)",
        },
        "recipe": {
            "steps": STEPS,
            "b_cap": B_CAP,
            "fm_weight": FM,
            "n_particles": N_PART,
            "particle_l2": PART_L2,
            "teacher": TEACHER,
            "music_parts_proxy": 0,
        },
        "covers": COVERS,
        "seeds": SEEDS,
        "rows": rows,
        "summary": summary,
        "recommended": rec_out,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2))

    lines = [
        "# Arm B pole/cover analogue sweep (2026-09-09)",
        "",
        "Recommendation **B** true #94 target: `faithful_guard_e` + leftover ON,",
        "FM=0, b_cap=1, steps=1200, `n_particles=1` (Music `--parts 0` proxy).",
        "Sweep toy `cover_weight` ∈ %s → Music modest `--pole_weight`."
        % COVERS,
        "Seeds: %s. Wall: %.1fs." % (SEEDS, blob["wall_s"]),
        "",
        "| cover | geom | pass | primary_mean | primary_min | leak_abs_max | knife |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for cover in COVERS:
        for geom in ("sheet", "field3d"):
            s = summary["cover_%.1f__%s" % (cover, geom)]
            lines.append(
                "| %.1f | %s | %s | %.4f | %.4f | %.4f | %s |"
                % (
                    cover,
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
        "## Recommended Music pole/cover",
        "",
        "- **cover_weight (toy)** = **%.1f**" % rec_out["cover_weight"],
        "- **Music `--pole_weight`** = **%.1f** (Arm B cover analogue)"
        % rec_out["music_pole_weight"],
    ]
    if rec_out.get("note"):
        lines.append("- Note: %s" % rec_out["note"])
    if rec_out.get("caution"):
        lines.append("- Caution: %s" % rec_out["caution"])
    if rec_out.get("rejected_reason"):
        lines.append("- Soft pick: %s" % rec_out["rejected_reason"])
    lines += [
        "",
        "## False-lock reject",
        "",
        "- High-cover (≥2.0) rejected when a lower modest cover already full-passes",
        "  sheet+Field3D (Fire #4 short×high-pin / cover3 pattern).",
        "- leftover_only (cover=0) known fail from dual_arm ablation — not re-swept.",
        "- Arm T lyric+tx is listen-only; do not claim #94 from lyrichold alone.",
        "",
        "## Paths",
        "",
        "- JSON: `arm_b_pole_cover_sweep_20260909.json`",
        "- Script: `arm_b_pole_cover_sweep_20260909.py`",
        "- Arm B smoke: `music_arm_b_locked_smoke_20260909.sh`",
        "- Arm T smoke: `music_locked_recipe_smoke_20260909.sh`",
        "- Strategy: `dual_arm_transfer_strategy_20260909.md`",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print("recommended cover/pole=", rec_out["cover_weight"], flush=True)
    print("wrote", OUT, MD, flush=True)


if __name__ == "__main__":
    main()
