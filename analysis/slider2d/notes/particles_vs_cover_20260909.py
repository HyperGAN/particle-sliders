"""Fire #5: particle-count vs residual cover at locked 1200+cover1.5 leftover sheet."""
from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.slider2d.gan import default_cfg, fit_adv, as_sheet_residual
from analysis.slider2d.scoreboard import cell_works
from analysis.slider2d.sheet import (
    leaky_field,
    hidden_sheet_report,
    sheet_verdicts,
    teacher_swings,
)
from analysis.slider2d.field import cosine
from conceptmod.textsliders.slider_targets import leftover_bipolar

SEED = 0
TEACHER = "faithful_guard_e"
STEPS = 1200
COVER = 1.5
B_CAP = 1.0
FM = 0.0
PARTICLES = (4, 8, 12, 24)
OUT_JSON = Path("analysis/slider2d/notes/particles_vs_cover_20260909.json")
OUT_MD = Path("analysis/slider2d/notes/particles_vs_cover_20260909.md")


def _sheet_from_residual(field, residual, stats, *, name="rpgan_bcap"):
    head = field.readout()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = [field.poles(r)[2] + d_plus for r in range(int(field.rows))]
    minus = [field.poles(r)[2] + d_minus for r in range(int(field.rows))]
    row = hidden_sheet_report(field, plus, minus, readout=head)
    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    ceiling = teacher_swings(field, head)
    leftover = leftover_bipolar(d_plus, d_minus)
    on_u = float(d_plus @ field.short_u())
    row.update(
        {
            "name": name,
            "pole_mode": "rpgan_bcap",
            "teacher": TEACHER,
            "student": "odd_even",
            "hold_weight": 0.0,
            "common": float(field.common),
            "common_beta": 0.0,
            "probe_cos": field.probe_cos(0),
            "pair_odd_cos": cosine(d_plus, a),
            "collapse": cosine(d_plus, d_minus),
            "pole_cos": cosine(d_plus, pos - neu),
            "sheet_dir_kept": float(d_plus @ field.sheet_dir())
            / float(field.common_vec(0).norm() + 1e-8),
            "leak_hidden": abs(float(d_plus @ field.leak_e())) / (abs(on_u) + 1e-8),
            "swing_kept": row["concept_swing"] / (abs(ceiling["concept_swing"]) + 1e-8),
            "on_sheet_kept": row["on_sheet"] / (ceiling["on_sheet"] + 1e-8),
            "teacher_on_sheet": ceiling["on_sheet"],
            "teacher_garble": ceiling["garble"],
            "teacher_leak_tok": ceiling["leak_tok"],
            "leak_frac": leftover["leak_frac"],
            "same_dir": leftover["same_dir"],
            **{k: v for k, v in stats.items() if k != "log"},
        }
    )
    row["axis"] = sheet_verdicts(row)
    row["pass"] = all(v == "right" for v in row["axis"].values())
    return row


def _row(n_particles: int) -> dict:
    cfg = default_cfg(
        steps=STEPS,
        seed=SEED,
        b_cap=B_CAP,
        fm_weight=FM,
        cover_weight=COVER,
        n_particles=n_particles,
    )
    field = leaky_field()
    leak_dir = field.leak_e() if float(field.leak) > 1e-8 else None
    t0 = time.time()
    residual, stats = fit_adv(field, teacher=TEACHER, leak_dir=leak_dir, cfg=cfg)
    sheet = _sheet_from_residual(field, residual, stats)
    dt = time.time() - t0
    ok = cell_works(
        leak=sheet.get("leak_tok"),
        on_sheet_kept=sheet.get("on_sheet_kept"),
        off_sheet=sheet.get("garble"),
        argmax_on_sheet=sheet.get("argmax_on_sheet"),
        swing_kept=sheet.get("swing_kept"),
    )
    odd_n = float(residual.w_odd.norm())
    even_n = float(residual.w_even.norm())
    res_n = float((residual.w_odd.pow(2).sum() + residual.w_even.pow(2).sum()).sqrt())
    row = {
        "n_particles": n_particles,
        "seed": SEED,
        "steps": STEPS,
        "cover_weight": COVER,
        "b_cap": B_CAP,
        "fm_weight": FM,
        "teacher": TEACHER,
        "leak_tok": sheet.get("leak_tok"),
        "on_sheet_kept": sheet.get("on_sheet_kept"),
        "garble": sheet.get("garble"),
        "swing_kept": sheet.get("swing_kept"),
        "argmax_on_sheet": sheet.get("argmax_on_sheet"),
        "pole_rel_err_plus": stats.get("pole_rel_err_plus"),
        "pole_rel_err_minus": stats.get("pole_rel_err_minus"),
        "pole_cos_plus": stats.get("pole_cos_plus"),
        "pole_cos_minus": stats.get("pole_cos_minus"),
        "covered": stats.get("covered"),
        "residual_norm": res_n,
        "w_odd_norm": odd_n,
        "w_even_norm": even_n,
        "g_loss": stats.get("g_loss"),
        "pass": bool(ok),
        "sheet_pass_internal": bool(sheet.get("pass")),
        "sec": round(dt, 2),
    }
    print(
        "n_particles=%-3d kept=%.4f leak=%+.4f covered=%s res_n=%.4f odd=%.4f even=%.4f pass=%s (%.1fs)"
        % (
            n_particles,
            float(row["on_sheet_kept"] or 0),
            float(row["leak_tok"] or 0),
            row["covered"],
            res_n,
            odd_n,
            even_n,
            ok,
            dt,
        ),
        flush=True,
    )
    return row


def main() -> None:
    print(
        "=== particles vs residual cover @%d cover=%.1f leftover sheet seed=%d ==="
        % (STEPS, COVER, SEED),
        flush=True,
    )
    rows = [_row(n) for n in PARTICLES]
    kepts = [float(r["on_sheet_kept"]) for r in rows]
    summary = {
        "n": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "kept_mean": sum(kepts) / len(kepts),
        "kept_min": min(kepts),
        "kept_max": max(kepts),
        "kept_span": max(kepts) - min(kepts),
    }
    payload = {
        "sha_hint": "435e873",
        "fire": 5,
        "recipe": "locked 1200+cover1.5 leftover sheet; sweep n_particles",
        "summary": summary,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Particle-count vs residual cover (2026-09-09 fire #5)",
        "",
        "SHA `435e873` pop-os `/ml2/music/sliders-conceptmod` CPU. GPUs left alone.",
        "",
        "Locked recipe: steps=1200, cover_weight=1.5, teacher=`faithful_guard_e`,",
        "b_cap=1.0, fm=0, seed=0. Sweep `n_particles` ∈ {4, 8, 12, 24} (default 12).",
        "",
        "## leftover sheet",
        "",
        "| n_particles | kept | leak | garble | swing | covered | residual_norm | w_odd | w_even | pass |",
        "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            "| %d | %.4f | %+.4f | %.4f | %.4f | %s | %.4f | %.4f | %.4f | %s |"
            % (
                r["n_particles"],
                float(r["on_sheet_kept"]),
                float(r["leak_tok"] or 0),
                float(r["garble"] or 0),
                float(r["swing_kept"] or 0),
                "yes" if r["covered"] else "no",
                float(r["residual_norm"]),
                float(r["w_odd_norm"]),
                float(r["w_even_norm"]),
                "PASS" if r["pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "Summary: **%d/%d PASS**, kept mean `%.4f`, span `%.4f` (min `%.4f`, max `%.4f`)."
        % (
            summary["n_pass"],
            summary["n"],
            summary["kept_mean"],
            summary["kept_span"],
            summary["kept_min"],
            summary["kept_max"],
        ),
        "",
        "## Takeaway",
        "",
    ]
    span = summary["kept_span"]
    if span < 0.005 and summary["n_pass"] == summary["n"]:
        lines.append(
            "Particle count is **nullspace** at the locked recipe: kept stays ≥0.90 "
            "across 4–24 particles (span < 0.005). Do not chase n_particles; keep default 12."
        )
    elif summary["n_pass"] < summary["n"]:
        fails = [r["n_particles"] for r in rows if not r["pass"]]
        lines.append(
            "Particle count **moves the leftover gate**: FAIL at n_particles=%s. "
            "Prefer the PASS cells; check whether low count starves cover or high count steals the residual."
            % fails
        )
    else:
        lines.append(
            "All PASS; see kept/residual_norm trend vs n_particles for whether particles steal mode from residual."
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_JSON, OUT_MD, flush=True)
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
