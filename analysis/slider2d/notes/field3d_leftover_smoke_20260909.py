#!/usr/bin/env python3
"""Fire #9 (2D->3D): Field3D leftover smoke @ locked recipe 1200+c1.5.

Locked: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles=12, particle_l2=0.02, b_cap=1.

Ask: does the portable 2D leftover recipe transfer to an R3 leftover toy
(u-hat, content/intended-off-u, leftover e-hat; poles = neu +/- a)?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import leftover_field3d, score_adv_field3d
from analysis.slider2d.gan import default_cfg

OUT = Path(__file__).resolve().parent / "field3d_leftover_smoke_20260909.json"
MD = Path(__file__).resolve().parent / "field3d_leftover_smoke_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"


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
    rows = []
    t0 = time.time()
    print("Field3D leftover smoke @ locked 1200+c1.5", flush=True)
    print("seeds=%s teacher=%s" % (SEEDS, TEACHER), flush=True)
    field = leftover_field3d()
    print(
        "geometry dim=%d probe_cos=%.4f slider=%.2f content=%.2f leak=%.2f"
        % (field.dim, field.probe_cos(), field.slider, field.content, field.leak),
        flush=True,
    )
    for seed in SEEDS:
        st = time.time()
        row = score_adv_field3d(
            leftover_field3d(),
            teacher=TEACHER,
            cfg=locked_cfg(seed),
            name="field3d_s%d" % seed,
        )
        row["wall_s"] = round(time.time() - st, 2)
        rows.append(row)
        print(
            "seed=%2d pass=%s u_kept=%.4f content_kept=%.4f leak_ratio=%.4f "
            "resid=%.3f collapse=%.3f (%.1fs)"
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
    c_vals = [r["content_kept"] for r in rows]
    leak_vals = [r["leak_ratio"] for r in rows]
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
        "geometry": {
            "dim": field.dim,
            "axes": ["u_concept", "content_intended_off_u", "e_leftover", "lyric"],
            "poles": "neu +/- a",
            "slider": field.slider,
            "content": field.content,
            "leak": field.leak,
            "probe_cos": field.probe_cos(),
        },
        "seeds": SEEDS,
        "n_pass": n_pass,
        "n_total": len(SEEDS),
        "u_kept_mean": sum(u_vals) / len(u_vals),
        "u_kept_span": max(u_vals) - min(u_vals),
        "content_kept_mean": sum(c_vals) / len(c_vals),
        "content_kept_span": max(c_vals) - min(c_vals),
        "leak_ratio_mean": sum(leak_vals) / len(leak_vals),
        "leak_ratio_max": max(leak_vals),
        "wall_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2, default=float) + "\n")

    lines = [
        "# Field3D leftover smoke — 2026-09-09 (Fire #9 / 2D→3D)",
        "",
        "Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,",
        "fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.",
        "",
        "Geometry: orthonormal u-hat (concept), c-hat (content/intended-off-u), e-hat (leftover),",
        "plus one lyric row -> dim=4. Poles h+/- = neu +/- a (no common s-hat).",
        "",
        "Seeds: %s." % (SEEDS,),
        "",
        "| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | s |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| %d | %s | %.4f | %.4f | %.4f | %.3f | %.3f | %.1f |"
            % (
                r["seed"],
                "PASS" if r["pass"] else "FAIL",
                r["u_kept"],
                r["content_kept"],
                r["leak_ratio"],
                r["residual_norm"],
                r["collapse"],
                r["wall_s"],
            )
        )
    lines += [
        "",
        "Summary: **%d/%d PASS**; u_kept mean=%.4f span=%.4f; content_kept mean=%.4f;"
        % (n_pass, len(SEEDS), summary["u_kept_mean"], summary["u_kept_span"], summary["content_kept_mean"]),
        "leak_ratio mean=%.4f max=%.4f; wall=%.1fs."
        % (summary["leak_ratio_mean"], summary["leak_ratio_max"], summary["wall_s"]),
        "",
        "### Finding",
        "",
    ]
    if n_pass == len(SEEDS):
        lines.append(
            "- **LOCKED RECIPE TRANSFERS** to R³ leftover: û+content recovered, leftover ê gated."
        )
        verdict = "field3d_leftover_transfers"
        next_v = "harden Field3D gates / exam-port rollout on same geometry, or stress content vs leftover entanglement"
    elif n_pass == 0:
        lines.append(
            "- **Recipe does not transfer yet** — inspect which gate fails (u / content / leak)."
        )
        verdict = "field3d_leftover_broken"
        next_v = "diagnose gate failures; check faithful_guard_e on content-entangled a"
    else:
        lines.append(
            "- **Partial transfer** %d/%d — seed-sensitive; not portable yet."
            % (n_pass, len(SEEDS))
        )
        verdict = "field3d_leftover_fragile"
        next_v = "widen margin on failing gate or check particle/cover interaction in R3"
    fail_u = sum(1 for r in rows if not r["pass_u"])
    fail_c = sum(1 for r in rows if not r["pass_content"])
    fail_l = sum(1 for r in rows if not r["pass_leak"])
    lines += [
        "- Gate fails: u=%d content=%d leak=%d (of %d)."
        % (fail_u, fail_c, fail_l, len(SEEDS)),
        "- Verdict: `%s`" % verdict,
        "- Next: %s" % next_v,
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    summary["verdict"] = verdict
    summary["next"] = next_v
    OUT.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(
        "DONE %d/%d PASS verdict=%s wall=%.1fs -> %s"
        % (n_pass, len(SEEDS), verdict, summary["wall_s"], OUT),
        flush=True,
    )


if __name__ == "__main__":
    main()
