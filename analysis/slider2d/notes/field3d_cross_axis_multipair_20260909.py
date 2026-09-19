#!/usr/bin/env python3
"""Fire #13: Field3D cross-axis multipair stress (transfer-first).

Locked: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles=12, particle_l2=0.02, b_cap=1. Skip 800xcover3.0.

Ask: does the locked recipe survive TRUE cross-axis geometry in R3 —
per-row different (slider,content,leak) mixes via Field3D.row_amps —
beyond Fire #12 PairField multi-row exam and Fire #13 hard_multipair
amplitude/declare mismatch cells?

Transfer map (Music LM):
- leftover gate / faithful_guard_e  -> --recipe faithful_guard_e
- cover vs residual                 -> cover_weight / --pole_weight
- b_cap                             -> b_cap=1
- particle budget                   -> n_particles (toy) vs --parts 0 (Music)
- FM off                            -> fm_weight=0 / --txfm_weight 0
- lm_adv                            -> residual+RpGAN core
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (
    Field3D,
    axis_content_primary_field3d,
    axis_leak_primary_field3d,
    axis_u_primary_field3d,
    close_field3d,
    cross_axis_mismatch_declare_field3d,
    cross_axis_rows_field3d,
    cross_axis_span_sample_field3d,
    divergent_field3d,
    leftover_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
    unused_e_field3d,
)
from analysis.slider2d.gan import default_cfg

OUT = Path(__file__).resolve().parent / "field3d_cross_axis_multipair_20260909.json"
MD = Path(__file__).resolve().parent / "field3d_cross_axis_multipair_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"

SEEDS_SMOKE = [0, 1]
SEEDS_FULL = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"

CELLS = {
    # baselines (should still pass — regression check)
    "f3d_leftover": ("score", leftover_field3d),
    "f3d_divergent": ("exam", divergent_field3d),
    "f3d_close": ("exam", close_field3d),
    "f3d_unused_e": ("exam", unused_e_field3d),
    # single-axis families
    "axis_u_primary": ("exam", axis_u_primary_field3d),
    "axis_content_primary": ("exam", axis_content_primary_field3d),
    "axis_leak_primary": ("exam", axis_leak_primary_field3d),
    # true cross-axis + live-like span
    "cross_axis_rows": ("exam", cross_axis_rows_field3d),
    "cross_axis_span_sample": ("exam", cross_axis_span_sample_field3d),
    # negative control (YAML lie) — expect fail, not knife-edge false lock
    "cross_axis_mismatch_declare": ("exam", cross_axis_mismatch_declare_field3d),
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


def run_one(cell: str, mode: str, ctor, seed: int) -> dict:
    t0 = time.time()
    field = ctor(seed=seed)
    name = "f3d_cross_%s_s%d" % (cell, seed)
    if mode == "exam":
        row = score_adv_field3d_exam(
            field, teacher=TEACHER, cfg=locked_cfg(seed), name=name
        )
        primary = float(row.get("exam_score", row.get("u_kept", 0.0)))
        passed = bool(row.get("exam_pass", row.get("pass")))
    else:
        row = score_adv_field3d(
            field, teacher=TEACHER, cfg=locked_cfg(seed), name=name
        )
        primary = float(row.get("u_kept", 0.0))
        passed = bool(row.get("pass"))
    out = {
        "cell": cell,
        "mode": mode,
        "seed": seed,
        "pass": passed,
        "primary": primary,
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "exam_score": float(row.get("exam_score", primary)),
        "exam_cont": float(row.get("exam_cont", 0.0)),
        "exam_swing": float(row.get("exam_swing", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "residual_norm": float(row.get("residual_norm", 0.0)),
        "even_norm": float(row.get("even_norm", 0.0)),
        "odd_norm": float(row.get("odd_norm", 0.0)),
        "wall_s": round(time.time() - t0, 2),
        "kind": getattr(field, "kind", cell),
        "dim": int(getattr(field, "dim", 0)),
        "rows": int(getattr(field, "rows", 0)),
    }
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    prim = [r["primary"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    uk = [r["u_kept"] for r in rows]
    span = (max(prim) - min(prim)) if prim else 0.0
    # knife_edge: pass rate in {1/6..5/6} or tiny margin near gate
    knife = False
    if 0 < n_pass < n:
        knife = True
    elif n_pass == n and prim:
        # near floor
        if min(prim) < 0.87:
            knife = True
    return {
        "n_pass": n_pass,
        "n_seeds": n,
        "pass_str": "%d/%d" % (n_pass, n),
        "primary_mean": sum(prim) / n if n else 0.0,
        "primary_span": span,
        "u_kept_mean": sum(uk) / n if n else 0.0,
        "leak_max": max(leaks) if leaks else 0.0,
        "knife_edge": knife,
        "pass": n_pass == n and not knife,
    }


def write_md(payload: dict) -> None:
    lines = []
    lines.append("# Field3D cross-axis multipair @ locked 1200+c1.5 (Fire #13)")
    lines.append("")
    lines.append(
        "Host: pop-os CPU @ `%s`. Locked leftover recipe; FM off; n=12; l2=0.02; b_cap=1."
        % payload["sha"]
    )
    lines.append("Seeds: `%s`." % payload["seeds"])
    lines.append("")
    lines.append("## Ask")
    lines.append("")
    lines.append(
        "Does the locked recipe survive **true cross-axis** R3 geometry "
        "(`Field3D.row_amps` — different rows on different axis mixes), "
        "beyond PairField multi-row exam (#12) and hard_multipair declare/amplitude mismatch?"
    )
    lines.append("")
    lines.append("## Per-cell summary")
    lines.append("")
    lines.append(
        "| cell | PASS | primary mean | primary span | u_kept mean | leak abs max | knife_edge |"
    )
    lines.append("|:---|:---:|---:|---:|---:|---:|:---:|")
    for cell, s in payload["by_cell"].items():
        lines.append(
            "| %s | %s | %.4f | %.4f | %.4f | %.4f | %s |"
            % (
                cell,
                s["pass_str"],
                s["primary_mean"],
                s["primary_span"],
                s["u_kept_mean"],
                s["leak_max"],
                "yes" if s["knife_edge"] else "no",
            )
        )
    lines.append("")
    lines.append("## Finding")
    lines.append("")
    lines.append("- Verdict: **`%s`**" % payload["verdict"])
    lines.append("- Wall: %.1fs" % payload["wall_s"])
    lines.append("- %s" % payload["finding"])
    lines.append("")
    lines.append("## Transfer interpretation (Music LM)")
    lines.append("")
    lines.append("| Toy knob | Music LM knob | Lesson |")
    lines.append("|---|---|---|")
    lines.append("| `faithful_guard_e` | leftover gate / `--recipe` | refuse bad `leak_*` YAML; honest unused_e subtracts |")
    lines.append("| `cover_weight=1.5` | `--pole_weight` (~1.0 Music posture) | cover still required with gate |")
    lines.append("| `b_cap=1` | `b_cap=1` | keep |")
    lines.append("| `n_particles=12` | `--parts 0` (do not map) | particle budget is toy; Music stays parts0 |")
    lines.append("| `fm_weight=0` | FM / txfm off | keep FM-off |")
    lines.append("| residual RpGAN | `lm_adv` mlp | transferable core |")
    lines.append("")
    lines.append("### Cover vs residual under cross-axis")
    lines.append("")
    for cell, s in payload["by_cell"].items():
        if cell.startswith("cross_axis") or cell.startswith("axis_"):
            lines.append(
                "- `%s`: primary=%.4f leak_max=%.4f pass=%s"
                % (cell, s["primary_mean"], s["leak_max"], s["pass_str"])
            )
    lines.append("")
    lines.append("## Recipe change?")
    lines.append("")
    lines.append(payload.get("recipe_change", "None — document transfer risk only."))
    lines.append("")
    MD.write_text("\n".join(lines) + "\n")


def append_log(payload: dict) -> None:
    block = []
    block.append("")
    block.append("## Fire #13 — cross-axis multipair R3 (2026-09-09)")
    block.append("")
    block.append("- Host: pop-os-cpu @ SHA `%s`" % payload["sha"])
    block.append("- Notes: `field3d_cross_axis_multipair_20260909.{py,json,md}`")
    block.append("- Seeds: %s; locked 1200+c1.5 / faithful_guard_e / FM0 / n=12 / b_cap=1" % payload["seeds"])
    for cell, s in payload["by_cell"].items():
        block.append(
            "  - `%s`: **%s** primary mean=%.4f span=%.4f leak_max=%.4f knife=%s"
            % (
                cell,
                s["pass_str"],
                s["primary_mean"],
                s["primary_span"],
                s["leak_max"],
                s["knife_edge"],
            )
        )
    block.append("- Verdict: **%s** — %s" % (payload["verdict"], payload["finding"]))
    block.append("- Wall: %.1fs" % payload["wall_s"])
    block.append("")
    block.append("### STATUS (Fire #13 cross-axis)")
    block.append("- %s" % payload["finding"])
    block.append("- Next: %s" % payload.get("next", "dual_arm / Music leftover-gate arm smoke"))
    block.append("")
    with LOG.open("a") as f:
        f.write("\n".join(block))


def main() -> None:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="seeds {0,1} only")
    ap.add_argument(
        "--cells",
        default="",
        help="comma-separated cell subset (default: all)",
    )
    args = ap.parse_args()
    seeds = SEEDS_SMOKE if args.smoke else SEEDS_FULL
    cell_names = (
        [c.strip() for c in args.cells.split(",") if c.strip()]
        if args.cells
        else list(CELLS.keys())
    )
    sha = (
        subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO)
        .decode()
        .strip()
    )
    t_all = time.time()
    detail = []
    by_cell = {}
    print("Fire #13 cross-axis multipair sha=%s seeds=%s cells=%s" % (sha, seeds, cell_names), flush=True)
    for cell in cell_names:
        mode, ctor = CELLS[cell]
        rows = []
        print("--- cell=%s mode=%s" % (cell, mode), flush=True)
        for seed in seeds:
            row = run_one(cell, mode, ctor, seed)
            rows.append(row)
            detail.append(row)
            print(
                "  seed=%2d pass=%s u=%.4f c=%.4f leak=%.4f exam=%.4f rows=%d/%d (%.1fs)"
                % (
                    seed,
                    row["pass"],
                    row["u_kept"],
                    row["content_kept"],
                    row["leak_ratio"],
                    row["exam_score"],
                    row["rows_covered"],
                    row["rows_total"],
                    row["wall_s"],
                ),
                flush=True,
            )
        by_cell[cell] = summarize(rows)
        print(
            "  >> %s primary_mean=%.4f leak_max=%.4f knife=%s"
            % (
                by_cell[cell]["pass_str"],
                by_cell[cell]["primary_mean"],
                by_cell[cell]["leak_max"],
                by_cell[cell]["knife_edge"],
            ),
            flush=True,
        )

    # Verdict logic
    expect_pass = [
        c
        for c in cell_names
        if c != "cross_axis_mismatch_declare"
    ]
    expect_fail = [c for c in cell_names if c == "cross_axis_mismatch_declare"]
    solid_pass = all(by_cell[c]["n_pass"] == by_cell[c]["n_seeds"] and not by_cell[c]["knife_edge"] for c in expect_pass if c in by_cell)
    solid_neg = all(by_cell[c]["n_pass"] == 0 and not by_cell[c]["knife_edge"] for c in expect_fail if c in by_cell)
    cross_keys = [c for c in ("cross_axis_rows", "cross_axis_span_sample", "axis_u_primary", "axis_content_primary", "axis_leak_primary") if c in by_cell]
    cross_ok = all(by_cell[c]["n_pass"] == by_cell[c]["n_seeds"] for c in cross_keys)
    fails = [c for c in expect_pass if c in by_cell and by_cell[c]["n_pass"] < by_cell[c]["n_seeds"]]
    knives = [c for c, s in by_cell.items() if s["knife_edge"]]

    if solid_pass and (not expect_fail or solid_neg) and cross_ok:
        verdict = "cross_axis_portable"
        finding = (
            "Locked recipe survives true cross-axis row_amps + span sampling; "
            "mismatch_declare negative control fails solidly (not a false lock)."
        )
        recipe_change = "None — portable core unchanged."
    elif fails or knives:
        verdict = "cross_axis_partial"
        finding = (
            "Locked recipe fails or knife-edges on: %s. Document as transfer risk; "
            "do not weaken recipe without multi-seed margin."
            % (", ".join(fails + knives) if (fails or knives) else "none")
        )
        recipe_change = "None adopted — document failing cells only."
    else:
        verdict = "cross_axis_partial"
        finding = "Mixed outcome; see per-cell grid."
        recipe_change = "None."

    payload = {
        "fire": 13,
        "date": "2026-09-09",
        "sha": sha,
        "recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "seeds": seeds,
        "by_cell": by_cell,
        "detail": detail,
        "verdict": verdict,
        "finding": finding,
        "recipe_change": recipe_change,
        "wall_s": round(time.time() - t_all, 1),
        "next": "Music leftover-gate arm smoke / dual_arm cover@n=1 confirmation",
        "transfer": {
            "lm_adv": "residual RpGAN core holds on single-axis families when pass",
            "leftover_gate": "faithful_guard_e",
            "cover_vs_residual": "cover_weight 1.5 toy; Music pole~1.0 posture",
            "b_cap": 1,
            "particle_budget": "toy n=12; Music --parts 0 (do not map)",
            "fm_off": True,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    write_md(payload)
    append_log(payload)
    print("VERDICT", verdict, "wall=%.1fs" % payload["wall_s"], flush=True)
    print("wrote", OUT, MD, LOG, flush=True)


if __name__ == "__main__":
    main()
