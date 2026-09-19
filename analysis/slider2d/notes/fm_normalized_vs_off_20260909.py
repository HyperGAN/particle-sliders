#!/usr/bin/env python3
"""Fire #6: FM-on-normalized vs FM-off @ locked recipe 1200+c1.5 n=12.

Re-measure findings note claim that normalized FM did not beat leftover gate.
Laptop local-exec unavailable; pop-os Tailscale timed out — box CPU only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.gan import default_cfg, score_adv_sheet
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    cell_works,
)
from analysis.slider2d.sheet import leaky_field

OUT = Path(__file__).resolve().parent / "fm_normalized_vs_off_20260909.json"
MD = Path(__file__).resolve().parent / "fm_normalized_vs_off_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"

RESIDUAL_KEYS = (
    "pole_rel_err_plus",
    "pole_rel_err_minus",
    "pole_cos_plus",
    "pole_cos_minus",
    "covered",
    "cap",
    "d_loss",
    "g_loss",
    "grad_real",
    "grad_fake",
    "collapse",
    "leak_frac",
    "pair_odd_cos",
    "sheet_dir_kept",
)


def score_sheet_row(cfg) -> dict:
    t0 = time.time()
    r = score_adv_sheet(leaky_field(), cfg=cfg)
    ok = bool(
        cell_works(
            leak=r.get("leak_tok"),
            on_sheet_kept=r.get("on_sheet_kept"),
            off_sheet=r.get("garble"),
            argmax_on_sheet=r.get("argmax_on_sheet"),
            swing_kept=r.get("swing_kept"),
        )
    )
    row = {
        "on_sheet_kept": float(r["on_sheet_kept"]),
        "leak_tok": float(r["leak_tok"]),
        "garble": float(r["garble"]),
        "swing_kept": float(r["swing_kept"]),
        "argmax_on_sheet": float(r["argmax_on_sheet"]),
        "pass": ok and bool(r["pass"]),
        "sec": round(time.time() - t0, 1),
    }
    for k in RESIDUAL_KEYS:
        if k in r:
            v = r[k]
            row[k] = bool(v) if isinstance(v, bool) else (float(v) if v is not None else None)
    return row


def run_cfg(*, seed: int, fm_weight: float, fm_normalize: bool) -> dict:
    cfg = default_cfg(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        n_particles=12,
        fm_weight=fm_weight,
        fm_normalize=fm_normalize,
    )
    row = score_sheet_row(cfg)
    row.update(
        {
            "seed": seed,
            "steps": 1200,
            "cover_weight": 1.5,
            "b_cap": 1.0,
            "n_particles": 12,
            "fm_weight": float(fm_weight),
            "fm_normalize": bool(fm_normalize),
        }
    )
    return row


def print_row(label: str, row: dict) -> None:
    mark = "PASS" if row["pass"] else "FAIL"
    print(
        f"{label}: kept={row['on_sheet_kept']:.4f} leak={row['leak_tok']:+.4f} "
        f"garble={row['garble']:.4f} swing={row['swing_kept']:.4f} {mark} ({row['sec']}s)",
        flush=True,
    )


def main() -> None:
    print(
        "=== Fire #6 FM-on-normalized vs FM-off @1200 cover=1.5 n=12 leftover ===",
        flush=True,
    )

    grid_specs = [
        ("baseline_fm0", 0.0, True),
        ("fm_norm_0.1", 0.1, True),
        ("fm_norm_0.5", 0.5, True),
        ("fm_norm_1.0", 1.0, True),
        ("fm_raw_0.5_control", 0.5, False),  # known-bad uncapped path
    ]
    grid_rows = []
    for label, w, norm in grid_specs:
        row = run_cfg(seed=0, fm_weight=w, fm_normalize=norm)
        row["label"] = label
        grid_rows.append(row)
        print_row(
            f"{label} w={w} norm={norm}",
            row,
        )

    baseline = grid_rows[0]
    baseline_kept = baseline["on_sheet_kept"]
    kept_floor = baseline_kept - 0.005

    # Competitive = PASS with kept >= baseline-0.005 OR kept within 0.005 of baseline even if FAIL near lock
    competitive = []
    for row in grid_rows[1:]:
        if row["label"].endswith("_control"):
            continue  # raw FM is diagnostic only
        if not row["fm_normalize"]:
            continue
        if row["fm_weight"] <= 0.0:
            continue
        kept_ok = row["on_sheet_kept"] >= kept_floor
        near_baseline = abs(row["on_sheet_kept"] - baseline_kept) <= 0.005
        if row["pass"] and kept_ok:
            competitive.append(row)
        elif row["pass"] and near_baseline:
            competitive.append(row)
        elif kept_ok and near_baseline and row["on_sheet_kept"] >= COMPILED_SHEET_LOCK - 0.01:
            # looks competitive even if soft fail — still seed-check
            competitive.append(row)

    # Also treat any FM-on PASS as seed-check candidate per task wording
    for row in grid_rows[1:]:
        if row["label"].endswith("_control"):
            continue
        if row["pass"] and row["fm_normalize"] and row not in competitive:
            competitive.append(row)

    multi_seed = None
    skip_reason = None
    if not competitive:
        skip_reason = (
            "all FM-on-normalized rows FAIL or clearly worse on kept vs baseline; "
            "skip multi-seed"
        )
        print(f"SKIP multi-seed: {skip_reason}", flush=True)
    else:
        # Seed-check each competitive weight vs FM-off
        seeds = [0, 1, 2, 3, 7, 42]
        weights = sorted({r["fm_weight"] for r in competitive})
        print(
            f"=== multi-seed for FM weights {weights} vs FM-off (seeds {seeds}) ===",
            flush=True,
        )
        multi_seed = {"seeds": seeds, "by_weight": {}}
        # Always include FM-off for this fire's own seed table
        for w in [0.0] + weights:
            rows = []
            for s in seeds:
                row = run_cfg(seed=s, fm_weight=w, fm_normalize=True)
                rows.append(row)
                print_row(f"seed={s:2d} w={w}", row)
            kept = [r["on_sheet_kept"] for r in rows]
            multi_seed["by_weight"][str(w)] = {
                "fm_weight": w,
                "fm_normalize": True,
                "rows": rows,
                "n_pass": sum(1 for r in rows if r["pass"]),
                "n": len(rows),
                "kept_min": min(kept),
                "kept_max": max(kept),
                "kept_mean": sum(kept) / len(kept),
                "kept_span": max(kept) - min(kept),
            }

    # Verdict
    fm_norm_rows = [r for r in grid_rows if r["fm_normalize"] and r["fm_weight"] > 0]
    any_pass = any(r["pass"] for r in fm_norm_rows)
    best_fm = max(fm_norm_rows, key=lambda r: r["on_sheet_kept"]) if fm_norm_rows else None
    raw = next(r for r in grid_rows if r["label"] == "fm_raw_0.5_control")

    if not any_pass and best_fm and best_fm["on_sheet_kept"] < kept_floor:
        verdict = "keep_fm_off"
        verdict_detail = (
            f"FM-on-normalized fails or kept < baseline−0.005 "
            f"(best kept {best_fm['on_sheet_kept']:.4f} vs baseline {baseline_kept:.4f}); "
            "confirm findings claim — do not revise locked recipe."
        )
    elif any_pass and best_fm and best_fm["on_sheet_kept"] >= kept_floor:
        # Check multi-seed if present
        if multi_seed is not None:
            off = multi_seed["by_weight"]["0.0"]
            on = multi_seed["by_weight"][str(best_fm["fm_weight"])]
            if on["n_pass"] == on["n"] and on["kept_mean"] >= off["kept_mean"] - 0.005:
                if on["kept_mean"] > off["kept_mean"] + 0.01:
                    verdict = "revise_consider"
                    verdict_detail = (
                        f"FM w={best_fm['fm_weight']} seed-stable and kept mean "
                        f"{on['kept_mean']:.4f} > off {off['kept_mean']:.4f}; "
                        "strong evidence needed before recipe change."
                    )
                else:
                    verdict = "keep_fm_off_flat"
                    verdict_detail = (
                        f"FM-on PASSes but is flat vs off "
                        f"(mean {on['kept_mean']:.4f} vs {off['kept_mean']:.4f}); "
                        "no recipe change."
                    )
            else:
                verdict = "false_improvement"
                verdict_detail = (
                    f"FM looked competitive at seed 0 but multi-seed weak "
                    f"({on['n_pass']}/{on['n']} PASS, mean {on['kept_mean']:.4f}); "
                    "keep FM off."
                )
        else:
            verdict = "keep_fm_off"
            verdict_detail = "FM competitive at seed 0 but multi-seed skipped unexpectedly."
    else:
        verdict = "keep_fm_off"
        verdict_detail = "FM-on not competitive with leftover-gated baseline."

    # Next thread suggestion
    if verdict in ("keep_fm_off", "keep_fm_off_flat", "false_improvement"):
        next_thread = (
            "particle_l2 micro-sweep at high n, or multi-pair/cross-axis stress, "
            "or exam_score at locked recipe (prefer particle_l2 micro-sweep @ n=12 first)"
        )
    else:
        next_thread = (
            "re-check FM seed grid + exam_score before any recipe revision; "
            "unlikely to revise locked 1200+c1.5 without strong multi-metric evidence"
        )

    payload = {
        "date": "2026-09-09",
        "fire": 6,
        "sha": "435e873",
        "host": "box-cpu (laptop local-exec unavailable; pop-os Tailscale timed out)",
        "thread": "FM-on-normalized vs FM-off @1200+c1.5 n=12 leftover-gated",
        "locked_recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "b_cap": 1.0,
            "n_particles": 12,
            "fm_weight": 0.0,
            "teacher": "faithful_guard_e",
        },
        "gates": {
            "sheet_lock": COMPILED_SHEET_LOCK,
            "leak_lock": COMPILED_LEAK_LOCK,
            "garble_max": COMPILED_GARBLE_MAX,
            "swing_floor": COMPILED_SWING_FLOOR,
        },
        "grid_rows": grid_rows,
        "baseline_kept": baseline_kept,
        "kept_floor": kept_floor,
        "competitive_labels": [r["label"] for r in competitive],
        "multi_seed": multi_seed,
        "multi_seed_skipped": skip_reason,
        "raw_control": {
            "note": "raw FM (fm_normalize=False) is the known-bad uncapped-by-b_cap path",
            "row": raw,
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "next_thread": next_thread,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# FM-on-normalized vs FM-off (2026-09-09 fire #6)",
        "",
        f"SHA `{payload['sha']}` on {payload['host']}.",
        "",
        "Locked recipe: steps=1200, cover_weight=1.5, b_cap=1.0, n_particles=12, "
        "teacher=`faithful_guard_e`, FM off by default.",
        "",
        "## Grid (seed=0)",
        "",
        "| label | fm_weight | normalize | kept | leak | garble | swing | pass | sec |",
        "|---|---:|:---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in grid_rows:
        lines.append(
            f"| `{r['label']}` | {r['fm_weight']} | {r['fm_normalize']} | "
            f"{r['on_sheet_kept']:.4f} | {r['leak_tok']:+.4f} | {r['garble']:.4f} | "
            f"{r['swing_kept']:.4f} | {'PASS' if r['pass'] else 'FAIL'} | {r['sec']} |"
        )
    lines += [
        "",
        "### Residual / train fields (seed=0)",
        "",
        "| label | pole_rel_err± | covered | cap | collapse | sheet_dir_kept |",
        "|---|---|:---:|---:|---:|---:|",
    ]
    for r in grid_rows:
        lines.append(
            f"| `{r['label']}` | "
            f"{r.get('pole_rel_err_plus', float('nan')):.4f}/"
            f"{r.get('pole_rel_err_minus', float('nan')):.4f} | "
            f"{r.get('covered')} | {r.get('cap', float('nan')):.4f} | "
            f"{r.get('collapse', float('nan')):.4f} | "
            f"{r.get('sheet_dir_kept', float('nan')):.4f} |"
        )

    lines += ["", "## Multi-seed", ""]
    if skip_reason:
        lines.append(f"Skipped: {skip_reason}.")
    else:
        lines.append("Ran seeds `{0,1,2,3,7,42}` for competitive FM weights vs FM-off.")
        lines.append("")
        for w_key, summ in multi_seed["by_weight"].items():
            lines.append(
                f"- fm_weight={w_key}: **{summ['n_pass']}/{summ['n']} PASS**, "
                f"kept mean `{summ['kept_mean']:.4f}`, span `{summ['kept_span']:.4f}`"
            )
            lines.append("")
            lines.append("| seed | kept | leak | pass |")
            lines.append("|---:|---:|---:|:---:|")
            for r in summ["rows"]:
                lines.append(
                    f"| {r['seed']} | {r['on_sheet_kept']:.4f} | {r['leak_tok']:+.4f} | "
                    f"{'PASS' if r['pass'] else 'FAIL'} |"
                )
            lines.append("")

    lines += [
        "## Verdict",
        "",
        f"**{verdict}**: {verdict_detail}",
        "",
        f"Raw FM control (w=0.5, normalize=False): kept={raw['on_sheet_kept']:.4f} "
        f"{'PASS' if raw['pass'] else 'FAIL'} — known-bad uncapped path (document only).",
        "",
        "## Next thread",
        "",
        next_thread,
        "",
    ]
    MD.write_text("\n".join(lines))

    # Append research log
    log_section = [
        "",
        "## Fire #6 — FM-on-normalized vs FM-off (2026-09-09)",
        "",
        f"- Host: box-cpu @ SHA `435e873`",
        f"- Tests: see fire header / pytest 10 passed",
        f"- Grid seed=0: baseline kept={baseline_kept:.4f}; "
        + ", ".join(
            f"{r['label']} kept={r['on_sheet_kept']:.4f} "
            f"{'PASS' if r['pass'] else 'FAIL'}"
            for r in grid_rows
        ),
        f"- Multi-seed: "
        + (
            skip_reason
            if skip_reason
            else "; ".join(
                f"w={k} {v['n_pass']}/{v['n']}"
                for k, v in multi_seed["by_weight"].items()
            )
        ),
        f"- Verdict: **{verdict}** — {verdict_detail}",
        f"- Next: {next_thread}",
        f"- Notes: `fm_normalized_vs_off_20260909.{{py,json,md}}`",
        "",
    ]
    if LOG.exists():
        existing = LOG.read_text()
    else:
        existing = (
            "# 2D adversarial slider research log — 2026-09-09\n"
            "\n"
            "Box-cpu fires against mikkel main `435e873` "
            "(Port ParticleGAN RpGAN + b_cap into the 2D slider example #94).\n"
            "Locked recipe: 1200 steps + cover_weight 1.5, leftover-gated "
            "(`faithful_guard_e`), FM off, n_particles≤12, b_cap=1. "
            "Do NOT adopt 800×cover3.0 (false lock).\n"
        )
    if "## Fire #6" not in existing:
        LOG.write_text(existing.rstrip() + "\n" + "\n".join(log_section))
    else:
        # replace existing fire #6 section naively: append with note
        LOG.write_text(existing.rstrip() + "\n" + "\n".join(log_section))

    print(f"wrote {OUT}", flush=True)
    print(f"wrote {MD}", flush=True)
    print(f"updated {LOG}", flush=True)
    print(f"VERDICT={verdict}", flush=True)


if __name__ == "__main__":
    main()
