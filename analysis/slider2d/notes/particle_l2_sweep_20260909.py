#!/usr/bin/env python3
"""Fire #7: particle_l2 micro-sweep @ locked recipe 1200+c1.5 n=12 leftover.

Default particle_l2=0.02. Check whether raising/lowering L2 at high n
helps kept / residual without breaking seed stability.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.gan import default_cfg, score_adv_sheet
from analysis.slider2d.scoreboard import cell_works
from analysis.slider2d.sheet import leaky_field

OUT = Path(__file__).resolve().parent / "particle_l2_sweep_20260909.json"
MD = Path(__file__).resolve().parent / "particle_l2_sweep_20260909.md"
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
    "residual_norm",
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


def run_cfg(*, seed: int, particle_l2: float, n_particles: int = 12) -> dict:
    cfg = default_cfg(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        n_particles=n_particles,
        fm_weight=0.0,
        particle_l2=particle_l2,
    )
    row = score_sheet_row(cfg)
    row.update(
        {
            "seed": seed,
            "steps": 1200,
            "cover_weight": 1.5,
            "b_cap": 1.0,
            "n_particles": n_particles,
            "fm_weight": 0.0,
            "particle_l2": float(particle_l2),
        }
    )
    return row


def print_row(label: str, row: dict) -> None:
    mark = "PASS" if row["pass"] else "FAIL"
    rnorm = row.get("residual_norm")
    rnorm_s = f" rnorm={rnorm:.4f}" if isinstance(rnorm, float) else ""
    print(
        f"{label}: kept={row['on_sheet_kept']:.4f} leak={row['leak_tok']:+.4f} "
        f"garble={row['garble']:.4f} swing={row['swing_kept']:.4f}{rnorm_s} {mark} ({row['sec']}s)",
        flush=True,
    )


def summarize(rows: list[dict]) -> dict:
    kept = [r["on_sheet_kept"] for r in rows]
    rnorms = [r["residual_norm"] for r in rows if isinstance(r.get("residual_norm"), float)]
    out = {
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n": len(rows),
        "kept_min": min(kept),
        "kept_max": max(kept),
        "kept_mean": sum(kept) / len(kept),
        "kept_span": max(kept) - min(kept),
    }
    if rnorms:
        out["residual_norm_mean"] = sum(rnorms) / len(rnorms)
        out["residual_norm_min"] = min(rnorms)
        out["residual_norm_max"] = max(rnorms)
    return out


def main() -> None:
    print(
        "=== Fire #7 particle_l2 micro-sweep @1200 cover=1.5 n=12 leftover ===",
        flush=True,
    )

    l2_grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    grid_rows = []
    for l2 in l2_grid:
        row = run_cfg(seed=0, particle_l2=l2)
        row["label"] = f"l2_{l2}"
        grid_rows.append(row)
        print_row(f"seed0 l2={l2}", row)

    baseline = next(r for r in grid_rows if abs(r["particle_l2"] - 0.02) < 1e-12)
    baseline_kept = baseline["on_sheet_kept"]

    # Always seed-check default (0.02) and any non-default that PASSes within 0.01 kept
    # or clearly better (+0.005), plus l2=0 (ablation) and the best kept.
    seeds = [0, 1, 2, 3, 7, 42]
    candidates = {0.02}
    best = max(grid_rows, key=lambda r: r["on_sheet_kept"])
    candidates.add(best["particle_l2"])
    candidates.add(0.0)  # ablation
    for r in grid_rows:
        if r["pass"] and abs(r["on_sheet_kept"] - baseline_kept) <= 0.01:
            candidates.add(r["particle_l2"])
        if r["pass"] and r["on_sheet_kept"] >= baseline_kept + 0.005:
            candidates.add(r["particle_l2"])

    # Cap seed-check set to avoid huge wall time: prefer 0, 0.02, best, and one other
    ordered = sorted(candidates)
    if len(ordered) > 4:
        # keep 0, 0.02, best, and farthest from 0.02 among remaining
        keep = {0.0, 0.02, best["particle_l2"]}
        rest = [x for x in ordered if x not in keep]
        rest.sort(key=lambda x: abs(x - 0.02), reverse=True)
        keep.add(rest[0])
        ordered = sorted(keep)

    print(f"=== multi-seed for particle_l2={ordered} seeds={seeds} ===", flush=True)
    multi_seed = {"seeds": seeds, "by_l2": {}}
    for l2 in ordered:
        rows = []
        for s in seeds:
            row = run_cfg(seed=s, particle_l2=l2)
            rows.append(row)
            print_row(f"seed={s:2d} l2={l2}", row)
        multi_seed["by_l2"][str(l2)] = {
            "particle_l2": l2,
            "rows": rows,
            **summarize(rows),
        }

    # Verdict
    base_ms = multi_seed["by_l2"].get("0.02")
    zero_ms = multi_seed["by_l2"].get("0.0")
    best_ms_key = max(
        multi_seed["by_l2"].keys(),
        key=lambda k: (
            multi_seed["by_l2"][k]["n_pass"],
            multi_seed["by_l2"][k]["kept_mean"],
        ),
    )
    best_ms = multi_seed["by_l2"][best_ms_key]

    verdict = "keep_default_0.02"
    detail_parts = []
    if base_ms and base_ms["n_pass"] == base_ms["n"]:
        detail_parts.append(
            f"default l2=0.02 seed-stable {base_ms['n_pass']}/{base_ms['n']} "
            f"mean kept {base_ms['kept_mean']:.4f} span {base_ms['kept_span']:.4f}"
        )
    else:
        verdict = "default_unstable_investigate"
        detail_parts.append("default l2=0.02 lost seed stability")

    if zero_ms:
        dkept = (zero_ms["kept_mean"] - (base_ms["kept_mean"] if base_ms else 0.0))
        detail_parts.append(
            f"l2=0 ablation {zero_ms['n_pass']}/{zero_ms['n']} mean kept {zero_ms['kept_mean']:.4f} "
            f"(Δ vs 0.02 = {dkept:+.4f})"
        )
        if (
            base_ms
            and zero_ms["n_pass"] == zero_ms["n"]
            and zero_ms["kept_mean"] > base_ms["kept_mean"] + 0.01
        ):
            verdict = "consider_lower_l2"
            detail_parts.append("l2=0 clearly better on kept — consider lowering default")

    if best_ms and abs(float(best_ms_key) - 0.02) > 1e-12:
        dkept = best_ms["kept_mean"] - (base_ms["kept_mean"] if base_ms else 0.0)
        detail_parts.append(
            f"best multi-seed l2={best_ms_key} {best_ms['n_pass']}/{best_ms['n']} "
            f"mean {best_ms['kept_mean']:.4f} (Δ={dkept:+.4f})"
        )
        if (
            base_ms
            and best_ms["n_pass"] == best_ms["n"]
            and best_ms["kept_mean"] >= base_ms["kept_mean"] + 0.01
        ):
            verdict = "consider_new_l2"
        elif abs(dkept) < 0.005 and best_ms["n_pass"] == best_ms["n"]:
            # flat — keep default
            if verdict == "keep_default_0.02":
                detail_parts.append("flat vs default (Δkept ≪ 0.01) — no recipe change")

    # Grid monotonicity note
    kept_by_l2 = [(r["particle_l2"], r["on_sheet_kept"], r["pass"]) for r in grid_rows]
    detail_parts.append("seed0 grid: " + ", ".join(
        f"{l2}->{k:.4f}{'P' if p else 'F'}" for l2, k, p in kept_by_l2
    ))

    payload = {
        "fire": 7,
        "date": "2026-09-09",
        "host": "pop-os-cpu",
        "sha": "435e873",
        "locked_recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "n_particles": 12,
            "fm_weight": 0.0,
            "b_cap": 1.0,
            "teacher": "faithful_guard_e",
            "default_particle_l2": 0.02,
        },
        "seed0_grid": grid_rows,
        "multi_seed": multi_seed,
        "verdict": verdict,
        "verdict_detail": "; ".join(detail_parts),
        "next_thread": (
            "multi-pair/cross-axis stress, or exam_score at locked recipe, "
            "or LR/b_cap micro-sweep that preserves seed stability"
        ),
    }

    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}", flush=True)

    lines = [
        "# particle_l2 micro-sweep @ locked 1200+c1.5 n=12 (Fire #7)",
        "",
        f"Host: pop-os CPU @ `435e873`. Default `particle_l2=0.02`.",
        "",
        "## Seed-0 grid",
        "",
        "| particle_l2 | kept | leak | garble | swing | residual_norm | pass | sec |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in grid_rows:
        rn = r.get("residual_norm")
        rn_s = f"{rn:.4f}" if isinstance(rn, float) else "—"
        lines.append(
            f"| {r['particle_l2']} | {r['on_sheet_kept']:.4f} | {r['leak_tok']:+.4f} | "
            f"{r['garble']:.4f} | {r['swing_kept']:.4f} | {rn_s} | "
            f"{'PASS' if r['pass'] else 'FAIL'} | {r['sec']} |"
        )

    lines += ["", "## Multi-seed", ""]
    for k, blk in multi_seed["by_l2"].items():
        lines.append(
            f"- l2={k}: {blk['n_pass']}/{blk['n']} PASS, kept mean {blk['kept_mean']:.4f} "
            f"(span {blk['kept_span']:.4f})"
            + (
                f", residual_norm mean {blk['residual_norm_mean']:.4f}"
                if "residual_norm_mean" in blk
                else ""
            )
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}** — {payload['verdict_detail']}",
        "",
        "## Next",
        "",
        payload["next_thread"],
        "",
    ]
    MD.write_text("\n".join(lines))
    print(f"Wrote {MD}", flush=True)

    # Append research log
    log_block = [
        "",
        "## Fire #7 — particle_l2 micro-sweep (2026-09-09)",
        "",
        "- Host: pop-os-cpu @ SHA `435e873`",
        f"- Seed0 grid l2∈{l2_grid}: "
        + ", ".join(
            f"{r['particle_l2']}→{r['on_sheet_kept']:.4f}{'P' if r['pass'] else 'F'}"
            for r in grid_rows
        ),
    ]
    for k, blk in multi_seed["by_l2"].items():
        log_block.append(
            f"- Multi-seed l2={k}: {blk['n_pass']}/{blk['n']} kept mean {blk['kept_mean']:.4f} "
            f"span {blk['kept_span']:.4f}"
        )
    log_block += [
        f"- Verdict: **{verdict}** — {payload['verdict_detail']}",
        f"- Next: {payload['next_thread']}",
        "- Notes: `particle_l2_sweep_20260909.{{py,json,md}}`",
        "",
    ]
    if LOG.exists():
        prev = LOG.read_text()
        if "## Fire #7" not in prev:
            LOG.write_text(prev.rstrip() + "\n" + "\n".join(log_block))
        else:
            print("research_log already has Fire #7; left unchanged", flush=True)
    else:
        LOG.write_text(
            "# 2D adversarial slider research log — 2026-09-09\n\n"
            + "\n".join(log_block)
        )
    print(f"Updated {LOG}", flush=True)
    print(f"VERDICT={verdict}", flush=True)
    print(payload["verdict_detail"], flush=True)


if __name__ == "__main__":
    main()
