#!/usr/bin/env python3
"""Deep thread A: Can per-row residual clear M21/M24/M27 WITHOUT clearing amp-lie M20?

Falsify overpowered heads: if per-row also clears M20 (YAML amp lie), heads are
too powerful / scoring too soft. Expected: M21/M24/M27 may clear (DoF / mix /
scale bites); M20 must STAY biting (declare lie is not a DoF problem).

Locked shared recipe UNCHANGED. merge=NO. CPU only. No Music GPU train.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
NOTES = _REPO / "analysis/slider2d/notes"

_spec = importlib.util.spec_from_file_location(
    "per_row_explore", NOTES / "per_row_residual_explore_20260909.py"
)
ex = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["per_row_explore"] = ex
_spec.loader.exec_module(ex)

# Reuse deepen gate recompute (kind-aware leftover_ok)
_dspec = importlib.util.spec_from_file_location(
    "per_row_deepen", NOTES / "per_row_residual_deepen_20260909.py"
)
dep = importlib.util.module_from_spec(_dspec)
assert _dspec.loader is not None
sys.modules["per_row_deepen"] = dep
_dspec.loader.exec_module(dep)

from analysis.slider2d.field3d import (  # noqa: E402
    amp_lie_leftover_declare_field3d,
    content_leak_flip_rows_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    scale_descent_homo_field3d,
)

OUT_JSON = NOTES / "per_row_falsify_m21_m24_m27_20260909.json"
OUT_MD = NOTES / "per_row_falsify_m21_m24_m27_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
LABEL = "NON_DEFAULT_falsify_A"
TEACHER = ex.TEACHER


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def score_scaffold(*args, **kwargs) -> dict:
    return dep._recompute_gates(ex.score_scaffold(*args, **kwargs))


def summarize(runs: list[dict]) -> dict:
    out = ex.summarize(runs)
    hm = [r["head_min_cos"] for r in runs if r.get("head_min_cos") is not None]
    out["mean_head_min_cos"] = round(sum(hm) / len(hm), 4) if hm else None
    out["min_head_min_cos"] = round(min(hm), 4) if hm else None
    return out


def grid(name: str, mode: str, field_fn, seeds, **score_kw) -> dict:
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s}", flush=True)
        try:
            f = field_fn(seed=s)
        except TypeError:
            f = field_fn()
        runs.append(
            score_scaffold(
                f, mode=mode, seed=s, name=f"{name}_s{s}", **score_kw
            )
        )
    out = summarize(runs)
    out["name"] = name
    out["mode"] = mode
    out["label"] = LABEL
    hm = out.get("mean_head_min_cos")
    print(
        f"  {name}: pass={out['pass']} multi={out['multi_row']} bite={out['bite_cleared']} "
        f"exam={out['mean_exam']} leak_max={out['leak_max']} "
        f"rows_cov≈{out['mean_rows_cov']} head_min_cos≈{hm} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def pack(cell: dict) -> dict:
    return {
        "pass": cell["pass"],
        "multi": cell["multi_row"],
        "bite": cell["bite_cleared"],
        "n_pass": cell["n_pass"],
        "n_multi": cell["n_multi"],
        "n_bite": cell.get("n_bite"),
        "n": cell["n"],
        "mean_exam": cell["mean_exam"],
        "leak_max": cell["leak_max"],
        "mean_rows_cov": cell["mean_rows_cov"],
        "fail": cell["fail_seeds"],
        "mean_head_min_cos": cell.get("mean_head_min_cos"),
    }


def main() -> None:
    t_wall = time.time()
    sha = git_sha()
    print(f"=== {LABEL} @ {sha} ===", flush=True)
    print(
        "Hypothesis: per-row clears DoF/mix/scale bites (M21/M24/M27) "
        "but NOT amp-lie M20 (YAML declare). If M20 clears → overpowered.",
        flush=True,
    )

    cells: list[dict] = []
    results: dict = {}

    # leftover CTRL under both modes
    print("\n[CTRL] leftover shared + per_row", flush=True)
    left_s = grid("CTRL_leftover_shared", "shared", leftover_field3d, SEEDS_SMOKE)
    left_p = grid("CTRL_leftover_per_row", "per_row", leftover_field3d, SEEDS_SMOKE)
    cells.extend([left_s, left_p])
    results["leftover_ctrl"] = {
        "shared": pack(left_s),
        "per_row": pack(left_p),
        "ok": left_s["n_pass"] == left_s["n"] and left_p["n_pass"] == left_p["n"],
    }

    specs = [
        ("M21", "hold_e_lyric_mix", hold_e_lyric_mix_field3d, False),
        ("M24", "content_leak_flip_rows", content_leak_flip_rows_field3d, False),
        ("M27", "scale_descent_homo", scale_descent_homo_field3d, False),
        ("M20", "amp_lie_leftover_declare", amp_lie_leftover_declare_field3d, True),
    ]

    print("\n[A] shared vs per_row on M21/M24/M27 + M20 falsifier", flush=True)
    for mid, name, ctor, expect_stay_bite in specs:
        print(f"\n--- {mid} {name} (expect_stay_bite={expect_stay_bite}) ---", flush=True)
        shared = grid(
            f"{mid}_{name}_shared",
            "shared",
            lambda seed, _c=ctor: _c(seed=seed),
            SEEDS_SMOKE,
        )
        per = grid(
            f"{mid}_{name}_per_row",
            "per_row",
            lambda seed, _c=ctor: _c(seed=seed),
            SEEDS_SMOKE,
        )
        cells.extend([shared, per])

        per_full = None
        # Expand promising clears (multi≥2/3) to full seeds — including M20
        # if it unexpectedly clears, so we can document overpowered.
        if per["n_multi"] >= 2 or per["n_pass"] >= 2:
            print(f"  expanding {mid} per_row to full seeds…", flush=True)
            per_full = grid(
                f"{mid}_{name}_per_row_full",
                "per_row",
                lambda seed, _c=ctor: _c(seed=seed),
                SEEDS_FULL,
            )
            cells.append(per_full)

        use = per_full or per
        shared_bites = shared["n_pass"] == 0
        per_clears = use["n_pass"] == use["n"] and use.get("n_bite", 0) == use["n"]
        per_saves_multi = (
            use["n_multi"] == use["n"]
            and use.get("n_bite", 0) == use["n"]
            and shared["n_multi"] < shared["n"]
        )
        results[mid] = {
            "cell": name,
            "expect_stay_bite": expect_stay_bite,
            "shared": pack(shared),
            "per_row": pack(use),
            "per_row_expanded": per_full is not None,
            "shared_bites": shared_bites,
            "per_row_clears": per_clears,
            "per_row_saves_multi": per_saves_multi,
        }

    wall = round(time.time() - t_wall, 1)

    # ---- Verdict ----
    m21 = results["M21"]
    m24 = results["M24"]
    m27 = results["M27"]
    m20 = results["M20"]
    ctrl_ok = results["leftover_ctrl"]["ok"]

    cleared_geom = [
        mid
        for mid in ("M21", "M24", "M27")
        if results[mid]["per_row_clears"] or results[mid]["per_row_saves_multi"]
    ]
    uncleared_geom = [
        mid
        for mid in ("M21", "M24", "M27")
        if mid not in cleared_geom
    ]
    m20_clears = bool(m20["per_row_clears"])
    m20_still_bites = not m20_clears and m20["shared_bites"]
    # Overpowered if M20 clears under per-row (YAML lie should not heal via DoF)
    overpowered = m20_clears
    # Ideal: at least one geom bite clears AND M20 stays biting
    ideal = bool(cleared_geom) and m20_still_bites and ctrl_ok

    if overpowered:
        verdict = (
            "OVERPOWERED — per-row cleared amp-lie M20 (YAML declare); "
            "DoF heads too strong / gates soft — do NOT merge"
        )
    elif ideal:
        verdict = (
            f"YES — per-row clears {cleared_geom} without clearing M20 "
            f"(amp-lie still bites); uncleared={uncleared_geom}; CTRL ok; merge=NO"
        )
    elif m20_still_bites and not cleared_geom:
        verdict = (
            f"PARTIAL — M20 stays bite (good falsify) but M21/M24/M27 "
            f"not fully cleared under per-row: { {m: results[m]['per_row']['pass'] for m in ('M21','M24','M27')} }; merge=NO"
        )
    else:
        verdict = (
            f"MIXED — cleared={cleared_geom} uncleared={uncleared_geom} "
            f"m20_clears={m20_clears} ctrl_ok={ctrl_ok}; merge=NO"
        )

    payload = {
        "label": LABEL,
        "sha": sha,
        "wall_s": wall,
        "teacher": TEACHER,
        "recipe_change": False,
        "merge_to_trainer": False,
        "hypothesis": (
            "per-row clears M21/M24/M27 (DoF/mix/scale) without clearing "
            "amp-lie M20 (YAML declare falsifier)"
        ),
        "verdict": verdict,
        "ideal": ideal,
        "overpowered": overpowered,
        "cleared_geom": cleared_geom,
        "uncleared_geom": uncleared_geom,
        "m20_still_bites": m20_still_bites,
        "m20_clears": m20_clears,
        "ctrl_ok": ctrl_ok,
        "results": results,
        "cells": cells,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        f"# Per-row falsify A — M21/M24/M27 vs M20 — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. **{LABEL}**.",
        "Locked recipe unchanged. merge_to_trainer=NO.",
        "",
        "## Hypothesis",
        "",
        "Per-row AdvResidual can clear DoF / content↔leak flip / scale-descent",
        "bites (M21/M24/M27) **without** clearing amp-lie declare (M20).",
        "If M20 clears → heads overpowered (falsified).",
        "",
        "## Grid",
        "",
        "| cell | mid | mode | pass | multi | bite | exam | leak_max | fail |",
        "|---|---|---|:---:|:---:|:---:|---:|---:|---|",
    ]
    for mid in ("M21", "M24", "M27", "M20"):
        r = results[mid]
        for mode_key in ("shared", "per_row"):
            p = r[mode_key]
            lines.append(
                f"| `{r['cell']}` | {mid} | {mode_key} | **{p['pass']}** | "
                f"{p['multi']} | {p['bite']} | {p['mean_exam']} | {p['leak_max']} | "
                f"{p['fail']} |"
            )
    ls, lp = results["leftover_ctrl"]["shared"], results["leftover_ctrl"]["per_row"]
    lines.append(
        f"| `leftover` | CTRL | shared | **{ls['pass']}** | {ls['multi']} | "
        f"{ls['bite']} | {ls['mean_exam']} | {ls['leak_max']} | {ls['fail']} |"
    )
    lines.append(
        f"| `leftover` | CTRL | per_row | **{lp['pass']}** | {lp['multi']} | "
        f"{lp['bite']} | {lp['mean_exam']} | {lp['leak_max']} | {lp['fail']} |"
    )
    lines += [
        "",
        "## Verdict",
        "",
        f"- **{verdict}**",
        f"- ideal={ideal}; overpowered={overpowered}; m20_still_bites={m20_still_bites}",
        f"- cleared_geom={cleared_geom}; uncleared_geom={uncleared_geom}",
        f"- leftover CTRL ok={ctrl_ok}",
        f"- recipe_change=NO; merge_to_trainer=NO",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Append research_log
    log_block = (
        f"\n## Fire — per-row falsify A (M21/M24/M27 vs M20) (2026-09-09)\n"
        f"\n"
        f"- Host: box-cpu @ SHA `{sha}`\n"
        f"- Dig: `per_row_falsify_m21_m24_m27_20260909.{{py,json,md}}` wall={wall}s\n"
        f"- Scoreboard: `MUSIC_TO_TOY_SCOREBOARD_20260909.md`\n"
        f"- cleared_geom={cleared_geom}; uncleared={uncleared_geom}; "
        f"m20_still_bites={m20_still_bites}; overpowered={overpowered}\n"
        f"- Verdict: {verdict}\n"
        f"- recipe_change=NO; merge_to_trainer=NO; ping_user=YES\n"
        f"- No Music GPU train; servers untouched.\n"
    )
    with LOG.open("a") as fh:
        fh.write(log_block)

    # Append scoreboard deep-thread section
    sb_block = (
        f"\n## Deep thread A result (2026-09-09)\n"
        f"\n"
        f"**Script:** `per_row_falsify_m21_m24_m27_20260909` wall={wall}s\n"
        f"\n"
        f"| mid | shared pass | per_row pass | clears? |\n"
        f"|---|---|---|---|\n"
    )
    for mid in ("M21", "M24", "M27", "M20"):
        r = results[mid]
        sb_block += (
            f"| {mid} | {r['shared']['pass']} | {r['per_row']['pass']} | "
            f"{'YES' if r['per_row_clears'] else 'no'} |\n"
        )
    sb_block += (
        f"\n**Verdict:** {verdict}\n"
        f"**merge=NO.** Locked shared recipe unchanged.\n"
    )
    with SCOREBOARD.open("a") as fh:
        fh.write(sb_block)

    print("\n=== VERDICT ===", flush=True)
    print(verdict, flush=True)
    print(f"wrote {OUT_MD}", flush=True)
    print(f"wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
