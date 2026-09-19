#!/usr/bin/env python3
"""Fire #20 — harden dig for lyric_span_entangle (M1) multi-row fail.

Hypothesis: shared AdvResidual + cover-all-rows cannot satisfy heterogeneous
row_amps when e_on_content>0 (Music span-entangled leftover). Isolate
cross-axis vs entanglement; try cover/n/steps; document hard boundary if
locked recipe never recovers multi-row.

Locked: 1200 / c1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1.
CPU only. No Music GPU train.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    Field3D,
    leftover_field3d,
    lyric_span_entangle_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = Path(__file__).resolve().parent
OUT_JSON = NOTES / "lyric_span_harden_fire20_20260909.json"
OUT_MD = NOTES / "lyric_span_harden_fire20_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"

# Catalog M1 defaults
M1_AMPS = (
    (1.05, 0.55, 0.35),
    (0.60, 1.10, 0.45),
    (0.75, 0.50, 0.90),
    (1.10, 0.85, 0.55),
    (0.90, 0.70, 0.65),
)
M1_SCALES = (0.75, 0.95, 1.05, 1.2, 1.35)
HOMO_AMPS = tuple((1.0, 0.6, 0.45) for _ in range(5))


def cfg(seed: int, *, cover: float = 1.5, n: int = 12, steps: int = 1200):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n,
        particle_l2=0.02,
    )


def make_field(*, amps, e_on_content: float, e_unused: float = 0.7, kind: str) -> Field3D:
    return Field3D(
        kind=kind,
        rows=5,
        row_scales=M1_SCALES,
        row_amps=amps,
        slider=1.0,
        content=0.7,
        leak=0.55,
        e_on_u=0.05,
        e_on_content=e_on_content,
        e_unused=e_unused,
    )


def run_one(field, *, seed: int, cover: float, n: int, steps: int, name: str) -> dict:
    t0 = time.time()
    row = score_adv_field3d_exam(
        field,
        teacher=TEACHER,
        cfg=cfg(seed, cover=cover, n=n, steps=steps),
        name=name,
    )
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": float(row.get("exam_score", 0.0)),
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "pass_multi_row": bool(row.get("pass_multi_row", False)),
        "pass_leftover_gate": bool(row.get("pass_leftover_gate", False)),
        "pass_cont": bool(row.get("pass_cont", False)),
        "pass_swing": bool(row.get("pass_swing", False)),
        "wall_s": round(time.time() - t0, 2),
    }


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    npass = sum(1 for r in runs if r["pass"])
    multi = sum(1 for r in runs if r["pass_multi_row"])
    return {
        "n": n,
        "pass": f"{npass}/{n}",
        "n_pass": npass,
        "multi_row": f"{multi}/{n}",
        "mean_exam": round(sum(r["exam_score"] for r in runs) / n, 4),
        "mean_u": round(sum(r["u_kept"] for r in runs) / n, 4),
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4),
        "mean_rows_cov": round(sum(r["rows_covered"] for r in runs) / n, 2),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "multi_fail_seeds": [r["seed"] for r in runs if not r["pass_multi_row"]],
        "runs": runs,
    }


def grid(name: str, field_fn, seeds, **kw) -> dict:
    runs = []
    for s in seeds:
        runs.append(run_one(field_fn(seed=s), seed=s, name=f"{name}_s{s}", **kw))
    out = summarize(runs)
    out["name"] = name
    out["kw"] = kw
    print(
        f"  {name}: pass={out['pass']} multi={out['multi_row']} "
        f"exam={out['mean_exam']} leak_max={out['leak_max']} "
        f"rows_cov≈{out['mean_rows_cov']} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def main() -> None:
    t_wall = time.time()
    cells = []

    # A) Regression: homogeneous leftover still solid
    cells.append(
        grid(
            "leftover_n12_c1.5",
            lambda seed: leftover_field3d(seed=seed),
            SEEDS_SMOKE,
            cover=1.5,
            n=12,
            steps=1200,
        )
    )

    # B) Locked M1 bite confirm (full seeds)
    cells.append(
        grid(
            "lyric_m1_locked_n12_c1.5",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_FULL,
            cover=1.5,
            n=12,
            steps=1200,
        )
    )

    # C) Harden probes on seed0 + smoke — isolate factors
    # C1: e_on_content ablation (keep hetero amps)
    for eoc in (0.0, 0.15, 0.25, 0.45):
        cells.append(
            grid(
                f"hetero_eoc{eoc}_n12_c1.5",
                lambda seed, e=eoc: make_field(
                    amps=M1_AMPS, e_on_content=e, kind=f"hetero_eoc{e}"
                ),
                SEEDS_SMOKE,
                cover=1.5,
                n=12,
                steps=1200,
            )
        )

    # C2: homogeneous amps + e_on_content (entangle without cross-axis)
    for eoc in (0.0, 0.45):
        cells.append(
            grid(
                f"homo_eoc{eoc}_n12_c1.5",
                lambda seed, e=eoc: make_field(
                    amps=HOMO_AMPS, e_on_content=e, kind=f"homo_eoc{e}"
                ),
                SEEDS_SMOKE,
                cover=1.5,
                n=12,
                steps=1200,
            )
        )

    # C3: cover / n / steps on locked M1 (seed0 knife + smoke)
    for label, cover, n, steps in (
        ("m1_n4_c1.5", 1.5, 4, 1200),
        ("m1_n12_c2.0", 2.0, 12, 1200),
        ("m1_n12_steps1600", 1.5, 12, 1600),
        ("m1_n12_c1.5_eunused1", 1.5, 12, 1200),
    ):
        if label.endswith("eunused1"):
            field_fn = lambda seed: make_field(
                amps=M1_AMPS, e_on_content=0.45, e_unused=1.0, kind="lyric_eunused1"
            )
        else:
            field_fn = lambda seed: lyric_span_entangle_field3d(seed=seed)
        cells.append(
            grid(label, field_fn, SEEDS_SMOKE, cover=cover, n=n, steps=steps)
        )

    # D) Music-posture M1 (n=1 c1.0) smoke — still bite?
    cells.append(
        grid(
            "lyric_m1_music_n1_c1.0",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            cover=1.0,
            n=1,
            steps=1200,
        )
    )

    wall = round(time.time() - t_wall, 1)

    # Verdicts
    locked = next(c for c in cells if c["name"] == "lyric_m1_locked_n12_c1.5")
    leftover = next(c for c in cells if c["name"] == "leftover_n12_c1.5")
    hetero0 = next(c for c in cells if c["name"] == "hetero_eoc0.0_n12_c1.5")
    hetero45 = next(c for c in cells if c["name"] == "hetero_eoc0.45_n12_c1.5")
    homo0 = next(c for c in cells if c["name"] == "homo_eoc0.0_n12_c1.5")
    homo45 = next(c for c in cells if c["name"] == "homo_eoc0.45_n12_c1.5")

    any_harden = any(
        c["n_pass"] == c["n"] and "m1_" in c["name"]
        for c in cells
        if c["name"].startswith("m1_")
    )
    # Soft boundary: hetero eoc=0 recovers multi-row?
    eoc_soft = hetero0["n_pass"] == hetero0["n"] and hetero45["n_pass"] < hetero45["n"]
    # Cross-axis alone vs entangle: homo+eoc45 vs hetero+eoc45
    cross_axis_driver = homo45["n_pass"] > hetero45["n_pass"]
    entangle_alone_ok = homo45["n_pass"] == homo45["n"]

    recipe_change = False  # default: never revise locked from this dig
    hard_boundary = locked["n_pass"] == 0 and not any_harden
    ping = bool(locked["n_pass"] == 0)  # biting stressor confirmed / harden path found

    payload = {
        "fire": 20,
        "host": "box-cpu",
        "sha": "435e873",
        "wall_s": wall,
        "recipe_change": recipe_change,
        "hard_boundary_lyric_span": hard_boundary,
        "eoc_soft_boundary": eoc_soft,
        "cross_axis_driver": cross_axis_driver,
        "entangle_alone_ok": entangle_alone_ok,
        "any_m1_harden_probe_fullpass": any_harden,
        "ping_user": ping,
        "cells": [{k: v for k, v in c.items() if k != "runs"} | {"runs": c["runs"]} for c in cells],
        "summary": {
            "leftover_pass": leftover["pass"],
            "locked_m1_pass": locked["pass"],
            "locked_m1_multi": locked["multi_row"],
            "hetero_eoc0": hetero0["pass"],
            "hetero_eoc45": hetero45["pass"],
            "homo_eoc0": homo0["pass"],
            "homo_eoc45": homo45["pass"],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Fire #20 — lyric_span_entangle harden dig (2026-09-09)",
        "",
        f"Host: box-cpu @ `435e873` (+ local field3d.py). Wall {wall}s. CPU only.",
        "pop-os SSH hop DOWN (laptop+LAN timeout).",
        "",
        "## Hypothesis",
        "",
        "Shared `AdvResidual` + multi-row cover cannot hit heterogeneous `row_amps`",
        "when `e_on_content>0` (Music span-entangled leftover / lyric pool mix).",
        "",
        "## Grid",
        "",
        "| cell | PASS | multi_row | mean exam | leak max | mean rows_cov | fail |",
        "|---|:---:|:---:|---:|---:|---:|---|",
    ]
    for c in cells:
        lines.append(
            f"| `{c['name']}` | {c['pass']} | {c['multi_row']} | {c['mean_exam']} | "
            f"{c['leak_max']} | {c['mean_rows_cov']} | {c['fail_seeds']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- **leftover regression?** {'NO' if leftover['n_pass'] == leftover['n'] else 'YES'} ({leftover['pass']})",
        f"- **locked M1 still bites?** {'YES' if locked['n_pass'] == 0 else 'NO'} ({locked['pass']}, multi {locked['multi_row']})",
        f"- **e_on_content soft boundary?** {eoc_soft} (hetero eoc0 {hetero0['pass']} vs eoc0.45 {hetero45['pass']})",
        f"- **homo+eoc0.45 alone OK?** {entangle_alone_ok} ({homo45['pass']})",
        f"- **cross-axis drives fail more than entangle?** {cross_axis_driver}",
        f"- **any cover/n/steps harden full-pass?** {any_harden}",
        f"- **hard boundary (document)?** {hard_boundary}",
        f"- **Recipe change?** NO",
        f"- **Ping user?** {'YES' if ping else 'NO'} — Music→toy M1 bite + harden diagnosis",
        "",
        "JSON: `lyric_span_harden_fire20_20260909.json`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Append research log
    log_extra = [
        "",
        "## Fire #20 — lyric_span_entangle harden (2026-09-09)",
        "",
        f"- Host: box-cpu @ SHA `435e873` (pop-os hop DOWN)",
        f"- Tests: pytest test_lm_2d_adv + test_lm_highd_leftover + test_field3d → **59 passed** (~121s)",
        f"- Dig: `lyric_span_harden_fire20_20260909.{{py,json,md}}` wall={wall}s",
        f"- locked M1: {locked['pass']} multi={locked['multi_row']} exam={locked['mean_exam']} leak_max={locked['leak_max']}",
        f"- hetero eoc0→0.45: {hetero0['pass']} → {hetero45['pass']}; homo eoc0/0.45: {homo0['pass']}/{homo45['pass']}",
        f"- Verdict: hard_boundary={hard_boundary}; eoc_soft={eoc_soft}; cross_axis_driver={cross_axis_driver}; "
        f"any_harden={any_harden}; recipe_change=NO; ping_user={'YES' if ping else 'NO'}",
        "- Next: if hard_boundary, document Music multi-span shared-residual limit in catalog; "
        "optional scaffold per-row residual research (do NOT adopt without multi-seed Music).",
        "",
    ]
    with LOG.open("a") as f:
        f.write("\n".join(log_extra))

    print(json.dumps({k: payload[k] for k in (
        "wall_s", "hard_boundary_lyric_span", "eoc_soft_boundary",
        "cross_axis_driver", "entangle_alone_ok", "any_m1_harden_probe_fullpass",
        "ping_user", "summary",
    )}, indent=2))


if __name__ == "__main__":
    main()
