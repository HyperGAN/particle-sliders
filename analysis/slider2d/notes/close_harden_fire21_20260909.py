#!/usr/bin/env python3
"""Fire #21 (+ Fire #24 annotate_dig_rows wire)

Fire #21 — f3d_close / close_live_noise harden dig (Music M3/M11).

Prior: n=1 Music posture seed0-only knife (~5/6); follow-up said n≥4 fixes it.
This fire quantifies n_particles harden path + close_live_noise amp/span mu.
Locked recipe otherwise. CPU only. No Music train. No recipe change unless
multi-seed Music evidence (do not adopt).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.notes.music_posture_dig_util import annotate_dig_rows, posture_block_for_md  # Fire #24
from analysis.slider2d.field3d import (  # noqa: E402
    close_field3d,
    close_live_noise_field3d,
    leftover_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "close_harden_fire21_20260909.json"
OUT_MD = NOTES / "close_harden_fire21_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"
N_SWEEP = [1, 2, 4, 8, 12]
COVERS_CLOSE = [1.0, 1.5]


def locked_cfg(seed: int, *, cover: float = 1.5, n_particles: int = 12, steps: int = 1200):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n_particles,
        particle_l2=0.02,
    )


def run_exam(field, seed: int, cover: float, n_particles: int, name: str) -> dict:
    t0 = time.time()
    row = score_adv_field3d_exam(
        field,
        teacher=TEACHER,
        cfg=locked_cfg(seed, cover=cover, n_particles=n_particles),
        name=name,
    )
    primary = float(row.get("exam_score", row.get("u_kept", 0.0)))
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": primary,
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "fail_reason": str(row.get("fail_reason", row.get("reason", ""))),
        "wall_s": round(time.time() - t0, 2),
        "name": name,
        "n_particles": n_particles,
        "cell": getattr(field, "kind", None),
        "music_close_posture_warn": row.get("music_close_posture_warn"),  # Fire #24
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    fails = [r["seed"] for r in rows if not r["pass"]]
    prim = [r["exam_score"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    return {
        "n": n,
        "n_pass": sum(1 for r in rows if r["pass"]),
        "pass_str": f"{sum(1 for r in rows if r['pass'])}/{n}",
        "fail_seeds": fails,
        "mean_exam": round(sum(prim) / n, 4) if n else 0.0,
        "exam_span": round(max(prim) - min(prim), 4) if prim else 0.0,
        "leak_max": round(max(leaks), 4) if leaks else 0.0,
        "seed0_fail": 0 in fails,
        "only_seed0": fails == [0],
    }


def main() -> None:
    t_wall = time.time()
    cells: dict[str, dict] = {}

    # Regression: leftover @ locked n12 c1.5
    rows = [
        run_exam(leftover_field3d(), s, 1.5, 12, f"leftover_s{s}")
        for s in SEEDS
    ]
    cells["leftover_n12_c1.5"] = {"rows": rows, "summary": summarize(rows)}
    print(f"leftover_n12_c1.5: {cells['leftover_n12_c1.5']['summary']['pass_str']}", flush=True)

    # f3d_close n × cover sweep
    for cover in COVERS_CLOSE:
        for n in N_SWEEP:
            key = f"close_n{n}_c{cover}"
            rows = [
                run_exam(close_field3d(), s, cover, n, key)
                for s in SEEDS
            ]
            cells[key] = {"rows": rows, "summary": summarize(rows)}
            sm = cells[key]["summary"]
            print(
                f"{key}: {sm['pass_str']} mean={sm['mean_exam']} "
                f"fail={sm['fail_seeds']} seed0_only={sm['only_seed0']}",
                flush=True,
            )

    # close_live_noise default noise @ Music cover 1.0, n sweep
    for n in [1, 2, 4, 8, 12]:
        key = f"live_n{n}_c1.0_default"
        rows = [
            run_exam(close_live_noise_field3d(seed=s), s, 1.0, n, key)
            for s in SEEDS
        ]
        cells[key] = {"rows": rows, "summary": summarize(rows)}
        sm = cells[key]["summary"]
        print(
            f"{key}: {sm['pass_str']} mean={sm['mean_exam']} fail={sm['fail_seeds']}",
            flush=True,
        )

    # noise mu @ n=1 (full) and n=4 (coarse); seeds trimmed for wall
    noise_seeds = [0, 1, 2, 42]
    for n, amps, spans in (
        (1, (0.02, 0.04, 0.08), (0.04, 0.08, 0.16)),
        (4, (0.04, 0.08), (0.08, 0.16)),
    ):
        for amp in amps:
            for span in spans:
                key = f"live_n{n}_c1.0_a{amp}_s{span}"
                rows = [
                    run_exam(
                        close_live_noise_field3d(seed=s, amp_noise=amp, span_noise=span),
                        s,
                        1.0,
                        n,
                        key,
                    )
                    for s in noise_seeds
                ]
                cells[key] = {"rows": rows, "summary": summarize(rows)}
                sm = cells[key]["summary"]
                print(
                    f"{key}: {sm['pass_str']} fail={sm['fail_seeds']}",
                    flush=True,
                )

    # Verdict helpers
    close_n1_c10 = cells["close_n1_c1.0"]["summary"]
    close_n1_c15 = cells["close_n1_c1.5"]["summary"]
    close_n4_c10 = cells["close_n4_c1.0"]["summary"]
    close_n4_c15 = cells["close_n4_c1.5"]["summary"]
    close_n12_c15 = cells["close_n12_c1.5"]["summary"]
    leftover = cells["leftover_n12_c1.5"]["summary"]

    n_min_full_c10 = None
    n_min_full_c15 = None
    for n in N_SWEEP:
        if cells[f"close_n{n}_c1.0"]["summary"]["n_pass"] == 6 and n_min_full_c10 is None:
            n_min_full_c10 = n
        if cells[f"close_n{n}_c1.5"]["summary"]["n_pass"] == 6 and n_min_full_c15 is None:
            n_min_full_c15 = n

    live_n1 = cells["live_n1_c1.0_default"]["summary"]
    live_n4 = cells["live_n4_c1.0_default"]["summary"]

    # Which noise cells still bite at n=1 / recover at n=4
    noise_bite_n1 = [
        k for k, v in cells.items()
        if k.startswith("live_n1_c1.0_a") and v["summary"]["n_pass"] < 6
    ]
    noise_full_n4 = [
        k for k, v in cells.items()
        if k.startswith("live_n4_c1.0_a") and v["summary"]["n_pass"] == 6
    ]
    noise_bite_n4 = [
        k for k, v in cells.items()
        if k.startswith("live_n4_c1.0_a") and v["summary"]["n_pass"] < 6
    ]

    seed0_knife_n1 = bool(close_n1_c10["only_seed0"] or close_n1_c15["only_seed0"])
    harden_n4 = (
        close_n4_c10["n_pass"] == 6 and close_n4_c15["n_pass"] == 6
    )
    regression = leftover["n_pass"] < 6
    recipe_change = False  # documenting n≥4 Music close rule is NOT a locked-recipe change

    ping = bool(
        seed0_knife_n1
        or harden_n4
        or noise_bite_n1
        or (not regression and (n_min_full_c10 or n_min_full_c15))
    )
    # Real finding: harden path confirmed or live noise still bites beyond n=4
    actionable = bool(
        (seed0_knife_n1 and harden_n4)
        or bool(noise_bite_n4)
        or regression
    )

    wall = round(time.time() - t_wall, 1)
    verdict = {
        "wall_s": wall,
        "host": "box-cpu",
        "sha": "435e873",
        "popos_ssh": "DOWN",
        "leftover_pass": leftover["pass_str"],
        "close_n1_c1.0": close_n1_c10["pass_str"],
        "close_n1_c1.5": close_n1_c15["pass_str"],
        "close_n4_c1.0": close_n4_c10["pass_str"],
        "close_n4_c1.5": close_n4_c15["pass_str"],
        "close_n12_c1.5": close_n12_c15["pass_str"],
        "n_min_full_pass_c1.0": n_min_full_c10,
        "n_min_full_pass_c1.5": n_min_full_c15,
        "seed0_only_knife_n1": seed0_knife_n1,
        "harden_n_ge_4": harden_n4,
        "live_n1_default": live_n1["pass_str"],
        "live_n4_default": live_n4["pass_str"],
        "noise_bite_n1_keys": noise_bite_n1,
        "noise_full_n4_keys": noise_full_n4,
        "noise_bite_n4_keys": noise_bite_n4,
        "leftover_regression": regression,
        "recipe_change": recipe_change,
        "ping_user": actionable,
        "actionable_finding": actionable,
    }

    payload = {"verdict": verdict, "cells": {k: v["summary"] for k, v in cells.items()}}
    # Keep full rows only for knife cells to limit JSON size
    payload["detail_rows"] = {
        k: cells[k]["rows"]
        for k in (
            "leftover_n12_c1.5",
            "close_n1_c1.0",
            "close_n1_c1.5",
            "close_n4_c1.0",
            "close_n4_c1.5",
            "close_n12_c1.5",
            "live_n1_c1.0_default",
            "live_n4_c1.0_default",
        )
        if k in cells
    }
    # Fire #24: posture annotate over detail_rows
    flat = []
    for _k, rs in (payload.get("detail_rows") or {}).items():
        flat.extend(rs or [])
    annotate_dig_rows(flat, payload=payload)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Fire #21 — f3d_close / close_live_noise harden (2026-09-09)",
        "",
        f"Host: box-cpu @ `435e873` (+ local field3d.py). Wall {wall}s. CPU only.",
        *posture_block_for_md(payload.get("music_close_posture") or {}),
        "pop-os SSH hop DOWN (laptop+LAN timeout).",
        "",
        "## Hypothesis",
        "",
        "Music-posture `n_particles=1` (parts0 proxy) causes the f3d_close seed0-only",
        "knife; bumping `n_particles≥4` restores multi-seed close. `close_live_noise`",
        "amp/span mu should stress M11 caption jitter beyond clean close.",
        "",
        "## Grid (pass / fail_seeds)",
        "",
        "| cell | PASS | mean exam | leak max | fail | seed0-only |",
        "|---|:---:|---:|---:|---|:---:|",
    ]
    for k in sorted(cells.keys()):
        sm = cells[k]["summary"]
        lines.append(
            f"| `{k}` | {sm['pass_str']} | {sm['mean_exam']} | {sm['leak_max']} | "
            f"{sm['fail_seeds']} | {sm['only_seed0']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- **leftover regression?** {'YES' if regression else 'NO'} ({leftover['pass_str']})",
        f"- **seed0-only knife @ n=1?** {'YES' if seed0_knife_n1 else 'NO'} "
        f"(c1.0 {close_n1_c10['pass_str']}, c1.5 {close_n1_c15['pass_str']})",
        f"- **n≥4 harden full 6/6?** {'YES' if harden_n4 else 'NO'} "
        f"(min full c1.0={n_min_full_c10}, c1.5={n_min_full_c15})",
        f"- **live default n1→n4?** {live_n1['pass_str']} → {live_n4['pass_str']}",
        f"- **noise mu still bites @ n=4?** {noise_bite_n4 or 'none'}",
        f"- **Recipe change?** NO (document Music close: prefer n≥4; locked n≤12 stands)",
        f"- **Ping user?** {'YES' if actionable else 'NO'}",
        "",
        "JSON: `close_harden_fire21_20260909.json`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Append research log
    log_extra = "\n".join(
        [
            "",
            "## Fire #21 — close / live_noise harden (2026-09-09)",
            "",
            "- Host: box-cpu @ SHA `435e873` (pop-os hop DOWN)",
            f"- Dig: `close_harden_fire21_20260909.{{py,json,md}}` wall={wall}s",
            f"- leftover n12 c1.5: {leftover['pass_str']}",
            f"- close n1: c1.0 {close_n1_c10['pass_str']} c1.5 {close_n1_c15['pass_str']} "
            f"(seed0_only={seed0_knife_n1})",
            f"- close n4: c1.0 {close_n4_c10['pass_str']} c1.5 {close_n4_c15['pass_str']} "
            f"(harden_n_ge_4={harden_n4})",
            f"- n_min_full_pass: c1.0={n_min_full_c10} c1.5={n_min_full_c15}",
            f"- live default: n1 {live_n1['pass_str']} n4 {live_n4['pass_str']}",
            f"- noise_bite_n4: {noise_bite_n4}",
            f"- Verdict: recipe_change=NO; ping_user={'YES' if actionable else 'NO'}; "
            f"document Music close n≥4 harden path",
            "- Next: catalog M3 harden note; optional per-row residual (out of recipe) or "
            "scaffold field3d close Music-posture warning",
            "",
        ]
    )
    with LOG.open("a") as f:
        f.write(log_extra)

    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
