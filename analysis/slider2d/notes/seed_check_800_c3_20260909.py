#!/usr/bin/env python3
"""Fire #4: seed-check knife-edge flat_800_c3.0 leftover sheet (vs locked 1200+c1.5)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.slider2d.gan import default_cfg, score_adv_sheet
from analysis.slider2d.scoreboard import cell_works
from analysis.slider2d.sheet import leaky_field

OUT = Path("analysis/slider2d/notes/seed_check_800_c3_20260909.json")
MD = Path("analysis/slider2d/notes/seed_check_800_c3_20260909.md")
SEEDS = [0, 1, 2, 3, 7, 42]
STEPS = 800
COVER = 3.0
TEACHER = "faithful_guard_e"


def main() -> None:
    rows = []
    print("=== seed-check @800 cover=3.0 leftover sheet ===", flush=True)
    for s in SEEDS:
        cfg = default_cfg(
            steps=STEPS,
            seed=s,
            b_cap=1.0,
            fm_weight=0.0,
            cover_weight=COVER,
        )
        t0 = time.time()
        r = score_adv_sheet(leaky_field(), teacher=TEACHER, cfg=cfg)
        dt = time.time() - t0
        ok = bool(
            cell_works(
                leak=r.get("leak_tok"),
                on_sheet_kept=r.get("on_sheet_kept"),
                off_sheet=r.get("garble"),
                argmax_on_sheet=r.get("argmax_on_sheet"),
                swing_kept=r.get("swing_kept"),
            )
        ) and bool(r.get("pass"))
        row = {
            "seed": s,
            "steps": STEPS,
            "cover_weight": COVER,
            "teacher": TEACHER,
            "on_sheet_kept": float(r["on_sheet_kept"]),
            "leak_tok": float(r["leak_tok"]),
            "garble": float(r["garble"]),
            "swing_kept": float(r["swing_kept"]),
            "argmax_on_sheet": float(r["argmax_on_sheet"]),
            "pass": ok,
            "sec": round(dt, 1),
        }
        rows.append(row)
        mark = "PASS" if ok else "FAIL"
        print(
            f"seed={s:2d} kept={row['on_sheet_kept']:.4f} leak={row['leak_tok']:+.4f} "
            f"garble={row['garble']:.4f} swing={row['swing_kept']:.4f} {mark} ({row['sec']}s)",
            flush=True,
        )

    kept = [r["on_sheet_kept"] for r in rows]
    summary = {
        "n": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "kept_min": min(kept),
        "kept_max": max(kept),
        "kept_mean": sum(kept) / len(kept),
        "kept_span": max(kept) - min(kept),
    }
    payload = {
        "sha_hint": "435e873",
        "fire": 4,
        "recipe": "flat_800_c3.0 leftover sheet",
        "summary": summary,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Seed-check 800xcover3.0 (2026-09-09 fire #4)",
        "",
        "SHA `435e873` pop-os `/ml2/music/sliders-conceptmod` CPU. GPUs left alone.",
        "",
        "## flat_800_c3.0 leftover sheet across seeds",
        "",
        "| seed | kept | leak | garble | swing | pass |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        lines.append(
            f"| {r['seed']} | {r['on_sheet_kept']:.4f} | {r['leak_tok']:+.4f} | "
            f"{r['garble']:.4f} | {r['swing_kept']:.4f} | {mark} |"
        )
    lines += [
        "",
        f"Summary: **{summary['n_pass']}/{summary['n']} PASS**, kept mean "
        f"`{summary['kept_mean']:.4f}`, span `{summary['kept_span']:.4f}` "
        f"(min `{summary['kept_min']:.4f}`, max `{summary['kept_max']:.4f}`).",
        "",
        "## Takeaway",
        "",
    ]
    if summary["n_pass"] == summary["n"] and summary["kept_min"] >= 0.9:
        lines.append(
            "Knife-edge from fire #3 seed0 **holds across seeds** — still treat as "
            "noisy lock-edge vs locked `1200+c1.5` (mean kept ~0.93, span ~0.002)."
        )
    elif summary["n_pass"] == 0:
        lines.append(
            "Fire #3 seed0 PASS was **not reproducible** — 800xc3.0 is a false lock. "
            "Keep `1200+cover1.5` as default."
        )
    else:
        lines.append(
            f"Mixed {summary['n_pass']}/{summary['n']} — 800xc3.0 is **unstable** at the "
            "0.90 gate. Prefer locked `1200+cover1.5` (6/6 PASS, span 0.0017)."
        )
    lines.append("")
    MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, MD, flush=True)
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
