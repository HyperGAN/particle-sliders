"""Tiny CPU ablation: leftover-gated reals vs cover_weight / b_cap on sheet leftover."""
from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.slider2d.gan import default_cfg, score_adv_sheet, score_field2d
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    cell_works,
)
from analysis.slider2d.sheet import leaky_field

STEPS = 400  # tiny budget vs recipe 1200
SEED = 0
OUT = Path("analysis/slider2d/_ablation_leftover_cover_bcap_20260909.json")


def _sheet_row(teacher: str, cover: float, b_cap: float, tag: str) -> dict:
    cfg = default_cfg(
        steps=STEPS, seed=SEED, b_cap=b_cap, fm_weight=0.0, cover_weight=cover
    )
    t0 = time.time()
    sheet = score_adv_sheet(leaky_field(), teacher=teacher, cfg=cfg)
    dt = time.time() - t0
    ok = cell_works(
        leak=sheet.get("leak_tok"),
        on_sheet_kept=sheet.get("on_sheet_kept"),
        off_sheet=sheet.get("garble"),
        argmax_on_sheet=sheet.get("argmax_on_sheet"),
        swing_kept=sheet.get("swing_kept"),
    )
    row = {
        "tag": tag,
        "teacher": teacher,
        "cover_weight": cover,
        "b_cap": b_cap,
        "steps": STEPS,
        "leak_tok": sheet.get("leak_tok"),
        "on_sheet_kept": sheet.get("on_sheet_kept"),
        "garble": sheet.get("garble"),
        "swing_kept": sheet.get("swing_kept"),
        "pass": bool(ok),
        "sec": round(dt, 2),
    }
    print(
        "teacher=%-18s cover=%.1f b_cap=%.1f leak=%s kept=%s garble=%s pass=%s (%.1fs)"
        % (
            teacher,
            cover,
            b_cap,
            row["leak_tok"],
            row["on_sheet_kept"],
            row["garble"],
            ok,
            dt,
        ),
        flush=True,
    )
    return row


def main() -> None:
    rows = []
    for teacher in ("faithful_guard_e", "faithful"):
        for cover in (0.0, 1.0, 1.5):
            rows.append(_sheet_row(teacher, cover, 1.0, "teacher_x_cover"))
    for bcap in (0.5, 1.0, 2.0):
        rows.append(_sheet_row("faithful_guard_e", 1.5, bcap, "bcap_sweep"))

    cfg = default_cfg(steps=STEPS, seed=SEED, b_cap=1.0, fm_weight=0.0, cover_weight=1.5)
    t0 = time.time()
    f2 = score_field2d(cfg)
    dt = time.time() - t0
    f2_pass = f2.get("pass")
    print(
        "field2d steps=%s: slider_cos=%s leak_ratio=%s pm1=%s pass=%s (%.1fs)"
        % (STEPS, f2.get("slider_cos"), f2.get("leak_ratio"), f2.get("pm1_cos"), f2_pass, dt),
        flush=True,
    )
    blob = {
        "sha_hint": "435e873",
        "gates": {
            "leak_max": COMPILED_LEAK_LOCK,
            "sheet_lock": COMPILED_SHEET_LOCK,
            "garble_max": COMPILED_GARBLE_MAX,
        },
        "rows": rows,
        "field2d_400": {
            "slider_cos": f2.get("slider_cos"),
            "leak_ratio": f2.get("leak_ratio"),
            "pm1_cos": f2.get("pm1_cos"),
            "pass": f2_pass,
            "sec": round(dt, 2),
        },
    }
    OUT.write_text(json.dumps(blob, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
