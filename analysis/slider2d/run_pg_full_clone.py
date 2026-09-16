#!/usr/bin/env python3
"""Gate/smoke: pg_full_clone (propose-only) vs locked_shared toy (CPU-only).

Runs the same three cells both recipes must face — Field2D polarity,
sheet leftover, exam divergent — and prints the side-by-side gate table
plus the KEEP/HOLD/DROP ledger. Writes a metrics JSON for the note in
``docs/pg-full-clone.md``.

Neither row claims Music 3 audio. The locked row is the current default
shape; the clone row is the propose-only closest ParticleGAN port. A
failing clone cell is a documented HOLD/DROP candidate, never a silent
retrain of the locked row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.adv import PG_FULL_CLONE_LEDGER, effective_delay, pg_full_clone_cfg
from analysis.slider2d.exam import divergent_field
from analysis.slider2d.gan import default_cfg, score_adv_exam, score_adv_sheet, score_field2d
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
)
from analysis.slider2d.sheet import leaky_field


def _fmt(value, spec: str, empty: str = "N/A") -> str:
    if value is None:
        return empty
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _pass(ok: bool | None) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "—"


def run(*, steps: int, seed: int) -> dict:
    locked = default_cfg(steps=steps, seed=seed)
    clone = pg_full_clone_cfg(steps=steps, seed=seed)
    rows = {}
    for name, cfg in (("locked_shared", locked), ("pg_full_clone", clone)):
        rows[name] = {
            "cfg": {
                "recipe": cfg.recipe,
                "steps": cfg.steps,
                "seed": cfg.seed,
                "beta2": cfg.beta2,
                "n_particles": cfg.n_particles,
                "vicreg_weight": cfg.vicreg_weight,
                "vicreg_sim_weight": cfg.vicreg_sim_weight,
                "particle_l2": cfg.particle_l2,
                "ema_scope": cfg.ema_scope,
                "delay": effective_delay(cfg),
                "cover_weight": cfg.cover_weight,
            },
            "field2d": score_field2d(cfg),
            "sheet_leftover": score_adv_sheet(leaky_field(), cfg=cfg),
            "exam_divergent": score_adv_exam(divergent_field(seed=seed), cfg=cfg),
        }
    return rows


def _slim(row: dict) -> dict:
    return {
        k: v for k, v in row.items() if isinstance(v, (int, float, str, bool)) or v is None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_REPO / "docs" / "pg-full-clone-metrics.json")
    args = parser.parse_args(argv)

    rows = run(steps=args.steps, seed=args.seed)
    slim = {
        name: {
            "cfg": row["cfg"],
            "field2d": _slim(row["field2d"]),
            "sheet_leftover": _slim(row["sheet_leftover"]),
            "exam_divergent": _slim(row["exam_divergent"]),
        }
        for name, row in rows.items()
    }
    slim["ledger"] = [list(r) for r in PG_FULL_CLONE_LEDGER]
    slim["gates"] = {
        "leak_lock": COMPILED_LEAK_LOCK,
        "sheet_lock": COMPILED_SHEET_LOCK,
        "garble_max": COMPILED_GARBLE_MAX,
        "swing_floor": COMPILED_SWING_FLOOR,
    }
    args.out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    lock, clone = rows["locked_shared"], rows["pg_full_clone"]
    for label, get in (
        ("field2d slider cos (≥0.90)", lambda r: r["field2d"].get("cos_slider_plus")),
        ("field2d leak_ratio (|.|≤0.20)", lambda r: r["field2d"].get("leak_ratio")),
        ("field2d ±1 cos (≤−0.85)", lambda r: r["field2d"].get("cos_plus_minus")),
        ("sheet leak_tok (|.|≤0.20)", lambda r: r["sheet_leftover"].get("leak_tok")),
        ("sheet on_sheet_kept (≥0.90)", lambda r: r["sheet_leftover"].get("on_sheet_kept")),
        ("exam overlap (≥0.85)", lambda r: r["exam_divergent"].get("roll_overlap")),
        ("exam swing_kept (≥0.60)", lambda r: r["exam_divergent"].get("roll_swing_kept")),
    ):
        print(
            f"{label:32s} locked={_fmt(get(lock), '+.3f'):>8s} "
            f"clone={_fmt(get(clone), '+.3f'):>8s}"
        )
    for label, key in (
        ("field2d", "field2d"),
        ("sheet_leftover", "sheet_leftover"),
        ("exam_divergent", "exam_divergent"),
    ):
        print(
            f"{label:32s} locked={_pass(lock[key].get('pass')):>8s} "
            f"clone={_pass(clone[key].get('pass')):>8s}"
        )
    print(f"metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
