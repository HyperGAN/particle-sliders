#!/usr/bin/env python3
"""Head-to-head: locked RpGAN+b_cap (#94) vs distribution-matching (FM) vs supervised.

CPU-only. Mirrors analysis/slider2d/run_lm_adv.collect cell wiring.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from analysis.slider2d.exam import close_field, divergent_field, score_exam, unused_e_field
from analysis.slider2d.gan import default_cfg, score_adv_exam, score_adv_sheet, score_field2d
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    cell_works,
    compiled_verdict,
    exam_score,
)
from analysis.slider2d.sheet import gender_like_field, leaky_field, score_sheet

OUT_JSON = Path(__file__).with_suffix(".json")
STEPS = 1200
EXAM_STEPS = 1200
BASE_STEPS = 200
SEED = 0


def _sheet_ok(row: dict) -> bool:
    return bool(
        cell_works(
            leak=row.get("leak_tok"),
            on_sheet_kept=row.get("on_sheet_kept"),
            off_sheet=row.get("garble"),
            argmax_on_sheet=row.get("argmax_on_sheet"),
            swing_kept=row.get("swing_kept"),
        )
    )


def _sheet_bits(row: dict) -> dict:
    return {
        "pass": _sheet_ok(row),
        "kept": float(row.get("on_sheet_kept", float("nan"))),
        "leak": float(row.get("leak_tok", float("nan"))),
        "garble": float(row.get("garble", float("nan"))),
        "swing_kept": float(row.get("swing_kept", float("nan"))),
    }


def _exam_bits(row: dict) -> dict:
    return {
        "pass": bool(row.get("pass")),
        "overlap": float(row.get("roll_overlap", float("nan"))),
        "swing": float(row.get("roll_swing_kept", float("nan"))),
        "coherence": float(row.get("roll_coherence", float("nan"))),
    }


def run_adv(label: str, **cfg_kw) -> dict:
    cfg = default_cfg(steps=STEPS, seed=SEED, **cfg_kw)
    exam_cfg = default_cfg(steps=EXAM_STEPS, seed=SEED, **cfg_kw)
    t0 = time.time()
    leftover = score_adv_sheet(leaky_field(), cfg=cfg)
    gender = score_adv_sheet(gender_like_field(), cfg=cfg)
    divergent = score_adv_exam(divergent_field(seed=SEED), cfg=exam_cfg)
    close = score_adv_exam(close_field(seed=SEED), cfg=exam_cfg)
    unused = score_adv_exam(unused_e_field(seed=SEED), cfg=exam_cfg)
    field = score_field2d(cfg)
    cells = {
        "exam_divergent": bool(divergent["pass"]),
        "exam_close": bool(close["pass"]),
        "exam_unused_e": bool(unused["pass"]),
        "sheet_leftover": _sheet_ok(leftover),
        "sheet_gender": _sheet_ok(gender),
    }
    overlap = {
        "exam_divergent": divergent.get("roll_overlap"),
        "exam_close": close.get("roll_overlap"),
        "exam_unused_e": unused.get("roll_overlap"),
    }
    swing = {
        "exam_divergent": divergent.get("roll_swing_kept"),
        "exam_close": close.get("roll_swing_kept"),
        "exam_unused_e": unused.get("roll_swing_kept"),
    }
    score = exam_score(overlap, swing)
    verdict = compiled_verdict(cells=cells)
    return {
        "label": label,
        "cfg": {
            "steps": STEPS,
            "exam_steps": EXAM_STEPS,
            "seed": SEED,
            "b_cap": float(cfg.b_cap),
            "fm_weight": float(cfg.fm_weight),
            "fm_normalize": bool(cfg.fm_normalize),
            "cover_weight": float(cfg.cover_weight),
            "teacher": "faithful_guard_e",
        },
        "sec": round(time.time() - t0, 2),
        "cells": cells,
        "exam_score": score,
        "compiled": verdict,
        "sheet_leftover": _sheet_bits(leftover),
        "sheet_gender": _sheet_bits(gender),
        "exam_divergent": _exam_bits(divergent),
        "exam_close": _exam_bits(close),
        "exam_unused_e": _exam_bits(unused),
        "field2d": {
            "pass": bool(field["pass"]),
            "slider_cos": float(field.get("slider_cos", float("nan"))),
            "leak_ratio": float(field.get("leak_ratio", float("nan"))),
        },
        "gates": {
            "leak_lock": COMPILED_LEAK_LOCK,
            "sheet_lock": COMPILED_SHEET_LOCK,
            "garble_max": COMPILED_GARBLE_MAX,
            "swing_floor": COMPILED_SWING_FLOOR,
        },
    }


def run_supervised() -> dict:
    t0 = time.time()
    base_left = score_sheet(
        "faithful_raw", leaky_field(), pole_mode="hidden", teacher="faithful",
        steps=BASE_STEPS, seed=SEED,
    )
    base_mid = score_sheet(
        "v9_hidden", leaky_field(), pole_mode="hidden", teacher="pair_odd",
        steps=BASE_STEPS, seed=SEED,
    )
    base_div = score_exam(
        "faithful_raw", divergent_field(seed=SEED), pole_mode="hidden", teacher="faithful",
        steps=BASE_STEPS, seed=SEED,
    )
    base_close = score_exam(
        "pair_odd_midpoint", close_field(seed=SEED), pole_mode="hidden", teacher="pair_odd",
        steps=BASE_STEPS, seed=SEED,
    )
    return {
        "label": "supervised_baselines",
        "baseline_steps": BASE_STEPS,
        "sec": round(time.time() - t0, 2),
        "sheet_faithful_raw": _sheet_bits(base_left) | {"recipe": "faithful_raw/v6"},
        "sheet_pair_odd": _sheet_bits(base_mid) | {"recipe": "pair_odd_midpoint/v9"},
        "exam_faithful_divergent": _exam_bits(base_div) | {"recipe": "faithful_raw"},
        "exam_pair_odd_close": _exam_bits(base_close) | {"recipe": "pair_odd_midpoint"},
    }


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    arms = [
        run_adv("locked_rpgan_bcap_fm0", b_cap=1.0, fm_weight=0.0, cover_weight=1.5),
        run_adv(
            "distmatch_fm_norm_1",
            b_cap=1.0, fm_weight=1.0, cover_weight=1.5, fm_normalize=True,
        ),
        run_adv(
            "distmatch_fm_raw_0p5",
            b_cap=1.0, fm_weight=0.5, cover_weight=1.5, fm_normalize=False,
        ),
    ]
    supervised = run_supervised()
    blob = {"sha": sha, "arms": arms, "supervised": supervised}
    OUT_JSON.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    for a in arms:
        sl = a["sheet_leftover"]
        print(
            f"{a['label']}: compiled={a['compiled']} exam_score={a['exam_score']:.4f} "
            f"leftover kept={sl['kept']:.4f} leak={sl['leak']:+.4f} pass={sl['pass']} "
            f"cells={a['cells']} sec={a['sec']}"
        )
    print("supervised:", json.dumps(supervised, indent=2))
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
