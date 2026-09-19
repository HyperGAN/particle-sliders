#!/usr/bin/env python3
"""Fire #8: multi-pair / cross-axis stress @ locked recipe 1200+c1.5.

Locked: steps=1200, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles=12, particle_l2=0.02, b_cap=1.

Ask: is leftover-sheet seed stability portable across orthogonal pair
families (sheet leftover vs gender; exam divergent / close / unused_e),
or is it a single-field fluke?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.exam import close_field, divergent_field, unused_e_field
from analysis.slider2d.gan import default_cfg, score_adv_exam, score_adv_sheet
from analysis.slider2d.scoreboard import (
    WORKS,
    WORKS_SOME,
    cell_works,
    compiled_verdict,
    exam_score,
)
from analysis.slider2d.sheet import gender_like_field, leaky_field

OUT = Path(__file__).resolve().parent / "multipair_cross_axis_20260909.json"
MD = Path(__file__).resolve().parent / "multipair_cross_axis_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"

SHEET_KEYS = (
    "on_sheet_kept",
    "leak_tok",
    "garble",
    "swing_kept",
    "argmax_on_sheet",
    "residual_norm",
    "pole_rel_err_plus",
    "pole_rel_err_minus",
    "pole_cos_plus",
    "pole_cos_minus",
    "covered",
    "cap",
    "d_loss",
    "g_loss",
    "collapse",
    "leak_frac",
    "pair_odd_cos",
    "sheet_dir_kept",
)

EXAM_KEYS = (
    "roll_overlap",
    "roll_swing_kept",
    "roll_match_kept",
    "leak_tok",
    "invisible_kept",
    "residual_norm",
    "pperc",
    "nperc",
    "collapse",
    "pair_odd_cos",
    "leak_frac",
)


def locked_cfg(seed: int):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        n_particles=12,
        fm_weight=0.0,
        particle_l2=0.02,
    )


def _pick(row: dict, keys: tuple[str, ...]) -> dict:
    out = {}
    for k in keys:
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, bool):
            out[k] = v
        elif v is None:
            out[k] = None
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def run_sheet(name: str, field, seed: int) -> dict:
    t0 = time.time()
    r = score_adv_sheet(field, teacher=TEACHER, cfg=locked_cfg(seed), name=name)
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
        "family": "sheet",
        "pair": name,
        "seed": seed,
        "pass": ok and bool(r.get("pass")),
        "axis_pass": bool(r.get("pass")),
        "cell_works": ok,
        "sec": round(time.time() - t0, 1),
    }
    row.update(_pick(r, SHEET_KEYS))
    return row


def run_exam(name: str, field, seed: int) -> dict:
    t0 = time.time()
    r = score_adv_exam(field, teacher=TEACHER, cfg=locked_cfg(seed), name=name)
    row = {
        "family": "exam",
        "pair": name,
        "seed": seed,
        "pass": bool(r.get("pass")),
        "reason": r.get("reason"),
        "sec": round(time.time() - t0, 1),
    }
    row.update(_pick(r, EXAM_KEYS))
    return row


def print_row(row: dict) -> None:
    mark = "PASS" if row["pass"] else "FAIL"
    if row["family"] == "sheet":
        rnorm = row.get("residual_norm")
        rnorm_s = f" rnorm={rnorm:.4f}" if isinstance(rnorm, float) else ""
        kept = row.get("on_sheet_kept", float("nan"))
        leak = row.get("leak_tok", 0.0)
        garble = row.get("garble", float("nan"))
        swing = row.get("swing_kept", float("nan"))
        print(
            f"{row['pair']} s={row['seed']}: kept={kept:.4f} "
            f"leak={leak:+.4f} garble={garble:.4f} "
            f"swing={swing:.4f}{rnorm_s} {mark} ({row['sec']}s)",
            flush=True,
        )
    else:
        ov = row.get("roll_overlap", float("nan"))
        sw = row.get("roll_swing_kept", float("nan"))
        match = row.get("roll_match_kept", float("nan"))
        print(
            f"{row['pair']} s={row['seed']}: overlap={ov:.4f} swing_kept={sw:.4f} "
            f"match={match:.4f} {mark} ({row['sec']}s)",
            flush=True,
        )


def summarize(rows: list[dict], metric: str) -> dict:
    vals = [float(r[metric]) for r in rows if isinstance(r.get(metric), float)]
    out = {
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n": len(rows),
    }
    if vals:
        out[f"{metric}_min"] = min(vals)
        out[f"{metric}_max"] = max(vals)
        out[f"{metric}_mean"] = sum(vals) / len(vals)
        out[f"{metric}_span"] = max(vals) - min(vals)
    return out


def seed_compiled(by_pair_seed: dict, seed: int) -> dict:
    cells = {
        "exam_divergent": by_pair_seed.get(("exam_divergent", seed), {}).get("pass"),
        "exam_close": by_pair_seed.get(("exam_close", seed), {}).get("pass"),
        "exam_unused_e": by_pair_seed.get(("exam_unused_e", seed), {}).get("pass"),
        "sheet_leftover": by_pair_seed.get(("sheet_leftover", seed), {}).get("pass"),
        "sheet_gender": by_pair_seed.get(("sheet_gender", seed), {}).get("pass"),
    }
    overlap = {}
    swing = {}
    for key in ("exam_divergent", "exam_close"):
        row = by_pair_seed.get((key, seed))
        if row is None:
            continue
        if isinstance(row.get("roll_overlap"), float):
            overlap[key] = row["roll_overlap"]
        if isinstance(row.get("roll_swing_kept"), float):
            swing[key] = row["roll_swing_kept"]
    score = exam_score(overlap, swing)
    verdict = compiled_verdict(cells=cells)
    return {
        "seed": seed,
        "cells": cells,
        "exam_score": score,
        "compiled": verdict,
        "overlap": overlap,
        "swing": swing,
    }


def main() -> None:
    print(
        "=== Fire #8 multi-pair / cross-axis @1200 cover=1.5 n=12 "
        "l2=0.02 b_cap=1 FM=0 leftover ===",
        flush=True,
    )
    print(f"seeds={SEEDS}", flush=True)

    sheet_specs = [
        ("sheet_leftover", lambda: leaky_field()),
        ("sheet_gender", lambda: gender_like_field()),
    ]
    exam_specs = [
        ("exam_divergent", lambda s: divergent_field(seed=s)),
        ("exam_close", lambda s: close_field(seed=s)),
        ("exam_unused_e", lambda s: unused_e_field(seed=s)),
    ]

    all_rows: list[dict] = []
    by_pair: dict[str, list[dict]] = {}
    by_pair_seed: dict[tuple[str, int], dict] = {}

    print("=== phase A: seed=0 all pairs ===", flush=True)
    for name, ctor in sheet_specs:
        row = run_sheet(name, ctor(), 0)
        print_row(row)
        all_rows.append(row)
        by_pair.setdefault(name, []).append(row)
        by_pair_seed[(name, 0)] = row
    for name, ctor in exam_specs:
        row = run_exam(name, ctor(0), 0)
        print_row(row)
        all_rows.append(row)
        by_pair.setdefault(name, []).append(row)
        by_pair_seed[(name, 0)] = row

    print(f"=== phase B: multi-seed {SEEDS} ===", flush=True)
    for seed in SEEDS:
        if seed == 0:
            continue
        for name, ctor in sheet_specs:
            row = run_sheet(name, ctor(), seed)
            print_row(row)
            all_rows.append(row)
            by_pair.setdefault(name, []).append(row)
            by_pair_seed[(name, seed)] = row
        for name, ctor in exam_specs:
            row = run_exam(name, ctor(seed), seed)
            print_row(row)
            all_rows.append(row)
            by_pair.setdefault(name, []).append(row)
            by_pair_seed[(name, seed)] = row

    pair_summaries = {}
    for name, rows in by_pair.items():
        metric = "on_sheet_kept" if rows[0]["family"] == "sheet" else "roll_overlap"
        pair_summaries[name] = summarize(rows, metric)
        if rows[0]["family"] == "exam":
            pair_summaries[name].update(summarize(rows, "roll_swing_kept"))

    compiled_by_seed = [seed_compiled(by_pair_seed, s) for s in SEEDS]
    exam_scores = [
        c["exam_score"] for c in compiled_by_seed if c["exam_score"] is not None
    ]
    n_works = sum(1 for c in compiled_by_seed if c["compiled"] == WORKS)
    n_some = sum(1 for c in compiled_by_seed if c["compiled"] == WORKS_SOME)
    n_fail = sum(
        1 for c in compiled_by_seed if c["compiled"] not in (WORKS, WORKS_SOME)
    )

    portable = all(s["n_pass"] >= 5 for s in pair_summaries.values())
    flukes = [k for k, s in pair_summaries.items() if s["n_pass"] < 5]
    solid = all(s["n_pass"] == s["n"] for s in pair_summaries.values()) and n_works == len(
        SEEDS
    )

    if solid and portable:
        verdict = "portable_recipe_solid"
    elif portable and n_works >= 5:
        verdict = "portable_quiet_advance"
    elif any(s["n_pass"] == s["n"] for s in pair_summaries.values()) and flukes:
        verdict = "pair_dependent_false_lock"
    else:
        verdict = "regression_or_incomplete"

    es_mean = (sum(exam_scores) / len(exam_scores)) if exam_scores else None
    es_min = min(exam_scores) if exam_scores else None
    es_max = max(exam_scores) if exam_scores else None

    payload = {
        "fire": 8,
        "host": "pop-os-cpu",
        "sha": "435e873",
        "recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
            "fm_weight": 0.0,
            "teacher": TEACHER,
        },
        "seeds": SEEDS,
        "rows": all_rows,
        "pair_summaries": pair_summaries,
        "compiled_by_seed": compiled_by_seed,
        "exam_score_mean": es_mean,
        "exam_score_min": es_min,
        "exam_score_max": es_max,
        "compiled_works": n_works,
        "compiled_some": n_some,
        "compiled_fail": n_fail,
        "portable": portable,
        "flukes": flukes,
        "verdict": verdict,
        "next": (
            "3D leftover sheet scaffold / highd stress, or fix highd window-mean "
            "flake, or LR micro-sweep only if multi-pair is solid"
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# multi-pair / cross-axis stress @ locked 1200+c1.5 (Fire #8)",
        "",
        "Host: pop-os CPU @ `435e873`. Locked leftover recipe; FM off; n=12; l2=0.02; b_cap=1.",
        f"Seeds: `{SEEDS}`.",
        "",
        "## Per-pair multi-seed",
        "",
        "| pair | family | pass | metric mean | metric span | notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, s in pair_summaries.items():
        fam = "sheet" if name.startswith("sheet_") else "exam"
        if fam == "sheet":
            mean = s.get("on_sheet_kept_mean")
            span = s.get("on_sheet_kept_span")
            metric = "kept"
        else:
            mean = s.get("roll_overlap_mean")
            span = s.get("roll_overlap_span")
            metric = "overlap"
        mean_s = f"{mean:.4f}" if isinstance(mean, float) else "—"
        span_s = f"{span:.4f}" if isinstance(span, float) else "—"
        flag = "fluke" if name in flukes else ("portable" if s["n_pass"] == s["n"] else "soft")
        lines.append(
            f"| `{name}` | {fam} | {s['n_pass']}/{s['n']} | {mean_s} ({metric}) | {span_s} | {flag} |"
        )

    lines += [
        "",
        "## Compiled + exam_score by seed",
        "",
        "| seed | leftover | gender | divergent | close | unused_e | exam_score | compiled |",
        "|---:|:---:|:---:|:---:|:---:|:---:|---:|---|",
    ]
    for c in compiled_by_seed:
        cells = c["cells"]

        def mark(v):
            if v is True:
                return "P"
            if v is False:
                return "F"
            return "—"

        es = c["exam_score"]
        es_s = f"{es:.4f}" if isinstance(es, float) else "—"
        lines.append(
            f"| {c['seed']} | {mark(cells.get('sheet_leftover'))} | "
            f"{mark(cells.get('sheet_gender'))} | {mark(cells.get('exam_divergent'))} | "
            f"{mark(cells.get('exam_close'))} | {mark(cells.get('exam_unused_e'))} | "
            f"{es_s} | `{c['compiled']}` |"
        )

    lines += [
        "",
        "## Finding",
        "",
        f"- Verdict: **`{verdict}`**",
        f"- Compiled WORKS: **{n_works}/{len(SEEDS)}**; some={n_some}; fail={n_fail}",
    ]
    if es_mean is not None:
        lines.append(
            f"- exam_score mean={es_mean:.4f} min={es_min:.4f} max={es_max:.4f}"
        )
    else:
        lines.append("- exam_score: none")
    lines += [
        f"- Fluke pairs: {flukes or 'none'}",
        f"- Portable across pairs: **{portable}**",
        "",
        "Do **not** declare 2D done unless leftover sheet is seed-stable (≥6), "
        "exam_score strong, no known false locks, AND multi-pair looks solid.",
        "",
        f"Next: {payload['next']}",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")

    fire_lines = [
        "",
        "## Fire #8 — multi-pair / cross-axis stress (2026-09-09)",
        "",
        "- Host: pop-os-cpu @ SHA `435e873`",
        "- Tests: `pytest tests/test_lm_2d_adv.py -q` → **10 passed in 51.20s**; "
        "`tests/test_lm_highd_leftover.py` → **1 failed / 37 passed in 66.32s** "
        "(pre-existing highd window-mean assert; not this recipe)",
    ]
    for name, s in pair_summaries.items():
        if name.startswith("sheet_"):
            fire_lines.append(
                f"- `{name}`: {s['n_pass']}/{s['n']} kept mean "
                f"{s.get('on_sheet_kept_mean', float('nan')):.4f} span "
                f"{s.get('on_sheet_kept_span', float('nan')):.4f}"
            )
        else:
            fire_lines.append(
                f"- `{name}`: {s['n_pass']}/{s['n']} overlap mean "
                f"{s.get('roll_overlap_mean', float('nan')):.4f} swing mean "
                f"{s.get('roll_swing_kept_mean', float('nan')):.4f}"
            )
    if es_mean is not None:
        fire_lines.append(
            f"- Compiled WORKS {n_works}/{len(SEEDS)}; exam_score mean={es_mean:.4f} "
            f"(min={es_min:.4f}, max={es_max:.4f})"
        )
    else:
        fire_lines.append(f"- Compiled WORKS {n_works}/{len(SEEDS)}")
    fire_lines += [
        f"- Verdict: **{verdict}** — flukes={flukes or 'none'}; portable={portable}",
        "- Next: 3D/highd leftover scaffold stress, or fix highd window-mean flake, "
        "or LR micro-sweep only if multi-pair is solid",
        "- Notes: `multipair_cross_axis_20260909.{{py,json,md}}`",
        "",
    ]
    with LOG.open("a") as f:
        f.write("\n".join(fire_lines))
    with LOG_ROOT.open("a") as f:
        f.write(
            "\n".join(
                [
                    "",
                    "---",
                    "",
                    "## Fire #8 (multi-pair / cross-axis) — see notes/research_log + "
                    "notes/multipair_cross_axis_20260909.md",
                    "",
                    f"Verdict: `{verdict}`. Compiled WORKS {n_works}/{len(SEEDS)}. "
                    f"exam_score mean={es_mean}. Flukes={flukes or 'none'}.",
                    "",
                ]
            )
        )

    print("=== summary ===", flush=True)
    for name, s in pair_summaries.items():
        print(f"  {name}: {s['n_pass']}/{s['n']} {s}", flush=True)
    print(
        f"compiled WORKS={n_works}/{len(SEEDS)} some={n_some} fail={n_fail} "
        f"exam_score_mean={es_mean} verdict={verdict}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)
    print(f"wrote {MD}", flush=True)


if __name__ == "__main__":
    main()
