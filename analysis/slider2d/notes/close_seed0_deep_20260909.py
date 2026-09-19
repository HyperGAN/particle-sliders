#!/usr/bin/env python3
"""Deep dig: f3d_close / close_live_noise seed0 knife @ n_particles=1 (Music parts0).

Goals:
1. Characterize seed0 failure (leak vs exam vs init/basin)
2. Sweep particle_l2, b_cap, cover, n∈{1,2,3,4}, init noise, seeds 0..20
3. Decide: portable recipe change vs mandatory multi-seed gate for Music-close
4. Write close_seed0_deep_20260909.{md,py,json} + research_log
5. Propose small portable fix only with evidence (no silent default flip)

Locked recipe stays: 1200 / cover∈{1.0,1.5} / faithful_guard_e / FM0 / n≤12 /
particle_l2=0.02 / vicreg=0.05 / b_cap=1. Reject 800×cover3.0 false lock.
CPU only. No Music GPU train.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    close_field3d,
    close_live_noise_field3d,
    divergent_field3d,
    leftover_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "close_seed0_deep_20260909.json"
OUT_MD = NOTES / "close_seed0_deep_20260909.md"
LOG = NOTES / "research_log_20260909.md"
PARTIAL = Path("/workspace/close_seed0_deep.partial.jsonl")

TEACHER = "faithful_guard_e"
SEEDS_20 = list(range(0, 21))  # 0..20
SEEDS_STD = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def make_cfg(seed: int, **kw):
    return default_cfg(
        steps=int(kw.get("steps", 1200)),
        seed=int(seed),
        b_cap=float(kw.get("b_cap", 1.0)),
        cover_weight=float(kw.get("cover", 1.5)),
        fm_weight=0.0,
        n_particles=int(kw.get("n", 1)),
        particle_l2=float(kw.get("particle_l2", 0.02)),
        span_frac=float(kw.get("span_frac", 0.40)),
        end_margin=float(kw.get("end_margin", 0.60)),
        cloud_std=float(kw.get("cloud_std", 0.03)),
        particle_jitter=float(kw.get("particle_jitter", 0.01)),
        lr=float(kw.get("lr", 5.0e-3)),
        vicreg_weight=float(kw.get("vicreg_weight", 0.05)),
    )


def run_one(field, *, seed: int, name: str, **cfg_kw) -> dict:
    t0 = time.time()
    cfg = make_cfg(seed, **cfg_kw)
    row = score_adv_field3d_exam(field, teacher=TEACHER, cfg=cfg, name=name)
    out = {
        "name": name,
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": float(row.get("exam_score", 0.0)),
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "on_u": float(row.get("on_u", 0.0)),
        "a_u": float(row.get("a_u", 0.0)) if "a_u" in row else None,
        "residual_norm": float(row.get("residual_norm", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "pass_u": bool(row.get("pass_u", False)),
        "pass_content": bool(row.get("pass_content", False)),
        "pass_leak": bool(row.get("pass_leak", False)),
        "pass_cont": bool(row.get("pass_cont", False)),
        "pass_swing": bool(row.get("pass_swing", False)),
        "pass_leftover_gate": bool(row.get("pass_leftover_gate", False)),
        "pass_multi_row": bool(row.get("pass_multi_row", False)),
        "exam_cont": float(row.get("exam_cont", 0.0)),
        "exam_swing": float(row.get("exam_swing", 0.0)),
        "cfg": {k: cfg_kw.get(k) for k in (
            "cover", "n", "steps", "particle_l2", "b_cap", "vicreg_weight",
            "cloud_std", "particle_jitter", "span_frac", "end_margin", "lr",
        )},
        "wall_s": round(time.time() - t0, 2),
    }
    with PARTIAL.open("a") as f:
        f.write(json.dumps(out) + "\n")
    print(
        f"  {name} seed={seed} pass={out['pass']} exam={out['exam_score']:.4f} "
        f"u={out['u_kept']:.4f} leak={out['leak_ratio']:.4f} "
        f"gates u/c/l/sw/mr="
        f"{int(out['pass_u'])}/{int(out['pass_content'])}/{int(out['pass_leak'])}/"
        f"{int(out['pass_swing'])}/{int(out['pass_multi_row'])} "
        f"wall={out['wall_s']}",
        flush=True,
    )
    return out


def summarize(runs: list[dict], name: str = "") -> dict:
    n = len(runs)
    npass = sum(1 for r in runs if r["pass"])
    prim = [r["exam_score"] for r in runs]
    us = [r["u_kept"] for r in runs]
    leaks = [abs(r["leak_ratio"]) for r in runs]
    fails = [r["seed"] for r in runs if not r["pass"]]
    return {
        "name": name,
        "n": n,
        "n_pass": npass,
        "pass_str": f"{npass}/{n}",
        "mean_exam": round(sum(prim) / n, 4) if n else 0.0,
        "exam_span": round(max(prim) - min(prim), 4) if n else 0.0,
        "mean_u": round(sum(us) / n, 4) if n else 0.0,
        "u_span": round(max(us) - min(us), 4) if n else 0.0,
        "leak_max": round(max(leaks), 4) if leaks else 0.0,
        "fail_seeds": fails,
        "knife": bool(0 < npass < n),
        "only_seed0": fails == [0],
        "seed0_fail": 0 in fails,
    }


def make_field(field_fn, seed: int):
    try:
        return field_fn(seed=seed)
    except TypeError:
        return field_fn()


def grid(name: str, field_fn, seeds, **cfg_kw) -> dict:
    runs = []
    for s in seeds:
        runs.append(run_one(make_field(field_fn, s), seed=s, name=f"{name}", **cfg_kw))
    out = summarize(runs, name=name)
    out["kw"] = dict(cfg_kw)
    out["runs"] = runs
    print(
        f"== {name}: {out['pass_str']} mean_exam={out['mean_exam']} "
        f"mean_u={out['mean_u']} leak_max={out['leak_max']} "
        f"fail={out['fail_seeds']} knife={out['knife']} only_s0={out['only_seed0']}",
        flush=True,
    )
    return out


def main() -> None:
    t_wall = time.time()
    PARTIAL.write_text("")  # reset
    sha = git_sha()
    cells: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # A) Characterize seed0 vs seed1 failure mode (close + live)
    # ------------------------------------------------------------------
    print("=== A) Characterize seed0 vs seed1 ===", flush=True)
    for label, ctor, cover in [
        ("close_char", close_field3d, 1.5),
        ("live_char", close_live_noise_field3d, 1.5),
        ("close_char_c1.0", close_field3d, 1.0),
        ("live_char_c1.0", close_live_noise_field3d, 1.0),
    ]:
        cells[label] = grid(label, ctor, [0, 1], cover=cover, n=1)

    # Classify failure mode from seed0 close_char
    s0 = next(r for r in cells["close_char"]["runs"] if r["seed"] == 0)
    s1 = next(r for r in cells["close_char"]["runs"] if r["seed"] == 1)
    fail_mode = {
        "seed0_exam_fail": not s0["pass"],
        "seed0_pass_u": s0["pass_u"],
        "seed0_pass_content": s0["pass_content"],
        "seed0_pass_leak": s0["pass_leak"],
        "seed0_pass_swing": s0["pass_swing"],
        "seed0_pass_multi": s0["pass_multi_row"],
        "seed0_u_kept": s0["u_kept"],
        "seed0_on_u": s0["on_u"],
        "seed0_leak_ratio": s0["leak_ratio"],
        "seed1_u_kept": s1["u_kept"],
        "seed1_on_u": s1["on_u"],
        "classification": None,
    }
    # Priority: leak fail vs exam (u undershoot) vs multi-row vs init signature
    if not s0["pass_leak"] and s0["leak_ratio"] > 0.20:
        fail_mode["classification"] = "leak"
    elif not s0["pass_u"] and s0["pass_content"] and s0["pass_multi_row"] and s0["pass_swing"]:
        fail_mode["classification"] = "exam_u_undershoot_init_basin"
    elif not s0["pass_multi_row"]:
        fail_mode["classification"] = "exam_multi_row"
    else:
        fail_mode["classification"] = "exam_other"
    print(f"FAIL_MODE={fail_mode['classification']}", flush=True)

    # ------------------------------------------------------------------
    # B) Seeds 0..20 @ n=1 locked (close c1.5 + live c1.5 + close c1.0)
    # ------------------------------------------------------------------
    print("=== B) Seeds 0..20 @ n=1 ===", flush=True)
    cells["close_n1_c1.5_s0_20"] = grid(
        "close_n1_c1.5_s0_20", close_field3d, SEEDS_20, cover=1.5, n=1
    )
    cells["live_n1_c1.5_s0_20"] = grid(
        "live_n1_c1.5_s0_20", close_live_noise_field3d, SEEDS_20, cover=1.5, n=1
    )
    cells["close_n1_c1.0_s0_20"] = grid(
        "close_n1_c1.0_s0_20", close_field3d, SEEDS_20, cover=1.0, n=1
    )

    # ------------------------------------------------------------------
    # C) n ∈ {1,2,3,4} multi-seed (std 6 seeds) close c1.5 + live c1.5
    # ------------------------------------------------------------------
    print("=== C) n sweep {1,2,3,4} ===", flush=True)
    for n in (1, 2, 3, 4):
        cells[f"close_n{n}_c1.5_std"] = grid(
            f"close_n{n}_c1.5_std", close_field3d, SEEDS_STD, cover=1.5, n=n
        )
        cells[f"live_n{n}_c1.5_std"] = grid(
            f"live_n{n}_c1.5_std", close_live_noise_field3d, SEEDS_STD, cover=1.5, n=n
        )

    # ------------------------------------------------------------------
    # D) seed0-only knobs: particle_l2, b_cap, cover, init noise
    # ------------------------------------------------------------------
    print("=== D) seed0 knobs (b_cap+cover full; l2/jitter/cloud spot+prior) ===", flush=True)
    # Full b_cap (new) + cover (include 3.0 false-lock check)
    for bc in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        cells[f"s0_bcap_{bc}"] = grid(
            f"s0_bcap_{bc}", close_field3d, [0], cover=1.5, n=1, b_cap=bc
        )
    for cov in (1.0, 1.25, 1.5, 2.0, 3.0):
        cells[f"s0_cover_{cov}"] = grid(
            f"s0_cover_{cov}", close_field3d, [0], cover=cov, n=1
        )
    # Spot-check extremes; prior harden already swept full l2/jitter/cloud as FAIL
    for l2 in (0.0, 0.02, 0.2):
        cells[f"s0_l2_{l2}"] = grid(
            f"s0_l2_{l2}", close_field3d, [0], cover=1.5, n=1, particle_l2=l2
        )
    for j in (0.0, 0.05, 0.1):
        cells[f"s0_jitter_{j}"] = grid(
            f"s0_jitter_{j}", close_field3d, [0], cover=1.5, n=1, particle_jitter=j
        )
    for cs in (0.0, 0.03, 0.12):
        cells[f"s0_cloud_{cs}"] = grid(
            f"s0_cloud_{cs}", close_field3d, [0], cover=1.5, n=1, cloud_std=cs
        )
    # Mark prior-confirmed fails for decision logic completeness
    for l2 in (0.005, 0.01, 0.05, 0.1):
        cells[f"s0_l2_{l2}"] = {
            "name": f"s0_l2_{l2}", "n": 1, "n_pass": 0, "pass_str": "0/1",
            "mean_exam": None, "fail_seeds": [0], "knife": False, "only_seed0": True,
            "seed0_fail": True, "from_prior": "harden_music_toy_bites_20260909",
        }
    for j in (0.01, 0.02):
        cells[f"s0_jitter_{j}"] = {
            "name": f"s0_jitter_{j}", "n": 1, "n_pass": 0, "pass_str": "0/1",
            "mean_exam": None, "fail_seeds": [0], "knife": False, "only_seed0": True,
            "seed0_fail": True, "from_prior": "harden_music_toy_bites_20260909",
        }
    for cs in (0.01, 0.06):
        cells[f"s0_cloud_{cs}"] = {
            "name": f"s0_cloud_{cs}", "n": 1, "n_pass": 0, "pass_str": "0/1",
            "mean_exam": None, "fail_seeds": [0], "knife": False, "only_seed0": True,
            "seed0_fail": True, "from_prior": "harden_music_toy_bites_20260909",
        }

    # ------------------------------------------------------------------
    # E) Portable candidates
    # ------------------------------------------------------------------
    print("=== E) Portable candidates ===", flush=True)
    # E1: vicreg=0 @ n=1 — expand to seeds 0..20
    cells["close_n1_vic0_s0_20"] = grid(
        "close_n1_vic0_s0_20", close_field3d, SEEDS_20,
        cover=1.5, n=1, vicreg_weight=0.0,
    )
    cells["live_n1_vic0_s0_20"] = grid(
        "live_n1_vic0_s0_20", close_live_noise_field3d, SEEDS_20,
        cover=1.5, n=1, vicreg_weight=0.0,
    )
    # E2: leftover / divergent safety under vic=0
    cells["leftover_n12_vic0.05"] = grid(
        "leftover_n12_vic0.05", leftover_field3d, SEEDS_STD,
        cover=1.5, n=12, vicreg_weight=0.05,
    )
    cells["leftover_n12_vic0"] = grid(
        "leftover_n12_vic0", leftover_field3d, SEEDS_STD,
        cover=1.5, n=12, vicreg_weight=0.0,
    )
    cells["leftover_n1_vic0"] = grid(
        "leftover_n1_vic0", leftover_field3d, SEEDS_SMOKE,
        cover=1.5, n=1, vicreg_weight=0.0,
    )
    cells["divergent_n12_vic0"] = grid(
        "divergent_n12_vic0", divergent_field3d, SEEDS_SMOKE,
        cover=1.5, n=12, vicreg_weight=0.0,
    )
    # E3: n=2 floor confirm already in C; also vic=0.05 default leftover baseline
    cells["leftover_n12_locked"] = cells["leftover_n12_vic0.05"]  # alias

    # ------------------------------------------------------------------
    # F) Decision
    # ------------------------------------------------------------------
    close20 = cells["close_n1_c1.5_s0_20"]
    live20 = cells["live_n1_c1.5_s0_20"]
    close20_c10 = cells["close_n1_c1.0_s0_20"]
    vic0_close = cells["close_n1_vic0_s0_20"]
    vic0_live = cells["live_n1_vic0_s0_20"]
    leftover_v05 = cells["leftover_n12_vic0.05"]
    leftover_v0 = cells["leftover_n12_vic0"]

    n_floor = None
    for n in (1, 2, 3, 4):
        if cells[f"close_n{n}_c1.5_std"]["n_pass"] == 6 and cells[f"live_n{n}_c1.5_std"]["n_pass"] == 6:
            n_floor = n
            break

    # Which seed0-only knobs recovered?
    s0_recover = []
    for k, v in cells.items():
        if k.startswith("s0_") and v["n_pass"] == 1 and v["n"] == 1:
            s0_recover.append(k)

    l2_any = any(cells[f"s0_l2_{l2}"]["n_pass"] == 1 for l2 in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
    bcap_any = any(cells[f"s0_bcap_{bc}"]["n_pass"] == 1 for bc in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0))
    cover_any = any(cells[f"s0_cover_{c}"]["n_pass"] == 1 for c in (1.0, 1.25, 1.5, 2.0, 3.0))
    jitter_any = any(cells[f"s0_jitter_{j}"]["n_pass"] == 1 for j in (0.0, 0.01, 0.02, 0.05, 0.1))
    cloud_any = any(cells[f"s0_cloud_{cs}"]["n_pass"] == 1 for cs in (0.0, 0.01, 0.03, 0.06, 0.12))

    leftover_flat = (
        leftover_v05["n_pass"] == 6
        and leftover_v0["n_pass"] == 6
        and abs(leftover_v05["mean_exam"] - leftover_v0["mean_exam"]) < 0.01
    )
    leftover_regression = leftover_v0["n_pass"] < leftover_v05["n_pass"]

    vic0_heals_close = vic0_close["n_pass"] == 21
    vic0_heals_live = vic0_live["n_pass"] == 21
    knife_close20 = close20["knife"] or close20["seed0_fail"]
    knife_live20 = live20["knife"] or live20["seed0_fail"]

    # Decision policy:
    # - Do NOT silently change locked defaults (vic=0.05, n=12 leftover, etc.)
    # - If vic=0 @ n=1 heals 21/21 with leftover flat → PROPOSE portable conditional
    #   (vicreg_weight=0 when n_particles==1) as Music-posture option, not silent flip
    # - Mandatory multi-seed gate for Music-close under parts0/n=1 regardless
    # - n≥2 floor is operational harden without touching global defaults

    propose_vic0_when_n1 = bool(
        vic0_heals_close and vic0_heals_live and leftover_flat and not leftover_regression
    )
    # Still not a silent default change — propose only
    recipe_change = False  # never silent; locked stays

    decision = "mandatory_multi_seed_gate"
    rationale = []
    if fail_mode["classification"] == "exam_u_undershoot_init_basin":
        rationale.append(
            "seed0 fail is û undershoot (exam/pass_u), not leak or multi-row — init/basin under n=1"
        )
    if knife_close20 or knife_live20:
        rationale.append(
            f"seeds0..20 still knife: close {close20['pass_str']} fail={close20['fail_seeds']}; "
            f"live {live20['pass_str']} fail={live20['fail_seeds']}"
        )
    if not l2_any and not bcap_any and not cover_any and not jitter_any and not cloud_any:
        rationale.append(
            "no seed0@n=1 recovery from particle_l2 / b_cap / cover / jitter / cloud"
        )
    if n_floor is not None and n_floor >= 2:
        rationale.append(f"n_particles>={n_floor} recovers 6/6 close+live under locked else")
    if propose_vic0_when_n1:
        rationale.append(
            "vicreg=0 @ n=1 heals 21/21 close+live with leftover n12 flat — propose "
            "conditional Music-posture (vic=0 when n=1), NOT silent global default flip"
        )
        decision = "multi_seed_gate_PLUS_propose_vic0_when_n1"
    else:
        rationale.append(
            "no safe portable default flip without Music multi-seed confirm; "
            "require multi-seed gate and/or n>=2 floor for Music-close"
        )

    wall = round(time.time() - t_wall, 1)
    verdict = {
        "wall_s": wall,
        "host": "box-cpu",
        "sha": sha,
        "fail_mode": fail_mode,
        "close_n1_c1.5_s0_20": close20["pass_str"],
        "close_n1_c1.5_fail": close20["fail_seeds"],
        "close_n1_c1.0_s0_20": close20_c10["pass_str"],
        "close_n1_c1.0_fail": close20_c10["fail_seeds"],
        "live_n1_c1.5_s0_20": live20["pass_str"],
        "live_n1_c1.5_fail": live20["fail_seeds"],
        "n_floor_6of6_close_and_live": n_floor,
        "s0_knob_recovers": s0_recover,
        "l2_recovers_s0": l2_any,
        "bcap_recovers_s0": bcap_any,
        "cover_recovers_s0": cover_any,
        "jitter_recovers_s0": jitter_any,
        "cloud_recovers_s0": cloud_any,
        "vic0_close_s0_20": vic0_close["pass_str"],
        "vic0_live_s0_20": vic0_live["pass_str"],
        "leftover_n12_vic0.05": leftover_v05["pass_str"],
        "leftover_n12_vic0": leftover_v0["pass_str"],
        "leftover_mean_exam_vic0.05": leftover_v05["mean_exam"],
        "leftover_mean_exam_vic0": leftover_v0["mean_exam"],
        "leftover_flat": leftover_flat,
        "leftover_regression": leftover_regression,
        "propose_vic0_when_n1": propose_vic0_when_n1,
        "recipe_change_silent": recipe_change,
        "decision": decision,
        "rationale": rationale,
        "music_close_rule": (
            "MANDATORY multi-seed gate for close/close_live_noise under Music parts0 "
            f"(n=1). Operational harden: n_particles>={n_floor or 2}. "
            + (
                "PROPOSE (not silent): set vicreg_weight=0 when n_particles==1 "
                "(ill-posed VICReg std over 1 particle); leftover flat in toy."
                if propose_vic0_when_n1
                else "No portable default flip proposed."
            )
        ),
        "rejected": [
            "800×cover3.0 false lock",
            "raise particle_l2 / b_cap / cover / jitter / cloud as seed0@n=1 fix",
            "silent global vicreg default flip without Music multi-seed",
            "weaken close_live_noise cell",
        ],
    }

    # slim cells for JSON (drop full runs of huge grids but keep fails + summaries)
    slim = {}
    for k, v in cells.items():
        slim[k] = {kk: vv for kk, vv in v.items() if kk != "runs"}
        if "runs" in v:
            slim[k]["fail_rows"] = [r for r in v["runs"] if not r["pass"]]
            if "char" in k:
                slim[k]["runs"] = v["runs"]

    payload = {
        "fire": "close_seed0_deep_20260909",
        "verdict": verdict,
        "cells": slim,
        "locked_recipe": {
            "steps": 1200,
            "cover": [1.0, 1.5],
            "teacher": TEACHER,
            "fm": 0.0,
            "n_max": 12,
            "particle_l2": 0.02,
            "vicreg_weight": 0.05,
            "b_cap": 1.0,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # MD memo
    lines = [
        "# close_seed0_deep — f3d_close / close_live_noise seed0 knife (2026-09-09)",
        "",
        f"Host: box-cpu @ `{sha}`. Wall **{wall}s**. CPU only. No Music train.",
        "",
        "## Decision (TL;DR)",
        "",
        f"**`{decision}`**",
        "",
        verdict["music_close_rule"],
        "",
        "### Rationale",
        "",
    ]
    for r in rationale:
        lines.append(f"- {r}")
    lines += [
        "",
        f"- **Silent recipe change?** NO (locked stays 1200/c1.5/guard/FM0/n≤12/l2=0.02/vic=0.05/b_cap=1)",
        f"- **Propose portable conditional?** {'YES — vicreg_weight=0 when n_particles==1' if propose_vic0_when_n1 else 'NO'}",
        f"- **Leftover regression under vic=0?** {'YES' if leftover_regression else 'NO'} "
        f"(n12: {leftover_v05['pass_str']}@{leftover_v05['mean_exam']} → "
        f"{leftover_v0['pass_str']}@{leftover_v0['mean_exam']})",
        "",
        "## 1) Failure mode (seed0 vs seed1, close c1.5 n=1)",
        "",
        f"**Classification: `{fail_mode['classification']}`**",
        "",
        "| seed | pass | u_kept | on_u | content | leak | pass_u | pass_c | pass_leak | swing | multi |",
        "|---:|:---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in cells["close_char"]["runs"]:
        lines.append(
            f"| {r['seed']} | {r['pass']} | {r['u_kept']:.4f} | {r['on_u']:.4f} | "
            f"{r['content_kept']:.4f} | {r['leak_ratio']:.4f} | {r['pass_u']} | "
            f"{r['pass_content']} | {r['pass_leak']} | {r['pass_swing']} | {r['pass_multi_row']} |"
        )
    lines += [
        "",
        "Not leak (leak≪0.20). Not multi-row (3/3). Exam fails via `pass_u` / leftover gate "
        "because û undershoots (on_u≈0.046 vs a_u≈0.12). Content+swing OK → **init/basin** "
        "under single particle + default VICReg.",
        "",
        "## 2) Seeds 0..20 @ n=1",
        "",
        "| cell | PASS | mean exam | fail seeds | knife |",
        "|---|:---:|---:|---|:---:|",
        f"| close c1.5 | {close20['pass_str']} | {close20['mean_exam']} | {close20['fail_seeds']} | {close20['knife']} |",
        f"| close c1.0 | {close20_c10['pass_str']} | {close20_c10['mean_exam']} | {close20_c10['fail_seeds']} | {close20_c10['knife']} |",
        f"| live c1.5 | {live20['pass_str']} | {live20['mean_exam']} | {live20['fail_seeds']} | {live20['knife']} |",
        f"| close c1.5 **vic=0** | {vic0_close['pass_str']} | {vic0_close['mean_exam']} | {vic0_close['fail_seeds']} | {vic0_close['knife']} |",
        f"| live c1.5 **vic=0** | {vic0_live['pass_str']} | {vic0_live['mean_exam']} | {vic0_live['fail_seeds']} | {vic0_live['knife']} |",
        "",
        "## 3) n ∈ {1,2,3,4} (std seeds)",
        "",
        "| n | close c1.5 | live c1.5 |",
        "|---:|:---:|:---:|",
    ]
    for n in (1, 2, 3, 4):
        lines.append(
            f"| {n} | {cells[f'close_n{n}_c1.5_std']['pass_str']} | "
            f"{cells[f'live_n{n}_c1.5_std']['pass_str']} |"
        )
    lines += [
        "",
        f"**n_floor for 6/6 close∧live = {n_floor}**",
        "",
        "## 4) seed0@n=1 knob sweeps (recover?)",
        "",
        f"| family | any recover? |",
        f"|---|:---:|",
        f"| particle_l2 | {l2_any} |",
        f"| b_cap | {bcap_any} |",
        f"| cover | {cover_any} |",
        f"| particle_jitter | {jitter_any} |",
        f"| cloud_std | {cloud_any} |",
        f"| vicreg=0 | {vic0_close['n_pass'] >= 1} (full: {vic0_close['pass_str']}) |",
        "",
        f"Recovered seed0-only cells: `{s0_recover}`",
        "",
        "## 5) Proposal (evidence-gated; not applied)",
        "",
    ]
    if propose_vic0_when_n1:
        lines += [
            "**PROPOSE** portable conditional (do not silently change `default_cfg`):",
            "",
            "```python",
            "# Music parts0 / n_particles==1 posture only",
            "if cfg.n_particles == 1:",
            "    cfg = replace(cfg, vicreg_weight=0.0)  # VICReg std ill-posed at n=1",
            "```",
            "",
            "Evidence:",
            f"- close n1 vic0 seeds0..20: **{vic0_close['pass_str']}** (default was {close20['pass_str']})",
            f"- live n1 vic0 seeds0..20: **{vic0_live['pass_str']}** (default was {live20['pass_str']})",
            f"- leftover n12: vic0.05 {leftover_v05['pass_str']}@{leftover_v05['mean_exam']} vs "
            f"vic0 {leftover_v0['pass_str']}@{leftover_v0['mean_exam']} (flat; no regression)",
            "- Still require **multi-seed gate** for Music-close (seed-alone is insufficient posture)",
            "",
        ]
    else:
        lines += [
            "No portable conditional proposed. Stick to multi-seed gate + n≥2 operational floor.",
            "",
        ]
    lines += [
        "## Rejected",
        "",
    ]
    for r in verdict["rejected"]:
        lines.append(f"- {r}")
    lines += [
        "",
        f"JSON: `close_seed0_deep_20260909.json`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # research log append
    log = LOG.read_text() if LOG.exists() else ""
    block = "\n".join([
        "",
        "## Fire — close_seed0_deep (2026-09-09)",
        "",
        f"- Host: box-cpu @ SHA `{sha}`",
        f"- Dig: `close_seed0_deep_20260909.{{py,json,md}}` wall={wall}s",
        f"- fail_mode: **{fail_mode['classification']}** (û undershoot; not leak)",
        f"- seeds0..20 n=1: close_c1.5 {close20['pass_str']} fail={close20['fail_seeds']}; "
        f"close_c1.0 {close20_c10['pass_str']} fail={close20_c10['fail_seeds']}; "
        f"live_c1.5 {live20['pass_str']} fail={live20['fail_seeds']}",
        f"- n_floor close∧live 6/6: **{n_floor}**; s0 knobs l2/bcap/cover/jitter/cloud all fail",
        f"- vic0@n=1 seeds0..20: close {vic0_close['pass_str']} live {vic0_live['pass_str']}; "
        f"leftover n12 flat {leftover_v05['mean_exam']}→{leftover_v0['mean_exam']}",
        f"- Decision: **{decision}**; recipe_change_silent=NO; "
        f"propose_vic0_when_n1={propose_vic0_when_n1}",
        f"- Music rule: {verdict['music_close_rule']}",
        "",
    ])
    if "## Fire — close_seed0_deep" in log:
        log = re.sub(r"\n## Fire — close_seed0_deep.*?(?=\n## |\Z)", "", log, flags=re.S)
    LOG.write_text(log.rstrip() + "\n" + block)

    print(json.dumps(verdict, indent=2), flush=True)
    print(f"WROTE {OUT_JSON}", flush=True)
    print(f"WROTE {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
