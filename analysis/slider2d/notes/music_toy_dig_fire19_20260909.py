#!/usr/bin/env python3
"""Fire #19 Music→toy dig (box-cpu @ 435e873 + local field3d.py).

Cells:
  a) leftover / close @ n=12 cover=1.5 (regression)
  b) close @ n=1 cover∈{1.0,1.5} (f3d_close knife)
  c) lyric_span_entangle @ n=12 cover=1.5 AND n=1 cover=1.0 (M1 NEW bite)
  d) cross_axis_rows @ n=1 cover=1.0 (Music-posture hard boundary)
  e) dual_arm leftover geom: L (guard+c1.5), listen (faithful+c1.5), gate-only (guard+c0)

Locked recipe: 1200 / cover=1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1.
No Music GPU train. CPU only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    close_field3d,
    cross_axis_rows_field3d,
    dual_arm_leftover_geom_field3d,
    leftover_field3d,
    lyric_span_entangle_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = Path(__file__).resolve().parent
OUT_JSON = NOTES / "music_toy_dig_fire19_20260909.json"
OUT_MD = NOTES / "music_toy_dig_fire19_20260909.md"
LOG = NOTES / "research_log_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER_GUARD = "faithful_guard_e"


def cfg(seed: int, *, cover: float, n_particles: int):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n_particles,
        particle_l2=0.02,
    )


def run_one(field, *, seed: int, cover: float, n: int, teacher: str, name: str, mode: str) -> dict:
    t0 = time.time()
    kw = dict(teacher=teacher, cfg=cfg(seed, cover=cover, n_particles=n), name=name)
    if mode == "score":
        row = score_adv_field3d(field, **kw)
        primary = float(row.get("u_kept", 0.0))
        passed = bool(row.get("pass"))
    else:
        row = score_adv_field3d_exam(field, **kw)
        primary = float(row.get("exam_score", row.get("u_kept", 0.0)))
        passed = bool(row.get("exam_pass", row.get("pass")))
    return {
        "seed": seed,
        "pass": passed,
        "primary": primary,
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "exam_score": float(row.get("exam_score", primary)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "wall_s": round(time.time() - t0, 2),
        "teacher": teacher,
        "cover": cover,
        "n_particles": n,
        "name": name,
        "mode": mode,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    prim = [r["primary"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    fail_seeds = [r["seed"] for r in rows if not r["pass"]]
    return {
        "n_pass": n_pass,
        "n_seeds": n,
        "pass_str": f"{n_pass}/{n}",
        "primary_mean": (sum(prim) / n) if n else 0.0,
        "primary_span": (max(prim) - min(prim)) if prim else 0.0,
        "leak_max": max(leaks) if leaks else 0.0,
        "fail_seeds": fail_seeds,
        "knife_edge": 0 < n_pass < n,
        "pass_all": n_pass == n and n > 0,
        "fail_all": n_pass == 0 and n > 0,
    }


def grid(label: str, make_field, *, cover: float, n: int, teacher=TEACHER_GUARD, seeds=SEEDS_FULL, mode="exam") -> dict:
    rows = []
    print(f"=== {label} c={cover} n={n} teacher={teacher} mode={mode} seeds={seeds} ===", flush=True)
    for seed in seeds:
        field = make_field(seed)
        r = run_one(
            field,
            seed=seed,
            cover=cover,
            n=n,
            teacher=teacher,
            name=f"{label}_c{cover}_n{n}_s{seed}",
            mode=mode,
        )
        rows.append(r)
        print(
            f"  seed={seed} pass={r['pass']} prim={r['primary']:.4f} "
            f"u={r['u_kept']:.4f} leak={r['leak_ratio']:.4f} ({r['wall_s']}s)",
            flush=True,
        )
    return {"summary": summarize(rows), "rows": rows, "cover": cover, "n": n, "teacher": teacher, "mode": mode}


def main() -> None:
    t_all = time.time()
    sha = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    NOTES.mkdir(parents=True, exist_ok=True)

    # Prefer full 6-seed for critical; smoke-first for heavy cells if needed.
    # Strategy: smoke {0,1,2} on all, then expand to full for a/b/c critical.
    results: dict = {}

    # --- a) regression leftover + close @ locked n=12 cover=1.5 ---
    results["leftover_n12_c1.5"] = grid(
        "leftover",
        lambda seed: leftover_field3d(seed=seed),
        cover=1.5,
        n=12,
        seeds=SEEDS_FULL,
        mode="score",
    )
    results["close_n12_c1.5"] = grid(
        "close",
        lambda seed: close_field3d(seed=seed),
        cover=1.5,
        n=12,
        seeds=SEEDS_FULL,
        mode="exam",
    )

    # --- b) f3d_close knife Music posture ---
    results["close_n1_c1.5"] = grid(
        "close",
        lambda seed: close_field3d(seed=seed),
        cover=1.5,
        n=1,
        seeds=SEEDS_FULL,
        mode="exam",
    )
    results["close_n1_c1.0"] = grid(
        "close",
        lambda seed: close_field3d(seed=seed),
        cover=1.0,
        n=1,
        seeds=SEEDS_FULL,
        mode="exam",
    )

    # --- c) lyric_span_entangle M1 NEW bite ---
    results["lyric_span_entangle_n12_c1.5"] = grid(
        "lyric_span_entangle",
        lambda seed: lyric_span_entangle_field3d(seed=seed),
        cover=1.5,
        n=12,
        seeds=SEEDS_FULL,
        mode="exam",
    )
    results["lyric_span_entangle_n1_c1.0"] = grid(
        "lyric_span_entangle",
        lambda seed: lyric_span_entangle_field3d(seed=seed),
        cover=1.0,
        n=1,
        seeds=SEEDS_FULL,
        mode="exam",
    )

    # --- d) cross_axis_rows Music posture hard boundary ---
    results["cross_axis_rows_n1_c1.0"] = grid(
        "cross_axis_rows",
        lambda seed: cross_axis_rows_field3d(seed=seed),
        cover=1.0,
        n=1,
        seeds=SEEDS_FULL,
        mode="exam",
    )

    # --- e) dual_arm (3 seeds) ---
    dual_seeds = SEEDS_SMOKE
    results["dual_arm_L_guard_c1.5_n1"] = grid(
        "dual_arm_L",
        lambda seed: dual_arm_leftover_geom_field3d(seed=seed),
        cover=1.5,
        n=1,
        teacher=TEACHER_GUARD,
        seeds=dual_seeds,
        mode="score",
    )
    results["dual_arm_listen_faithful_c1.5_n1"] = grid(
        "dual_arm_listen",
        lambda seed: dual_arm_leftover_geom_field3d(seed=seed),
        cover=1.5,
        n=1,
        teacher="faithful",
        seeds=dual_seeds,
        mode="score",
    )
    results["dual_arm_gate_only_guard_c0_n1"] = grid(
        "dual_arm_gate_only",
        lambda seed: dual_arm_leftover_geom_field3d(seed=seed),
        cover=0.0,
        n=1,
        teacher=TEACHER_GUARD,
        seeds=dual_seeds,
        mode="score",
    )

    wall = round(time.time() - t_all, 1)

    # Verdict helpers
    def S(key: str) -> dict:
        return results[key]["summary"]

    lyric_locked = S("lyric_span_entangle_n12_c1.5")
    lyric_music = S("lyric_span_entangle_n1_c1.0")
    close_knife_c15 = S("close_n1_c1.5")
    close_knife_c10 = S("close_n1_c1.0")
    leftover_reg = S("leftover_n12_c1.5")
    close_reg = S("close_n12_c1.5")
    cross = S("cross_axis_rows_n1_c1.0")
    dual_L = S("dual_arm_L_guard_c1.5_n1")
    dual_listen = S("dual_arm_listen_faithful_c1.5_n1")
    dual_gate = S("dual_arm_gate_only_guard_c0_n1")

    lyric_bites = (not lyric_locked["pass_all"]) or (not lyric_music["pass_all"])
    close_seed0_only = (
        close_knife_c10["knife_edge"]
        and close_knife_c10["fail_seeds"] == [0]
    ) or (
        close_knife_c15["knife_edge"]
        and close_knife_c15["fail_seeds"] == [0]
    )
    regression = (not leftover_reg["pass_all"]) or (not close_reg["pass_all"])

    verdict = {
        "lyric_span_entangle_bites": lyric_bites,
        "lyric_locked_pass": lyric_locked["pass_str"],
        "lyric_music_pass": lyric_music["pass_str"],
        "f3d_close_seed0_only_knife": close_seed0_only,
        "close_n1_c1.0": close_knife_c10["pass_str"],
        "close_n1_c1.0_fail_seeds": close_knife_c10["fail_seeds"],
        "close_n1_c1.5": close_knife_c15["pass_str"],
        "close_n1_c1.5_fail_seeds": close_knife_c15["fail_seeds"],
        "leftover_n12_regression": not leftover_reg["pass_all"],
        "close_n12_regression": not close_reg["pass_all"],
        "cross_axis_n1_c1.0": cross["pass_str"],
        "dual_L": dual_L["pass_str"],
        "dual_listen_expect_fail": dual_listen["pass_str"],
        "dual_gate_expect_fail": dual_gate["pass_str"],
        "recipe_change": False,
        "ping_user": bool(lyric_bites or regression or close_seed0_only),
    }

    payload = {
        "fire": 19,
        "date": "2026-09-09",
        "host": "box-cpu",
        "sha": sha,
        "locked_recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER_GUARD,
            "fm_weight": 0.0,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "music_posture": {"n_particles": 1, "cover_weight": 1.0},
        "seeds_full": SEEDS_FULL,
        "results": results,
        "verdict": verdict,
        "wall_s": wall,
        "catalog": str(CATALOG.name) if CATALOG.exists() else None,
        "fixes": [
            "odd() now honors row_amps (Fire #13 / M1/M2)",
            "test_leftover_basis_orthonormalish renamed from poles",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", OUT_JSON, flush=True)

    lines = [
        "# Fire #19 Music→toy dig — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}` (+ local field3d.py). Wall {wall}s. CPU only.",
        "",
        "## Recipe",
        "",
        "- Locked: 1200 / cover=1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1",
        "- Music-posture: n_particles=1 (parts0 proxy), cover∈{1.0,1.5}",
        "- Skip 800×cover3.0",
        "",
        "## Fixes this fire",
        "",
        "- `Field3D.odd()` now uses `row_amps` when set (required for M1/M2/cross-axis)",
        "- `test_leftover_basis_orthonormalish` (was misnamed poles check)",
        "",
        "## Pass grid",
        "",
        "| cell | cover | n | PASS | primary mean | leak max | fail seeds |",
        "|---|---:|---:|:---:|---:|---:|---|",
    ]
    order = [
        "leftover_n12_c1.5",
        "close_n12_c1.5",
        "close_n1_c1.5",
        "close_n1_c1.0",
        "lyric_span_entangle_n12_c1.5",
        "lyric_span_entangle_n1_c1.0",
        "cross_axis_rows_n1_c1.0",
        "dual_arm_L_guard_c1.5_n1",
        "dual_arm_listen_faithful_c1.5_n1",
        "dual_arm_gate_only_guard_c0_n1",
    ]
    for k in order:
        block = results[k]
        s = block["summary"]
        lines.append(
            f"| {k} | {block['cover']} | {block['n']} | {s['pass_str']} | "
            f"{s['primary_mean']:.4f} | {s['leak_max']:.4f} | {s['fail_seeds']} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"- **lyric_span_entangle bites?** {'YES' if lyric_bites else 'NO'} — "
        f"locked {lyric_locked['pass_str']} (mean prim {lyric_locked['primary_mean']:.4f}, "
        f"leak_max {lyric_locked['leak_max']:.4f}); music {lyric_music['pass_str']}",
        f"- **f3d_close seed0-only knife @ n=1?** {'YES' if close_seed0_only else 'NO/OTHER'} — "
        f"c1.0 {close_knife_c10['pass_str']} fail={close_knife_c10['fail_seeds']}; "
        f"c1.5 {close_knife_c15['pass_str']} fail={close_knife_c15['fail_seeds']}",
        f"- **leftover/close @ n=12 regression?** {'YES' if regression else 'NO'} — "
        f"leftover {leftover_reg['pass_str']}, close {close_reg['pass_str']}",
        f"- **cross_axis Music posture:** {cross['pass_str']}",
        f"- **dual_arm:** L {dual_L['pass_str']}; listen(expect FAIL) {dual_listen['pass_str']}; "
        f"gate-only(expect FAIL) {dual_gate['pass_str']}",
        "- **Recipe change?** NO (default)",
        f"- **Ping user?** {'YES' if verdict['ping_user'] else 'NO'}",
        "",
        f"JSON: `{OUT_JSON.name}`  Catalog: `{CATALOG.name}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_MD, flush=True)

    # Append research_log Fire #19
    block = [
        "",
        "## Fire #19 — Music→toy dig (lyric_span_entangle + f3d_close knife) (2026-09-09)",
        "",
        f"- Host: box-cpu @ SHA `{sha}` (+ local field3d.py row_amps fix)",
        f"- Tests: see parent report (pytest after odd()/basis rename)",
        f"- Dig: `music_toy_dig_fire19_20260909.{{py,json,md}}` wall={wall}s",
        f"- Catalog synced: `music_to_toy_stressor_catalog_20260909.md`",
        f"- lyric_span_entangle: locked {lyric_locked['pass_str']} mean_prim={lyric_locked['primary_mean']:.4f} "
        f"leak_max={lyric_locked['leak_max']:.4f}; music {lyric_music['pass_str']}",
        f"- f3d_close n=1: c1.0 {close_knife_c10['pass_str']} fail={close_knife_c10['fail_seeds']}; "
        f"c1.5 {close_knife_c15['pass_str']} fail={close_knife_c15['fail_seeds']}",
        f"- leftover/close n=12: {leftover_reg['pass_str']} / {close_reg['pass_str']}",
        f"- cross_axis n=1 c1.0: {cross['pass_str']}; dual L/listen/gate: "
        f"{dual_L['pass_str']}/{dual_listen['pass_str']}/{dual_gate['pass_str']}",
        f"- Verdict: lyric_bite={'YES' if lyric_bites else 'NO'}; "
        f"seed0_knife={'YES' if close_seed0_only else 'NO'}; "
        f"regression={'YES' if regression else 'NO'}; recipe_change=NO; "
        f"ping_user={'YES' if verdict['ping_user'] else 'NO'}",
        "- No Music GPU train; servers untouched.",
        "",
    ]
    if LOG.exists():
        prev = LOG.read_text()
        if "Fire #19 — Music→toy dig" not in prev:
            LOG.write_text(prev.rstrip() + "\n" + "\n".join(block))
            print("appended", LOG, flush=True)
        else:
            print("Fire #19 already in log; skip append", flush=True)
    else:
        LOG.write_text("# 2D adversarial slider research log — 2026-09-09\n" + "\n".join(block))
        print("created", LOG, flush=True)

    print("VERDICT:", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
