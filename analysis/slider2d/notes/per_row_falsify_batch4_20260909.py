#!/usr/bin/env python3
"""Falsify: per-row clears M29 (DoF) but NOT M28/M30/M31 (eoc/teacher floor).

Mirrors falsify_A (M24/M27 clear, M20/M21 stay). Locked recipe UNCHANGED. merge=NO.
Also CTRL leftover under per-row. CPU only. No Music GPU.
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

_dspec = importlib.util.spec_from_file_location(
    "per_row_deepen", NOTES / "per_row_residual_deepen_20260909.py"
)
dep = importlib.util.module_from_spec(_dspec)
assert _dspec.loader is not None
sys.modules["per_row_deepen"] = dep
_dspec.loader.exec_module(dep)

from analysis.slider2d.field3d import (  # noqa: E402
    content_cascade_rows_field3d,
    eoc_threshold_edge_field3d,
    guard_refuse_hot_eoc_field3d,
    leftover_field3d,
    leftover_hot_eoc_declare_field3d,
)

OUT_JSON = NOTES / "per_row_falsify_batch4_20260909.json"
OUT_MD = NOTES / "per_row_falsify_batch4_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
LABEL = "NON_DEFAULT_falsify_batch4"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def score_scaffold(*args, **kwargs) -> dict:
    return dep._recompute_gates(ex.score_scaffold(*args, **kwargs))


def summarize(runs):
    out = ex.summarize(runs)
    hm = [r["head_min_cos"] for r in runs if r.get("head_min_cos") is not None]
    out["mean_head_min_cos"] = round(sum(hm) / len(hm), 4) if hm else None
    return out


def grid(name, mode, field_fn, seeds, **score_kw):
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s}", flush=True)
        try:
            f = field_fn(seed=s)
        except TypeError:
            f = field_fn()
        runs.append(score_scaffold(f, mode=mode, seed=s, name=f"{name}_s{s}", **score_kw))
    out = summarize(runs)
    out["name"] = name
    out["mode"] = mode
    print(
        f"  {name}: pass={out['pass']} multi={out['multi_row']} bite={out['bite_cleared']} "
        f"exam={out['mean_exam']} leak_max={out['leak_max']} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def pack(cell):
    return {
        "pass": cell["pass"],
        "multi": cell["multi_row"],
        "bite": cell["bite_cleared"],
        "n_pass": cell["n_pass"],
        "n": cell["n"],
        "mean_exam": cell["mean_exam"],
        "leak_max": cell["leak_max"],
        "fail": cell["fail_seeds"],
        "mean_head_min_cos": cell.get("mean_head_min_cos"),
    }


def main():
    t0 = time.time()
    sha = git_sha()
    print(f"=== {LABEL} @ {sha} ===", flush=True)
    cells = [
        ("M28", "guard_refuse_hot_eoc", guard_refuse_hot_eoc_field3d, False),  # expect no clear
        ("M29", "content_cascade_rows", content_cascade_rows_field3d, True),   # expect clear
        ("M30", "eoc_threshold_edge", eoc_threshold_edge_field3d, False),
        ("M31", "leftover_hot_eoc_declare", leftover_hot_eoc_declare_field3d, False),
    ]
    results = {}
    for mid, name, fn, expect_clear in cells:
        print(f"\n[{mid}] {name} shared smoke + per_row full", flush=True)
        shared = grid(f"{mid}_shared", "shared", fn, SEEDS_SMOKE)
        per = grid(f"{mid}_per_row", "per_row", fn, SEEDS_FULL, coupling_weight=0.0)
        clears = per.get("n_pass", 0) == per.get("n", 0) and per.get("n", 0) > 0
        # n_pass from summarize
        if "n_pass" not in per and isinstance(per.get("pass"), str) and "/" in per["pass"]:
            per["n_pass"] = int(per["pass"].split("/")[0])
            per["n"] = int(per["pass"].split("/")[1])
            clears = per["n_pass"] == per["n"]
        results[mid] = {
            "name": name,
            "expect_clear": expect_clear,
            "shared": pack(shared),
            "per_row": pack(per),
            "clears": bool(clears),
            "match_expect": bool(clears) == bool(expect_clear),
        }
        print(f"  → clears={clears} expect={expect_clear} match={clears == expect_clear}", flush=True)

    print("\n[CTRL] leftover per_row", flush=True)
    ctrl = grid("CTRL_leftover_per_row", "per_row", leftover_field3d, SEEDS_SMOKE)
    ctrl_ok = ctrl.get("n_pass", 0) == ctrl.get("n", 0) or (
        isinstance(ctrl.get("pass"), str) and ctrl["pass"].startswith(str(ctrl.get("n", 3)))
    )
    if isinstance(ctrl.get("pass"), str) and "/" in ctrl["pass"]:
        ctrl_ok = ctrl["pass"].split("/")[0] == ctrl["pass"].split("/")[1]

    cleared = [m for m, r in results.items() if r["clears"]]
    uncleared = [m for m, r in results.items() if not r["clears"]]
    overpowered = "M28" in cleared or "M30" in cleared or "M31" in cleared
    expected_m29 = results["M29"]["clears"]
    falsify_ok = expected_m29 and not overpowered and ctrl_ok and all(
        results[m]["match_expect"] for m in results
    )

    verdict = (
        "YES — per-row clears M29 (DoF) without clearing M28/M30/M31 (eoc/teacher); CTRL ok; merge=NO"
        if falsify_ok
        else "PARTIAL/UNEXPECTED — see table; check overpowered or M29 miss"
    )
    wall = round(time.time() - t0, 1)
    payload = {
        "label": LABEL,
        "sha": sha,
        "wall_s": wall,
        "recipe_change": False,
        "merge_to_trainer": False,
        "results": results,
        "ctrl": pack(ctrl),
        "ctrl_ok": ctrl_ok,
        "cleared": cleared,
        "uncleared": uncleared,
        "overpowered": overpowered,
        "falsify_ok": falsify_ok,
        "verdict": verdict,
        "m14_note": "M14=content_deleted_under_declare_lie (M20 family) — not in this grid",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        f"# Per-row falsify batch4 (M28–M31) — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. **Locked recipe unchanged. merge=NO.**",
        "",
        "Expect: **M29 clears** (DoF cascade); **M28/M30/M31 stay biting** (eoc/teacher).",
        "M14 = content_deleted_under_declare_lie (M20 family) — out of scope here.",
        "",
        "| mid | shared pass | per_row pass | clears? | expect | match |",
        "|---|---|---|:---:|:---:|:---:|",
    ]
    for mid, r in results.items():
        lines.append(
            f"| {mid} | {r['shared']['pass']} | {r['per_row']['pass']} | "
            f"{'YES' if r['clears'] else 'no'} | {'clear' if r['expect_clear'] else 'bite'} | "
            f"{r['match_expect']} |"
        )
    lines += [
        "",
        f"CTRL leftover per_row: {ctrl['pass']} ok={ctrl_ok}",
        f"cleared={cleared}; uncleared={uncleared}; overpowered={overpowered}",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    with LOG.open("a") as fh:
        fh.write(
            f"""
## Fire — per-row falsify batch4 (2026-09-09)

- Host: box-cpu @ SHA `{sha}`
- Dig: `per_row_falsify_batch4_20260909.{{py,json,md}}` wall={wall}s
- cleared={cleared}; uncleared={uncleared}; overpowered={overpowered}; CTRL ok={ctrl_ok}
- Verdict: {verdict}
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- M14 note: content_deleted_under_declare_lie (M20 family)
- No Music GPU train; servers untouched.
"""
        )
    with SCOREBOARD.open("a") as fh:
        fh.write(
            f"""

## Per-row falsify batch4 (2026-09-09)

**Script:** `per_row_falsify_batch4_20260909` wall={wall}s

| mid | shared | per_row | clears? |
|---|---|---|---|
"""
            + "\n".join(
                f"| {m} | {r['shared']['pass']} | {r['per_row']['pass']} | {r['clears']} |"
                for m, r in results.items()
            )
            + f"""

**Verdict:** {verdict}
**merge=NO.** M14 remains content_deleted_under_declare_lie (M20 family).
"""
        )
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"DONE wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
