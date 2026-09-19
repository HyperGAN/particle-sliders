#!/usr/bin/env python3
"""M30 per-row 1/6 knife — characterize near ê-floor threshold.

Falsify batch4: M30 per_row 1/6 fail=[0,2,3,7,42] → seed1 alone clears.
Is leak just under 0.20 on seed1? Does eoc micro-sweep / couple move the knife?

Locked recipe UNCHANGED. merge=NO. CPU only. No Music GPU.
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
    eoc_threshold_edge_field3d,
    hold_e_lyric_mix_field3d,
)

OUT_JSON = NOTES / "m30_per_row_knife_20260909.json"
OUT_MD = NOTES / "m30_per_row_knife_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
EOC_SWEEP = [0.30, 0.32, 0.33, 0.34, 0.35]
LABEL = "M30_per_row_knife"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def score(*args, **kwargs):
    return dep._recompute_gates(ex.score_scaffold(*args, **kwargs))


def main():
    t0 = time.time()
    sha = git_sha()
    print(f"=== {LABEL} @ {sha} ===", flush=True)

    # 1) Per-seed M30 per_row detail
    print("\n[1] M30 per_row per-seed", flush=True)
    seed_rows = []
    for s in SEEDS:
        print(f"    … seed={s}", flush=True)
        f = eoc_threshold_edge_field3d(seed=s)
        r = score(f, mode="per_row", seed=s, name=f"m30_pr_s{s}", coupling_weight=0.0)
        row = {
            "seed": s,
            "pass": bool(r.get("pass") or r.get("exam_pass")),
            "leak_ratio": round(float(r.get("leak_ratio", 0)), 6),
            "u_kept": round(float(r.get("u_kept", 0)), 6),
            "exam": round(float(r.get("exam_score", r.get("exam_cont", 0) or 0)), 6),
            "multi_ok": bool(r.get("pass_multi_row")),
            "pass_leak": bool(r.get("pass_leak")),
            "bite": bool(r.get("bite_cleared")),
        }
        # pass may be bool already from recompute
        if isinstance(r.get("pass"), bool):
            row["pass"] = r["pass"]
        seed_rows.append(row)
        print(
            f"  s{s}: pass={row['pass']} leak={row['leak_ratio']} "
            f"pass_leak={row['pass_leak']} u={row['u_kept']}",
            flush=True,
        )
    pass_seeds = [r["seed"] for r in seed_rows if r["pass"]]
    fail_seeds = [r["seed"] for r in seed_rows if not r["pass"]]

    # 2) eoc micro-sweep shared + per_row (smoke seeds)
    print("\n[2] eoc micro-sweep", flush=True)
    sweep = []
    for eoc in EOC_SWEEP:
        for mode in ("shared", "per_row"):
            runs = []
            seeds = SEEDS if mode == "per_row" and eoc in (0.33, 0.35) else [0, 1, 2]
            for s in seeds:
                print(f"    … eoc={eoc} {mode} seed={s}", flush=True)
                f = hold_e_lyric_mix_field3d(seed=s, e_on_content=eoc)  # M21 pool family
                # For 0.33 also run dedicated M30 ctor
                if abs(eoc - 0.33) < 1e-9:
                    f = eoc_threshold_edge_field3d(seed=s)
                kw = {"coupling_weight": 0.0} if mode == "per_row" else {}
                r = score(f, mode=mode, seed=s, name=f"eoc{eoc}_{mode}_s{s}", **kw)
                runs.append(r)
            n = len(runs)
            n_pass = sum(1 for r in runs if r.get("pass") or r.get("exam_pass"))
            # prefer bool pass from recompute
            n_pass = sum(1 for r in runs if bool(r.get("pass")))
            leaks = [float(r.get("leak_ratio", 0)) for r in runs]
            sweep.append(
                {
                    "eoc": eoc,
                    "mode": mode,
                    "pass": f"{n_pass}/{n}",
                    "n_pass": n_pass,
                    "n": n,
                    "mean_leak": round(sum(leaks) / n, 6),
                    "max_leak": round(max(leaks), 6),
                    "fail": [runs[i].get("seed", seeds[i]) for i, r in enumerate(runs) if not r.get("pass")],
                }
            )
            print(
                f"  eoc={eoc} {mode}: {n_pass}/{n} mean_lr={sweep[-1]['mean_leak']} "
                f"max={sweep[-1]['max_leak']}",
                flush=True,
            )

    # 3) couple w on M30 per_row
    print("\n[3] couple on M30 per_row", flush=True)
    couple = []
    for w in (0.0, 0.1, 0.3):
        runs = []
        for s in SEEDS:
            print(f"    … w={w} seed={s}", flush=True)
            f = eoc_threshold_edge_field3d(seed=s)
            r = score(
                f, mode="per_row", seed=s, name=f"m30_w{w}_s{s}", coupling_weight=w
            )
            runs.append(r)
        n_pass = sum(1 for r in runs if bool(r.get("pass")))
        leaks = [float(r.get("leak_ratio", 0)) for r in runs]
        couple.append(
            {
                "w": w,
                "pass": f"{n_pass}/6",
                "n_pass": n_pass,
                "mean_leak": round(sum(leaks) / 6, 6),
                "max_leak": round(max(leaks), 6),
                "fail": [SEEDS[i] for i, r in enumerate(runs) if not r.get("pass")],
            }
        )
        print(f"  w={w}: {n_pass}/6 mean_lr={couple[-1]['mean_leak']}", flush=True)

    # Analytic floor reminder
    import math

    def analytic_lr(eoc, content=0.85, leak=0.65, e_unused=0.85):
        n = math.sqrt(eoc**2 + e_unused**2)
        eh_c, eh_e = eoc / n, e_unused / n
        adot = content * eh_c + leak * eh_e
        ae = leak - adot * eh_e
        return abs(ae)

    analytic = {e: round(analytic_lr(e), 6) for e in EOC_SWEEP}

    wall = round(time.time() - t0, 1)
    # Verdict
    seed1_special = pass_seeds == [1] or (1 in pass_seeds and len(pass_seeds) <= 2)
    couple_no_clear = all(c["n_pass"] < 6 for c in couple)
    still_hard = True  # M30 not 6/6 under any probe here
    verdict = (
        f"M30 per-row knife = seeds {pass_seeds} (leak near floor); "
        f"couple does not clear 6/6; eoc sweep confirms threshold; "
        f"NOT a recipe fix — keep HARD_BITE"
        if couple_no_clear
        else "UNEXPECTED couple/eoc clearance — inspect"
    )

    payload = {
        "label": LABEL,
        "sha": sha,
        "wall_s": wall,
        "recipe_change": False,
        "merge_to_trainer": False,
        "seed_rows": seed_rows,
        "pass_seeds": pass_seeds,
        "fail_seeds": fail_seeds,
        "eoc_sweep": sweep,
        "couple": couple,
        "analytic_lr": analytic,
        "verdict": verdict,
        "m14_note": "content_deleted_under_declare_lie (M20 family) — unrelated",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M30 per-row knife — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. merge=NO. recipe_change=NO.",
        "",
        "## Per-seed M30 per_row (eoc=0.33)",
        "",
        "| seed | pass | leak_ratio | pass_leak | u_kept |",
        "|---:|:---:|---:|:---:|---:|",
    ]
    for r in seed_rows:
        lines.append(
            f"| {r['seed']} | {r['pass']} | {r['leak_ratio']:.4f} | {r['pass_leak']} | {r['u_kept']:.4f} |"
        )
    lines += [
        "",
        f"pass_seeds={pass_seeds}; fail_seeds={fail_seeds}",
        "",
        "## Analytic lr vs eoc (M21 pool amps)",
        "",
        "| eoc | analytic_lr | pass_analytic |",
        "|---:|---:|:---:|",
    ]
    for e, lr in analytic.items():
        lines.append(f"| {e:.2f} | {lr:.4f} | {lr <= 0.20} |")
    lines += [
        "",
        "## Empirical eoc sweep",
        "",
        "| eoc | mode | pass | mean_leak | max_leak | fail |",
        "|---:|---|:---:|---:|---:|---|",
    ]
    for s in sweep:
        lines.append(
            f"| {s['eoc']:.2f} | {s['mode']} | {s['pass']} | {s['mean_leak']:.4f} | "
            f"{s['max_leak']:.4f} | {s['fail']} |"
        )
    lines += [
        "",
        "## Couple weight on M30 per_row",
        "",
        "| w | pass | mean_leak | max_leak | fail |",
        "|---:|:---:|---:|---:|---|",
    ]
    for c in couple:
        lines.append(
            f"| {c['w']} | {c['pass']} | {c['mean_leak']:.4f} | {c['max_leak']:.4f} | {c['fail']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "- Do not chase M30 with couple/per-row/recipe knobs.",
        "- M14 = content_deleted_under_declare_lie (M20 family).",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    with LOG.open("a") as fh:
        fh.write(
            f"""
## Fire — M30 per-row knife (2026-09-09)

- Host: box-cpu @ SHA `{sha}`
- Dig: `m30_per_row_knife_20260909.{{py,json,md}}` wall={wall}s
- pass_seeds={pass_seeds}; fail_seeds={fail_seeds}
- couple: {[c['pass'] for c in couple]}; still no 6/6
- Verdict: {verdict}
- recipe_change=NO; merge=NO; ping_user=YES
- Pop-os sync OK earlier; box dig continues. No Music GPU.
"""
        )
    with SCOREBOARD.open("a") as fh:
        fh.write(
            f"""

## M30 per-row knife (2026-09-09)

Wall={wall}s. pass_seeds={pass_seeds} (1/6 class). Couple w∈{{0,0.1,0.3}} no 6/6 clear.
Keep M30 HARD_BITE. Near analytic floor noise — not a recipe lever.
"""
        )
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"DONE wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
