#!/usr/bin/env python3
"""NON-DEFAULT deepen: per-row AdvResidual couple band + other Music→toy bites.

Prior (explore): per-row clears lyric_span_entangle multi-span 6/6; shared 0/6.
Soft couple w0.3 ok, w1.0 collapses. Locked shared recipe UNCHANGED — do NOT
merge per-row into trainer defaults. No Music GPU train.

Depth:
  1) Couple weight sweep {0, 0.1, 0.3, 0.5, 0.7} × seeds vs head_min_cos
  2) Per-row on hold_e_lyric_mix (M21), cross_axis_rows, amp_lie (expect YAML fail)
  3) Music-posture n=1 / cover=1.0 with per-row on lyric — still clear?
  4) Leftover / close smoke under per-row (must stay green)
  5) Music mapping sketch (propose-only) in companion .md
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

# Load explore harness (MultiResidual / fit / score / grid)
_spec = importlib.util.spec_from_file_location(
    "per_row_explore", NOTES / "per_row_residual_explore_20260909.py"
)
ex = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["per_row_explore"] = ex  # required before dataclass exec
_spec.loader.exec_module(ex)

from analysis.slider2d.field3d import (  # noqa: E402
    amp_lie_leftover_declare_field3d,
    close_field3d,
    cross_axis_rows_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    lyric_span_entangle_field3d,
)

OUT_JSON = NOTES / "per_row_residual_deepen_20260909.json"
OUT_MD = NOTES / "per_row_residual_deepen_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
COUPLE_WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7]
MUSIC_CFG = {"n_particles": 1, "cover_weight": 1.0}
LABEL = "NON_DEFAULT_deepen"
TEACHER = ex.TEACHER


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _recompute_gates(out: dict) -> dict:
    """Align leftover_ok with field3d.score_adv_field3d_exam kind rules."""
    kind = out.get("cell") or ""
    multi_ok = bool(out.get("pass_multi_row"))
    pass_u = bool(out.get("pass_u"))
    pass_leak = bool(out.get("pass_leak"))
    content_kept = float(out.get("content_kept") or 0.0)
    # amp_lie / declare-lie family: content must be kept
    if kind in (
        "amp_lie_leftover_declare",
        "cross_axis_mismatch_declare",
        "e_on_u_declare_lie",
        "declare_split_three",
        "divergent",
        "axis_content_primary",
    ):
        leftover_ok = bool(pass_u and content_kept >= 0.75)
    elif kind in (
        "unused_e",
        "lyric_span_entangle",
        "cross_axis_rows",
        "cross_axis_span_sample",
        "axis_u_primary",
        "axis_leak_primary",
        "hold_e_lyric_mix",
        "stagger_mild_cross",
        "multipair_corr_seed",
        "scale_stagger_homo",
        "roles_split_proxy",
        "prefix_shared_proxy",
        "close_with_leak",
        "content_leak_flip_rows",
        "lyric_neu_heavy_gate",
        "scale_descent_homo",
        "guard_refuse_hot_eoc",
        "content_cascade_rows",
        "eoc_threshold_edge",
        "leftover_hot_eoc_declare",
    ):
        leftover_ok = bool(pass_leak and pass_u)
    else:
        # close / leftover
        leftover_ok = bool(pass_u and content_kept >= 0.75)

    pass_cont_t = bool(out.get("pass_cont"))
    pass_swing_t = bool(out.get("pass_swing"))
    exam_pass = bool(pass_cont_t and pass_swing_t and leftover_ok and multi_ok)
    bite_cleared = bool(multi_ok and leftover_ok and pass_u)
    out = dict(out)
    out["pass_leftover_gate"] = leftover_ok
    out["exam_pass"] = exam_pass
    out["pass"] = exam_pass
    out["bite_cleared"] = bite_cleared
    return out


def score_scaffold(*args, **kwargs) -> dict:
    return _recompute_gates(ex.score_scaffold(*args, **kwargs))


def summarize(runs: list[dict]) -> dict:
    out = ex.summarize(runs)
    hm = [r["head_min_cos"] for r in runs if r.get("head_min_cos") is not None]
    hmean = [r["head_mean_cos"] for r in runs if r.get("head_mean_cos") is not None]
    out["mean_head_min_cos"] = round(sum(hm) / len(hm), 4) if hm else None
    out["min_head_min_cos"] = round(min(hm), 4) if hm else None
    out["mean_head_mean_cos"] = round(sum(hmean) / len(hmean), 4) if hmean else None
    return out


def grid(name: str, mode: str, field_fn, seeds, **score_kw) -> dict:
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s}", flush=True)
        field = field_fn(seed=s) if callable(field_fn) else field_fn
        # leftover/close ignore seed kw sometimes
        try:
            f = field_fn(seed=s) if callable(field_fn) else field_fn
        except TypeError:
            f = field_fn() if callable(field_fn) else field_fn
        runs.append(
            score_scaffold(
                f,
                mode=mode,
                seed=s,
                name=f"{name}_s{s}",
                **score_kw,
            )
        )
    out = summarize(runs)
    out["name"] = name
    out["mode"] = mode
    out["label"] = LABEL
    out["coupling_weight"] = score_kw.get("coupling_weight", 0.0)
    out["cfg_kw"] = score_kw.get("cfg_kw") or {}
    hm = out.get("mean_head_min_cos")
    print(
        f"  {name}: pass={out['pass']} multi={out['multi_row']} bite={out['bite_cleared']} "
        f"exam={out['mean_exam']} leak_max={out['leak_max']} "
        f"rows_cov≈{out['mean_rows_cov']} head_min_cos≈{hm} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def main() -> None:
    t_wall = time.time()
    sha = git_sha()
    print(f"=== per_row_residual_deepen {LABEL} @ {sha} ===", flush=True)
    cells: list[dict] = []

    # ------------------------------------------------------------------
    # 1) Couple weight sweep on lyric M1
    # ------------------------------------------------------------------
    print("\n[1] couple weight sweep × seeds (lyric_span_entangle)", flush=True)
    couple_smoke = []
    for w in COUPLE_WEIGHTS:
        c = grid(
            f"couple_w{w}_lyric_smoke",
            "per_row",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_SMOKE,
            coupling_weight=float(w),
        )
        couple_smoke.append(c)
        cells.append(c)

    # Robust band: multi==n on smoke AND head_min_cos not collapsed to shared
    # (prior: w1.0 → head_min≈0.89 + multi 0/3). Prefer multi full + head_min < 0.85
    band_candidates = []
    for c in couple_smoke:
        w = float(c.get("coupling_weight") or 0.0)
        multi_ok = c["n_multi"] == c["n"]
        bite_ok = c.get("n_bite", 0) == c["n"]
        hm = c.get("mean_head_min_cos")
        # fragmentation floor: unconstrained ≈0.56; collapse ≈0.89
        not_collapsed = hm is None or hm < 0.85
        if multi_ok and bite_ok and not_collapsed:
            band_candidates.append(w)

    print(f"  smoke band candidates: {band_candidates}", flush=True)

    # Expand promising band + edges (0 and nearest fail) to full seeds
    expand_ws = sorted(set(band_candidates) | {0.0})
    # Also expand the first fail above band if any
    for c in couple_smoke:
        w = float(c.get("coupling_weight") or 0.0)
        if c["n_multi"] < c["n"] and w not in expand_ws:
            # only the lowest failing weight above band max
            if band_candidates and w > max(band_candidates):
                expand_ws.append(w)
                break
    expand_ws = sorted(set(expand_ws))
    print(f"  expanding to full seeds: {expand_ws}", flush=True)

    couple_full = []
    for w in expand_ws:
        # skip re-run if smoke already used full? always re-run full for clean stats
        c = grid(
            f"couple_w{w}_lyric_full",
            "per_row",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_FULL,
            coupling_weight=float(w),
        )
        couple_full.append(c)
        cells.append(c)

    robust_band = []
    for c in couple_full:
        w = float(c.get("coupling_weight") or 0.0)
        hm = c.get("mean_head_min_cos")
        if (
            c["n_multi"] == c["n"]
            and c.get("n_bite", 0) == c["n"]
            and c["n_pass"] == c["n"]
            and (hm is None or hm < 0.85)
        ):
            robust_band.append(
                {
                    "w": w,
                    "pass": c["pass"],
                    "multi": c["multi_row"],
                    "bite": c["bite_cleared"],
                    "mean_head_min_cos": hm,
                    "min_head_min_cos": c.get("min_head_min_cos"),
                    "mean_exam": c["mean_exam"],
                    "leak_max": c["leak_max"],
                }
            )

    # ------------------------------------------------------------------
    # 2) Other bites: shared vs per_row @ locked recipe
    # ------------------------------------------------------------------
    print("\n[2] other Music→toy bites (shared vs per_row)", flush=True)
    bite_specs = [
        ("hold_e_lyric_mix", hold_e_lyric_mix_field3d, "M21"),
        ("cross_axis_rows", cross_axis_rows_field3d, "cross"),
        ("amp_lie_leftover_declare", amp_lie_leftover_declare_field3d, "M20"),
    ]
    other_bites = {}
    for name, ctor, mid in bite_specs:
        shared = grid(
            f"{mid}_{name}_shared",
            "shared",
            lambda seed, _c=ctor: _c(seed=seed),
            SEEDS_SMOKE,
        )
        per = grid(
            f"{mid}_{name}_per_row",
            "per_row",
            lambda seed, _c=ctor: _c(seed=seed),
            SEEDS_SMOKE,
        )
        cells.append(shared)
        cells.append(per)
        # Expand per_row to full if smoke multi≥2/3 (promising)
        per_full = None
        if per["n_multi"] >= 2:
            print(f"  expanding {name} per_row to full seeds…", flush=True)
            per_full = grid(
                f"{mid}_{name}_per_row_full",
                "per_row",
                lambda seed, _c=ctor: _c(seed=seed),
                SEEDS_FULL,
            )
            cells.append(per_full)
        other_bites[name] = {
            "mid": mid,
            "shared": {
                "pass": shared["pass"],
                "multi": shared["multi_row"],
                "bite": shared["bite_cleared"],
                "n_pass": shared["n_pass"],
                "n_multi": shared["n_multi"],
                "n_bite": shared.get("n_bite"),
                "mean_exam": shared["mean_exam"],
                "leak_max": shared["leak_max"],
                "fail": shared["fail_seeds"],
            },
            "per_row": {
                "pass": (per_full or per)["pass"],
                "multi": (per_full or per)["multi_row"],
                "bite": (per_full or per)["bite_cleared"],
                "n_pass": (per_full or per)["n_pass"],
                "n_multi": (per_full or per)["n_multi"],
                "n_bite": (per_full or per).get("n_bite"),
                "mean_exam": (per_full or per)["mean_exam"],
                "leak_max": (per_full or per)["leak_max"],
                "fail": (per_full or per)["fail_seeds"],
                "mean_head_min_cos": (per_full or per).get("mean_head_min_cos"),
                "expanded": per_full is not None,
            },
            "per_row_saves": bool(
                (per_full or per)["n_multi"] == (per_full or per)["n"]
                and (per_full or per).get("n_bite", 0) == (per_full or per)["n"]
                and shared["n_multi"] < shared["n"]
            ),
            "per_row_clears": bool(
                (per_full or per)["n_multi"] == (per_full or per)["n"]
                and (per_full or per).get("n_bite", 0) == (per_full or per)["n"]
                and (per_full or per)["n_pass"] == (per_full or per)["n"]
            ),
        }

    # ------------------------------------------------------------------
    # 3) Music-posture n=1 / cover=1.0 with per-row on lyric
    # ------------------------------------------------------------------
    print("\n[3] Music-posture n=1 cover=1.0 (lyric)", flush=True)
    music_shared = grid(
        "music_shared_lyric",
        "shared",
        lambda seed: lyric_span_entangle_field3d(seed=seed),
        SEEDS_SMOKE,
        cfg_kw=dict(MUSIC_CFG),
    )
    music_per = grid(
        "music_per_row_lyric",
        "per_row",
        lambda seed: lyric_span_entangle_field3d(seed=seed),
        SEEDS_SMOKE,
        cfg_kw=dict(MUSIC_CFG),
    )
    cells.append(music_shared)
    cells.append(music_per)
    music_per_full = None
    if music_per["n_multi"] >= 2:
        print("  expanding music per_row lyric to full seeds…", flush=True)
        music_per_full = grid(
            "music_per_row_lyric_full",
            "per_row",
            lambda seed: lyric_span_entangle_field3d(seed=seed),
            SEEDS_FULL,
            cfg_kw=dict(MUSIC_CFG),
        )
        cells.append(music_per_full)
    music_clears = bool(
        (music_per_full or music_per)["n_multi"] == (music_per_full or music_per)["n"]
        and (music_per_full or music_per).get("n_bite", 0)
        == (music_per_full or music_per)["n"]
    )

    # ------------------------------------------------------------------
    # 4) Leftover / close smoke under per_row (locked)
    # ------------------------------------------------------------------
    print("\n[4] leftover/close smoke under per_row (locked)", flush=True)
    left = grid(
        "smoke_per_row_leftover",
        "per_row",
        lambda seed: leftover_field3d(),
        SEEDS_SMOKE,
    )
    close = grid(
        "smoke_per_row_close",
        "per_row",
        lambda seed: close_field3d(),
        SEEDS_SMOKE,
    )
    cells.append(left)
    cells.append(close)
    controls_ok = bool(
        left["n_pass"] == left["n"] and close["n_pass"] == close["n"]
    )

    wall = round(time.time() - t_wall, 1)

    # ------------------------------------------------------------------
    # Verdict + Music mapping sketch
    # ------------------------------------------------------------------
    saved = [k for k, v in other_bites.items() if v["per_row_saves"] or v["per_row_clears"]]
    still_fail = [
        k
        for k, v in other_bites.items()
        if not v["per_row_clears"] and not v["per_row_saves"]
    ]
    amp_still_fail = bool(
        other_bites.get("amp_lie_leftover_declare", {}).get("per_row_clears") is False
        and other_bites.get("amp_lie_leftover_declare", {}).get("shared", {}).get(
            "n_multi", 0
        )
        is not None
    )
    # amp expected fail: no full clear
    amp_expected = not other_bites.get("amp_lie_leftover_declare", {}).get(
        "per_row_clears", True
    )

    band_lo = min((b["w"] for b in robust_band), default=None)
    band_hi = max((b["w"] for b in robust_band), default=None)

    music_mapping = {
        "propose_only": True,
        "no_music_train": True,
        "summary": (
            "Per-row AdvResidual on the toy = one odd/even head per lyric row "
            "(or multipair). In lm_adv / Music this maps to multi-residual / "
            "per-span students: each lyric span (or caption-pair row) owns a "
            "residual head rather than one shared δ forced to cover hetero "
            "row_amps. Soft couple keeps a shared slider identity without "
            "collapsing to the shared DoF floor."
        ),
        "toy_to_music": [
            {
                "toy": "row r / Field3D.odd(r) with hetero row_amps",
                "music": "lyric span / multipair caption row with distinct ûĉê mix",
            },
            {
                "toy": "shared AdvResidual δ ≈ mean(a_r) — cover gate floor",
                "music": "single student residual across all spans — Fire #20 limit",
            },
            {
                "toy": "MultiResidual heads[r].delta(scale)",
                "music": "per-span / per-row residual bank; apply head matching active span",
            },
            {
                "toy": "coupling_weight on (1-cos) of heads in R3",
                "music": "soft shared-identity loss across span heads (one slider personality)",
            },
            {
                "toy": "amp_lie YAML declared_e lie — per-row does NOT save",
                "music": "bad leak / hold YAML still requires target fix, not more DoF",
            },
            {
                "toy": "hold_e_lyric_mix / cross_axis — DoF vs gate/mix bites",
                "music": "test whether Music bite is amp-mix DoF or teacher/YAML",
            },
        ],
        "lm_adv_sketch": (
            "Scaffold only: AdvResidual → list[AdvResidual] keyed by lyric-span id "
            "(or row index in multipair batch). Cover loss sums per-span "
            "(neu_s + δ_s - teacher_s). Optional couple: mean pairwise "
            "(1 - cos(δ_s, δ_t)) in the held leftover subspace. Infer: pick "
            "head by active span or mean-pool heads. Do NOT change locked "
            "shared defaults until multi-seed Music GPU confirms."
        ),
        "risks": [
            "Head fragmentation → multiple personalities (monitor head_min_cos)",
            "Over-coupling → shared floor returns (toy w≥1.0)",
            "YAML/teacher lies unchanged by DoF (amp_lie)",
            "VRAM × n_spans for per-row heads",
        ],
    }

    if robust_band and controls_ok:
        verdict_str = (
            f"YES — couple robust band w∈[{band_lo}, {band_hi}] clears lyric multi; "
            f"per_row saves {saved}; amp_lie expected_fail={amp_expected}; "
            f"music_posture_clears={music_clears}; controls_ok"
        )
    elif controls_ok:
        verdict_str = (
            "MIXED — controls ok but couple band empty or partial; see numbers"
        )
    else:
        verdict_str = "NO — leftover/close smoke broke under per_row"

    payload = {
        "sha": sha,
        "label": LABEL,
        "locked_recipe": "1200 / c1.5 / faithful_guard_e / FM0 / n≤12 / b_cap=1 (UNCHANGED)",
        "merge_to_trainer": False,
        "recipe_change": False,
        "wall_s": wall,
        "couple_sweep": {
            "weights": COUPLE_WEIGHTS,
            "smoke_band_candidates": band_candidates,
            "expanded_weights": expand_ws,
            "robust_band": robust_band,
            "band_lo": band_lo,
            "band_hi": band_hi,
        },
        "other_bites": other_bites,
        "music_posture": {
            "cfg": MUSIC_CFG,
            "shared": {
                "pass": music_shared["pass"],
                "multi": music_shared["multi_row"],
                "bite": music_shared["bite_cleared"],
                "fail": music_shared["fail_seeds"],
            },
            "per_row": {
                "pass": (music_per_full or music_per)["pass"],
                "multi": (music_per_full or music_per)["multi_row"],
                "bite": (music_per_full or music_per)["bite_cleared"],
                "fail": (music_per_full or music_per)["fail_seeds"],
                "mean_head_min_cos": (music_per_full or music_per).get(
                    "mean_head_min_cos"
                ),
                "expanded": music_per_full is not None,
            },
            "clears": music_clears,
        },
        "controls": {
            "leftover": left["pass"],
            "close": close["pass"],
            "ok": controls_ok,
        },
        "cells": cells,
        "music_mapping": music_mapping,
        "verdict": {
            "str": verdict_str,
            "robust_band_lo": band_lo,
            "robust_band_hi": band_hi,
            "saved_cells": saved,
            "still_fail_cells": still_fail,
            "amp_lie_still_fail": amp_expected,
            "music_posture_clears": music_clears,
            "controls_ok": controls_ok,
            "recipe_change": False,
            "merge_to_trainer": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))

    # MD
    lines = [
        "# Per-row residual deepen (NON-DEFAULT) — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. **Locked recipe unchanged.**",
        "Label: `NON_DEFAULT_deepen` — analysis-only; do **not** merge to trainer.",
        "",
        "## 1) Couple weight band vs head_min_cos",
        "",
        "| w | seeds | PASS | multi | bite | mean exam | leak max | head_min_cos | fail |",
        "|---|---|:---:|:---:|:---:|---:|---:|---:|---|",
    ]
    for c in cells:
        if not c["name"].startswith("couple_w"):
            continue
        lines.append(
            f"| {c.get('coupling_weight')} | {c['n']} | {c['pass']} | {c['multi_row']} | "
            f"{c['bite_cleared']} | {c['mean_exam']} | {c['leak_max']} | "
            f"{c.get('mean_head_min_cos')} | {c['fail_seeds']} |"
        )
    lines += [
        "",
        f"**Robust band (full seeds, multi+bite+exam, head_min_cos<0.85):** "
        f"w ∈ [{band_lo}, {band_hi}] → {robust_band}",
        "",
        "## 2) Other Music→toy cells — does per-row save?",
        "",
        "| cell | mid | shared multi | per_row multi | per_row pass | saves? | notes |",
        "|---|---|:---:|:---:|:---:|:---:|---|",
    ]
    for name, block in other_bites.items():
        note = (
            "YAML lie — expect fail"
            if name.startswith("amp_lie")
            else ("DoF / amp-mix" if "cross" in name or "lyric" in name else "")
        )
        lines.append(
            f"| `{name}` | {block['mid']} | {block['shared']['multi']} | "
            f"{block['per_row']['multi']} | {block['per_row']['pass']} | "
            f"{'YES' if block['per_row_saves'] or block['per_row_clears'] else 'NO'} | {note} |"
        )
    lines += [
        "",
        f"Saved by per-row: **{saved}**. Still fail: **{still_fail}**. "
        f"amp_lie still fail (expected YAML): **{amp_expected}**.",
        "",
        "## 3) Music-posture n=1 / cover=1.0 + per-row lyric",
        "",
        f"- shared: pass={music_shared['pass']} multi={music_shared['multi_row']} "
        f"bite={music_shared['bite_cleared']} fail={music_shared['fail_seeds']}",
        f"- per_row: pass={(music_per_full or music_per)['pass']} "
        f"multi={(music_per_full or music_per)['multi_row']} "
        f"bite={(music_per_full or music_per)['bite_cleared']} "
        f"head_min_cos≈{(music_per_full or music_per).get('mean_head_min_cos')} "
        f"fail={(music_per_full or music_per)['fail_seeds']}",
        f"- **clears?** {music_clears}",
        "",
        "## 4) Leftover / close smoke (per-row, locked)",
        "",
        f"- leftover: {left['pass']}",
        f"- close: {close['pass']}",
        f"- controls_ok: **{controls_ok}**",
        "",
        "## 5) Music mapping sketch (propose-only, no Music train)",
        "",
        music_mapping["summary"],
        "",
        "### Toy → Music",
        "",
    ]
    for row in music_mapping["toy_to_music"]:
        lines.append(f"- **Toy:** {row['toy']}")
        lines.append(f"  - **Music:** {row['music']}")
    lines += [
        "",
        "### lm_adv scaffold sketch",
        "",
        music_mapping["lm_adv_sketch"],
        "",
        "### Risks",
        "",
    ]
    for r in music_mapping["risks"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict_str}**",
        "",
        "- recipe_change=NO; merge_to_trainer=NO; locked shared AdvResidual stays default",
        "- Next: multi-seed Music GPU smoke with per-row student (later); keep couple in band",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))

    # Append research log
    log_block = f"""
## Fire — per-row residual deepen (NON-DEFAULT) (2026-09-09)

- Host: box-cpu @ SHA `{sha}` (no Music GPU; locked recipe UNCHANGED)
- Dig: `per_row_residual_deepen_20260909.{{py,json,md}}` wall={wall}s
- Couple band: w∈[{band_lo}, {band_hi}] robust={robust_band}
- Other bites saved by per-row: {saved}; still_fail={still_fail}; amp_lie_expected_fail={amp_expected}
- Music-posture n=1 c1.0 per-row lyric clears={music_clears}
- Controls leftover/close: {left['pass']} / {close['pass']} (ok={controls_ok})
- Verdict: {verdict_str}
- recipe_change=NO; merge_to_trainer=NO; Music mapping sketch in md (propose-only)
"""
    with LOG.open("a") as f:
        f.write(log_block)

    print("\n=== VERDICT ===", flush=True)
    print(verdict_str, flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"wrote {OUT_MD}", flush=True)
    print(f"appended {LOG}", flush=True)
    print(f"wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
