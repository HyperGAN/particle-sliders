#!/usr/bin/env python3
"""Music→toy CPU stressors: locked recipe + Music-posture variant.

New Music-derived cells:
  1) lyric_span_entangle — span-heterogeneous leftover (cross-axis lyric analogue)
  2) dual_arm leftover_only vs listen/cover_only — separate cells, both must pass
  3) close_live_noise — close-pair multi-seed knife with live-like amp/span noise
  4) parts0 posture — n_particles=1 cover=1.0 vs toy n=12 cover=1.5

No Music GPU train. Do not touch nano-work-server / ComfyUI / run_server.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

# Apply scaffold hooks first (idempotent).
_hooks = Path(__file__).resolve().parent / "field3d_music_stressor_hooks.py"
if _hooks.exists():
    sys.path.insert(0, str(_hooks.parent))
    import field3d_music_stressor_hooks as _h  # noqa: E402

    try:
        _h.apply()
    except Exception as e:  # noqa: BLE001
        print("hooks apply skip/warn:", e, flush=True)

from analysis.slider2d.field3d import (  # noqa: E402
    Field3D,
    close_field3d,
    leftover_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

# Optional Fire #13 ctors
try:
    from analysis.slider2d.field3d import (  # noqa: E402
        cross_axis_rows_field3d,
        cross_axis_span_sample_field3d,
        lyric_span_entangle_field3d,
        close_live_noise_field3d,
        dual_arm_leftover_geom_field3d,
    )
except ImportError:
    cross_axis_rows_field3d = None  # type: ignore
    cross_axis_span_sample_field3d = None  # type: ignore

    def lyric_span_entangle_field3d(**kwargs) -> Field3D:  # type: ignore
        amps = (
            (1.05, 0.55, 0.35),
            (0.60, 1.10, 0.45),
            (0.75, 0.50, 0.90),
            (1.10, 0.85, 0.55),
            (0.90, 0.70, 0.65),
        )
        base = dict(
            kind="lyric_span_entangle",
            rows=5,
            row_scales=(0.75, 0.95, 1.05, 1.2, 1.35),
            row_amps=amps,
            slider=1.0,
            content=0.7,
            leak=0.55,
            e_on_u=0.05,
            e_on_content=0.45,
            e_unused=0.7,
        )
        base.update(kwargs)
        return Field3D(**base)

    def close_live_noise_field3d(seed: int = 0, amp_noise: float = 0.04, span_noise: float = 0.08, **kwargs) -> Field3D:  # type: ignore
        j = ((seed * 37) % 11) - 5
        k = ((seed * 53) % 9) - 4
        content = 0.90 + amp_noise * (j / 5.0)
        leak = max(0.02, 0.10 + amp_noise * (k / 4.0) * 0.5)
        scale = 1.0 + span_noise * (j / 10.0)
        base = dict(
            kind="close_live_noise",
            rows=1,
            row_scales=(scale,),
            slider=1.0,
            content=content,
            leak=leak,
            e_on_u=0.0,
            e_on_content=0.0,
            e_unused=1.0,
        )
        try:
            c = close_field3d(seed=seed)
            base["rows"] = int(getattr(c, "rows", 1))
            base["row_scales"] = tuple(float(s) * scale for s in getattr(c, "row_scales", (1.0,))) or (scale,)
            for attr in ("slider", "e_on_u", "e_on_content", "e_unused"):
                if hasattr(c, attr):
                    base[attr] = getattr(c, attr)
            base["content"] = content
            base["leak"] = leak
            base["kind"] = "close_live_noise"
        except Exception:
            pass
        base.update(kwargs)
        return Field3D(**base)

    def dual_arm_leftover_geom_field3d(**kwargs) -> Field3D:  # type: ignore
        base = dict(
            kind="dual_arm_leftover_geom",
            rows=3,
            row_scales=(1.0, 1.1, 0.9),
            slider=1.0,
            content=0.55,
            leak=0.45,
            e_on_u=0.0,
            e_on_content=0.0,
            e_unused=1.0,
        )
        base.update(kwargs)
        return Field3D(**base)


NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "music_to_toy_stressors_run_20260909.json"
OUT_MD = NOTES / "music_to_toy_stressors_run_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER_GUARD = "faithful_guard_e"


def cfg(seed: int, *, cover: float, n_particles: int, steps: int = 1200):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n_particles,
        particle_l2=0.02,
    )


def run_exam(field, *, seed: int, cover: float, n: int, teacher: str, name: str, mode: str = "exam") -> dict:
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
        "residual_norm": float(row.get("residual_norm", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "wall_s": round(time.time() - t0, 2),
        "teacher": teacher,
        "cover": cover,
        "n_particles": n,
        "name": name,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    prim = [r["primary"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    return {
        "n_pass": n_pass,
        "n_seeds": n,
        "pass_str": f"{n_pass}/{n}",
        "primary_mean": sum(prim) / n if n else 0.0,
        "primary_span": (max(prim) - min(prim)) if prim else 0.0,
        "leak_max": max(leaks) if leaks else 0.0,
        "knife_edge": 0 < n_pass < n,
        "pass_all": n_pass == n and n > 0,
        "fail_all": n_pass == 0 and n > 0,
    }


def grid(label: str, make_field, *, covers, ns, teacher=TEACHER_GUARD, seeds=SEEDS, mode="exam") -> dict:
    out = {}
    print(f"=== {label} ===", flush=True)
    for cover in covers:
        for n in ns:
            rows = []
            key = f"c{cover}_n{n}"
            for seed in seeds:
                field = make_field(seed)
                r = run_exam(
                    field,
                    seed=seed,
                    cover=cover,
                    n=n,
                    teacher=teacher,
                    name=f"{label}_{key}_s{seed}",
                    mode=mode,
                )
                rows.append(r)
                print(
                    f"  {key} seed={seed} pass={r['pass']} prim={r['primary']:.4f} leak={r['leak_ratio']:.4f} ({r['wall_s']}s)",
                    flush=True,
                )
            out[key] = {"summary": summarize(rows), "rows": rows}
    return out


def main() -> None:
    t_all = time.time()
    sha = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()
    NOTES.mkdir(parents=True, exist_ok=True)

    # Copy catalog into notes if present beside script
    src_cat = Path(__file__).resolve().parent / "music_to_toy_stressor_catalog_20260909.md"
    if src_cat.exists() and (not CATALOG.exists() or CATALOG.read_text() != src_cat.read_text()):
        CATALOG.write_text(src_cat.read_text())
        print("synced catalog ->", CATALOG, flush=True)

    results = {}

    # --- Stressor 1: lyric_span_entangle (M1) ---
    results["lyric_span_entangle"] = grid(
        "lyric_span_entangle",
        lambda seed: lyric_span_entangle_field3d(seed=seed),
        covers=(1.5, 1.0),
        ns=(12, 1),
    )

    # --- Stressor 2: dual-arm leftover vs listen (M4/M7/M8) ---
    print("=== dual_arm cells ===", flush=True)
    dual = {}
    # leftover_only: guard + cover=0
    rows = []
    for seed in SEEDS[:3]:
        r = run_exam(
            dual_arm_leftover_geom_field3d(seed=seed),
            seed=seed,
            cover=0.0,
            n=1,
            teacher=TEACHER_GUARD,
            name=f"arm_leftover_only_s{seed}",
            mode="score",
        )
        rows.append(r)
        print(f"  leftover_only seed={seed} pass={r['pass']} prim={r['primary']:.4f} leak={r['leak_ratio']:.4f}", flush=True)
    dual["leftover_only_guard_cover0_n1"] = {"summary": summarize(rows), "rows": rows}

    # listen/cover_only: no guard + cover=1.5
    rows = []
    for seed in SEEDS[:3]:
        r = run_exam(
            dual_arm_leftover_geom_field3d(seed=seed),
            seed=seed,
            cover=1.5,
            n=1,
            teacher="faithful",
            name=f"arm_listen_cover_only_s{seed}",
            mode="score",
        )
        rows.append(r)
        print(f"  listen_cover_only seed={seed} pass={r['pass']} prim={r['primary']:.4f} leak={r['leak_ratio']:.4f}", flush=True)
    dual["listen_cover_only_faithful_c1.5_n1"] = {"summary": summarize(rows), "rows": rows}

    # locked both: guard + cover=1.5
    rows = []
    for seed in SEEDS[:3]:
        r = run_exam(
            dual_arm_leftover_geom_field3d(seed=seed),
            seed=seed,
            cover=1.5,
            n=1,
            teacher=TEACHER_GUARD,
            name=f"arm_locked_s{seed}",
            mode="score",
        )
        rows.append(r)
        print(f"  locked_both seed={seed} pass={r['pass']} prim={r['primary']:.4f} leak={r['leak_ratio']:.4f}", flush=True)
    dual["locked_guard_cover1.5_n1"] = {"summary": summarize(rows), "rows": rows}
    results["dual_arm"] = dual

    # --- Stressor 3: close_live_noise knife (M3/M11) ---
    results["close_live_noise"] = grid(
        "close_live_noise",
        lambda seed: close_live_noise_field3d(seed=seed),
        covers=(1.5, 1.0),
        ns=(12, 1),
    )

    # Baseline close for delta
    results["close_clean"] = grid(
        "close_clean",
        lambda seed: close_field3d(seed=seed),
        covers=(1.5, 1.0),
        ns=(12, 1),
    )

    # leftover regression @ locked vs music posture
    results["leftover_baseline"] = grid(
        "leftover_baseline",
        lambda seed: leftover_field3d(),
        covers=(1.5, 1.0),
        ns=(12, 1),
        mode="score",
    )

    # Harden existing cross_axis if present
    if cross_axis_rows_field3d is not None:
        results["cross_axis_rows"] = grid(
            "cross_axis_rows",
            lambda seed: cross_axis_rows_field3d(seed=seed),
            covers=(1.5,),
            ns=(12, 1),
        )
    if cross_axis_span_sample_field3d is not None:
        results["cross_axis_span_sample"] = grid(
            "cross_axis_span_sample",
            lambda seed: cross_axis_span_sample_field3d(seed=seed),
            covers=(1.5,),
            ns=(12, 1),
        )

    # Wins: Music bugs that bite in-toy under locked or music posture
    wins = []
    bites = []

    def note_cell(path: str, summ: dict, expect_fail: bool, msg: str):
        bites.append({"cell": path, **summ, "expect_fail": expect_fail, "msg": msg})
        if expect_fail and (summ.get("fail_all") or summ.get("knife_edge") or not summ.get("pass_all")):
            wins.append(f"{path}: {summ['pass_str']} — {msg}")

    le = results["lyric_span_entangle"]
    for k, v in le.items():
        note_cell(f"lyric_span_entangle/{k}", v["summary"], True, "M1 span-entangled leftover bites")

    for k, v in dual.items():
        expect = k != "locked_guard_cover1.5_n1"
        note_cell(f"dual_arm/{k}", v["summary"], expect, "M4/M7/M8 dual-arm arm must not pass alone" if expect else "locked both should pass")
        if k == "locked_guard_cover1.5_n1" and v["summary"].get("pass_all"):
            wins.append(f"dual_arm/{k}: {v['summary']['pass_str']} — gate∧cover required (positive control)")

    for k, v in results["close_live_noise"].items():
        note_cell(f"close_live_noise/{k}", v["summary"], True, "M3/M11 close+live noise knife")

    payload = {
        "date": "2026-09-09",
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
        "music_posture": {"n_particles": 1, "cover_weight": 1.0, "parts_proxy": True},
        "results": results,
        "bites": bites,
        "wins_music_bug_now_fails_in_toy": wins,
        "wall_s": round(time.time() - t_all, 1),
    }
    # Strip bulky nested for md; full in json
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", OUT_JSON, flush=True)

    lines = [
        "# Music→toy stressors run — 2026-09-09",
        "",
        f"Host: pop-os CPU @ `{sha}`. Wall {payload['wall_s']}s. No Music train.",
        "",
        "## Recipe",
        "",
        "- Locked: 1200 / cover=1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1",
        "- Music-posture: n_particles=1 (parts0 proxy), cover=1.0",
        "",
        "## Pass/fail grid (summary)",
        "",
        "| cell | recipe | PASS | primary mean | leak max | knife |",
        "|---|---|:---:|---:|---:|:---:|",
    ]

    def add_rows(group: str, block: dict):
        if group == "dual_arm":
            for k, v in block.items():
                s = v["summary"]
                lines.append(
                    f"| {group}/{k} | dual | {s['pass_str']} | {s['primary_mean']:.4f} | {s['leak_max']:.4f} | {'yes' if s['knife_edge'] else 'no'} |"
                )
            return
        for k, v in block.items():
            s = v["summary"]
            recipe = "locked" if "c1.5_n12" in k else ("music-posture" if "c1.0_n1" in k else k)
            lines.append(
                f"| {group}/{k} | {recipe} | {s['pass_str']} | {s['primary_mean']:.4f} | {s['leak_max']:.4f} | {'yes' if s['knife_edge'] else 'no'} |"
            )

    for g in (
        "lyric_span_entangle",
        "dual_arm",
        "close_live_noise",
        "close_clean",
        "leftover_baseline",
        "cross_axis_rows",
        "cross_axis_span_sample",
    ):
        if g in results:
            add_rows(g, results[g])

    lines += [
        "",
        "## Wins — Music bug now fails (or knives) in-toy",
        "",
    ]
    if wins:
        for w in wins:
            lines.append(f"- {w}")
    else:
        lines.append("- (none yet — check JSON)")

    lines += [
        "",
        "## Still hard to encode",
        "",
        "- tx∩guard SystemExit / SpanTransformerD lyric positions",
        "- parts ↔ n_particles object mismatch (proxy only)",
        "- lyrichold ≠ cover/pole without live listen metrics",
        "",
        f"Catalog: `{CATALOG.name}`",
        f"JSON: `{OUT_JSON.name}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_MD, flush=True)

    # Append research_log
    block = [
        "",
        "## Fire — Music→toy stressors (2026-09-09)",
        "",
        f"- Host: pop-os-cpu @ SHA `{sha}`",
        f"- Catalog: `music_to_toy_stressor_catalog_20260909.md`",
        f"- Run: `music_to_toy_stressors_run_20260909.{{py,json,md}}` wall={payload['wall_s']}s",
        f"- Wins: {wins[:8] if wins else 'see JSON'}",
        "- No Music GPU train; servers untouched.",
        "",
    ]
    if LOG.exists():
        prev = LOG.read_text()
        if "Music→toy stressors (2026-09-09)" not in prev:
            LOG.write_text(prev.rstrip() + "\n" + "\n".join(block))
            print("appended", LOG, flush=True)
    else:
        LOG.write_text("# 2D adversarial slider research log — 2026-09-09\n" + "\n".join(block))
        print("created", LOG, flush=True)

    print("WINS:", json.dumps(wins, indent=2), flush=True)


if __name__ == "__main__":
    main()
