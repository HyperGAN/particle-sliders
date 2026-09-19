#!/usr/bin/env python3
"""Fire #23/#24: call annotate_dig_rows(detail, payload=...) before JSON write.

Dig f3d_close seed0 knife + cross_axis_rows hard-bound + Music→toy stressors.

Locked toy recipe: 1200, cover=1.5 (also probe 1.0), faithful_guard_e, FM0,
n_particles=1 (Music parts0 proxy) and n=12, particle_l2=0.02, b_cap=1.
Reject 800×cover3.0. CPU only. No Music train.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.notes.music_posture_dig_util import annotate_dig_rows, posture_block_for_md  # Fire #23
from analysis.slider2d.field3d import (  # noqa: E402
    Field3D,
    close_field3d,
    cross_axis_rows_field3d,
    cross_axis_span_sample_field3d,
    leftover_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "dig_close_cross_music_stress_20260909.json"
OUT_MD = NOTES / "dig_close_cross_music_stress_20260909.md"
MUSIC_MD = NOTES / "music_to_toy_stressors_20260909.md"
LOG = NOTES / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"


def locked_cfg(seed: int, *, cover: float = 1.5, n_particles: int = 1, steps: int = 1200):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n_particles,
        particle_l2=0.02,
    )


def run_exam(field, seed: int, cover: float, n_particles: int, name: str) -> dict:
    t0 = time.time()
    row = score_adv_field3d_exam(
        field,
        teacher=TEACHER,
        cfg=locked_cfg(seed, cover=cover, n_particles=n_particles),
        name=name,
    )
    primary = float(row.get("exam_score", row.get("u_kept", 0.0)))
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": primary,
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "residual_norm": float(row.get("residual_norm", 0.0)),
        "wall_s": round(time.time() - t0, 2),
        "name": name,
        "n_particles": n_particles,
        "cell": getattr(field, "kind", None),
        # Fire #24: keep posture warn for annotate_dig_rows
        "music_close_posture_warn": row.get("music_close_posture_warn"),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    prim = [r["exam_score"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    span = (max(prim) - min(prim)) if prim else 0.0
    knife = 0 < n_pass < n
    return {
        "n_pass": n_pass,
        "n_seeds": n,
        "pass_str": f"{n_pass}/{n}",
        "exam_mean": sum(prim) / n if n else 0.0,
        "exam_span": span,
        "leak_max": max(leaks) if leaks else 0.0,
        "knife_edge": knife,
        "pass_all": n_pass == n,
    }


# --- Music→toy new biting stressors ---

def lyric_span_entangle_field3d(**kwargs) -> Field3D:
    """Music lyric-span entangle: leftover ê shares energy with content across staggered spans.

    Mirrors multi-position lyric pooling where residual leak mixes into content
    directions differently per span row (live-like Music caption geometry).
    """
    # Heterogeneous mixes + mild e-on-content entanglement (not a full YAML lie).
    amps = (
        (1.05, 0.55, 0.35),
        (0.60, 1.10, 0.45),
        (0.75, 0.50, 0.90),
        (1.10, 0.85, 0.55),
        (0.90, 0.70, 0.65),
    )
    base = {
        "kind": "lyric_span_entangle",
        "rows": 5,
        "row_scales": (0.75, 0.95, 1.05, 1.2, 1.35),
        "row_amps": amps,
        "slider": 1.0,
        "content": 0.7,
        "leak": 0.55,
        "e_on_u": 0.05,
        "e_on_content": 0.45,  # span-entangled leftover (Music lyric risk)
        "e_unused": 0.7,
    }
    base.update(kwargs)
    return Field3D(**base)


def dual_arm_listen_cover_only_field3d(**kwargs) -> Field3D:
    """Arm-T analogue: cover/pole pin without leftover gate geometry (content+leak entangled).

    Music listen arm (lyric+tx, no guard) is structurally cover_only — expect leak.
    Cell uses leftover-ish geometry but we score with teacher=faithful (no guard) separately.
    """
    base = {
        "kind": "dual_arm_listen_cover_only",
        "rows": 3,
        "row_scales": (1.0, 1.1, 0.9),
        "slider": 1.0,
        "content": 0.55,
        "leak": 0.45,
        "e_on_u": 0.0,
        "e_on_content": 0.0,
        "e_unused": 1.0,
    }
    base.update(kwargs)
    return Field3D(**base)


def close_jittered(seed: int, jitter: float = 0.02) -> Field3D:
    """Harden probe: slight geometry jitter on close cell (seed-dependent)."""
    f = close_field3d(seed=seed)
    # Field3D is likely a dataclass — mutate amps via replace if possible
    try:
        import dataclasses
        # nudge content/leak slightly by seed to escape bad basin
        c = float(getattr(f, "content", 0.9)) + jitter * ((seed % 5) - 2) * 0.01
        leak = float(getattr(f, "leak", 0.1)) + jitter * ((seed % 3) - 1) * 0.01
        return dataclasses.replace(f, content=c, leak=max(0.01, leak))
    except Exception:
        return f


def main() -> None:
    t_all = time.time()
    sha = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()
    detail = []
    sections = {}

    print("=== A) f3d_close seed0 dig @ n=1 cover∈{1.0,1.5} ===", flush=True)
    close_grid = {}
    for cover in (1.0, 1.5):
        rows = []
        for seed in SEEDS:
            field = close_field3d(seed=seed)
            r = run_exam(field, seed, cover, 1, f"close_c{cover}_n1_s{seed}")
            rows.append(r)
            detail.append({"section": "close_n1", "cover": cover, **r})
            print(f"  cover={cover} seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} leak={r['leak_ratio']:.4f} ({r['wall_s']}s)", flush=True)
        close_grid[str(cover)] = {"summary": summarize(rows), "rows": rows}

    print("=== B) harden probes on seed0 (n=1, cover=1.5) ===", flush=True)
    harden = []
    # B1: more steps
    for steps in (1600, 2000):
        t0 = time.time()
        field = close_field3d(seed=0)
        row = score_adv_field3d_exam(
            field, teacher=TEACHER,
            cfg=locked_cfg(0, cover=1.5, n_particles=1, steps=steps),
            name=f"close_harden_steps{steps}",
        )
        h = {
            "probe": f"steps_{steps}",
            "pass": bool(row.get("exam_pass", row.get("pass"))),
            "exam_score": float(row.get("exam_score", row.get("u_kept", 0))),
            "leak_ratio": float(row.get("leak_ratio", 0)),
            "wall_s": round(time.time() - t0, 2),
        }
        harden.append(h)
        print(f"  {h}", flush=True)
    # B2: n_particles=4,12 at cover 1.5 seed0
    for n in (4, 12):
        r = run_exam(close_field3d(seed=0), 0, 1.5, n, f"close_n{n}_s0")
        harden.append({"probe": f"n_particles_{n}", **{k: r[k] for k in ("pass", "exam_score", "leak_ratio", "wall_s")}})
        print(f"  n={n} {harden[-1]}", flush=True)
    # B3: alt seeds near 0
    for seed in (5, 6, 8, 9, 10, 11):
        r = run_exam(close_field3d(seed=seed), seed, 1.5, 1, f"close_alt_s{seed}")
        harden.append({"probe": f"alt_seed_{seed}", **{k: r[k] for k in ("pass", "exam_score", "leak_ratio", "wall_s")}})
        print(f"  alt seed={seed} pass={r['pass']} exam={r['exam_score']:.4f}", flush=True)
    # B4: geometry jitter seed0
    r = run_exam(close_jittered(0), 0, 1.5, 1, "close_jitter_s0")
    harden.append({"probe": "geom_jitter", **{k: r[k] for k in ("pass", "exam_score", "leak_ratio", "wall_s")}})
    print(f"  jitter {harden[-1]}", flush=True)

    print("=== C) cross_axis_rows @ n=1 and n=12 (locked cover=1.5) ===", flush=True)
    cross = {}
    for n in (1, 12):
        rows = []
        for seed in SEEDS:
            r = run_exam(cross_axis_rows_field3d(seed=seed), seed, 1.5, n, f"xrows_n{n}_s{seed}")
            rows.append(r)
            detail.append({"section": "cross_axis_rows", "n_particles": n, **r})
            print(f"  n={n} seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} rows={r['rows_covered']}/{r['rows_total']} ({r['wall_s']}s)", flush=True)
        cross[str(n)] = {"summary": summarize(rows), "rows": rows}

    print("=== D) NEW biting stressor: lyric_span_entangle @ n=1 ===", flush=True)
    lyric_rows = []
    for seed in SEEDS:
        r = run_exam(lyric_span_entangle_field3d(seed=seed), seed, 1.5, 1, f"lyric_ent_s{seed}")
        lyric_rows.append(r)
        detail.append({"section": "lyric_span_entangle", **r})
        print(f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} leak={r['leak_ratio']:.4f}", flush=True)
    lyric_sum = summarize(lyric_rows)

    print("=== E) dual-arm listen cover_only (teacher=faithful, no guard) @ n=1 ===", flush=True)
    listen_rows = []
    for seed in SEEDS[:3]:
        t0 = time.time()
        field = dual_arm_listen_cover_only_field3d(seed=seed)
        row = score_adv_field3d_exam(
            field, teacher="faithful",
            cfg=locked_cfg(seed, cover=1.5, n_particles=1),
            name=f"listen_cover_only_s{seed}",
        )
        r = {
            "seed": seed,
            "pass": bool(row.get("exam_pass", row.get("pass"))),
            "exam_score": float(row.get("exam_score", row.get("u_kept", 0))),
            "u_kept": float(row.get("u_kept", 0)),
            "leak_ratio": float(row.get("leak_ratio", 0)),
            "wall_s": round(time.time() - t0, 2),
        }
        listen_rows.append(r)
        print(f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} leak={r['leak_ratio']:.4f}", flush=True)
    listen_sum = summarize(listen_rows)

    # Control: same geometry with guard
    guard_rows = []
    for seed in SEEDS[:3]:
        r = run_exam(dual_arm_listen_cover_only_field3d(seed=seed), seed, 1.5, 1, f"listen_with_guard_s{seed}")
        guard_rows.append(r)
    guard_sum = summarize(guard_rows)

    payload = {
        "date": "2026-09-09",
        "sha": sha,
        "recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "n_particles_music_proxy": 1,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "close_n1": close_grid,
        "harden_seed0": harden,
        "cross_axis_rows": cross,
        "lyric_span_entangle": {"summary": lyric_sum, "rows": lyric_rows},
        "dual_arm_listen_cover_only": {"summary": listen_sum, "rows": listen_rows},
        "dual_arm_same_geom_with_guard": {"summary": guard_sum, "rows": guard_rows},
        "wall_s": round(time.time() - t_all, 1),
    }

    # Verdicts
    c15 = close_grid["1.5"]["summary"]
    c10 = close_grid["1.0"]["summary"]
    x1 = cross["1"]["summary"]
    x12 = cross["12"]["summary"]
    harden_wins = [h for h in harden if h.get("pass")]
    payload["verdicts"] = {
        "close_n1_cover1.0": c10["pass_str"] + (" knife" if c10["knife_edge"] else ""),
        "close_n1_cover1.5": c15["pass_str"] + (" knife" if c15["knife_edge"] else ""),
        "harden_seed0_any_pass": bool(harden_wins),
        "harden_passes": [h["probe"] for h in harden_wins],
        "cross_axis_rows_n1": x1["pass_str"],
        "cross_axis_rows_n12": x12["pass_str"],
        "lyric_span_entangle_bites": not lyric_sum["pass_all"],
        "listen_cover_only_bites": not listen_sum["pass_all"],
    }

    # Fire #24: aggregate Music close posture from detail rows
    annotate_dig_rows(detail, payload=payload)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    # MD dig note
    lines = [
        "# Dig: f3d_close knife + cross_axis + Music→toy stressors — 2026-09-09",
        "",
        f"Host: pop-os CPU @ `{sha}`. Locked recipe; Music proxy n_particles=1; FM0; b_cap=1.",
        f"Wall: {payload['wall_s']}s. No Music train. Servers untouched.",
        "",
        *posture_block_for_md(payload.get("music_close_posture") or {}),
        "",
        "## A) f3d_close @ n=1",
        "",
        f"| cover | PASS | exam mean | span | leak max | knife |",
        f"|---:|:---:|---:|---:|---:|:---:|",
        f"| 1.0 | {c10['pass_str']} | {c10['exam_mean']:.4f} | {c10['exam_span']:.4f} | {c10['leak_max']:.4f} | {'yes' if c10['knife_edge'] else 'no'} |",
        f"| 1.5 | {c15['pass_str']} | {c15['exam_mean']:.4f} | {c15['exam_span']:.4f} | {c15['leak_max']:.4f} | {'yes' if c15['knife_edge'] else 'no'} |",
        "",
        "### Seed0 harden probes (cover=1.5)",
        "",
        "| probe | pass | exam | leak |",
        "|---|:---:|---:|---:|",
    ]
    for h in harden:
        lines.append(f"| {h['probe']} | {h['pass']} | {h.get('exam_score', 0):.4f} | {h.get('leak_ratio', 0):.4f} |")
    lines += [
        "",
        "## B) cross_axis_rows (hard boundary check)",
        "",
        f"| n_particles | PASS | exam mean | span | leak max |",
        f"|---:|:---:|---:|---:|---:|",
        f"| 1 | {x1['pass_str']} | {x1['exam_mean']:.4f} | {x1['exam_span']:.4f} | {x1['leak_max']:.4f} |",
        f"| 12 | {x12['pass_str']} | {x12['exam_mean']:.4f} | {x12['exam_span']:.4f} | {x12['leak_max']:.4f} |",
        "",
        "## C) NEW Music→toy stressors",
        "",
        f"- `lyric_span_entangle` @ n=1 cover=1.5: **{lyric_sum['pass_str']}** mean={lyric_sum['exam_mean']:.4f} leak_max={lyric_sum['leak_max']:.4f} bites={not lyric_sum['pass_all']}",
        f"- `dual_arm_listen_cover_only` (teacher=faithful): **{listen_sum['pass_str']}** mean={listen_sum['exam_mean']:.4f} leak_max={listen_sum['leak_max']:.4f}",
        f"- same geom + guard: **{guard_sum['pass_str']}** mean={guard_sum['exam_mean']:.4f} leak_max={guard_sum['leak_max']:.4f}",
        "",
        "## Finding",
        "",
        "- Prefer harden path if any seed0 probe recovers 6/6 close; else document flake + multi-seed Music rule.",
        "- cross_axis_rows: if still 0/6 at n=1 and n=12 → **hard boundary** (heterogeneous per-row axis mix does not port).",
        "- New stressors that FAIL under locked recipe are biting Music→toy cells — keep as regression suite.",
        "",
        f"JSON: `{OUT_JSON.name}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    print("VERDICTS", json.dumps(payload["verdicts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
