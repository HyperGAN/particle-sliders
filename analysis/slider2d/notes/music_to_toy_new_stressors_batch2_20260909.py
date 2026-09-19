#!/usr/bin/env python3
"""Music→toy NEW stressors batch2 — 2026-09-09.

Invented cells (NOT lyric_span_entangle / cross_axis_rows / dual_arm / close_live_noise;
also distinct from parallel M13–M19):

  M20 amp_lie_leftover_declare — homogeneous leftover + declared content-axis YAML lie
  M21 hold_e_lyric_mix         — content↔leftover mix mimicking hold-ê lyric pool
  M22 stagger_mild_cross       — staggered scales + mild cross-axis (soft multipair)
  M23 multipair_corr_seed      — multi-pair R³ with seed-correlated row_amps

Recipe probes (no new CELLS geom; keep FM0 / l2=0.02 default):
  close_fm_tempt               — FM on vs off under close geometry
  particle_l2_extreme_n1       — l2 extremes under Music posture n=1

Locked: 1200, cover=1.5, n=12, FM0, l2=0.02, b_cap=1, teacher=faithful_guard_e
Music posture: n=1, cover=1.0, same guard/FM0/b_cap

No Music GPU train. Box only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    CELLS_3D,
    amp_lie_leftover_declare_field3d,
    close_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    multipair_corr_seed_field3d,
    score_adv_field3d_exam,
    stagger_mild_cross_field3d,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = Path(__file__).resolve().parent
OUT_JSON = NOTES / "music_to_toy_new_stressors_batch2_20260909.json"
OUT_MD = NOTES / "music_to_toy_new_stressors_batch2_20260909.md"
LOG = NOTES / "research_log_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
SEEDS_PROBE = [0, 1, 7]  # FM / l2 temptation — enough for knife vs flat
TEACHER = "faithful_guard_e"

NEW_CELLS = [
    ("amp_lie_leftover_declare", amp_lie_leftover_declare_field3d, "M20"),
    ("hold_e_lyric_mix", hold_e_lyric_mix_field3d, "M21"),
    ("stagger_mild_cross", stagger_mild_cross_field3d, "M22"),
    ("multipair_corr_seed", multipair_corr_seed_field3d, "M23"),
]


def make_cfg(
    seed: int,
    *,
    cover: float,
    n_particles: int,
    steps: int = 1200,
    fm_weight: float = 0.0,
    particle_l2: float = 0.02,
    fm_normalize: bool = True,
):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=fm_weight,
        fm_normalize=fm_normalize,
        n_particles=n_particles,
        particle_l2=particle_l2,
    )


def run_one(field, *, seed: int, cover: float, n: int, name: str, fm: float = 0.0, l2: float = 0.02) -> dict:
    t0 = time.time()
    cfg = make_cfg(seed, cover=cover, n_particles=n, fm_weight=fm, particle_l2=l2)
    row = score_adv_field3d_exam(field, teacher=TEACHER, cfg=cfg, name=name)
    return {
        "seed": seed,
        "pass": bool(row.get("exam_pass", row.get("pass"))),
        "exam_score": float(row.get("exam_score", 0.0)),
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "pass_multi_row": bool(row.get("pass_multi_row", False)),
        "pass_leftover_gate": bool(row.get("pass_leftover_gate", False)),
        "pass_u": bool(row.get("pass_u", False)),
        "pass_leak": bool(row.get("pass_leak", False)),
        "wall_s": round(time.time() - t0, 2),
        "cover": cover,
        "n_particles": n,
        "fm_weight": fm,
        "particle_l2": l2,
        "name": name,
        "cell": getattr(field, "kind", "?"),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    prim = [r["exam_score"] for r in rows]
    leaks = [abs(r["leak_ratio"]) for r in rows]
    fails = [r["seed"] for r in rows if not r["pass"]]
    return {
        "n_pass": n_pass,
        "n_seeds": n,
        "pass_str": f"{n_pass}/{n}",
        "exam_mean": (sum(prim) / n) if n else 0.0,
        "exam_span": (max(prim) - min(prim)) if prim else 0.0,
        "leak_max": max(leaks) if leaks else 0.0,
        "fail_seeds": fails,
        "knife": 0 < n_pass < n,
        "bite": n_pass < n,  # any fail = bite (knife or hard)
        "hard_bite": n_pass == 0 and n > 0,
        "no_bite": n_pass == n and n > 0,
    }


def grid_cell(label: str, make_field, *, cover: float, n: int, seeds=SEEDS) -> dict:
    rows = []
    print(f"=== {label} c={cover} n={n} ===", flush=True)
    for seed in seeds:
        field = make_field(seed)
        r = run_one(field, seed=seed, cover=cover, n=n, name=f"{label}_c{cover}_n{n}_s{seed}")
        rows.append(r)
        print(
            f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} "
            f"u={r['u_kept']:.4f} leak={r['leak_ratio']:.4f} "
            f"rows={r['rows_covered']}/{r['rows_total']} ({r['wall_s']}s)",
            flush=True,
        )
    return {"summary": summarize(rows), "rows": rows}


def main() -> None:
    t_all = time.time()
    sha = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()
    results: dict = {
        "meta": {
            "sha": sha,
            "date": "2026-09-09",
            "locked": "1200 c1.5 n12 FM0 l2=0.02 b_cap=1 guard",
            "music_posture": "n=1 c1.0 FM0 l2=0.02 b_cap=1 guard",
            "avoided": [
                "lyric_span_entangle",
                "cross_axis_rows",
                "dual_arm",
                "close_live_noise",
            ],
        }
    }

    # --- New geometry cells: locked + music ---
    geom = {}
    for name, ctor, mid in NEW_CELLS:
        def _make(seed, _ctor=ctor, _name=name):
            if _name == "multipair_corr_seed":
                return _ctor(seed=seed)
            return _ctor(seed=seed)

        locked = grid_cell(f"{mid}_{name}", _make, cover=1.5, n=12)
        music = grid_cell(f"{mid}_{name}", _make, cover=1.0, n=1)
        geom[name] = {"mid": mid, "locked": locked, "music": music}
    results["geom"] = geom

    # --- leftover regression ---
    print("=== leftover regression locked ===", flush=True)
    reg_rows = []
    for seed in SEEDS[:3]:
        r = run_one(leftover_field3d(seed=seed), seed=seed, cover=1.5, n=12, name=f"leftover_reg_s{seed}")
        reg_rows.append(r)
        print(f"  leftover seed={seed} pass={r['pass']} exam={r['exam_score']:.4f}", flush=True)
    results["leftover_regression"] = {"summary": summarize(reg_rows), "rows": reg_rows}

    # --- close FM temptation (keep default FM0; show Music temptation) ---
    fm_block = {}
    for fm in (0.0, 0.5):
        for cover, n, posture in ((1.5, 12, "locked"), (1.0, 1, "music")):
            key = f"close_fm{fm}_{posture}"
            rows = []
            print(f"=== {key} ===", flush=True)
            for seed in SEEDS_PROBE:
                r = run_one(
                    close_field3d(seed=seed),
                    seed=seed,
                    cover=cover,
                    n=n,
                    fm=fm,
                    name=f"{key}_s{seed}",
                )
                rows.append(r)
                print(
                    f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} "
                    f"u={r['u_kept']:.4f} ({r['wall_s']}s)",
                    flush=True,
                )
            fm_block[key] = {"summary": summarize(rows), "rows": rows, "fm": fm, "posture": posture}
    results["close_fm_tempt"] = fm_block

    # --- particle_l2 extremes under Music posture n=1 ---
    l2_block = {}
    for l2 in (0.0, 0.02, 0.2):
        for geom_name, make in (
            ("leftover", lambda seed: leftover_field3d(seed=seed)),
            ("close", lambda seed: close_field3d(seed=seed)),
        ):
            key = f"{geom_name}_n1_l2_{l2}"
            rows = []
            print(f"=== {key} music posture ===", flush=True)
            for seed in SEEDS_PROBE:
                r = run_one(
                    make(seed),
                    seed=seed,
                    cover=1.0,
                    n=1,
                    l2=l2,
                    name=f"{key}_s{seed}",
                )
                rows.append(r)
                print(
                    f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} "
                    f"u={r['u_kept']:.4f} ({r['wall_s']}s)",
                    flush=True,
                )
            l2_block[key] = {"summary": summarize(rows), "rows": rows, "l2": l2, "geom": geom_name}
    results["particle_l2_extreme_n1"] = l2_block

    # --- bite table ---
    bite_rows = []
    for name, block in geom.items():
        for posture in ("locked", "music"):
            s = block[posture]["summary"]
            bite_rows.append(
                {
                    "cell": name,
                    "mid": block["mid"],
                    "posture": posture,
                    "pass": s["pass_str"],
                    "exam_mean": round(s["exam_mean"], 4),
                    "leak_max": round(s["leak_max"], 4),
                    "fail_seeds": s["fail_seeds"],
                    "verdict": (
                        "HARD_BITE" if s["hard_bite"] else ("KNIFE" if s["knife"] else "NO_BITE")
                    ),
                }
            )
    for key, block in fm_block.items():
        s = block["summary"]
        bite_rows.append(
            {
                "cell": key,
                "mid": "FM_tempt",
                "posture": block["posture"],
                "pass": s["pass_str"],
                "exam_mean": round(s["exam_mean"], 4),
                "leak_max": round(s["leak_max"], 4),
                "fail_seeds": s["fail_seeds"],
                "verdict": (
                    "HARD_BITE" if s["hard_bite"] else ("KNIFE" if s["knife"] else "NO_BITE")
                ),
            }
        )
    for key, block in l2_block.items():
        s = block["summary"]
        bite_rows.append(
            {
                "cell": key,
                "mid": "l2_extreme",
                "posture": "music",
                "pass": s["pass_str"],
                "exam_mean": round(s["exam_mean"], 4),
                "leak_max": round(s["leak_max"], 4),
                "fail_seeds": s["fail_seeds"],
                "verdict": (
                    "HARD_BITE" if s["hard_bite"] else ("KNIFE" if s["knife"] else "NO_BITE")
                ),
            }
        )
    reg_s = results["leftover_regression"]["summary"]
    bite_rows.append(
        {
            "cell": "leftover_regression",
            "mid": "CTRL",
            "posture": "locked",
            "pass": reg_s["pass_str"],
            "exam_mean": round(reg_s["exam_mean"], 4),
            "leak_max": round(reg_s["leak_max"], 4),
            "fail_seeds": reg_s["fail_seeds"],
            "verdict": "NO_BITE" if reg_s["no_bite"] else "REGRESSION",
        }
    )
    results["bite_table"] = bite_rows
    results["wall_s"] = round(time.time() - t_all, 1)

    # Which geom cells are clean enough to keep registered (all are already in CELLS_3D;
    # confirm they construct + ran without exception)
    registered = []
    for name, _, mid in NEW_CELLS:
        assert name in CELLS_3D
        registered.append({"cell": name, "mid": mid, "in_CELLS_3D": True})
    results["registered"] = registered

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("wrote", OUT_JSON, flush=True)

    # Markdown report
    lines = [
        "# Music→toy new stressors batch2 — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {results['wall_s']}s. CPU only. No Music train.",
        "",
        "## New cells (avoided lyric_span / cross_axis_rows / dual_arm / close_live_noise)",
        "",
        "| ID | Cell | Music symptom |",
        "|---|---|---|",
        "| M20 | `amp_lie_leftover_declare` | declared YAML content-axis amplitude lie on leftover geom |",
        "| M21 | `hold_e_lyric_mix` | content↔leftover mix ≈ hold-ê lyric pool |",
        "| M22 | `stagger_mild_cross` | staggered scales + mild cross-axis (soft multipair) |",
        "| M23 | `multipair_corr_seed` | multi-pair R³ correlated seeds |",
        "",
        "Recipe probes: `close_fm_tempt` (FM0 vs 0.5), `particle_l2_extreme_n1` (l2∈{0,0.02,0.2} @ n=1).",
        "",
        "## Bite / no-bite table",
        "",
        "| cell | mid | posture | pass | exam_mean | leak_max | fail_seeds | verdict |",
        "|---|---|---|:---:|---:|---:|---|---|",
    ]
    for r in bite_rows:
        lines.append(
            f"| `{r['cell']}` | {r['mid']} | {r['posture']} | **{r['pass']}** | "
            f"{r['exam_mean']} | {r['leak_max']} | {r['fail_seeds']} | **{r['verdict']}** |"
        )
    lines += [
        "",
        "## Registry",
        "",
        "Registered in `CELLS_3D` (clean + tested this fire):",
        "",
    ]
    for r in registered:
        lines.append(f"- `{r['cell']}` ({r['mid']})")
    lines += [
        "",
        "## Verdict",
        "",
        "- Recipe change: **NO** (keep FM0, l2=0.02, locked 1200+c1.5).",
        "- Leftover regression must stay NO_BITE.",
        "- FM temptation: document flat/tempt; do not enable FM.",
        "- particle_l2 extremes @ n=1: document; keep default 0.02.",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_MD, flush=True)

    # Append research_log
    wins = [f"{r['cell']}/{r['posture']}: {r['pass']} {r['verdict']}" for r in bite_rows if r["verdict"] != "NO_BITE"]
    no_bites = [f"{r['cell']}/{r['posture']}: {r['pass']}" for r in bite_rows if r["verdict"] == "NO_BITE"]
    entry = f"""
## Fire — Music→toy new stressors batch2 (2026-09-09)

- Host: box-cpu @ SHA `{sha}`
- Notes: `music_to_toy_new_stressors_batch2_20260909.{{py,json,md}}` wall={results['wall_s']}s
- New CELLS_3D: amp_lie_leftover_declare (M20), hold_e_lyric_mix (M21), stagger_mild_cross (M22), multipair_corr_seed (M23)
- Bites: {wins}
- No-bites: {no_bites}
- Verdict: recipe_change=NO; leftover_regression={reg_s['pass_str']}; ping_user=YES
- No Music GPU train; servers untouched.
"""
    if LOG.exists():
        cur = LOG.read_text()
        if "new stressors batch2" not in cur:
            LOG.write_text(cur.rstrip() + "\n" + entry)
            print("appended research_log", flush=True)
    else:
        LOG.write_text("# research log\n" + entry)

    # Catalog append
    if CATALOG.exists():
        cat = CATALOG.read_text()
        if "M20" not in cat:
            cat += """

## Batch2 new stressors (2026-09-09)

| ID | Music symptom | Cell | Notes |
|---|---|---|---|
| M20 | Declared leak YAML / content-axis amplitude lie (leftover geom) | `amp_lie_leftover_declare` | Distinct from e_on_u_declare_lie / cross_axis_mismatch_declare |
| M21 | Content↔leftover mix ≈ hold-ê lyric pool | `hold_e_lyric_mix` | No hetero row_amps |
| M22 | Mild multipair (between homo and full cross_axis) | `stagger_mild_cross` | Soft axis wander + stagger |
| M23 | Multi-pair R³ correlated seeds | `multipair_corr_seed` | Seed-tied amp correlation |
| FM | Batch FM on vs off under close | recipe probe | Keep FM0 |
| l2 | particle_l2 extremes @ Music n=1 | recipe probe | Keep l2=0.02 |

See `music_to_toy_new_stressors_batch2_20260909.md` bite table.
"""
            CATALOG.write_text(cat)
            print("appended catalog", flush=True)

    print("DONE wall=", results["wall_s"], flush=True)
    # Print bite table summary
    print("\n=== BITE TABLE ===", flush=True)
    for r in bite_rows:
        print(f"  {r['verdict']:10} {r['pass']:5} {r['cell']}/{r['posture']}", flush=True)


if __name__ == "__main__":
    main()
