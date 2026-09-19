#!/usr/bin/env python3
"""Music→toy batch3 continuous grind — 2026-09-09.

1. Catalog already lists M20–M23; this fire invents M24+ and deepens M21.
2. New cells (NOT lyric_span / cross_axis_rows / dual_arm / close_live /
   M20–M23 / M13–M19 duplicates):

  M24 content_leak_flip_rows — û-primary + content↔leak flip across rows
  M25 lyric_neu_heavy_gate   — heavy lyric neu vs leftover unused-ê gate
  M26 declare_split_three    — ambiguous declared ê split û/content/unused
  M27 scale_descent_homo     — homo amps + descending scales (traj reverse)

3. M21 deepen: portable knobs (n/cover/steps/vic/e_on_content/false-lock)
   — can any recover hold_e_lyric_mix without false lock?

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
    content_leak_flip_rows_field3d,
    declare_split_three_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    lyric_neu_heavy_gate_field3d,
    scale_descent_homo_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = Path(__file__).resolve().parent
OUT_JSON = NOTES / "music_to_toy_batch3_20260909.json"
OUT_MD = NOTES / "music_to_toy_batch3_20260909.md"
LOG = NOTES / "research_log_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
SEEDS_DEEP = [0, 1, 2]  # deepen probes
TEACHER = "faithful_guard_e"

NEW_CELLS = [
    ("content_leak_flip_rows", content_leak_flip_rows_field3d, "M24"),
    ("lyric_neu_heavy_gate", lyric_neu_heavy_gate_field3d, "M25"),
    ("declare_split_three", declare_split_three_field3d, "M26"),
    ("scale_descent_homo", scale_descent_homo_field3d, "M27"),
]


def make_cfg(
    seed: int,
    *,
    cover: float,
    n_particles: int,
    steps: int = 1200,
    fm_weight: float = 0.0,
    particle_l2: float = 0.02,
    vicreg_weight: float = 0.05,
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
        vicreg_weight=vicreg_weight,
    )


def run_one(
    field,
    *,
    seed: int,
    cover: float,
    n: int,
    name: str,
    steps: int = 1200,
    fm: float = 0.0,
    l2: float = 0.02,
    vic: float = 0.05,
) -> dict:
    t0 = time.time()
    cfg = make_cfg(
        seed,
        cover=cover,
        n_particles=n,
        steps=steps,
        fm_weight=fm,
        particle_l2=l2,
        vicreg_weight=vic,
    )
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
        "steps": steps,
        "vicreg_weight": vic,
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
        "bite": n_pass < n,
        "hard_bite": n_pass == 0 and n > 0,
        "no_bite": n_pass == n and n > 0,
    }


def grid_cell(label: str, make_field, *, cover: float, n: int, seeds=SEEDS, **kw) -> dict:
    rows = []
    print(f"=== {label} c={cover} n={n} ===", flush=True)
    for seed in seeds:
        field = make_field(seed)
        r = run_one(
            field,
            seed=seed,
            cover=cover,
            n=n,
            name=f"{label}_c{cover}_n{n}_s{seed}",
            **kw,
        )
        rows.append(r)
        print(
            f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} "
            f"u={r['u_kept']:.4f} leak={r['leak_ratio']:.4f} "
            f"rows={r['rows_covered']}/{r['rows_total']} ({r['wall_s']}s)",
            flush=True,
        )
    return {"summary": summarize(rows), "rows": rows}


def verdict_of(s: dict) -> str:
    if s["hard_bite"]:
        return "HARD_BITE"
    if s["knife"]:
        return "KNIFE"
    if s["no_bite"]:
        return "NO_BITE"
    return "UNKNOWN"


def main() -> None:
    t_all = time.time()
    sha = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()
    results: dict = {
        "meta": {
            "sha": sha,
            "date": "2026-09-09",
            "locked": "1200 c1.5 n12 FM0 l2=0.02 b_cap=1 guard vic=0.05",
            "music_posture": "n=1 c1.0 FM0 l2=0.02 b_cap=1 guard",
            "avoided": [
                "lyric_span_entangle",
                "cross_axis_rows",
                "dual_arm",
                "close_live_noise",
                "amp_lie_leftover_declare",
                "hold_e_lyric_mix",
                "stagger_mild_cross",
                "multipair_corr_seed",
            ],
        }
    }

    # --- New geometry cells: locked + music ---
    geom = {}
    for name, ctor, mid in NEW_CELLS:

        def _make(seed, _ctor=ctor):
            return _ctor(seed=seed)

        locked = grid_cell(f"{mid}_{name}", _make, cover=1.5, n=12)
        music = grid_cell(f"{mid}_{name}", _make, cover=1.0, n=1)
        geom[name] = {"mid": mid, "locked": locked, "music": music}
    results["geom"] = geom

    # --- leftover regression (must stay green) ---
    print("=== leftover regression locked ===", flush=True)
    reg_rows = []
    for seed in SEEDS[:3]:
        r = run_one(
            leftover_field3d(seed=seed),
            seed=seed,
            cover=1.5,
            n=12,
            name=f"leftover_reg_s{seed}",
        )
        reg_rows.append(r)
        print(
            f"  leftover seed={seed} pass={r['pass']} exam={r['exam_score']:.4f}",
            flush=True,
        )
    results["leftover_regression"] = {"summary": summarize(reg_rows), "rows": reg_rows}

    # --- M21 deepen: portable knobs ---
    print("=== M21 hold_e_lyric_mix deepen ===", flush=True)
    deepen = {}
    # Baseline locked + music (confirm HARD_BITE)
    deepen["baseline_locked"] = grid_cell(
        "M21_baseline",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed),
        cover=1.5,
        n=12,
        seeds=SEEDS_DEEP,
    )
    deepen["baseline_music"] = grid_cell(
        "M21_baseline",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed),
        cover=1.0,
        n=1,
        seeds=SEEDS_DEEP,
    )

    # n floor under locked cover
    for n in (1, 2, 4, 12):
        key = f"n{n}_c1.5"
        deepen[key] = grid_cell(
            f"M21_{key}",
            lambda seed: hold_e_lyric_mix_field3d(seed=seed),
            cover=1.5,
            n=n,
            seeds=SEEDS_DEEP,
        )

    # cover sweep @ n=12
    for cover in (1.0, 1.5, 2.0):
        key = f"n12_c{cover}"
        if key in deepen:
            continue
        deepen[key] = grid_cell(
            f"M21_{key}",
            lambda seed: hold_e_lyric_mix_field3d(seed=seed),
            cover=cover,
            n=12,
            seeds=SEEDS_DEEP,
        )

    # steps / vic
    deepen["steps1600_c1.5_n12"] = grid_cell(
        "M21_steps1600",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed),
        cover=1.5,
        n=12,
        seeds=SEEDS_DEEP,
        steps=1600,
    )
    deepen["vic0_c1.5_n12"] = grid_cell(
        "M21_vic0",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed),
        cover=1.5,
        n=12,
        seeds=SEEDS_DEEP,
        vic=0.0,
    )
    deepen["vic0_music_n1"] = grid_cell(
        "M21_vic0_music",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed),
        cover=1.0,
        n=1,
        seeds=SEEDS_DEEP,
        vic=0.0,
    )

    # e_on_content ablation (geom kwargs — isolation like Fire #20)
    deepen["eoc0_locked"] = grid_cell(
        "M21_eoc0",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed, e_on_content=0.0),
        cover=1.5,
        n=12,
        seeds=SEEDS_DEEP,
    )
    deepen["eoc0_music"] = grid_cell(
        "M21_eoc0",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed, e_on_content=0.0),
        cover=1.0,
        n=1,
        seeds=SEEDS_DEEP,
    )
    # soft content/leak with eoc kept (is eoc the driver?)
    deepen["soft_mix_eoc035"] = grid_cell(
        "M21_soft_mix",
        lambda seed: hold_e_lyric_mix_field3d(
            seed=seed, content=0.55, leak=0.45, e_on_content=0.35
        ),
        cover=1.5,
        n=12,
        seeds=SEEDS_DEEP,
    )
    deepen["hot_mix_eoc0"] = grid_cell(
        "M21_hot_eoc0",
        lambda seed: hold_e_lyric_mix_field3d(
            seed=seed, content=0.85, leak=0.65, e_on_content=0.0
        ),
        cover=1.5,
        n=12,
        seeds=SEEDS_DEEP,
    )

    # False-lock temptation (Fire #4 shape) — expect do-not-adopt even if green
    deepen["false_lock_800_c3"] = grid_cell(
        "M21_false_lock",
        lambda seed: hold_e_lyric_mix_field3d(seed=seed),
        cover=3.0,
        n=12,
        seeds=SEEDS_DEEP,
        steps=800,
    )

    results["m21_deepen"] = deepen

    # recovery analysis
    recoveries = []
    for key, block in deepen.items():
        s = block["summary"]
        if s["no_bite"]:
            recoveries.append(
                {
                    "key": key,
                    "pass": s["pass_str"],
                    "exam_mean": round(s["exam_mean"], 4),
                    "leak_max": round(s["leak_max"], 4),
                    "false_lock_shape": key.startswith("false_lock"),
                }
            )
    portable_ok = [
        r for r in recoveries if not r["false_lock_shape"]
    ]
    results["m21_recovery"] = {
        "any_full_pass": recoveries,
        "portable_recoveries": portable_ok,
        "hard_boundary": len(portable_ok) == 0,
        "false_lock_only": len(portable_ok) == 0 and any(
            r["false_lock_shape"] for r in recoveries
        ),
    }

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
                    "verdict": verdict_of(s),
                }
            )
    for key, block in deepen.items():
        s = block["summary"]
        bite_rows.append(
            {
                "cell": f"M21/{key}",
                "mid": "M21_deep",
                "posture": "probe",
                "pass": s["pass_str"],
                "exam_mean": round(s["exam_mean"], 4),
                "leak_max": round(s["leak_max"], 4),
                "fail_seeds": s["fail_seeds"],
                "verdict": verdict_of(s),
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

    registered = []
    for name, _, mid in NEW_CELLS:
        assert name in CELLS_3D
        registered.append({"cell": name, "mid": mid, "in_CELLS_3D": True})
    results["registered"] = registered

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("wrote", OUT_JSON, flush=True)

    # Markdown
    hb = results["m21_recovery"]["hard_boundary"]
    lines = [
        "# Music→toy batch3 — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {results['wall_s']}s. CPU only. No Music train.",
        "",
        "## New cells (M24–M27; avoided lyric_span / cross_axis / dual_arm / close_live / M20–M23)",
        "",
        "| ID | Cell | Music symptom |",
        "|---|---|---|",
        "| M24 | `content_leak_flip_rows` | û-primary + content↔leak dominance flip mid-caption |",
        "| M25 | `lyric_neu_heavy_gate` | heavy lyric neu vs leftover unused-ê gate |",
        "| M26 | `declare_split_three` | ambiguous declared ê split û/content/unused |",
        "| M27 | `scale_descent_homo` | homo amps + descending scales (traj/outro reverse) |",
        "",
        "## Bite / no-bite table (new cells + leftover CTRL)",
        "",
        "| cell | mid | posture | pass | exam_mean | leak_max | fail_seeds | verdict |",
        "|---|---|---|:---:|---:|---:|---|---|",
    ]
    for r in bite_rows:
        if r["mid"] == "M21_deep":
            continue
        lines.append(
            f"| `{r['cell']}` | {r['mid']} | {r['posture']} | **{r['pass']}** | "
            f"{r['exam_mean']} | {r['leak_max']} | {r['fail_seeds']} | **{r['verdict']}** |"
        )

    lines += [
        "",
        "## M21 deepen (`hold_e_lyric_mix`) — portable knobs",
        "",
        "| probe | pass | exam_mean | leak_max | fail_seeds | verdict |",
        "|---|:---:|---:|---:|---|---|",
    ]
    for r in bite_rows:
        if r["mid"] != "M21_deep":
            continue
        key = r["cell"].replace("M21/", "")
        lines.append(
            f"| `{key}` | **{r['pass']}** | {r['exam_mean']} | {r['leak_max']} | "
            f"{r['fail_seeds']} | **{r['verdict']}** |"
        )

    rec_txt = (
        "NONE — hard boundary (like lyric_span)"
        if hb
        else ", ".join(r["key"] for r in portable_ok)
    )
    lines += [
        "",
        f"**Portable recoveries (non-false-lock):** {rec_txt}",
        "",
        f"**Hard boundary:** **{hb}** — document like lyric_span if True; do not chase recipe.",
        "",
        "## Registry",
        "",
        "Registered in `CELLS_3D` this fire:",
        "",
    ]
    for r in registered:
        lines.append(f"- `{r['cell']}` ({r['mid']})")
    lines += [
        "",
        "## Verdict",
        "",
        "- Recipe change: **NO** (keep FM0, l2=0.02, locked 1200+c1.5).",
        f"- Leftover regression: **{reg_s['pass_str']}** (must stay NO_BITE).",
        f"- M21 hard boundary: **{hb}**.",
        "- False-lock 800×c3.0: document only; never adopt.",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]

    # Interpretation stubs filled from bite table
    lines += ["## Interpretation", ""]
    for name, block in geom.items():
        mid = block["mid"]
        lk = block["locked"]["summary"]
        mu = block["music"]["summary"]
        lines.append(
            f"- **{mid} `{name}`**: locked {verdict_of(lk)} {lk['pass_str']}; "
            f"music {verdict_of(mu)} {mu['pass_str']} "
            f"(exam_locked={lk['exam_mean']:.4f}, leak_max_music={mu['leak_max']:.4f})"
        )
    lines.append(
        f"- **M21 deepen**: hard_boundary={hb}; portable={ [r['key'] for r in portable_ok] }"
    )
    lines.append(
        f"- **leftover CTRL**: {reg_s['pass_str']} exam={reg_s['exam_mean']:.4f}"
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_MD, flush=True)

    # research_log append
    wins = [
        f"{r['cell']}/{r['posture']}: {r['pass']} {r['verdict']}"
        for r in bite_rows
        if r["verdict"] not in ("NO_BITE",) and r["mid"] != "M21_deep"
    ]
    m21_bites = [
        f"{r['cell']}: {r['pass']} {r['verdict']}"
        for r in bite_rows
        if r["mid"] == "M21_deep" and r["verdict"] != "NO_BITE"
    ]
    entry = f"""
## Fire — Music→toy batch3 (2026-09-09)

- Host: box-cpu @ SHA `{sha}`
- Notes: `music_to_toy_batch3_20260909.{{py,json,md}}` wall={results['wall_s']}s
- New CELLS_3D: content_leak_flip_rows (M24), lyric_neu_heavy_gate (M25), declare_split_three (M26), scale_descent_homo (M27)
- Geom bites: {wins}
- M21 deepen hard_boundary={hb}; portable_recoveries={[r['key'] for r in portable_ok]}; still_biting_probes={m21_bites[:8]}
- leftover_regression={reg_s['pass_str']}; recipe_change=NO; ping_user=YES
- No Music GPU train; servers untouched.
"""
    if LOG.exists():
        cur = LOG.read_text()
        if "Music→toy batch3" not in cur:
            LOG.write_text(cur.rstrip() + "\n" + entry)
            print("appended research_log", flush=True)
    else:
        LOG.write_text("# research log\n" + entry)

    # Catalog: ensure M20–M23 present; append Batch3
    if CATALOG.exists():
        cat = CATALOG.read_text()
        if "M20" not in cat or "amp_lie_leftover_declare" not in cat:
            cat += """

## Batch2 new stressors (2026-09-09)

| ID | Music symptom | Cell | Notes |
|---|---|---|---|
| M20 | Declared leak YAML / content-axis amplitude lie (leftover geom) | `amp_lie_leftover_declare` | Distinct from e_on_u_declare_lie / cross_axis_mismatch_declare |
| M21 | Content↔leftover mix ≈ hold-ê lyric pool | `hold_e_lyric_mix` | No hetero row_amps; HARD_BITE both postures |
| M22 | Mild multipair (between homo and full cross_axis) | `stagger_mild_cross` | Soft axis wander + stagger; music knife |
| M23 | Multi-pair R³ correlated seeds | `multipair_corr_seed` | Seed-tied amp correlation; NO_BITE |

See `music_to_toy_new_stressors_batch2_20260909.md` bite table.
"""
        if "Batch3 new stressors" not in cat:
            cat += f"""

## Batch3 new stressors (2026-09-09)

| ID | Music symptom | Cell | Notes |
|---|---|---|---|
| M24 | Mid-caption content↔leak attribute flip (û stays primary) | `content_leak_flip_rows` | Not roles_split / cross_axis / stagger_mild |
| M25 | Lyric-token neu heavy vs leftover unused-ê gate | `lyric_neu_heavy_gate` | Not prefix_shared / hold_e_lyric_mix |
| M26 | Ambiguous declared ê split û/content/unused | `declare_split_three` | Not hard amp lie M14/M20 |
| M27 | Descending span scales (traj/outro reverse) | `scale_descent_homo` | Mirror of M16 ascending |
| M21d | hold_e_lyric_mix portable-knob deepen | probes | hard_boundary={hb}; see batch3 md |

See `music_to_toy_batch3_20260909.md` bite table.
"""
            CATALOG.write_text(cat)
            print("appended catalog batch3", flush=True)
        else:
            CATALOG.write_text(cat)

    print("DONE wall=", results["wall_s"], flush=True)
    print("\n=== BITE TABLE ===", flush=True)
    for r in bite_rows:
        if r["mid"] == "M21_deep":
            continue
        print(f"  {r['verdict']:10} {r['pass']:5} {r['cell']}/{r['posture']}", flush=True)
    print("\n=== M21 DEEPEN ===", flush=True)
    for r in bite_rows:
        if r["mid"] != "M21_deep":
            continue
        print(f"  {r['verdict']:10} {r['pass']:5} {r['cell']}", flush=True)
    print(
        f"\nM21 hard_boundary={hb} portable={[r['key'] for r in portable_ok]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
