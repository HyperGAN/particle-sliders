#!/usr/bin/env python3
"""Music→toy batch4 — 2026-09-09.

New cells (avoid M1–M27 duplicates), driven by M21 mechanistic dig + Music gaps:

  M28 guard_refuse_hot_eoc     — eoc in blend-guard refuse band → raw-pole leak
  M29 content_cascade_rows     — û-stable ascending content (verse→chorus)
  M30 eoc_threshold_edge       — M21 pool, eoc=0.33 just over ê-floor
  M31 leftover_hot_eoc_declare — leftover amps + admitting hot eoc (ê-floor)

Also: leftover CTRL; M28 vs M21 contrast (refuse vs floor); M30 edge smoke.

Locked recipe unchanged. No Music GPU train. merge=NO for any NON_DEFAULT.
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
    content_cascade_rows_field3d,
    eoc_threshold_edge_field3d,
    field3d_teacher_points,
    guard_refuse_hot_eoc_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    leftover_hot_eoc_declare_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402
from conceptmod.textsliders.slider_targets import (  # noqa: E402
    lm_blend_guard,
    lm_faithful_sub_e,
)

NOTES = Path(__file__).resolve().parent
OUT_JSON = NOTES / "music_to_toy_batch4_20260909.json"
OUT_MD = NOTES / "music_to_toy_batch4_20260909.md"
LOG = NOTES / "research_log_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"

NEW_CELLS = [
    ("guard_refuse_hot_eoc", guard_refuse_hot_eoc_field3d, "M28"),
    ("content_cascade_rows", content_cascade_rows_field3d, "M29"),
    ("eoc_threshold_edge", eoc_threshold_edge_field3d, "M30"),
    ("leftover_hot_eoc_declare", leftover_hot_eoc_declare_field3d, "M31"),
]


def make_cfg(seed, *, cover, n_particles, steps=1200, vicreg_weight=0.05):
    return default_cfg(
        steps=steps,
        seed=seed,
        b_cap=1.0,
        cover_weight=cover,
        fm_weight=0.0,
        n_particles=n_particles,
        particle_l2=0.02,
        vicreg_weight=vicreg_weight,
    )


def run_one(field, *, seed, cover, n, name, vic=0.05) -> dict:
    t0 = time.time()
    cfg = make_cfg(seed, cover=cover, n_particles=n, vicreg_weight=vic)
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
        "vicreg_weight": vic,
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
        "leak_max": max(leaks) if leaks else 0.0,
        "fail_seeds": fails,
        "knife": 0 < n_pass < n,
        "bite": n_pass < n,
        "hard_bite": n_pass == 0 and n > 0,
        "no_bite": n_pass == n and n > 0,
    }


def verdict_of(s: dict) -> str:
    if s["hard_bite"]:
        return "HARD_BITE"
    if s["knife"]:
        return "KNIFE"
    if s["no_bite"]:
        return "NO_BITE"
    return "UNKNOWN"


def grid_cell(label, make_field, *, cover, n, seeds=SEEDS, **kw) -> dict:
    rows = []
    print(f"=== {label} c={cover} n={n} ===", flush=True)
    for seed in seeds:
        try:
            field = make_field(seed)
        except TypeError:
            field = make_field()
        r = run_one(
            field, seed=seed, cover=cover, n=n, name=f"{label}_c{cover}_n{n}_s{seed}", **kw
        )
        rows.append(r)
        print(
            f"  seed={seed} pass={r['pass']} exam={r['exam_score']:.4f} "
            f"u={r['u_kept']:.4f} leak={r['leak_ratio']:.4f} "
            f"rows={r['rows_covered']}/{r['rows_total']} ({r['wall_s']}s)",
            flush=True,
        )
    return {"summary": summarize(rows), "rows": rows}


def guard_status(field) -> dict:
    pos, neg, neu = field.poles(0)
    ld = field.declared_e()
    tp, tm = lm_faithful_sub_e(pos, neg, neu, ld, slider_dir=field.short_u())
    g = lm_blend_guard(tp, tm, pos, neg)
    tgp, _ = field3d_teacher_points(field, 0, teacher=TEACHER, leak_dir=ld)
    d = tgp - neu
    on_e = float(d @ field.leak_e())
    on_u = float(d @ field.short_u())
    lr = abs(on_e) / (abs(on_u) + 1e-8)
    return {
        "admissible": bool(g["admissible"]),
        "to_pole": round(float(g["to_pole"]), 4),
        "to_mid": round(float(g["to_mid"]), 4),
        "teacher_lr": round(lr, 4),
        "e_on_content": float(field.e_on_content),
    }


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
            "batch": 4,
            "ids": ["M28", "M29", "M30", "M31"],
        }
    }

    # Guard contrast (no train): M21 admits floor vs M28 refuse
    print("=== guard contrast M21 vs M28/M30/M31 ===", flush=True)
    contrast = {
        "M21": guard_status(hold_e_lyric_mix_field3d(seed=0)),
        "M28": guard_status(guard_refuse_hot_eoc_field3d(seed=0)),
        "M30": guard_status(eoc_threshold_edge_field3d(seed=0)),
        "M31": guard_status(leftover_hot_eoc_declare_field3d(seed=0)),
    }
    for k, v in contrast.items():
        print(f"  {k}: {v}", flush=True)
    results["guard_contrast"] = contrast

    geom = {}
    for name, ctor, mid in NEW_CELLS:

        def _make(seed, _ctor=ctor):
            try:
                return _ctor(seed=seed)
            except TypeError:
                return _ctor()

        locked = grid_cell(f"{mid}_{name}", _make, cover=1.5, n=12)
        music = grid_cell(f"{mid}_{name}", _make, cover=1.0, n=1)
        geom[name] = {"mid": mid, "locked": locked, "music": music}
    results["geom"] = geom

    # leftover CTRL
    print("=== leftover regression ===", flush=True)
    reg_rows = []
    for seed in SEEDS_SMOKE:
        r = run_one(
            leftover_field3d(),
            seed=seed,
            cover=1.5,
            n=12,
            name=f"leftover_reg_s{seed}",
        )
        reg_rows.append(r)
        print(f"  leftover seed={seed} pass={r['pass']} exam={r['exam_score']:.4f}", flush=True)
    results["leftover_regression"] = {"summary": summarize(reg_rows), "rows": reg_rows}

    # Music-posture vic0 smoke on M28/M30 (must still fail — no false fix)
    print("=== vic0@n1 hard-bite smoke (M28/M30) ===", flush=True)
    vic_smoke = {}
    for name, ctor, mid in NEW_CELLS:
        if mid not in ("M28", "M30"):
            continue

        def _make(seed, _ctor=ctor):
            try:
                return _ctor(seed=seed)
            except TypeError:
                return _ctor()

        vic_smoke[name] = grid_cell(
            f"{mid}_{name}_vic0",
            _make,
            cover=1.0,
            n=1,
            seeds=SEEDS_SMOKE,
            vic=0.0,
        )
    results["vic0_n1_smoke"] = vic_smoke

    wall = round(time.time() - t_all, 1)
    results["wall_s"] = wall
    results["recipe_change"] = False

    # Wins / bites table
    bites, knives, no_bites = [], [], []
    for name, pack in geom.items():
        mid = pack["mid"]
        for posture in ("locked", "music"):
            s = pack[posture]["summary"]
            v = verdict_of(s)
            line = f"{mid}_{name}/{posture}: {s['pass_str']} {v}"
            if v == "HARD_BITE":
                bites.append(line)
            elif v == "KNIFE":
                knives.append(line)
            else:
                no_bites.append(line)
    results["bites"] = bites
    results["knives"] = knives
    results["no_bites"] = no_bites
    leftover_ok = results["leftover_regression"]["summary"]["no_bite"]
    results["leftover_ok"] = leftover_ok

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str) + "\n")

    # MD
    lines = [
        "# Music→toy batch4 — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. No Music train.",
        "",
        "## New cells (M28–M31)",
        "",
        "| ID | Cell | Music / mech symptom |",
        "|---|---|---|",
        "| M28 | `guard_refuse_hot_eoc` | hot eoc → blend guard refuses → raw-pole leak (M21 dig refuse band) |",
        "| M29 | `content_cascade_rows` | û-stable ascending content (verse→chorus) |",
        "| M30 | `eoc_threshold_edge` | M21 pool, eoc=0.33 just over analytic ê-floor |",
        "| M31 | `leftover_hot_eoc_declare` | leftover amps + admitting hot eoc (ê-floor, not YAML blowup) |",
        "",
        "## Guard contrast (teacher, no train)",
        "",
        "| cell | admissible | teacher_lr | eoc |",
        "|---|:---:|---:|---:|",
    ]
    for k, v in contrast.items():
        lines.append(
            f"| {k} | {v['admissible']} | {v['teacher_lr']:.4f} | {v['e_on_content']} |"
        )
    lines += [
        "",
        "## Bite / no-bite table",
        "",
        "| cell | mid | posture | pass | exam_mean | leak_max | fail_seeds | verdict |",
        "|---|---|---|:---:|---:|---:|---|---|",
    ]
    for name, pack in geom.items():
        mid = pack["mid"]
        for posture in ("locked", "music"):
            s = pack[posture]["summary"]
            lines.append(
                f"| `{name}` | {mid} | {posture} | **{s['pass_str']}** | "
                f"{s['exam_mean']:.4f} | {s['leak_max']:.4f} | {s['fail_seeds']} | "
                f"**{verdict_of(s)}** |"
            )
    sreg = results["leftover_regression"]["summary"]
    lines.append(
        f"| `leftover_regression` | CTRL | locked | **{sreg['pass_str']}** | "
        f"{sreg['exam_mean']:.4f} | {sreg['leak_max']:.4f} | {sreg['fail_seeds']} | "
        f"**{verdict_of(sreg)}** |"
    )
    lines += [
        "",
        "## vic0@n1 smoke (must not false-fix hard bites)",
        "",
    ]
    for name, pack in vic_smoke.items():
        s = pack["summary"]
        lines.append(f"- `{name}` vic0 n1: {s['pass_str']} leak_max={s['leak_max']:.4f} ({verdict_of(s)})")
    lines += [
        "",
        "## Registry",
        "",
        "Registered in `CELLS_3D` this fire:",
        "",
    ]
    for name, _, mid in NEW_CELLS:
        lines.append(f"- `{name}` ({mid})")
    lines += [
        "",
        "## Verdict",
        "",
        f"- Recipe change: **NO**",
        f"- Leftover regression: **{sreg['pass_str']}** (must stay NO_BITE)",
        f"- HARD_BITEs: {bites}",
        f"- KNIVEs: {knives}",
        f"- NO_BITEs: {no_bites}",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # research_log
    log_block = f"""
## Fire — Music→toy batch4 (2026-09-09)

- Host: box-cpu @ SHA `{sha}`
- Notes: `music_to_toy_batch4_20260909.{{py,json,md}}` wall={wall}s
- New CELLS_3D: {[n for n,_,_ in NEW_CELLS]}
- Guard contrast: M21 admit={contrast['M21']['admissible']} lr={contrast['M21']['teacher_lr']}; M28 admit={contrast['M28']['admissible']} lr={contrast['M28']['teacher_lr']}
- Bites: {bites}
- Knives: {knives}
- No-bites: {no_bites}
- leftover_regression={sreg['pass_str']}; recipe_change=NO; ping_user=YES
- No Music GPU train; servers untouched.
"""
    with LOG.open("a") as fh:
        fh.write(log_block)

    # catalog append
    cat_block = f"""

## Batch4 M28–M31 (2026-09-09) — RESULTS

| ID | Cell | locked | music | notes |
|---|---|---|---|---|
"""
    for name, pack in geom.items():
        mid = pack["mid"]
        sl = pack["locked"]["summary"]
        sm = pack["music"]["summary"]
        cat_block += (
            f"| {mid} | `{name}` | **{sl['pass_str']}** {verdict_of(sl)} | "
            f"**{sm['pass_str']}** {verdict_of(sm)} | "
            f"exam_l={sl['exam_mean']:.3f} leak_m={sm['leak_max']:.3f} |\n"
        )
    cat_block += f"| CTRL | leftover | **{sreg['pass_str']}** | — | must stay green |\n"
    cat_block += (
        "\nFrom M21 mech dig: M28 = guard-refuse band; M30 = ê-floor edge; "
        "M31 = admitting hot eoc on leftover amps. Recipe change=NO.\n"
    )
    with CATALOG.open("a") as fh:
        fh.write(cat_block)

    # scoreboard
    sb_block = f"""

## Batch4 M28–M31 (2026-09-09)

Wall={wall}s. New cells from M21 mech + Music gaps. Recipe change=NO.

| ID | locked | music | note |
|---|---|---|---|
"""
    for name, pack in geom.items():
        mid = pack["mid"]
        sl = pack["locked"]["summary"]
        sm = pack["music"]["summary"]
        sb_block += f"| {mid} `{name}` | {sl['pass_str']} {verdict_of(sl)} | {sm['pass_str']} {verdict_of(sm)} | |\n"
    sb_block += f"| CTRL leftover | {sreg['pass_str']} | — | must PASS |\n"
    with SCOREBOARD.open("a") as fh:
        fh.write(sb_block)

    # registry sanity
    for name, _, mid in NEW_CELLS:
        assert name in CELLS_3D, name

    print(f"\nDONE wall={wall}s leftover_ok={leftover_ok}", flush=True)
    print(f"bites={bites}", flush=True)
    print(f"knives={knives}", flush=True)
    print(f"no_bites={no_bites}", flush=True)


if __name__ == "__main__":
    main()
