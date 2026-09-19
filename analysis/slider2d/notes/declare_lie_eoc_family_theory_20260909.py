#!/usr/bin/env python3
"""Teacher / eoc-floor / declare-lie family — unified theory + cliffs — 2026-09-09.

Unifies three fail branches under faithful_guard_e × declared_ê:
  A) content_deleted_under_declare_lie — M14 / M20 (+ M26 sibling)
  B) ê-floor admit mode — M21 / M30 (+ M31 leftover-hot admit)
  C) guard-refuse / raw-pole — M28

Runs missing cliffs (teacher + short emp smokes). Reuses prior dig JSONs for
locked characterization. CPU only. No Music GPU. Locked recipe UNCHANGED.

Outputs: declare_lie_eoc_family_theory_20260909.{md,json}
         + scoreboard / catalog / research_log append.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))
NOTES = _REPO / "analysis/slider2d/notes"

# Reuse M21 analytic + batch4 guard_status
_m21_spec = importlib.util.spec_from_file_location(
    "m21_eoc", NOTES / "m21_eoc_leak_irreducible_20260909.py"
)
m21 = importlib.util.module_from_spec(_m21_spec)
assert _m21_spec.loader is not None
sys.modules["m21_eoc"] = m21
_m21_spec.loader.exec_module(m21)

_b4_spec = importlib.util.spec_from_file_location(
    "batch4", NOTES / "music_to_toy_batch4_20260909.py"
)
b4 = importlib.util.module_from_spec(_b4_spec)
assert _b4_spec.loader is not None
sys.modules["batch4"] = b4
_b4_spec.loader.exec_module(b4)

from analysis.slider2d.field3d import (  # noqa: E402
    amp_lie_leftover_declare_field3d,
    declare_split_three_field3d,
    e_on_u_declare_lie_field3d,
    eoc_threshold_edge_field3d,
    guard_refuse_hot_eoc_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    leftover_hot_eoc_declare_field3d,
    score_adv_field3d_exam,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

OUT_JSON = NOTES / "declare_lie_eoc_family_theory_20260909.json"
OUT_MD = NOTES / "declare_lie_eoc_family_theory_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"
CATALOG = NOTES / "music_to_toy_stressor_catalog_20260909.md"

SEEDS_SMOKE = [0, 1, 2]
LEAK_GATE = 0.20


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_json(name: str) -> dict | None:
    p = NOTES / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def shared_smoke(name, field_fn, teacher, seeds, *, eoc=None, eou=None, n=12, cover=1.5):
    runs = []
    for s in seeds:
        cfg = default_cfg(
            steps=1200, seed=s, b_cap=1.0, cover_weight=cover,
            fm_weight=0.0, n_particles=n, particle_l2=0.02,
        )
        kw = {}
        if eoc is not None:
            kw["e_on_content"] = float(eoc)
        if eou is not None:
            kw["e_on_u"] = float(eou)
            if eou == 0.0:
                kw["e_on_content"] = 0.0
                kw["e_unused"] = 1.0
        try:
            field = field_fn(seed=s, **kw) if kw else field_fn(seed=s)
        except TypeError:
            field = field_fn(**kw) if kw else field_fn()
        t0 = time.time()
        row = score_adv_field3d_exam(field, teacher=teacher, cfg=cfg, name=f"{name}_s{s}")
        runs.append({
            "seed": s,
            "pass": bool(row.get("exam_pass")),
            "exam_score": float(row["exam_score"]),
            "u_kept": float(row["u_kept"]),
            "content_kept": float(row["content_kept"]),
            "leak_ratio": float(row["leak_ratio"]),
            "pass_cont": bool(row.get("pass_cont")),
            "pass_leftover_gate": bool(row.get("pass_leftover_gate")),
            "pass_leak": bool(row.get("pass_leak")),
            "wall_s": round(time.time() - t0, 2),
        })
        print(
            f"    {name} s={s} pass={runs[-1]['pass']} exam={runs[-1]['exam_score']:.4f} "
            f"cont={runs[-1]['content_kept']:.4f} leak={runs[-1]['leak_ratio']:.4f} "
            f"({runs[-1]['wall_s']}s)",
            flush=True,
        )
    nn = len(runs)
    mean = lambda k: round(sum(float(r[k]) for r in runs) / nn, 4) if nn else 0.0
    return {
        "name": name, "teacher": teacher, "n": nn,
        "pass": f"{sum(1 for r in runs if r['pass'])}/{nn}",
        "n_pass": sum(1 for r in runs if r["pass"]),
        "mean_exam": mean("exam_score"),
        "mean_u": mean("u_kept"),
        "mean_content": mean("content_kept"),
        "mean_leak": mean("leak_ratio"),
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "runs": runs,
    }


def m21_pool_guard_cliff() -> list[dict]:
    """Fine eoc cliff on M21 content/leak pool: admit ê-floor → refuse raw-pole."""
    rows = []
    eocs = [
        0.0, 0.20, 0.30, 0.32, 0.33, 0.35, 0.40, 0.50,
        0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85, 1.0,
    ]
    for eoc in eocs:
        f = hold_e_lyric_mix_field3d(seed=0, e_on_content=eoc)
        g = b4.guard_status(f)
        ana = m21.analytic_teacher_residual(e_on_content=eoc)
        mode = (
            "pass_clean" if eoc < 0.33 and g["admissible"] and ana["pass_leak"]
            else "admit_e_floor" if g["admissible"] and not ana["pass_leak"]
            else "admit_near_floor" if g["admissible"]
            else "refuse_raw_pole"
        )
        # When analytic pass_leak but emp would still be near gate — classify admit_near
        if g["admissible"] and ana["leak_ratio"] > LEAK_GATE:
            mode = "admit_e_floor"
        elif g["admissible"] and ana["leak_ratio"] <= LEAK_GATE and eoc >= 0.30:
            mode = "admit_near_floor"
        elif g["admissible"] and ana["leak_ratio"] <= LEAK_GATE:
            mode = "pass_clean"
        elif not g["admissible"]:
            mode = "refuse_raw_pole"
        row = {
            "eoc": eoc,
            "admissible": g["admissible"],
            "teacher_lr": g["teacher_lr"],
            "to_pole": g["to_pole"],
            "to_mid": g["to_mid"],
            "analytic_lr": ana["leak_ratio"],
            "analytic_pass_leak": ana["pass_leak"],
            "analytic_content_kept": ana["content_kept"],
            "mode": mode,
            "cell_anchor": (
                "M30" if abs(eoc - 0.33) < 1e-9
                else "M21" if abs(eoc - 0.35) < 1e-9
                else "M28" if abs(eoc - 0.85) < 1e-9
                else None
            ),
        }
        rows.append(row)
        print(
            f"    eoc={eoc:.2f} admit={g['admissible']} teach_lr={g['teacher_lr']:.4f} "
            f"ana_lr={ana['leak_ratio']:.4f} mode={mode}",
            flush=True,
        )
    return rows


def m31_leftover_eoc_cliff() -> list[dict]:
    """Leftover amps (0.55/0.45) × eoc — M31 admit-hot vs M20 YAML blowup contrast."""
    rows = []
    for eoc, e_unused in [
        (0.0, 1.0),
        (0.35, 0.85),
        (0.55, 0.70),
        (0.85, 0.55),
        (1.15, 0.45),  # M31
        (1.25, 0.15),  # M20-like YAML blowup
    ]:
        f = leftover_hot_eoc_declare_field3d(
            seed=0, e_on_content=eoc, e_unused=e_unused, e_on_u=0.05 if eoc > 0 else 0.0,
        )
        g = b4.guard_status(f)
        ana = m21.analytic_teacher_residual(
            content=0.55, leak=0.45, e_on_u=0.05 if eoc > 0 else 0.0,
            e_on_content=eoc, e_unused=e_unused,
        )
        rows.append({
            "eoc": eoc, "e_unused": e_unused,
            "admissible": g["admissible"],
            "teacher_lr": g["teacher_lr"],
            "analytic_lr": ana["leak_ratio"],
            "analytic_content_kept": ana["content_kept"],
            "analytic_pass_leak": ana["pass_leak"],
            "anchor": (
                "M31" if abs(eoc - 1.15) < 1e-9
                else "M20ish" if abs(eoc - 1.25) < 1e-9 and abs(e_unused - 0.15) < 1e-9
                else None
            ),
        })
        print(
            f"    leftover eoc={eoc} eu={e_unused} admit={g['admissible']} "
            f"teach_lr={g['teacher_lr']:.4f} ana_lr={ana['leak_ratio']:.4f}",
            flush=True,
        )
    return rows


def find_cliffs(guard_rows: list[dict]) -> dict:
    """Extract cliff points from M21-pool guard cliff."""
    floor_on = None  # first eoc where analytic_lr > 0.20 and admit
    refuse_on = None  # first eoc where not admissible
    for r in guard_rows:
        if floor_on is None and r["admissible"] and r["analytic_lr"] > LEAK_GATE:
            floor_on = r["eoc"]
        if refuse_on is None and not r["admissible"]:
            refuse_on = r["eoc"]
    # last pass_clean / admit_near before floor
    last_pass = None
    for r in guard_rows:
        if r["mode"] in ("pass_clean", "admit_near_floor") and r["analytic_pass_leak"]:
            last_pass = r["eoc"]
    return {
        "e_floor_on_eoc": floor_on,
        "last_analytic_pass_eoc": last_pass,
        "refuse_on_eoc": refuse_on,
        "m30_edge_eoc": 0.33,
        "m21_default_eoc": 0.35,
        "m28_default_eoc": 0.85,
        "analytic_floor_m21": 0.20503,
    }


def main():
    t0 = time.time()
    sha = git_sha()
    print(f"=== declare-lie / eoc-floor family theory @ {sha} ===", flush=True)

    prior = {
        "m14_failmode": load_json("m14_failmode_20260909.json"),
        "m21": load_json("m21_eoc_leak_irreducible_20260909.json"),
        "batch4": load_json("music_to_toy_batch4_20260909.json"),
        "m30_knife": load_json("m30_per_row_knife_20260909.json"),
        "falsify_b4": load_json("per_row_falsify_batch4_20260909.json"),
        "declare_lie_family": load_json("declare_lie_family_20260909.json"),
    }

    # --- missing cliffs ---
    print("\n[A] M21-pool admit→refuse guard cliff (teacher)", flush=True)
    guard_cliff = m21_pool_guard_cliff()
    cliffs = find_cliffs(guard_cliff)

    print("\n[B] M31 leftover-amp eoc cliff (teacher)", flush=True)
    m31_cliff = m31_leftover_eoc_cliff()

    print("\n[C] emp smoke at cliff points (shared locked)", flush=True)
    emp = []
    # CTRL
    emp.append(shared_smoke("CTRL_leftover", leftover_field3d, "faithful_guard_e", SEEDS_SMOKE))
    # Branch A locked
    emp.append(shared_smoke("M14_locked", e_on_u_declare_lie_field3d, "faithful_guard_e", SEEDS_SMOKE))
    emp.append(shared_smoke("M20_locked", amp_lie_leftover_declare_field3d, "faithful_guard_e", SEEDS_SMOKE))
    # Branch B/C cliff eocs on M21 pool
    for eoc, tag in [
        (0.32, "pre_floor"),
        (0.33, "M30_edge"),
        (0.35, "M21_default"),
        (cliffs["refuse_on_eoc"] or 0.70, "refuse_onset"),
        (0.85, "M28_hot"),
    ]:
        emp.append(shared_smoke(
            f"M21pool_eoc{eoc}_{tag}",
            hold_e_lyric_mix_field3d, "faithful_guard_e", SEEDS_SMOKE, eoc=eoc,
        ))
    # M31 locked
    emp.append(shared_smoke(
        "M31_locked", leftover_hot_eoc_declare_field3d, "faithful_guard_e", SEEDS_SMOKE,
    ))
    # M14 eou fine cliff emp (fill 0.15 gap if declare_lie_family not done)
    for eou in (0.0, 0.15, 0.30):
        emp.append(shared_smoke(
            f"M14_eou{eou}",
            e_on_u_declare_lie_field3d, "faithful_guard_e", SEEDS_SMOKE, eou=eou,
        ))

    # Member map from prior + this emp
    by_emp = {c["name"]: c for c in emp}
    b4g = (prior["batch4"] or {}).get("guard_contrast") or {}
    m14c = None
    m20c = None
    if prior["m14_failmode"]:
        for c in prior["m14_failmode"].get("cells", []):
            if c.get("name") == "M14_locked_n12":
                m14c = c
            if c.get("name") == "M20_locked_n12":
                m20c = c

    members = {
        "M14": {
            "cell": "e_on_u_declare_lie",
            "branch": "content_deleted_under_declare_lie",
            "mode": "YAML û-restatement × guard",
            "locked_pass": (m14c or by_emp["M14_locked"])["pass"] if m14c else by_emp["M14_locked"]["pass"],
            "exam": (m14c or by_emp["M14_locked"]).get("mean_exam", by_emp["M14_locked"]["mean_exam"]),
            "content": (m14c or by_emp["M14_locked"]).get("mean_content", by_emp["M14_locked"]["mean_content"]),
            "leak": (m14c or by_emp["M14_locked"]).get("mean_leak", by_emp["M14_locked"]["mean_leak"]),
            "admit": True,
            "per_row_clears": False,
            "cliff": "e_on_u=0 PASS → e_on_u≥0.3 BITES",
        },
        "M20": {
            "cell": "amp_lie_leftover_declare",
            "branch": "content_deleted_under_declare_lie",
            "mode": "YAML content-axis amp lie × guard",
            "locked_pass": (m20c or by_emp["M20_locked"])["pass"] if m20c else by_emp["M20_locked"]["pass"],
            "exam": (m20c or by_emp["M20_locked"]).get("mean_exam", by_emp["M20_locked"]["mean_exam"]),
            "content": (m20c or by_emp["M20_locked"]).get("mean_content", by_emp["M20_locked"]["mean_content"]),
            "leak": (m20c or by_emp["M20_locked"]).get("mean_leak", by_emp["M20_locked"]["mean_leak"]),
            "admit": True,
            "per_row_clears": False,
            "cliff": "faithful (no guard) PASS; guard × eoc hot → content delete",
        },
        "M21": {
            "cell": "hold_e_lyric_mix",
            "branch": "e_floor_admit",
            "mode": "declared_e tilt; guard admits; exam pure-ê floor",
            "locked_pass": "0/6",
            "exam": None,
            "content": None,
            "leak": 0.2106,
            "admit": True,
            "per_row_clears": False,
            "cliff": f"eoc≤{cliffs['last_analytic_pass_eoc']} pass → eoc≥{cliffs['e_floor_on_eoc']} ê-floor (lr≥0.205)",
            "analytic_floor": 0.20503,
        },
        "M30": {
            "cell": "eoc_threshold_edge",
            "branch": "e_floor_admit",
            "mode": "M21 pool eoc=0.33 just over floor",
            "locked_pass": "0/6",
            "leak": 0.2065,
            "admit": True,
            "per_row_clears": False,
            "per_row_knife": "1/6 seed1",
            "cliff": "analytic eoc 0.32 pass / 0.33 fail",
        },
        "M28": {
            "cell": "guard_refuse_hot_eoc",
            "branch": "guard_refuse_raw_pole",
            "mode": "hot eoc → blend guard refuses → raw poles",
            "locked_pass": "0/6",
            "leak": 0.66,
            "admit": False,
            "per_row_clears": False,
            "cliff": f"refuse onset eoc≥{cliffs['refuse_on_eoc']} (M28 default 0.85)",
        },
        "M31": {
            "cell": "leftover_hot_eoc_declare",
            "branch": "e_floor_admit",
            "mode": "leftover amps + admitting hot eoc (ê-floor + exam collapse)",
            "locked_pass": by_emp["M31_locked"]["pass"],
            "exam": by_emp["M31_locked"]["mean_exam"],
            "leak": by_emp["M31_locked"]["mean_leak"],
            "admit": True,
            "per_row_clears": False,
            "cliff": "leftover×eoc=1.15 admits; lr≈0.20+; ≠ M20 YAML blowup",
        },
    }
    # optional M26 from declare_lie_family if present
    if prior["declare_lie_family"] and "natures" in prior["declare_lie_family"]:
        members["M26"] = {
            "cell": "declare_split_three",
            "branch": "content_deleted_under_declare_lie",
            "mode": "ambiguous ê split û/content/unused",
            "locked_pass": "0/6",
            "nature": prior["declare_lie_family"]["natures"].get("M26_locked"),
            "admit": True,
            "per_row_clears": False,
            "cliff": "sibling of M14/M20; not DoF",
        }
    else:
        # smoke M26 nature via one shared if family not done
        m26 = shared_smoke("M26_locked", declare_split_three_field3d, "faithful_guard_e", [0, 1])
        emp.append(m26)
        members["M26"] = {
            "cell": "declare_split_three",
            "branch": "content_deleted_under_declare_lie",
            "mode": "ambiguous ê split û/content/unused",
            "locked_pass": m26["pass"],
            "exam": m26["mean_exam"],
            "content": m26["mean_content"],
            "leak": m26["mean_leak"],
            "admit": True,
            "per_row_clears": False,
            "cliff": "sibling of M14/M20; not DoF",
        }

    wall = round(time.time() - t0, 1)

    # suite checks
    ctrl_ok = by_emp["CTRL_leftover"]["n_pass"] == by_emp["CTRL_leftover"]["n"]
    m14_bites = by_emp["M14_locked"]["n_pass"] == 0
    m20_bites = by_emp["M20_locked"]["n_pass"] == 0
    m31_bites = by_emp["M31_locked"]["n_pass"] == 0
    eou0_pass = by_emp["M14_eou0.0"]["n_pass"] == by_emp["M14_eou0.0"]["n"]
    eou03_bites = by_emp["M14_eou0.3"]["n_pass"] == 0
    floor_edge_bites = by_emp.get("M21pool_eoc0.33_M30_edge", {}).get("n_pass", 1) == 0
    refuse_bites = by_emp.get("M21pool_eoc0.85_M28_hot", {}).get("n_pass", 1) == 0
    suite_pass = (
        ctrl_ok and m14_bites and m20_bites and m31_bites
        and eou0_pass and eou03_bites and floor_edge_bites and refuse_bites
        and cliffs["e_floor_on_eoc"] is not None
        and cliffs["refuse_on_eoc"] is not None
    )

    verdict = (
        f"family map: A=content_deleted(M14/M20/M26) B=ê-floor-admit(M21/M30/M31) "
        f"C=refuse-raw(M28); cliffs eou 0→≥0.3; eoc floor_on={cliffs['e_floor_on_eoc']} "
        f"refuse_on={cliffs['refuse_on_eoc']}; suite_pass={suite_pass}; "
        f"n/vic/per_row clear? NO; recipe_change=NO; merge=NO"
    )

    payload = {
        "wall_s": wall, "sha": sha, "recipe_change": False, "merge_to_trainer": False,
        "cliffs": cliffs, "guard_cliff_m21_pool": guard_cliff, "m31_leftover_cliff": m31_cliff,
        "members": members, "emp_smoke": emp, "suite_pass": suite_pass, "verdict": verdict,
        "prior_sources": [k for k, v in prior.items() if v is not None],
        "ctrl_ok": ctrl_ok,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # --- one-page family map ---
    md = f"""# Declare-lie / eoc-floor family — unified theory (2026-09-09)

Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. **No Music GPU.**
Locked shared AdvResidual **unchanged**. per-row = analysis-only. merge=**NO**.

## One-page family map

```
faithful_guard_e  ×  declared_ê  (YAML / e_on_* tilt)
        │
        ├─► A  content_deleted_under_declare_lie
        │      M14 e_on_u_declare_lie   (ê ∥ û)
        │      M20 amp_lie_leftover     (ê ∥ content)
        │      M26 declare_split_three  (ê split)
        │      cliff: e_on_u=0 PASS → ≥0.3 BITES; faithful-no-guard PASS
        │      fix: YAML / declared target — NOT n/vic/per-row
        │
        ├─► B  ê-floor (admit mode)
        │      M21 hold_e_lyric_mix     eoc=0.35  lr_floor=0.20503
        │      M30 eoc_threshold_edge   eoc=0.33  (analytic edge)
        │      M31 leftover_hot_eoc     eoc=1.15 leftover amps (admits)
        │      cliff: eoc≤{cliffs['last_analytic_pass_eoc']} pass → eoc≥{cliffs['e_floor_on_eoc']} ê-floor
        │      exam scores pure leak_e; teacher subtracts tilted declared_e
        │      fix: declared_e / eoc geometry — NOT recipe / per-row
        │
        └─► C  guard-refuse / raw-pole
               M28 guard_refuse_hot_eoc eoc=0.85  lr≈0.66
               cliff: refuse onset eoc≥{cliffs['refuse_on_eoc']} (to_pole > to_mid)
               teacher falls back to raw poles → full geometry leak
```

| ID | cell | branch | admit? | locked | leak≈ | per-row | cliff |
|---|---|---|:---:|:---:|---:|:---:|---|
| **M14** | `e_on_u_declare_lie` | A content_deleted | Y | {members['M14']['locked_pass']} | {members['M14']['leak']} | NO | eou 0→≥0.3 |
| **M20** | `amp_lie_leftover_declare` | A content_deleted | Y | {members['M20']['locked_pass']} | {members['M20']['leak']} | NO | guard×YAML |
| **M26** | `declare_split_three` | A content_deleted | Y | {members['M26']['locked_pass']} | {members['M26'].get('leak','—')} | NO | sibling A |
| **M21** | `hold_e_lyric_mix` | B ê-floor admit | Y | 0/6 | 0.211 | NO | eoc≥{cliffs['e_floor_on_eoc']} |
| **M30** | `eoc_threshold_edge` | B ê-floor admit | Y | 0/6 | 0.207 | knife 1/6 | 0.32Y/0.33N |
| **M31** | `leftover_hot_eoc_declare` | B ê-floor admit | Y | {members['M31']['locked_pass']} | {members['M31']['leak']} | NO | leftover×hot |
| **M28** | `guard_refuse_hot_eoc` | C refuse raw | N | 0/6 | 0.66 | NO | eoc≥{cliffs['refuse_on_eoc']} |

**Not in family:** M16/M17/M2/M24/M27/M29 = **DoF** (per-row clears). M6 mismatch_declare = legacy hetero neg (leak blowup, not content-delete).

## Cliffs (this dig + prior)

### 1. M14 e_on_u ladder (branch A)

| eou | emp PASS | exam | content |
|---|:---:|---:|---:|
| 0.0 | {by_emp['M14_eou0.0']['pass']} | {by_emp['M14_eou0.0']['mean_exam']} | {by_emp['M14_eou0.0']['mean_content']} |
| 0.15 | {by_emp['M14_eou0.15']['pass']} | {by_emp['M14_eou0.15']['mean_exam']} | {by_emp['M14_eou0.15']['mean_content']} |
| 0.30 | {by_emp['M14_eou0.3']['pass']} | {by_emp['M14_eou0.3']['mean_exam']} | {by_emp['M14_eou0.3']['mean_content']} |

**Cliff:** any û-restatement on (`e_on_u≥0.3`) → content_deleted under guard.

### 2. M21-pool eoc ladder (branch B → C)

| eoc | admit | ana_lr | teach_lr | mode | anchor |
|---:|:---:|---:|---:|---|---|
"""
    for r in guard_cliff:
        anc = r["cell_anchor"] or ""
        md += (
            f"| {r['eoc']:.2f} | {'Y' if r['admissible'] else 'N'} | {r['analytic_lr']:.4f} | "
            f"{r['teacher_lr']:.4f} | {r['mode']} | {anc} |\n"
        )

    md += f"""
**Cliffs:**
- **ê-floor onset:** eoc ≥ **{cliffs['e_floor_on_eoc']}** (last analytic pass eoc={cliffs['last_analytic_pass_eoc']})
- **M30 edge:** eoc **0.32 pass / 0.33 fail** (analytic; emp shared 0/3 @ 0.33)
- **refuse onset:** eoc ≥ **{cliffs['refuse_on_eoc']}** → raw-pole lr≈leak/slider≈0.65
- **M21 default 0.35** sits in admit ê-floor; **M28 default 0.85** sits in refuse

### 3. Emp smoke at cliff anchors

| cell | PASS | exam | content | leak |
|---|:---:|---:|---:|---:|
"""
    for c in emp:
        md += (
            f"| `{c['name']}` | {c['pass']} | {c['mean_exam']} | "
            f"{c['mean_content']} | {c['mean_leak']} |\n"
        )

    md += f"""
### 4. M31 leftover-amp eoc (branch B vs A contrast)

| eoc | e_unused | admit | teach_lr | ana_lr | anchor |
|---:|---:|:---:|---:|---:|---|
"""
    for r in m31_cliff:
        md += (
            f"| {r['eoc']} | {r['e_unused']} | {'Y' if r['admissible'] else 'N'} | "
            f"{r['teacher_lr']:.4f} | {r['analytic_lr']:.4f} | {r['anchor'] or ''} |\n"
        )

    md += f"""
## Mechanism (shared spine)

1. Teacher `faithful_guard_e` subtracts along **declared_ê** (from YAML `e_on_*`).
2. Exam leak gate scores **pure `leak_e()`**.
3. When declared ≠ exam axis:
   - **A:** declared restates û/content → guard deletes content (leftover gate fails; û looks fine).
   - **B:** declared tilts off ê but guard **admits** → teacher target retains ê → **analytic floor** lr≈0.205.
   - **C:** declared so damaged guard **refuses** → raw poles → lr≈0.65.
4. n / cover / vic / per-row / couple **cannot** clear A/B/C (target/YAML geometry, not DoF).

## Clearance (prior digs — all NO for this family)

| mid | n2 | vic0@n1 | per_row | source |
|---|:---:|:---:|:---:|---|
| M14 | NO | NO | NO | m14_failmode |
| M20 | NO | NO | NO | m14_failmode / falsify |
| M21 | NO | NO | NO | m21_eoc_irreducible |
| M28 | — | NO | NO | batch4 / falsify_b4 |
| M30 | — | NO | knife 1/6 | m30_knife / falsify_b4 |
| M31 | — | — | NO | falsify_b4 |

## Verdict

{verdict}

- **suite_pass = {suite_pass}**
- recipe_change=**NO**; merge_to_trainer=**NO**; No Music train
- Keep HARD_BITEs: M14, M20, M21, M26, M28, M30, M31
- Fix path: correct declared ê / eoc YAML — not recipe knobs

## Sources

- `m14_failmode_20260909`, `m21_eoc_leak_irreducible_20260909`, `music_to_toy_batch4_20260909`
- `m30_per_row_knife_20260909`, `per_row_falsify_batch4_20260909`
- this dig cliffs: admit→refuse + M31 leftover + eou 0.15 + emp anchors

JSON: `{OUT_JSON.name}`
"""
    OUT_MD.write_text(md)

    log_block = (
        f"\n## Fire — declare-lie / eoc-floor family unified theory (2026-09-09)\n\n"
        f"- Host: box-cpu @ `{sha}`\n"
        f"- Dig: `declare_lie_eoc_family_theory_20260909.{{py,json,md}}` wall={wall}s\n"
        f"- Branches: A=content_deleted(M14/M20/M26); B=ê-floor-admit(M21/M30/M31); "
        f"C=refuse-raw(M28)\n"
        f"- Cliffs: eou 0→≥0.3; eoc floor_on={cliffs['e_floor_on_eoc']} "
        f"refuse_on={cliffs['refuse_on_eoc']}; M30 0.32Y/0.33N\n"
        f"- suite_pass={suite_pass}; ctrl_ok={ctrl_ok}\n"
        f"- Verdict: {verdict}\n"
        f"- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.\n"
    )
    with LOG.open("a") as f:
        f.write(log_block)

    fold = (
        f"\n\n## Declare-lie / eoc-floor family theory (2026-09-09)\n\n"
        f"Source: `declare_lie_eoc_family_theory_20260909` wall={wall}s. "
        f"**suite_pass={suite_pass}**.\n\n"
        f"| branch | members | cliff |\n|---|---|---|\n"
        f"| A content_deleted_under_declare_lie | M14, M20, M26 | eou 0→≥0.3; faithful-no-guard PASS |\n"
        f"| B ê-floor admit | M21, M30, M31 | eoc≥{cliffs['e_floor_on_eoc']} floor; M30 0.32Y/0.33N |\n"
        f"| C guard-refuse raw-pole | M28 | refuse eoc≥{cliffs['refuse_on_eoc']}; lr≈0.66 |\n\n"
        f"n/vic/per_row clear any? **NO**. Keep HARD_BITEs. Recipe change=NO. merge=NO.\n"
        f"One-page map: `declare_lie_eoc_family_theory_20260909.md`.\n"
    )
    with SCOREBOARD.open("a") as f:
        f.write(fold)

    with CATALOG.open("a") as f:
        f.write(
            f"\n\n## Declare-lie / eoc-floor family (2026-09-09)\n\n"
            f"Unified theory: `declare_lie_eoc_family_theory_20260909`. "
            f"A=M14/M20/M26 content_deleted; B=M21/M30/M31 ê-floor admit; "
            f"C=M28 refuse-raw. Cliffs: eou≥0.3; eoc floor_on={cliffs['e_floor_on_eoc']}; "
            f"refuse_on={cliffs['refuse_on_eoc']}. suite_pass={suite_pass}. "
            f"Fix=YAML/declared_ê not recipe.\n"
        )

    print(json.dumps({
        "wall_s": wall, "suite_pass": suite_pass, "cliffs": cliffs, "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
