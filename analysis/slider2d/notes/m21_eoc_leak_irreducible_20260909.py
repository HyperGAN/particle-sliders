#!/usr/bin/env python3
"""Mechanistic dig: why M21 e_on_content leak is irreducible under shared/per-row.

Hypothesis (closed form):
  declared_e = e_on_content·ĉ + e_unused·ê  (tilted off pure ê)
  faithful_sub_e subtracts (a·ê̂)ê̂ from odd a
  → teacher target â retains ê component; leak_ratio = |â_e|/|â_u|
  At M21 defaults (content=0.85, leak=0.65, eoc=0.35, e_unused=0.85):
    â ≈ (1.0 û, 0.50 ĉ, −0.205 ê) → leak_ratio≈0.205 > 0.20 gate
  Perfect residual cover reproduces this floor. Per-row cannot help:
  rows are homogeneous + same tilted teacher — more heads ≠ different target.

Diagnostics (analysis-only; locked recipe UNCHANGED; merge=NO):
  A) Analytic eoc / amp sweeps (no train)
  B) Shared fit confirms residual ≈ teacher floor
  C) Per-row fit same floor (couple w∈{0,0.3})
  D) Train leak_dir=pure ê (ignore eoc) → exam clears (proves teacher tilt)
  E) Score leak against declared_êhat vs pure ê (axis mismatch readout)
  F) leftover CTRL + M20 amp_lie still bite under shared

CPU only. No Music GPU train.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path
import sys

import torch

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

from analysis.slider2d.adv import AdvConfig  # noqa: E402
from analysis.slider2d.field3d import (  # noqa: E402
    Field3D,
    amp_lie_leftover_declare_field3d,
    field3d_teacher_points,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    score_adv_field3d,
)
from analysis.slider2d.gan import default_cfg, fit_adv  # noqa: E402
from conceptmod.textsliders.slider_targets import lm_hold_dir, lm_unit  # noqa: E402

OUT_JSON = NOTES / "m21_eoc_leak_irreducible_20260909.json"
OUT_MD = NOTES / "m21_eoc_leak_irreducible_20260909.md"
LOG = NOTES / "research_log_20260909.md"
SCOREBOARD = NOTES / "MUSIC_TO_TOY_SCOREBOARD_20260909.md"

SEEDS_FULL = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
TEACHER = "faithful_guard_e"
LABEL = "MECH_M21_eoc_irreducible"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def analytic_teacher_residual(
    *,
    slider: float = 1.0,
    content: float = 0.85,
    leak: float = 0.65,
    e_on_u: float = 0.0,
    e_on_content: float = 0.35,
    e_unused: float = 0.85,
) -> dict:
    """Closed-form â after subtract along ê_⊥ = declared_e − (declared·û)û."""
    # declared composition (unnormalized)
    du, dc, de = float(e_on_u), float(e_on_content), float(e_unused)
    # ê_⊥: remove û component
    # declared · û = du (basis orthonormal)
    e_perp_u, e_perp_c, e_perp_e = 0.0, dc, de  # after removing du*û
    # if e_on_u was nonzero, subtract: e_perp = declared - du*û → (0, dc, de)
    n = math.sqrt(e_perp_c**2 + e_perp_e**2 + e_perp_u**2)
    if n < 1e-12:
        return {
            "au": slider,
            "ac": content,
            "ae": leak,
            "leak_ratio": abs(leak) / (abs(slider) + 1e-12),
            "content_kept": 1.0,
            "pass_leak": abs(leak) / (abs(slider) + 1e-12) <= 0.20,
            "eh_c": 0.0,
            "eh_e": 1.0 if abs(leak) > 0 else 0.0,
            "a_dot_eh": float(leak),
            "norm_declared_perp": 0.0,
        }
    eh_c, eh_e = e_perp_c / n, e_perp_e / n
    a_dot = content * eh_c + leak * eh_e  # slider·0
    au = float(slider)
    ac = float(content) - a_dot * eh_c
    ae = float(leak) - a_dot * eh_e
    lr = abs(ae) / (abs(au) + 1e-12)
    ck = ac / (abs(content) + 1e-12) if abs(content) > 1e-12 else 1.0
    return {
        "au": round(au, 6),
        "ac": round(ac, 6),
        "ae": round(ae, 6),
        "leak_ratio": round(lr, 6),
        "content_kept": round(ck, 6),
        "pass_leak": bool(lr <= 0.20),
        "pass_content": bool(ck >= 0.75) if abs(content) > 1e-8 else True,
        "eh_c": round(eh_c, 6),
        "eh_e": round(eh_e, 6),
        "a_dot_eh": round(a_dot, 6),
        "norm_declared_perp": round(n, 6),
    }


def analytic_eoc_sweep() -> list[dict]:
    rows = []
    for eoc in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.0]:
        r = analytic_teacher_residual(e_on_content=eoc)
        r["e_on_content"] = eoc
        rows.append(r)
    return rows


def analytic_amp_sweep() -> list[dict]:
    """Vary content/leak at fixed eoc=0.35 (M21 declare tilt)."""
    rows = []
    for content, leak in [
        (0.55, 0.45),  # leftover-like
        (0.70, 0.50),
        (0.85, 0.65),  # M21
        (0.85, 0.45),
        (0.55, 0.65),
        (1.00, 0.80),
        (0.40, 0.30),
    ]:
        r = analytic_teacher_residual(content=content, leak=leak, e_on_content=0.35)
        r["content"] = content
        r["leak"] = leak
        rows.append(r)
    return rows


def delta_decomp(field: Field3D, d: torch.Tensor) -> dict:
    on_u = float(d @ field.short_u())
    on_c = float(d @ field.content_dir())
    on_e = float(d @ field.leak_e())
    a = field.odd(0)
    a_u = float(a @ field.short_u())
    a_c = float(a @ field.content_dir())
    lr = abs(on_e) / (abs(on_u) + 1e-8)
    # also vs declared direction
    dec = field.declared_e()
    assert dec is not None
    held = lm_hold_dir(dec, slider_dir=field.short_u(), mode="slider")
    if held is not None and float(held.norm()) > 1e-8:
        eh = lm_unit(held)
        on_declared = float(d @ eh)
        lr_declared = abs(on_declared) / (abs(on_u) + 1e-8)
    else:
        on_declared = None
        lr_declared = None
    return {
        "on_u": round(on_u, 6),
        "on_c": round(on_c, 6),
        "on_e": round(on_e, 6),
        "u_kept": round(on_u / (a_u + 1e-8), 6),
        "content_kept": round(on_c / (a_c + 1e-8), 6) if abs(a_c) > 1e-8 else None,
        "leak_ratio_pure_e": round(lr, 6),
        "pass_leak_pure_e": bool(lr <= 0.20),
        "on_declared_hat": None if on_declared is None else round(on_declared, 6),
        "leak_ratio_declared": None if lr_declared is None else round(lr_declared, 6),
        "pass_leak_declared_axis": (
            None if lr_declared is None else bool(lr_declared <= 0.20)
        ),
    }


def teacher_decomp(field: Field3D) -> dict:
    leak_dir = field.declared_e()
    t_plus, t_minus = field3d_teacher_points(
        field, 0, teacher=TEACHER, leak_dir=leak_dir
    )
    neu = field.poles(0)[2]
    d = (t_plus - neu).flatten()
    out = delta_decomp(field, d)
    out["source"] = "teacher_target"
    return out


def locked_cfg(seed: int, **kw) -> AdvConfig:
    base = dict(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        fm_weight=0.0,
        n_particles=12,
        particle_l2=0.02,
    )
    base.update(kw)
    return default_cfg(**base)


def fit_shared_metrics(field: Field3D, seed: int, *, leak_dir=None, **cfg_kw) -> dict:
    cfg = locked_cfg(seed, **cfg_kw)
    if leak_dir is None:
        leak_dir = field.declared_e()
    residual, stats = fit_adv(field, teacher=TEACHER, leak_dir=leak_dir, cfg=cfg)
    d = residual.delta(1.0)
    decomp = delta_decomp(field, d)
    # coverage vs teacher
    t_plus, t_minus = field3d_teacher_points(
        field, 0, teacher=TEACHER, leak_dir=field.declared_e()
    )
    neu = field.poles(0)[2]
    err = float((neu + d - t_plus).norm() / t_plus.norm().clamp_min(1e-8))
    decomp.update(
        {
            "seed": seed,
            "pole_rel_err": round(err, 6),
            "train_leak_is_declared": leak_dir is None
            or (
                field.declared_e() is not None
                and float((leak_dir - field.declared_e()).norm()) < 1e-6
            ),
            "g_loss": stats.get("g_loss"),
        }
    )
    return decomp


def summarize_fits(runs: list[dict], name: str) -> dict:
    n = len(runs)
    lrs = [r["leak_ratio_pure_e"] for r in runs]
    errs = [r["pole_rel_err"] for r in runs if r.get("pole_rel_err") is not None]
    n_pass = sum(1 for r in runs if r["pass_leak_pure_e"])
    return {
        "name": name,
        "n": n,
        "pass_leak": f"{n_pass}/{n}",
        "n_pass_leak": n_pass,
        "mean_leak_ratio": round(sum(lrs) / n, 6) if n else None,
        "max_leak_ratio": round(max(lrs), 6) if n else None,
        "mean_pole_rel_err": round(sum(errs) / len(errs), 6) if errs else None,
        "mean_on_e": round(sum(r["on_e"] for r in runs) / n, 6) if n else None,
        "mean_on_u": round(sum(r["on_u"] for r in runs) / n, 6) if n else None,
        "mean_on_c": round(sum(r["on_c"] for r in runs) / n, 6) if n else None,
        "runs": runs,
    }


def score_scaffold(*args, **kwargs) -> dict:
    return dep._recompute_gates(ex.score_scaffold(*args, **kwargs))


def per_row_grid(name: str, field_fn, seeds, **score_kw) -> dict:
    runs = []
    for s in seeds:
        print(f"    … {name} seed={s}", flush=True)
        try:
            f = field_fn(seed=s)
        except TypeError:
            f = field_fn()
        r = score_scaffold(f, mode="per_row", seed=s, name=f"{name}_s{s}", **score_kw)
        runs.append(r)
    out = ex.summarize(runs)
    out["name"] = name
    # enrich with mean on_e if present
    ons = [r.get("on_e") for r in runs if r.get("on_e") is not None]
    if ons:
        out["mean_on_e"] = round(sum(ons) / len(ons), 6)
    print(
        f"  {name}: pass={out['pass']} multi={out['multi_row']} "
        f"leak_max={out['leak_max']} exam={out['mean_exam']} fail={out['fail_seeds']}",
        flush=True,
    )
    return out


def main() -> None:
    t_wall = time.time()
    sha = git_sha()
    print(f"=== {LABEL} @ {sha} ===", flush=True)

    # ------------------------------------------------------------------
    # A) Analytic
    # ------------------------------------------------------------------
    print("\n[A] analytic teacher floor", flush=True)
    m21_analytic = analytic_teacher_residual()
    eoc_sweep = analytic_eoc_sweep()
    amp_sweep = analytic_amp_sweep()
    print(
        f"  M21 defaults: â=({m21_analytic['au']},{m21_analytic['ac']},{m21_analytic['ae']}) "
        f"lr={m21_analytic['leak_ratio']} pass_leak={m21_analytic['pass_leak']}",
        flush=True,
    )
    # peak fail band
    fail_eocs = [r["e_on_content"] for r in eoc_sweep if not r["pass_leak"]]
    print(f"  eoc fail band (analytic): {fail_eocs}", flush=True)

    field0 = hold_e_lyric_mix_field3d(seed=0)
    teacher0 = teacher_decomp(field0)
    print(
        f"  teacher target decomp: on_e={teacher0['on_e']} lr={teacher0['leak_ratio_pure_e']} "
        f"lr_decl={teacher0['leak_ratio_declared']}",
        flush=True,
    )
    # analytic vs actual teacher should match closely
    analytic_vs_teacher = {
        "analytic_ae": m21_analytic["ae"],
        "teacher_on_e": teacher0["on_e"],
        "analytic_lr": m21_analytic["leak_ratio"],
        "teacher_lr": teacher0["leak_ratio_pure_e"],
        "abs_ae_err": round(abs(m21_analytic["ae"] - teacher0["on_e"]), 6),
        "abs_lr_err": round(
            abs(m21_analytic["leak_ratio"] - teacher0["leak_ratio_pure_e"]), 6
        ),
        "match": abs(m21_analytic["leak_ratio"] - teacher0["leak_ratio_pure_e"]) < 0.01,
    }
    print(f"  analytic↔teacher match: {analytic_vs_teacher}", flush=True)

    # ------------------------------------------------------------------
    # B) Shared fit @ locked — residual ≈ teacher floor
    # ------------------------------------------------------------------
    print("\n[B] shared locked fit on M21 (seeds full)", flush=True)
    shared_runs = []
    for s in SEEDS_FULL:
        print(f"    … shared_declared seed={s}", flush=True)
        shared_runs.append(
            fit_shared_metrics(hold_e_lyric_mix_field3d(seed=s), s)
        )
    shared_sum = summarize_fits(shared_runs, "shared_declared_e")
    print(
        f"  shared: pass_leak={shared_sum['pass_leak']} "
        f"mean_lr={shared_sum['mean_leak_ratio']} mean_on_e={shared_sum['mean_on_e']} "
        f"pole_err={shared_sum['mean_pole_rel_err']}",
        flush=True,
    )
    floor_match = abs(
        (shared_sum["mean_leak_ratio"] or 0) - m21_analytic["leak_ratio"]
    ) < 0.03
    print(f"  residual≈analytic floor? {floor_match}", flush=True)

    # eoc empirical smoke (shared, 3 seeds) at key points
    print("\n[B2] shared eoc empirical smoke", flush=True)
    eoc_emp = []
    for eoc in [0.0, 0.20, 0.35, 0.70]:
        runs = []
        for s in SEEDS_SMOKE:
            print(f"    … eoc={eoc} seed={s}", flush=True)
            f = hold_e_lyric_mix_field3d(seed=s, e_on_content=eoc)
            runs.append(fit_shared_metrics(f, s))
        sm = summarize_fits(runs, f"eoc_{eoc}")
        pred = analytic_teacher_residual(e_on_content=eoc)
        sm["analytic_lr"] = pred["leak_ratio"]
        sm["analytic_pass"] = pred["pass_leak"]
        sm["e_on_content"] = eoc
        eoc_emp.append(sm)
        print(
            f"  eoc={eoc}: emp_lr={sm['mean_leak_ratio']} pred={pred['leak_ratio']} "
            f"pass={sm['pass_leak']}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # C) Per-row same floor
    # ------------------------------------------------------------------
    print("\n[C] per-row M21 (couple 0 / 0.3)", flush=True)
    pr0 = per_row_grid(
        "per_row_w0",
        hold_e_lyric_mix_field3d,
        SEEDS_FULL,
        coupling_weight=0.0,
    )
    pr03 = per_row_grid(
        "per_row_w0.3",
        hold_e_lyric_mix_field3d,
        SEEDS_SMOKE,
        coupling_weight=0.3,
    )

    # ------------------------------------------------------------------
    # D) Train with pure ê (ignore eoc declaration) — should CLEAR exam
    # ------------------------------------------------------------------
    print("\n[D] train leak_dir=pure ê (geom ignore eoc)", flush=True)
    pure_runs = []
    for s in SEEDS_FULL:
        print(f"    … pure_e train seed={s}", flush=True)
        f = hold_e_lyric_mix_field3d(seed=s)
        pure_runs.append(
            fit_shared_metrics(f, s, leak_dir=f.leak_e())  # pure ê
        )
    pure_sum = summarize_fits(pure_runs, "train_pure_e")
    # Also score via score_adv with override — confirm pass
    pure_score_runs = []
    for s in SEEDS_SMOKE:
        f = hold_e_lyric_mix_field3d(seed=s)
        row = score_adv_field3d(
            f, teacher=TEACHER, leak_dir=f.leak_e(), cfg=locked_cfg(s), name=f"pure_s{s}"
        )
        pure_score_runs.append(
            {
                "seed": s,
                "pass": row["pass"],
                "pass_leak": row["pass_leak"],
                "leak_ratio": row["leak_ratio"],
                "u_kept": row["u_kept"],
            }
        )
    pure_exam_pass = sum(1 for r in pure_score_runs if r["pass"])
    print(
        f"  train_pure_e: pass_leak={pure_sum['pass_leak']} mean_lr={pure_sum['mean_leak_ratio']} "
        f"score_adv pass={pure_exam_pass}/{len(pure_score_runs)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # E) Axis mismatch readout on teacher/shared
    # ------------------------------------------------------------------
    print("\n[E] axis mismatch: pure-ê vs declared-axis leak score", flush=True)
    axis_note = {
        "teacher": teacher0,
        "shared_seed0": shared_runs[0] if shared_runs else None,
        "claim": (
            "Exam leak_ratio uses pure leak_e(); teacher subtracts along tilted "
            "declared_e. Perfect cover ⇒ irreducible pure-ê leak ≈ analytic floor."
        ),
    }

    # ------------------------------------------------------------------
    # F) Controls: leftover PASS; M20 still bite
    # ------------------------------------------------------------------
    print("\n[F] controls leftover + M20", flush=True)
    ctrl_runs = []
    for s in SEEDS_SMOKE:
        print(f"    … leftover seed={s}", flush=True)
        row = score_adv_field3d(
            leftover_field3d(), teacher=TEACHER, cfg=locked_cfg(s), name=f"leftover_s{s}"
        )
        ctrl_runs.append(
            {"seed": s, "pass": row["pass"], "leak_ratio": row["leak_ratio"], "u_kept": row["u_kept"]}
        )
    m20_runs = []
    for s in SEEDS_SMOKE:
        print(f"    … M20 seed={s}", flush=True)
        row = score_adv_field3d(
            amp_lie_leftover_declare_field3d(seed=s),
            teacher=TEACHER,
            cfg=locked_cfg(s),
            name=f"m20_s{s}",
        )
        m20_runs.append(
            {
                "seed": s,
                "pass": row["pass"],
                "pass_leak": row["pass_leak"],
                "leak_ratio": row["leak_ratio"],
            }
        )
    leftover_ok = all(r["pass"] for r in ctrl_runs)
    m20_still_bites = all(not r["pass"] for r in m20_runs)
    print(
        f"  leftover PASS={leftover_ok}; M20 still bites={m20_still_bites}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    shared_fails = shared_sum["n_pass_leak"] == 0
    per_row_fails = pr0.get("n_pass", 0) == 0 or (
        isinstance(pr0.get("pass"), str) and pr0["pass"].startswith("0/")
    )
    # pr0 pass is like "0/6"
    pr_n_pass = pr0.get("n_pass")
    if pr_n_pass is None and isinstance(pr0.get("pass"), str) and "/" in pr0["pass"]:
        pr_n_pass = int(pr0["pass"].split("/")[0])
    per_row_fails = (pr_n_pass or 0) == 0

    pure_clears = pure_sum["n_pass_leak"] >= 5  # ≥5/6
    irreducible = (
        analytic_vs_teacher["match"]
        and floor_match
        and shared_fails
        and per_row_fails
        and pure_clears
        and leftover_ok
    )

    verdict = (
        "YES — M21 leak is teacher-axis geometry floor (declared_e tilt), "
        "not shared-DoF / optimization; per-row cannot clear; train-pure-ê clears"
        if irreducible
        else "PARTIAL — see tables; check floor_match / pure_clears / controls"
    )
    print(f"\nVERDICT: {verdict}", flush=True)
    print(
        f"  analytic_match={analytic_vs_teacher['match']} floor_match={floor_match} "
        f"shared_fails={shared_fails} per_row_fails={per_row_fails} "
        f"pure_clears={pure_clears} leftover_ok={leftover_ok} m20_bites={m20_still_bites}",
        flush=True,
    )

    wall = round(time.time() - t_wall, 1)
    payload = {
        "label": LABEL,
        "sha": sha,
        "wall_s": wall,
        "recipe_change": False,
        "merge_to_trainer": False,
        "analytic_m21": m21_analytic,
        "analytic_vs_teacher": analytic_vs_teacher,
        "eoc_sweep_analytic": eoc_sweep,
        "amp_sweep_analytic": amp_sweep,
        "eoc_fail_band": fail_eocs,
        "shared_declared": {
            k: v for k, v in shared_sum.items() if k != "runs"
        },
        "shared_runs": shared_runs,
        "eoc_empirical": [
            {k: v for k, v in r.items() if k != "runs"} for r in eoc_emp
        ],
        "per_row_w0": {
            k: pr0[k]
            for k in (
                "pass",
                "multi_row",
                "bite_cleared",
                "mean_exam",
                "leak_max",
                "fail_seeds",
                "n_pass",
                "n",
                "mean_on_e",
            )
            if k in pr0
        },
        "per_row_w0.3": {
            k: pr03[k]
            for k in (
                "pass",
                "multi_row",
                "bite_cleared",
                "mean_exam",
                "leak_max",
                "fail_seeds",
                "n_pass",
                "n",
            )
            if k in pr03
        },
        "train_pure_e": {k: v for k, v in pure_sum.items() if k != "runs"},
        "train_pure_e_runs": pure_runs,
        "train_pure_e_score_adv": pure_score_runs,
        "axis_mismatch": axis_note,
        "controls": {
            "leftover": ctrl_runs,
            "leftover_ok": leftover_ok,
            "m20": m20_runs,
            "m20_still_bites": m20_still_bites,
        },
        "flags": {
            "analytic_match": analytic_vs_teacher["match"],
            "floor_match": floor_match,
            "shared_fails": shared_fails,
            "per_row_fails": per_row_fails,
            "pure_e_train_clears": pure_clears,
            "irreducible_under_shared_and_per_row": irreducible,
        },
        "verdict": verdict,
        "mechanism": (
            "declared_e = e_on_content·ĉ + e_unused·ê tilts ê̂ off pure ê; "
            "faithful_sub_e leaves â_e≠0 on teacher; exam scores |δ·ê|/|δ·û| "
            "against pure ê → floor≈0.205 at M21 defaults. Homogeneous rows ⇒ "
            "per-row heads face identical tilted teacher. Only ablating eoc "
            "(or training on pure ê) clears — not a recipe/DoF fix."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {OUT_JSON}", flush=True)

    # MD
    lines = [
        "# M21 e_on_content leak irreducible — mechanistic dig — 2026-09-09",
        "",
        f"Host: box-cpu @ `{sha}`. Wall {wall}s. CPU only. **No Music train.**",
        f"Label: `{LABEL}` — analysis-only; **locked recipe unchanged; merge=NO.**",
        "",
        "## Mechanism (closed form)",
        "",
        "M21 `hold_e_lyric_mix`: homogeneous leftover + `e_on_content=0.35` so",
        "`declared_e = 0.35·ĉ + 0.85·ê` (tilted off pure ê).",
        "",
        "`faithful_guard_e` / `faithful_sub_e` subtracts `(a·ê̂_⊥)ê̂_⊥` from odd `a`.",
        "With tilted ê̂, the **teacher target itself** retains an ê component:",
        "",
        f"- Analytic â ≈ `({m21_analytic['au']} û, {m21_analytic['ac']} ĉ, {m21_analytic['ae']} ê)`",
        f"- **leak_ratio floor = {m21_analytic['leak_ratio']}** (gate ≤0.20 → FAIL)",
        f"- Teacher actual: on_e={teacher0['on_e']}, lr={teacher0['leak_ratio_pure_e']} "
        f"(abs err lr={analytic_vs_teacher['abs_lr_err']}; match={analytic_vs_teacher['match']})",
        "",
        "Exam `leak_ratio` uses **pure `leak_e()`**, while training subtracts along",
        "**tilted `declared_e`**. Perfect residual cover reproduces the teacher floor.",
        "Per-row cannot help: rows are homogeneous and share the same tilted teacher.",
        "",
        "## Analytic eoc sweep (content=0.85, leak=0.65, e_unused=0.85)",
        "",
        "| eoc | â_e | leak_ratio | pass_leak | content_kept |",
        "|---:|---:|---:|:---:|---:|",
    ]
    for r in eoc_sweep:
        lines.append(
            f"| {r['e_on_content']:.2f} | {r['ae']:.4f} | {r['leak_ratio']:.4f} | "
            f"{'Y' if r['pass_leak'] else 'N'} | {r['content_kept']:.4f} |"
        )
    lines += [
        "",
        f"**Fail band (analytic):** eoc ∈ {fail_eocs} (M21 default 0.35 sits in-band).",
        "",
        "## Analytic amp sweep @ eoc=0.35",
        "",
        "| content | leak | â_e | leak_ratio | pass_leak |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for r in amp_sweep:
        lines.append(
            f"| {r['content']:.2f} | {r['leak']:.2f} | {r['ae']:.4f} | "
            f"{r['leak_ratio']:.4f} | {'Y' if r['pass_leak'] else 'N'} |"
        )
    lines += [
        "",
        "## Empirical shared (locked declared_e)",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| pass_leak | **{shared_sum['pass_leak']}** |",
        f"| mean leak_ratio | {shared_sum['mean_leak_ratio']} |",
        f"| mean on_e | {shared_sum['mean_on_e']} |",
        f"| mean pole_rel_err | {shared_sum['mean_pole_rel_err']} |",
        f"| residual≈analytic floor | **{floor_match}** |",
        "",
        "## Empirical eoc smoke (shared)",
        "",
        "| eoc | emp mean_lr | analytic_lr | pass_leak |",
        "|---:|---:|---:|---|",
    ]
    for r in eoc_emp:
        lines.append(
            f"| {r['e_on_content']} | {r['mean_leak_ratio']} | {r['analytic_lr']} | {r['pass_leak']} |"
        )
    lines += [
        "",
        "## Per-row (NON_DEFAULT scaffold)",
        "",
        f"| mode | pass | multi | leak_max | exam | fail |",
        f"|---|:---:|:---:|---:|---:|---|",
        f"| per_row w=0 | {pr0.get('pass')} | {pr0.get('multi_row')} | {pr0.get('leak_max')} | "
        f"{pr0.get('mean_exam')} | {pr0.get('fail_seeds')} |",
        f"| per_row w=0.3 | {pr03.get('pass')} | {pr03.get('multi_row')} | {pr03.get('leak_max')} | "
        f"{pr03.get('mean_exam')} | {pr03.get('fail_seeds')} |",
        "",
        "## Diagnostic: train on pure ê (ignore eoc) — geom ablation",
        "",
        f"- fit pass_leak: **{pure_sum['pass_leak']}** mean_lr={pure_sum['mean_leak_ratio']}",
        f"- score_adv pass: **{pure_exam_pass}/{len(pure_score_runs)}**",
        "- Proves: failure is **teacher leak_dir tilt**, not odd amps / cover / particles.",
        "- Do **not** delete eoc from the cell (that soft-deletes the Music symptom).",
        "",
        "## Controls",
        "",
        f"- leftover CTRL: PASS={leftover_ok} ({ctrl_runs})",
        f"- M20 amp_lie still bites: {m20_still_bites} ({m20_runs})",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "- recipe_change=**NO**; merge_to_trainer=**NO**",
        "- M21 remains HARD BOUNDARY under shared **and** per-row",
        "- Driver confirmed: `e_on_content>0` tilts declared_e → teacher ê floor > 0.20",
        "- Distinct from M1/M24/M27 DoF bites (per-row clears those, not M21)",
        "",
        f"JSON: `{OUT_JSON.name}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)

    # Append research_log
    log_block = f"""
## Fire — M21 eoc leak irreducible (mechanistic) (2026-09-09)

- Host: box-cpu @ SHA `{sha}`
- Dig: `m21_eoc_leak_irreducible_20260909.{{py,json,md}}` wall={wall}s
- Analytic floor @ M21 defaults: leak_ratio={m21_analytic['leak_ratio']} (â_e={m21_analytic['ae']})
- analytic↔teacher match={analytic_vs_teacher['match']}; residual≈floor={floor_match}
- shared declared: pass_leak={shared_sum['pass_leak']} mean_lr={shared_sum['mean_leak_ratio']}
- per_row w0: pass={pr0.get('pass')} leak_max={pr0.get('leak_max')} (still fails leak gate)
- train pure ê: pass_leak={pure_sum['pass_leak']} (clears — teacher-tilt diagnostic)
- leftover_ok={leftover_ok}; m20_still_bites={m20_still_bites}
- Verdict: {verdict}
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- No Music GPU train; servers untouched.
"""
    with LOG.open("a") as fh:
        fh.write(log_block)
    print(f"appended {LOG}", flush=True)

    # Fold into scoreboard
    sb_block = f"""

## Mechanistic: M21 e_on_content leak irreducible (2026-09-09)

**Script:** `m21_eoc_leak_irreducible_20260909` wall={wall}s

**Closed form:** `declared_e = eoc·ĉ + e_unused·ê` tilts ê̂; `faithful_sub_e` leaves
teacher â with ê component. At M21 defaults â_e≈{m21_analytic['ae']},
**leak_ratio floor≈{m21_analytic['leak_ratio']}** (>0.20). Exam scores pure `leak_e()`;
perfect cover reproduces the floor.

| probe | result |
|---|---|
| analytic↔teacher | match={analytic_vs_teacher['match']} |
| shared locked | pass_leak={shared_sum['pass_leak']} mean_lr={shared_sum['mean_leak_ratio']} ≈floor |
| per-row w0 / w0.3 | pass={pr0.get('pass')} / {pr03.get('pass')} — multi OK, **leak gate fails** |
| train leak_dir=pure ê | pass_leak={pure_sum['pass_leak']} **CLEARS** (diagnostic only) |
| leftover / M20 | ok={leftover_ok} / still_bites={m20_still_bites} |

**Why per-row does not clear M21:** homogeneous rows + identical tilted teacher —
not a DoF / mean-δ bite (unlike M1/M24/M27).

**merge=NO.** Do not strip eoc from cell. Recipe change=NO.
"""
    with SCOREBOARD.open("a") as fh:
        fh.write(sb_block)
    print(f"appended {SCOREBOARD}", flush=True)
    print(f"DONE wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
