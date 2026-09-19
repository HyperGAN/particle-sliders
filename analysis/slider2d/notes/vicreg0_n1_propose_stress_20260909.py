#!/usr/bin/env python3
"""Stress-check propose: vicreg_weight=0 when n_particles==1 (Music posture only).

Propose-only finding (close_seed0_deep / vicreg_n1_close_knife):
  vic=0 @ n=1 clears close/live seed knives 21/21; leftover n12 flat.

THIS JOB (no locked-default change, no Music GPU train):
  1) Replicate close/live 21/21 at n=1: vic0 vs vic default (c1.5 propose + c1.0 Music)
  2) leftover / dual-arm positive / M22 stagger_mild / M23 multipair @ n=1 vic0
     — flag false lock or new fail
  3) Hard bites M20/M21/M24 must still FAIL under vic0 (no YAML-lie fix)
  4) Write vicreg0_n1_propose_stress_20260909.{md,py,json}
     recommend adopt vs reject for Music-posture n=1 only

Locked recipe UNCHANGED: 1200 / c1.5 / faithful_guard_e / FM0 / n≤12 /
particle_l2=0.02 / vicreg=0.05 / b_cap=1.
Music-posture scoring proxy: n=1, cover=1.0, same guard/FM0/l2/b_cap.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import sys

_REPO = Path("/workspace/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import (  # noqa: E402
    amp_lie_leftover_declare_field3d,
    close_field3d,
    close_live_noise_field3d,
    content_leak_flip_rows_field3d,
    dual_arm_leftover_geom_field3d,
    hold_e_lyric_mix_field3d,
    leftover_field3d,
    multipair_corr_seed_field3d,
    score_adv_field3d,
    score_adv_field3d_exam,
    stagger_mild_cross_field3d,
)
from analysis.slider2d.gan import default_cfg  # noqa: E402

NOTES = _REPO / "analysis/slider2d/notes"
OUT_JSON = NOTES / "vicreg0_n1_propose_stress_20260909.json"
OUT_MD = NOTES / "vicreg0_n1_propose_stress_20260909.md"
LOG = NOTES / "research_log_20260909.md"
PARTIAL = Path("/workspace/vicreg0_n1_propose_stress.partial.jsonl")
PRIOR_DEEP = NOTES / "close_seed0_deep_20260909.json"

TEACHER = "faithful_guard_e"
SEEDS_20 = list(range(0, 21))
SEEDS_STD = [0, 1, 2, 3, 7, 42]
SEEDS_SMOKE = [0, 1, 2]
# seeds that failed under default at c1.0 close (prior dig)
SEEDS_C10_KNIFE = [0, 4]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def make_cfg(seed: int, **kw):
    """Scoring cfg only — never mutates gan.default_cfg locked defaults."""
    return default_cfg(
        steps=int(kw.get("steps", 1200)),
        seed=int(seed),
        b_cap=float(kw.get("b_cap", 1.0)),
        cover_weight=float(kw.get("cover", 1.5)),
        fm_weight=0.0,
        n_particles=int(kw.get("n", 1)),
        particle_l2=float(kw.get("particle_l2", 0.02)),
        span_frac=float(kw.get("span_frac", 0.40)),
        end_margin=float(kw.get("end_margin", 0.60)),
        cloud_std=float(kw.get("cloud_std", 0.03)),
        particle_jitter=float(kw.get("particle_jitter", 0.01)),
        lr=float(kw.get("lr", 5.0e-3)),
        vicreg_weight=float(kw.get("vicreg_weight", 0.05)),
    )


def _load_done() -> set[str]:
    done: set[str] = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done.add(row["key"])
            except Exception:
                continue
    return done


def run_one(
    field,
    *,
    key: str,
    seed: int,
    name: str,
    mode: str = "exam",
    teacher: str = TEACHER,
    done: set[str] | None = None,
    **cfg_kw,
) -> dict:
    if done is not None and key in done:
        # reload from partial
        for line in reversed(PARTIAL.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("key") == key:
                print(f"  [skip] {key} pass={row['pass']} exam={row.get('exam_score', row.get('primary', 0)):.4f}", flush=True)
                return row
    t0 = time.time()
    cfg = make_cfg(seed, **cfg_kw)
    if mode == "exam":
        row = score_adv_field3d_exam(field, teacher=teacher, cfg=cfg, name=name)
        passed = bool(row.get("exam_pass", row.get("pass")))
        primary = float(row.get("exam_score", 0.0))
    else:
        row = score_adv_field3d(field, teacher=teacher, cfg=cfg, name=name)
        passed = bool(row.get("pass", False))
        primary = float(row.get("primary", row.get("exam_score", 0.0)))
    out = {
        "key": key,
        "name": name,
        "seed": seed,
        "mode": mode,
        "teacher": teacher,
        "pass": passed,
        "exam_score": primary,
        "primary": primary,
        "u_kept": float(row.get("u_kept", 0.0)),
        "content_kept": float(row.get("content_kept", 0.0)),
        "leak_ratio": float(row.get("leak_ratio", 0.0)),
        "on_u": float(row.get("on_u", 0.0)),
        "residual_norm": float(row.get("residual_norm", 0.0)),
        "rows_covered": int(row.get("rows_covered", 0)),
        "rows_total": int(row.get("rows_total", 0)),
        "pass_u": bool(row.get("pass_u", False)),
        "pass_content": bool(row.get("pass_content", False)),
        "pass_leak": bool(row.get("pass_leak", False)),
        "pass_cont": bool(row.get("pass_cont", False)),
        "pass_swing": bool(row.get("pass_swing", False)),
        "pass_leftover_gate": bool(row.get("pass_leftover_gate", False)),
        "pass_multi_row": bool(row.get("pass_multi_row", False)),
        "cfg": {
            "cover": cfg_kw.get("cover"),
            "n": cfg_kw.get("n"),
            "steps": cfg_kw.get("steps", 1200),
            "vicreg_weight": cfg_kw.get("vicreg_weight", 0.05),
            "particle_l2": cfg_kw.get("particle_l2", 0.02),
        },
        "wall_s": round(time.time() - t0, 2),
    }
    with PARTIAL.open("a") as f:
        f.write(json.dumps(out) + "\n")
    if done is not None:
        done.add(key)
    print(
        f"  {key} pass={out['pass']} exam={out['exam_score']:.4f} "
        f"u={out['u_kept']:.4f} leak={out['leak_ratio']:.4f} "
        f"wall={out['wall_s']}",
        flush=True,
    )
    return out


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    npass = sum(1 for r in runs if r["pass"])
    prim = [r["exam_score"] for r in runs]
    return {
        "n": n,
        "n_pass": npass,
        "pass": f"{npass}/{n}",
        "pass_str": f"{npass}/{n}",
        "mean_exam": round(sum(prim) / n, 4) if n else 0.0,
        "exam_span": round(max(prim) - min(prim), 4) if n else 0.0,
        "mean_u": round(sum(r["u_kept"] for r in runs) / n, 4) if n else 0.0,
        "leak_max": round(max(r["leak_ratio"] for r in runs), 4) if n else 0.0,
        "fail_seeds": [r["seed"] for r in runs if not r["pass"]],
        "knife": bool(0 < npass < n),
        "hard_bite": bool(npass == 0 and n > 0),
        "no_bite": bool(npass == n and n > 0),
        "pass_all": bool(npass == n and n > 0),
        "runs": runs,
    }


def grid(
    label: str,
    field_fn,
    seeds,
    *,
    done: set[str],
    mode: str = "exam",
    teacher: str = TEACHER,
    field_kw_fn=None,
    **cfg_kw,
) -> dict:
    print(
        f"=== {label} cover={cfg_kw.get('cover')} n={cfg_kw.get('n')} "
        f"vic={cfg_kw.get('vicreg_weight', 0.05)} ===",
        flush=True,
    )
    runs = []
    for s in seeds:
        if field_kw_fn is not None:
            field = field_fn(**field_kw_fn(s))
        else:
            try:
                field = field_fn(seed=s)
            except TypeError:
                field = field_fn()
        key = f"{label}|s{s}|c{cfg_kw.get('cover')}|n{cfg_kw.get('n')}|v{cfg_kw.get('vicreg_weight', 0.05)}|{teacher}|{mode}"
        runs.append(
            run_one(
                field,
                key=key,
                seed=s,
                name=f"{label}_s{s}",
                mode=mode,
                teacher=teacher,
                done=done,
                **cfg_kw,
            )
        )
    out = summarize(runs)
    out["name"] = label
    out["kw"] = dict(cfg_kw)
    out["teacher"] = teacher
    out["mode"] = mode
    print(
        f"  >> {label}: pass={out['pass']} exam={out['mean_exam']} "
        f"fail={out['fail_seeds']} knife={out['knife']} hard_bite={out['hard_bite']}",
        flush=True,
    )
    return out


def load_prior_c15() -> dict:
    """Cite prior close_seed0_deep c1.5 21-seed cells (same locked recipe)."""
    if not PRIOR_DEEP.exists():
        return {}
    d = json.loads(PRIOR_DEEP.read_text())
    cells = d.get("cells", {})
    v = d.get("verdict", {})
    out = {}
    for k in (
        "close_n1_c1.5_s0_20",
        "live_n1_c1.5_s0_20",
        "close_n1_vic0_s0_20",
        "live_n1_vic0_s0_20",
        "leftover_n12_vic0.05",
        "leftover_n12_vic0",
    ):
        c = cells.get(k)
        if not c:
            continue
        out[k] = {
            "pass_str": c.get("pass_str") or c.get("pass"),
            "mean_exam": c.get("mean_exam"),
            "fail_seeds": c.get("fail_seeds"),
            "knife": c.get("knife"),
            "source": "close_seed0_deep_20260909.json",
        }
    out["propose_vic0_when_n1"] = v.get("propose_vic0_when_n1")
    out["leftover_flat"] = v.get("leftover_flat")
    out["leftover_mean_exam_vic0.05"] = v.get("leftover_mean_exam_vic0.05")
    out["leftover_mean_exam_vic0"] = v.get("leftover_mean_exam_vic0")
    return out


def main() -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    t_wall = time.time()
    sha = git_sha()
    done = _load_done()
    print(f"=== vicreg0_n1_propose_stress @ {sha} partial_done={len(done)} ===", flush=True)
    print("LOCKED DEFAULTS UNCHANGED. No Music train. Stress scoring posture only.", flush=True)

    cells: dict = {}
    prior = load_prior_c15()
    cells["prior_c15_cite"] = prior

    # ------------------------------------------------------------------
    # 1) Replicate close/live 21/21 @ n=1: vic0 vs default
    #    A) Music posture c1.0 (primary for adopt) — full seeds 0..20
    #    B) Propose c1.5 — cite prior 21/21 + live reconfirm SEEDS_RECONFIRM
    # ------------------------------------------------------------------
    SEEDS_RECONFIRM = [0, 1, 2, 3, 4, 7, 42]  # knife + std + c1.0-fail seed4

    # A) Music posture full replicate
    for vic, vtag in ((0.05, "vic0.05"), (0.0, "vic0")):
        cells[f"close_n1_c1.0_{vtag}"] = grid(
            f"close_n1_c1.0_{vtag}",
            close_field3d,
            SEEDS_20,
            done=done,
            cover=1.0,
            n=1,
            vicreg_weight=vic,
        )
        cells[f"live_n1_c1.0_{vtag}"] = grid(
            f"live_n1_c1.0_{vtag}",
            close_live_noise_field3d,
            SEEDS_20,
            done=done,
            cover=1.0,
            n=1,
            vicreg_weight=vic,
        )

    # B) Propose c1.5 live reconfirm (prior JSON holds full 21/21)
    for vic, vtag in ((0.05, "vic0.05"), (0.0, "vic0")):
        cells[f"close_n1_c1.5_{vtag}"] = grid(
            f"close_n1_c1.5_{vtag}",
            close_field3d,
            SEEDS_RECONFIRM,
            done=done,
            cover=1.5,
            n=1,
            vicreg_weight=vic,
        )
        cells[f"live_n1_c1.5_{vtag}"] = grid(
            f"live_n1_c1.5_{vtag}",
            close_live_noise_field3d,
            SEEDS_RECONFIRM,
            done=done,
            cover=1.5,
            n=1,
            vicreg_weight=vic,
        )

    # ------------------------------------------------------------------
    # 2a) leftover controls — n12 flat + music n1
    # ------------------------------------------------------------------
    for vic, vtag in ((0.05, "vic0.05"), (0.0, "vic0")):
        cells[f"leftover_n12_c1.5_{vtag}"] = grid(
            f"leftover_n12_c1.5_{vtag}",
            leftover_field3d,
            SEEDS_STD,
            done=done,
            cover=1.5,
            n=12,
            vicreg_weight=vic,
        )
        cells[f"leftover_n1_c1.0_{vtag}"] = grid(
            f"leftover_n1_c1.0_{vtag}",
            leftover_field3d,
            SEEDS_STD,
            done=done,
            cover=1.0,
            n=1,
            vicreg_weight=vic,
        )

    # ------------------------------------------------------------------
    # 2b) dual-arm: positive control must PASS; arms alone must FAIL
    #     under vic0 @ n=1 (and Music cover variants)
    # ------------------------------------------------------------------
    # leftover_only: guard + cover=0 — expect FAIL
    cells["dual_leftover_only_n1_vic0"] = grid(
        "dual_leftover_only_n1_vic0",
        dual_arm_leftover_geom_field3d,
        SEEDS_SMOKE,
        done=done,
        mode="score",
        teacher=TEACHER,
        cover=0.0,
        n=1,
        vicreg_weight=0.0,
    )
    # listen/cover only: faithful + cover=1.5 — expect FAIL
    cells["dual_listen_only_n1_vic0"] = grid(
        "dual_listen_only_n1_vic0",
        dual_arm_leftover_geom_field3d,
        SEEDS_SMOKE,
        done=done,
        mode="score",
        teacher="faithful",
        cover=1.5,
        n=1,
        vicreg_weight=0.0,
    )
    # positive: guard + cover=1.5 — expect PASS
    cells["dual_pos_c1.5_n1_vic0"] = grid(
        "dual_pos_c1.5_n1_vic0",
        dual_arm_leftover_geom_field3d,
        SEEDS_SMOKE,
        done=done,
        mode="score",
        teacher=TEACHER,
        cover=1.5,
        n=1,
        vicreg_weight=0.0,
    )
    # Music posture positive: guard + cover=1.0 — expect PASS
    cells["dual_pos_c1.0_n1_vic0"] = grid(
        "dual_pos_c1.0_n1_vic0",
        dual_arm_leftover_geom_field3d,
        SEEDS_SMOKE,
        done=done,
        mode="score",
        teacher=TEACHER,
        cover=1.0,
        n=1,
        vicreg_weight=0.0,
    )
    # baseline dual pos under default vic for delta
    cells["dual_pos_c1.5_n1_vic0.05"] = grid(
        "dual_pos_c1.5_n1_vic0.05",
        dual_arm_leftover_geom_field3d,
        SEEDS_SMOKE,
        done=done,
        mode="score",
        teacher=TEACHER,
        cover=1.5,
        n=1,
        vicreg_weight=0.05,
    )

    # ------------------------------------------------------------------
    # 2c) M22 stagger_mild_cross / M23 multipair @ Music n=1
    # ------------------------------------------------------------------
    for vic, vtag in ((0.05, "vic0.05"), (0.0, "vic0")):
        cells[f"M22_stagger_music_{vtag}"] = grid(
            f"M22_stagger_music_{vtag}",
            stagger_mild_cross_field3d,
            SEEDS_STD,
            done=done,
            cover=1.0,
            n=1,
            vicreg_weight=vic,
        )
        cells[f"M23_multipair_music_{vtag}"] = grid(
            f"M23_multipair_music_{vtag}",
            multipair_corr_seed_field3d,
            SEEDS_STD,
            done=done,
            cover=1.0,
            n=1,
            vicreg_weight=vic,
        )

    # ------------------------------------------------------------------
    # 3) Hard bites M20/M21/M24 — must FAIL under Music n1 vic0
    # ------------------------------------------------------------------
    bite_specs = [
        ("M20_amp_lie", amp_lie_leftover_declare_field3d),
        ("M21_hold_e", hold_e_lyric_mix_field3d),
        ("M24_content_leak_flip", content_leak_flip_rows_field3d),
    ]
    for name, ctor in bite_specs:
        for vic, vtag in ((0.05, "vic0.05"), (0.0, "vic0")):
            cells[f"{name}_music_{vtag}"] = grid(
                f"{name}_music_{vtag}",
                ctor,
                SEEDS_STD,
                done=done,
                cover=1.0,
                n=1,
                vicreg_weight=vic,
            )

    # ------------------------------------------------------------------
    # Verdict / adopt vs reject for Music-posture n=1
    # ------------------------------------------------------------------
    def g(name: str) -> dict:
        return cells[name]

    close_m_v05 = g("close_n1_c1.0_vic0.05")
    close_m_v0 = g("close_n1_c1.0_vic0")
    live_m_v05 = g("live_n1_c1.0_vic0.05")
    live_m_v0 = g("live_n1_c1.0_vic0")
    close_p_v05 = g("close_n1_c1.5_vic0.05")
    close_p_v0 = g("close_n1_c1.5_vic0")
    live_p_v05 = g("live_n1_c1.5_vic0.05")
    live_p_v0 = g("live_n1_c1.5_vic0")

    left12_v05 = g("leftover_n12_c1.5_vic0.05")
    left12_v0 = g("leftover_n12_c1.5_vic0")
    left1_v05 = g("leftover_n1_c1.0_vic0.05")
    left1_v0 = g("leftover_n1_c1.0_vic0")

    dual_pos_v0 = g("dual_pos_c1.5_n1_vic0")
    dual_pos_m_v0 = g("dual_pos_c1.0_n1_vic0")
    dual_pos_v05 = g("dual_pos_c1.5_n1_vic0.05")
    dual_lo = g("dual_leftover_only_n1_vic0")
    dual_li = g("dual_listen_only_n1_vic0")

    m22_v05 = g("M22_stagger_music_vic0.05")
    m22_v0 = g("M22_stagger_music_vic0")
    m23_v05 = g("M23_multipair_music_vic0.05")
    m23_v0 = g("M23_multipair_music_vic0")

    bites_ok = True
    bite_rows = []
    for name, _ in bite_specs:
        b0 = g(f"{name}_music_vic0")
        b05 = g(f"{name}_music_vic0.05")
        still_fail = b0["hard_bite"] or (b0["n_pass"] == 0)
        # must not "fix" — if default fails and vic0 passes = BAD false lock
        false_fix = (b05["n_pass"] == 0) and (b0["n_pass"] > 0)
        if not still_fail or false_fix:
            bites_ok = False
        bite_rows.append(
            {
                "name": name,
                "vic0.05": b05["pass_str"],
                "vic0": b0["pass_str"],
                "still_hard_fail_vic0": still_fail,
                "false_fix": false_fix,
                "leak_max_vic0": b0["leak_max"],
                "fail_seeds_vic0": b0["fail_seeds"],
            }
        )

    leftover_flat = abs(left12_v05["mean_exam"] - left12_v0["mean_exam"]) < 1e-3 and left12_v0["pass_all"] and left12_v05["pass_all"]
    leftover_n1_ok = left1_v0["pass_all"] and left1_v05["pass_all"]
    leftover_regression = (not left12_v0["pass_all"]) or (not left1_v0["pass_all"])

    dual_pos_ok = dual_pos_v0["pass_all"] and dual_pos_m_v0["pass_all"]
    dual_arms_still_fail = dual_lo["n_pass"] == 0 and dual_li["n_pass"] == 0
    dual_ok = dual_pos_ok and dual_arms_still_fail

    # M22: knife under default is expected; vic0 clearing fully = potential false lock
    # if it becomes 6/6 while default was knife — flag as soft_false_lock_risk (not hard bite)
    m22_new_fail = m22_v0["n_pass"] < m22_v05["n_pass"] and m22_v0["n_pass"] < m22_v0["n"]
    m22_clears_knife = m22_v05["knife"] and m22_v0["pass_all"]
    m22_false_lock_risk = bool(m22_clears_knife)  # soft: documents knife heal
    m23_new_fail = not m23_v0["pass_all"]
    m23_ok = m23_v0["pass_all"] and m23_v05["pass_all"]

    # c1.5: prior full 21/21 + live reconfirm must heal seed0 and not regress others
    prior_close_v0 = (prior or {}).get("close_n1_vic0_s0_20") or {}
    prior_live_v0 = (prior or {}).get("live_n1_vic0_s0_20") or {}
    prior_close_v05 = (prior or {}).get("close_n1_c1.5_s0_20") or {}
    prior_live_v05 = (prior or {}).get("live_n1_c1.5_s0_20") or {}
    prior_c15_21 = (
        (prior_close_v0.get("pass_str") == "21/21")
        and (prior_live_v0.get("pass_str") == "21/21")
    )
    reconfirm_close_v0_ok = close_p_v0["pass_all"] and (0 not in close_p_v0["fail_seeds"])
    reconfirm_live_v0_ok = live_p_v0["pass_all"] and (0 not in live_p_v0["fail_seeds"])
    reconfirm_close_knife = (0 in close_p_v05["fail_seeds"]) or close_p_v05["knife"]
    reconfirm_live_knife = (0 in live_p_v05["fail_seeds"]) or live_p_v05["knife"]
    replicate_c15_close = bool(prior_c15_21 and reconfirm_close_v0_ok)
    replicate_c15_live = bool(prior_c15_21 and reconfirm_live_v0_ok)
    replicate_c15_knife = bool(reconfirm_close_knife and reconfirm_live_knife)

    music_close_healed = close_m_v0["n_pass"] == 21
    music_live_healed = live_m_v0["n_pass"] == 21
    music_close_default_knife = close_m_v05["n_pass"] < 21
    music_live_default_knife = live_m_v05["n_pass"] < 21

    # Adopt criteria (Music posture n=1 only; NOT silent locked default flip):
    # - replicates heal on close/live (c1.5 propose and/or Music c1.0)
    # - leftover flat / no regression
    # - dual-arm positive stays green; arms alone stay fail
    # - hard bites stay fail (no YAML lie fix)
    # - no new fail on M23; M22 soft-clear is OK to note but not auto-reject
    # - reject if dual breaks, leftover regresses, bites false-fix, or heal fails
    heal_ok = (
        (replicate_c15_close and replicate_c15_live)
        or (music_close_healed and music_live_healed)
    )
    music_heal_ok = music_close_healed and music_live_healed

    reject_reasons = []
    if not music_heal_ok:
        reject_reasons.append(
            f"Music c1.0 close/live not fully healed under vic0 "
            f"(close {close_m_v0['pass_str']} live {live_m_v0['pass_str']})"
        )
    if leftover_regression:
        reject_reasons.append("leftover regression under vic0")
    if not dual_ok:
        reject_reasons.append(
            f"dual-arm broken (pos_c15={dual_pos_v0['pass_str']} "
            f"pos_c10={dual_pos_m_v0['pass_str']} "
            f"lo={dual_lo['pass_str']} li={dual_li['pass_str']})"
        )
    if not bites_ok:
        reject_reasons.append("hard bite false-fix or not still failing under vic0")
    if m23_new_fail:
        reject_reasons.append(f"M23 new fail under vic0 ({m23_v0['pass_str']})")
    if m22_new_fail and m22_v0["n_pass"] == 0:
        reject_reasons.append(f"M22 hard new fail under vic0 ({m22_v0['pass_str']})")

    # Soft warnings (do not auto-reject)
    warnings = []
    if m22_false_lock_risk:
        warnings.append(
            f"M22 knife clears under vic0 ({m22_v05['pass_str']}→{m22_v0['pass_str']}) "
            "— soft false-lock risk; document as posture side-effect, not hard bite"
        )
    if not leftover_flat:
        warnings.append(
            f"leftover n12 exam not flat "
            f"({left12_v05['mean_exam']}→{left12_v0['mean_exam']}) though pass may hold"
        )
    if not (replicate_c15_close and replicate_c15_live):
        warnings.append(
            f"c1.5 propose replicate incomplete "
            f"(close {close_p_v0['pass_str']} live {live_p_v0['pass_str']})"
        )

    if reject_reasons:
        recommend = "REJECT"
    elif music_heal_ok and leftover_n1_ok and (not leftover_regression) and dual_ok and bites_ok and m23_ok:
        recommend = "ADOPT"
    else:
        recommend = "REJECT"

    # adopt means: Music-posture harness ONLY — if n_particles<=1: vicreg_weight=0
    # NOT a change to locked AdvConfig defaults
    adopt_scope = (
        "Music-posture scoring harness only: "
        "if n_particles<=1: set vicreg_weight=0. "
        "Do NOT change locked default vicreg_weight=0.05. "
        "Do NOT silence multi-seed gate. Fire #21 n>=2 remains primary harden."
    )

    wall = round(time.time() - t_wall, 1)
    verdict = {
        "recommend": recommend,
        "scope": "Music-posture n=1 only",
        "adopt_scope": adopt_scope if recommend == "ADOPT" else None,
        "reject_reasons": reject_reasons,
        "warnings": warnings,
        "locked_defaults_changed": False,
        "music_train": False,
        "replicate": {
            "c1.5_close_vic0_reconfirm": close_p_v0["pass_str"],
            "c1.5_live_vic0_reconfirm": live_p_v0["pass_str"],
            "c1.5_close_vic0.05_reconfirm": close_p_v05["pass_str"],
            "c1.5_live_vic0.05_reconfirm": live_p_v05["pass_str"],
            "c1.5_prior_vic0_close": prior_close_v0.get("pass_str"),
            "c1.5_prior_vic0_live": prior_live_v0.get("pass_str"),
            "c1.5_prior_vic0.05_close": prior_close_v05.get("pass_str"),
            "c1.5_prior_vic0.05_live": prior_live_v05.get("pass_str"),
            "c1.5_healed_21_21": bool(replicate_c15_close and replicate_c15_live),
            "c1.5_default_still_knife": bool(replicate_c15_knife),
            "c1.5_mode": "prior_21_cite + live_reconfirm_seeds",
            "music_c1.0_close_vic0": close_m_v0["pass_str"],
            "music_c1.0_live_vic0": live_m_v0["pass_str"],
            "music_c1.0_close_vic0.05": close_m_v05["pass_str"],
            "music_c1.0_live_vic0.05": live_m_v05["pass_str"],
            "music_c1.0_healed_21_21": bool(music_heal_ok),
            "music_c1.0_default_knife": bool(music_close_default_knife or music_live_default_knife),
            "prior_cite": {
                "close_vic0": (prior or {}).get("close_n1_vic0_s0_20"),
                "live_vic0": (prior or {}).get("live_n1_vic0_s0_20"),
            },
        },
        "controls": {
            "leftover_n12_vic0.05": left12_v05["pass_str"],
            "leftover_n12_vic0": left12_v0["pass_str"],
            "leftover_n12_mean_exam_vic0.05": left12_v05["mean_exam"],
            "leftover_n12_mean_exam_vic0": left12_v0["mean_exam"],
            "leftover_flat": leftover_flat,
            "leftover_n1_c1.0_vic0.05": left1_v05["pass_str"],
            "leftover_n1_c1.0_vic0": left1_v0["pass_str"],
            "leftover_regression": leftover_regression,
            "dual_pos_c1.5_vic0": dual_pos_v0["pass_str"],
            "dual_pos_c1.0_vic0": dual_pos_m_v0["pass_str"],
            "dual_pos_c1.5_vic0.05": dual_pos_v05["pass_str"],
            "dual_leftover_only_vic0": dual_lo["pass_str"],
            "dual_listen_only_vic0": dual_li["pass_str"],
            "dual_ok": dual_ok,
            "M22_vic0.05": m22_v05["pass_str"],
            "M22_vic0": m22_v0["pass_str"],
            "M22_false_lock_risk": m22_false_lock_risk,
            "M22_new_fail": m22_new_fail,
            "M23_vic0.05": m23_v05["pass_str"],
            "M23_vic0": m23_v0["pass_str"],
            "M23_ok": m23_ok,
        },
        "hard_bites": bite_rows,
        "bites_ok": bites_ok,
        "wall_s": wall,
        "sha": sha,
        "host": "box-cpu",
    }

    # strip heavy runs from cells for JSON size? keep summaries+runs for evidence
    cells_out = {}
    for k, v in cells.items():
        if k == "prior_c15_cite":
            cells_out[k] = v
            continue
        cells_out[k] = {kk: vv for kk, vv in v.items() if kk != "runs"}
        cells_out[k]["runs"] = [
            {
                "seed": r["seed"],
                "pass": r["pass"],
                "exam_score": r["exam_score"],
                "u_kept": r["u_kept"],
                "leak_ratio": r["leak_ratio"],
                "wall_s": r["wall_s"],
            }
            for r in v.get("runs", [])
        ]

    payload = {
        "fire": "vicreg0_n1_propose_stress_20260909",
        "meta": {
            "sha": sha,
            "date": "2026-09-09",
            "locked_defaults_unchanged": True,
            "music_train": False,
            "propose": "vicreg_weight=0 when n_particles==1 (Music posture)",
            "wall_s": wall,
        },
        "verdict": verdict,
        "cells": cells_out,
        "locked_recipe": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "n_particles_max": 12,
            "particle_l2": 0.02,
            "vicreg_weight": 0.05,
            "b_cap": 1.0,
        },
        "music_posture": {
            "n_particles": 1,
            "cover_weight": 1.0,
            "teacher": TEACHER,
            "fm_weight": 0.0,
            "particle_l2": 0.02,
            "b_cap": 1.0,
            "vicreg_weight_default": 0.05,
            "vicreg_weight_propose": 0.0,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # Evidence table markdown
    lines = [
        "# vicreg0 @ n=1 propose stress — 2026-09-09",
        "",
        f"**Host:** box-cpu @ `{sha}`. CPU harness only. **No Music train.**",
        "**Locked defaults UNCHANGED** (vicreg_weight stays 0.05 in AdvConfig).",
        "",
        f"## Recommendation: **{recommend}** (Music-posture n=1 only)",
        "",
    ]
    if recommend == "ADOPT":
        lines += [
            "Adopt as **Music-posture scoring harness** conditional only:",
            "",
            "```python",
            "# Music-posture harness — NOT locked AdvConfig default",
            "if n_particles <= 1:",
            "    cfg = replace(cfg, vicreg_weight=0.0)  # VICReg std ill-posed at n=1",
            "```",
            "",
            adopt_scope,
            "",
        ]
    else:
        lines += [
            "**Reject** for Music-posture n=1 conditional. Reasons:",
            "",
        ]
        for r in reject_reasons:
            lines.append(f"- {r}")
        lines.append("")

    if warnings:
        lines.append("### Warnings (non-blocking if ADOPT)")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Evidence table",
        "",
        "| cell | posture | vic=0.05 | vic=0.0 | note |",
        "|---|---|:---:|:---:|---|",
        f"| close n1 | Music c1.0 | {close_m_v05['pass_str']} fail={close_m_v05['fail_seeds']} | "
        f"**{close_m_v0['pass_str']}** | primary Music knife |",
        f"| close_live n1 | Music c1.0 | {live_m_v05['pass_str']} fail={live_m_v05['fail_seeds']} | "
        f"**{live_m_v0['pass_str']}** | primary Music knife |",
        f"| close n1 | propose c1.5 prior | {prior_close_v05.get('pass_str')} | "
        f"**{prior_close_v0.get('pass_str')}** | prior 21-seed cite |",
        f"| close_live n1 | propose c1.5 prior | {prior_live_v05.get('pass_str')} | "
        f"**{prior_live_v0.get('pass_str')}** | prior 21-seed cite |",
        f"| close n1 | propose c1.5 reconfirm | {close_p_v05['pass_str']} fail={close_p_v05['fail_seeds']} | "
        f"**{close_p_v0['pass_str']}** | seeds {SEEDS_RECONFIRM} |",
        f"| close_live n1 | propose c1.5 reconfirm | {live_p_v05['pass_str']} fail={live_p_v05['fail_seeds']} | "
        f"**{live_p_v0['pass_str']}** | seeds {SEEDS_RECONFIRM} |",
        f"| leftover n12 | locked c1.5 | {left12_v05['pass_str']}@{left12_v05['mean_exam']} | "
        f"{left12_v0['pass_str']}@{left12_v0['mean_exam']} | flat={leftover_flat} |",
        f"| leftover n1 | Music c1.0 | {left1_v05['pass_str']} | {left1_v0['pass_str']} | must stay green |",
        f"| dual-arm positive | c1.5 n1 | {dual_pos_v05['pass_str']} | {dual_pos_v0['pass_str']} | must PASS |",
        f"| dual-arm positive | Music c1.0 | — | {dual_pos_m_v0['pass_str']} | must PASS |",
        f"| dual leftover_only | n1 vic0 | — | {dual_lo['pass_str']} | must FAIL |",
        f"| dual listen_only | n1 vic0 | — | {dual_li['pass_str']} | must FAIL |",
        f"| M22 stagger_mild | Music n1 | {m22_v05['pass_str']} | {m22_v0['pass_str']} | "
        f"false_lock_risk={m22_false_lock_risk} |",
        f"| M23 multipair | Music n1 | {m23_v05['pass_str']} | {m23_v0['pass_str']} | must stay green |",
    ]
    for br in bite_rows:
        lines.append(
            f"| {br['name']} HARD_BITE | Music n1 | {br['vic0.05']} | {br['vic0']} | "
            f"still_fail={br['still_hard_fail_vic0']} false_fix={br['false_fix']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Gate checklist",
        "",
        f"| gate | result |",
        f"|---|---|",
        f"| Replicate c1.5 close/live 21/21 under vic0 | "
        f"{'YES' if replicate_c15_close and replicate_c15_live else 'NO'} "
        f"(prior {prior_close_v0.get('pass_str')}/{prior_live_v0.get('pass_str')}; "
        f"reconfirm {close_p_v0['pass_str']}/{live_p_v0['pass_str']}) |",
        f"| Music c1.0 close/live 21/21 under vic0 | "
        f"{'YES' if music_heal_ok else 'NO'} "
        f"({close_m_v0['pass_str']}/{live_m_v0['pass_str']}) |",
        f"| leftover flat / no regression | "
        f"{'YES' if leftover_flat and not leftover_regression else 'NO'} |",
        f"| dual-arm positive PASS / arms FAIL | {'YES' if dual_ok else 'NO'} |",
        f"| M22 no hard new fail | {'YES' if not (m22_new_fail and m22_v0['n_pass']==0) else 'NO'} "
        f"(soft clear risk={m22_false_lock_risk}) |",
        f"| M23 no new fail | {'YES' if m23_ok else 'NO'} |",
        f"| M20/M21/M24 still FAIL under vic0 | {'YES' if bites_ok else 'NO'} |",
        f"| Locked defaults changed? | **NO** |",
        "",
        "## Relation to Fire #21",
        "",
        "- **n≥2** remains primary harden for close-family (VICReg well-posed).",
        "- **vic=0 @ n=1** is complementary Music parts0 posture (ill-posed VICReg term).",
        "- Multi-seed gate still mandatory under parts0 even if vic0 adopted.",
        "",
        f"Wall: {wall}s. JSON: `{OUT_JSON.name}`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")

    # research log append
    log_block = [
        "",
        "## Note — vicreg0_n1_propose_stress (2026-09-09)",
        "",
        f"- Doc: `vicreg0_n1_propose_stress_20260909.md` / `.json` / `.py`",
        f"- **Recommend: {recommend}** for Music-posture n=1 only (locked defaults unchanged).",
        f"- Replicate c1.5: close {close_p_v05['pass_str']}→{close_p_v0['pass_str']}; "
        f"live {live_p_v05['pass_str']}→{live_p_v0['pass_str']}",
        f"- Music c1.0: close {close_m_v05['pass_str']}→{close_m_v0['pass_str']}; "
        f"live {live_m_v05['pass_str']}→{live_m_v0['pass_str']}",
        f"- leftover n12: {left12_v05['pass_str']}@{left12_v05['mean_exam']} vs "
        f"{left12_v0['pass_str']}@{left12_v0['mean_exam']} flat={leftover_flat}",
        f"- dual_ok={dual_ok}; M22 {m22_v05['pass_str']}→{m22_v0['pass_str']} "
        f"false_lock_risk={m22_false_lock_risk}; M23 {m23_v0['pass_str']}",
        f"- bites_ok={bites_ok}: " + ", ".join(
            f"{br['name']} {br['vic0']}" for br in bite_rows
        ),
        f"- reject_reasons={reject_reasons}; warnings={warnings}",
        f"- wall={wall}s sha={sha}",
        "",
    ]
    if LOG.exists():
        prev = LOG.read_text()
        if "vicreg0_n1_propose_stress" not in prev:
            LOG.write_text(prev.rstrip() + "\n" + "\n".join(log_block))
        else:
            # replace prior note block crudely by append with timestamp wall
            LOG.write_text(prev.rstrip() + "\n" + "\n".join(log_block))
    else:
        LOG.write_text("\n".join(log_block) + "\n")

    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"wrote {OUT_MD}", flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"recommend={recommend} wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
