"""Wide ParticleGAN-family matrix for the unipolar toy gates (UniPG-E).

Fans N arms (1-2 knobs each off the GAN-only unipolar locked shape) over a
fixed budget ladder and seeds, one scoreboard out. Every arm is GAN-only
(Rp logistic + GradRegularizer, cover/FM forced to 0) and propose_only:
nothing here touches the Music bipolar trainer, ``locked_shared``, or the
live ``--lm_target`` default.

Arm identity cards carry:
- ``delta``: exact ``AdvConfig`` overrides vs ``uni_locked``,
- ``vicreg_fn``: ``locked`` (toy sim+var+cov) or ``faithful`` (upstream
  var+cov, std_target 1.0),
- ``pg_tag``: MATCH (Rp + GradRegularizer family, GAN-only) or DRIFT,
- ``card``: one-line human identity.

Run: ``PYTHONPATH=. python3 analysis/slider2d/run_unipolar_gan_sweep.py
--out docs/unipolar-gan-sweep --workers 4`` (CPU only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.exam import close_field, divergent_field
from analysis.slider2d.unipolar_gan import (
    PROPOSE_ONLY,
    UNIPOLAR_EVAL_SCALES,
    score_uni_adv,
    uni_cfg,
)

assert PROPOSE_ONLY is True

BUDGETS = (600, 1200, 2400, 3400)
SEEDS = (0, 1, 7)
CELLS = ("divergent", "close")

CELL_CTOR = {"divergent": divergent_field, "close": close_field}

# arm name -> {delta, vicreg_fn, pg_tag, card}
# deltas are AdvConfig overrides vs uni_locked (which itself is AdvConfig()
# defaults with cover_weight=0 + fm_weight=0). vicreg_fn in {"locked",
# "faithful"} selects the particle-regularizer transcription.
ARMS: dict[str, dict] = {
    # -- baseline: locked #94 shape, supervision off (production-analogue) --
    "uni_locked": {
        "delta": {},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "locked b_cap k=1 coeff=1, n=12, vic=0.05, pl2=0.02, LR 5e-3 shared; GAN-only baseline",
    },
    # -- GradRegularizer axis --
    "k05": {
        "delta": {"kappa": 0.5},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "tighter cap k=0.5 (D flatter, G signal weaker?)",
    },
    "k2": {
        "delta": {"kappa": 2.0},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "looser cap k=2 (D may stay steep longer)",
    },
    "coeff05": {
        "delta": {"b_cap": 0.5},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "half-strength cap coeff=0.5",
    },
    "coeff2": {
        "delta": {"b_cap": 2.0},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "double-strength cap coeff=2",
    },
    "ginterp": {
        "delta": {"grad_arm": "g_interp_cap"},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "interp-path cap (high-dim ParticleGAN geometry)",
    },
    "delayed": {
        "delta": {"target_anneal": "delayed"},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "delayed center anneal (hold k then slide to R1/R2-like)",
    },
    "lazy2": {
        "delta": {"grad_lazy": 2},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "lazy reg every 2nd step (StyleGAN2-style, coeff x2)",
    },
    # -- particles / VICReg axis --
    "slowprior": {
        "delta": {"prior_lr_mult": 0.1},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "slow prior 0.1x (residual must carry the mode)",
    },
    "frozenprior": {
        "delta": {"prior_lr_mult": 0.01, "vicreg_weight": 0.0},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "near-frozen prior 0.01x + VICReg off (pure jitter prior)",
    },
    "n32": {
        "delta": {"n_particles": 32},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "big particles n=32 (toward ParticleGAN scale)",
    },
    "vic0": {
        "delta": {"vicreg_weight": 0.0},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "VICReg off (particle spread unconstrained)",
    },
    "vicfaithful": {
        "delta": {"vicreg_weight": 1.0},
        "vicreg_fn": "faithful",
        "pg_tag": "MATCH",
        "card": "upstream VICReg (var+cov, std_target=1.0, weight=1.0)",
    },
    "pl2off": {
        "delta": {"particle_l2": 0.0},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "particle L2 off (particles free to roam)",
    },
    # -- LR / EMA / critic axis --
    "d15": {
        "delta": {"d_lr_mult": 1.5},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "ParticleGAN D mult x1.5",
    },
    "lr2x": {
        "delta": {"lr": 1.0e-2, "d_lr_mult": 1.5},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "#116 pg_2x_lr clone (2x base + D 1.5x, prior HOLD 1.0)",
    },
    "beta999": {
        "delta": {"beta2": 0.999},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "Adam beta2 0.999 (upstream momentum)",
    },
    "emaoff": {
        "delta": {"ema": 0.0},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "EMA off (raw last-step residual)",
    },
    "critic128": {
        "delta": {"critic_hidden": 128},
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "thicker critic hidden=128",
    },
    # -- #116 full-clone preset (thin, existing knobs only) --
    "fullclone": {
        "delta": {
            "beta2": 0.999,
            "n_particles": 32,
            "vicreg_weight": 1.0,
            "particle_l2": 0.0,
        },
        "vicreg_fn": "faithful",
        "pg_tag": "MATCH",
        "card": "#116 pg_full_clone thin preset (b2 .999, n=32, vic=1 faithful, pl2=0)",
    },
    # -- early-pass hunt winners (frozen jitter prior x cap geometry) --
    # Shared particle treatment (prior 0.01x, VICReg off, L2 off: particles
    # stay a pure jitter prior); the arms differ only in cap geometry.
    "frozen_k2": {
        "delta": {
            "prior_lr_mult": 0.01,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
            "kappa": 2.0,
        },
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "HUNT: frozen jitter prior + loose cap k=2",
    },
    "frozen_gi": {
        "delta": {
            "prior_lr_mult": 0.01,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
            "grad_arm": "g_interp_cap",
        },
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "HUNT: frozen jitter prior + interp-path cap",
    },
    "frozen_c05": {
        "delta": {
            "prior_lr_mult": 0.01,
            "vicreg_weight": 0.0,
            "particle_l2": 0.0,
            "b_cap": 0.5,
        },
        "vicreg_fn": "locked",
        "pg_tag": "MATCH",
        "card": "HUNT: frozen jitter prior + half-strength cap",
    },
}


def _resolve_vicreg_fn(name: str):
    if name == "locked":
        return None
    if name == "faithful":
        from analysis.slider2d.pg_clone_propose import vicreg_faithful_loss

        return vicreg_faithful_loss
    raise ValueError(f"unknown vicreg_fn {name!r}")


def run_one(job: dict) -> dict:
    """One (arm, budget, seed, cell) run. Top-level for multiprocessing."""
    import torch

    torch.set_num_threads(1)
    arm = ARMS[job["arm"]]
    cfg = uni_cfg(steps=job["steps"], seed=job["seed"], **arm["delta"])
    field = CELL_CTOR[job["cell"]](seed=job["seed"])
    row = score_uni_adv(
        field, cfg=cfg, vicreg_fn=_resolve_vicreg_fn(arm["vicreg_fn"]), name=job["arm"]
    )
    return {
        "arm": job["arm"],
        "steps": job["steps"],
        "seed": job["seed"],
        "cell": job["cell"],
        "cover": row["cover"],
        "off_caption": row["off_caption"],
        "neu_hold": row["neu_hold"],
        "overlap_pos": row["overlap_pos"],
        "blend_toward_mid": row["blend_toward_mid"],
        "half_cover": row["half_cover"],
        "hit": bool(row["hit"]),
        "pg_tag": arm["pg_tag"],
        "d_loss": row.get("d_loss"),
        "g_loss": row.get("g_loss"),
        "canary_landed": row["canary"]["minus_landed"],
        "canary_off": row["canary"]["minus_off_caption"],
        "canary_dangerous": bool(row["canary"]["dangerous"]),
    }


def run_grid(*, workers: int = 4) -> list[dict]:
    jobs = [
        {"arm": arm, "steps": steps, "seed": seed, "cell": cell}
        for arm in ARMS
        for steps in BUDGETS
        for seed in SEEDS
        for cell in CELLS
    ]
    t0 = time.time()
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(run_one, jobs))
    else:
        rows = [run_one(j) for j in jobs]
    dt = time.time() - t0
    print(f"sweep: {len(rows)} runs in {dt:.0f}s ({dt / max(1, len(rows)):.2f}s/run)", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Per (arm, budget): hits across seeds x cells; PASS = 6/6."""
    summary = {}
    for arm in ARMS:
        per_budget = {}
        for steps in BUDGETS:
            sub = [r for r in rows if r["arm"] == arm and r["steps"] == steps]
            hits = sum(1 for r in sub if r["hit"])
            cov = {c: [r for r in sub if r["cell"] == c] for c in CELLS}
            per_budget[str(steps)] = {
                "hits": f"{hits}/6",
                "pass": bool(hits == 6),
                "mean_cover": sum(r["cover"] for r in sub) / len(sub),
                "max_off": max(r["off_caption"] for r in sub),
                "min_hold": min(r["neu_hold"] for r in sub),
                "by_cell_seed": {
                    f"{r['cell']}/s{r['seed']}": {
                        "cover": round(r["cover"], 3),
                        "off": round(r["off_caption"], 3),
                        "hit": r["hit"],
                    }
                    for r in sorted(sub, key=lambda r: (r["cell"], r["seed"]))
                },
                "seeds": sorted({r["seed"] for r in sub}),
            }
        passes = [s for s, b in per_budget.items() if b["pass"]]
        summary[arm] = {
            "card": ARMS[arm]["card"],
            "delta": ARMS[arm]["delta"],
            "vicreg_fn": ARMS[arm]["vicreg_fn"],
            "pg_tag": ARMS[arm]["pg_tag"],
            "budgets": per_budget,
            "pass_budgets": passes,
            "earliest_pass": min((int(s) for s in passes), default=None),
        }
    return summary


def write_scoreboard(rows: list[dict], out: Path) -> dict:
    summary = summarize(rows)
    blob = {
        "polarity": "uni",
        "eval_scales": list(UNIPOLAR_EVAL_SCALES),
        "gates": {"cover_min": 0.85, "off_caption_max": 0.05, "neu_hold_min": 0.85},
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "loss": "Rp logistic (paired), GAN-only: no MSE/cover/FM/lyric-hold/plan/anchor on G",
        "teacher": "raw_positive (pos pole; lm_faithful_plus_neu contract for +)",
        "scale_zero": "exact base by construction (delta(0)==0)",
        "canary_note": "scale -1 logged only, never a gate; antipodal cos never consulted",
        "propose_only": True,
        "merge_to_trainer": False,
        "live_default_unchanged": True,
        "music_bipolar_untouched": True,
        "arms": summary,
        "runs": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "scoreboard.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    (out.parent / (out.name + ".md")).write_text(render_markdown(blob) + "\n", encoding="utf-8")
    return blob


def render_markdown(blob: dict) -> str:
    L = [
        "# Unipolar ParticleGAN toy sweep (UniPG-E, GAN-only)",
        "",
        "CPU harness: zero-at-zero residual, paired Rp logistic + GradRegularizer,",
        "raw-positive teacher cloud only. **No supervised MSE / cover / FM on G.**",
        "Scale 0 is exact base by construction; scale -1 is a canary (never a gate);",
        "antipodal cos is never consulted. All arms `propose_only` — Music bipolar",
        "(`ARM_B` / `locked_shared` / live `--lm_target`) untouched.",
        "",
        "Gates (both cells `divergent` + `close`): cover@+1 >= 0.85, leak@+1 <= 0.05,",
        "neu_hold@0 >= 0.85. PASS = 6/6 (2 cells x seeds 0/1/7) at that budget.",
        "",
        "## Budget ladder",
        "",
        "| arm | 600 | 1200 | 2400 | 3400 | earliest |",
        "|---|---|---|---|---|---|",
    ]
    for arm, s in blob["arms"].items():
        cells = []
        for b in ("600", "1200", "2400", "3400"):
            bb = s["budgets"][b]
            cells.append(f"**PASS** ({bb['hits']})" if bb["pass"] else bb["hits"])
        L.append(
            f"| `{arm}` | {' | '.join(cells)} | {s['earliest_pass'] or '—'} |"
        )
    L += ["", "## Arm cards", ""]
    for arm, s in blob["arms"].items():
        L.append(
            f"- `{arm}` [{s['pg_tag']}] vicreg_fn={s['vicreg_fn']} delta={json.dumps(s['delta'], sort_keys=True)} — {s['card']}"
        )
    L += ["", "## Per-budget detail (cover/off per cell/seed)", ""]
    for arm, s in blob["arms"].items():
        L.append(f"### `{arm}` — {s['card']}")
        L.append("")
        L.append("| budget | cell/seed | cover | off | hit |")
        L.append("|---|---|---|---|---|")
        for b in ("600", "1200", "2400", "3400"):
            for k, v in s["budgets"][b]["by_cell_seed"].items():
                L.append(
                    f"| {b} | {k} | {v['cover']:.3f} | {v['off']:.3f} | {'HIT' if v['hit'] else '—'} |"
                )
        L.append("")
    L += ["", "## Read", ""]
    winners = [(a, s["earliest_pass"]) for a, s in blob["arms"].items()
               if s["earliest_pass"] is not None]
    winners.sort(key=lambda kv: (kv[1], kv[0]))
    if winners:
        best = ", ".join(f"`{a}`@{b}" for a, b in winners)
        L.append(f"- PASS arms (earliest budget): {best}.")
    else:
        L.append("- No arm passes 6/6 at any budget yet; closest arms first:")
        closest = sorted(
            blob["arms"].items(),
            key=lambda kv: (-max(
                int(h.split("/")[0]) for h in
                [v["hits"] for v in kv[1]["budgets"].values()]), kv[0]),
        )
        for arm, s in closest[:3]:
            best_b = max(s["budgets"].items(), key=lambda kv: kv[1]["hits"])
            L.append(f"  - `{arm}`: best {best_b[1]['hits']} at {best_b[0]}.")
    L += [
        "- Why most arms fail at low budgets: live particles eat the pole modes",
        "  (the fake cloud covers the teacher even while the residual undershoots);",
        "  slowing the prior forces the residual to carry +1. Seed-1 divergent can",
        "  additionally need a looser cap (k=2 / interp geometry) or D stalls G.",
        "- No supervised MSE on any G: `cover_weight=0`, `fm_weight=0` (fail-closed).",
        "- Music bipolar untouched: no trainer row, no default, no `--lm_target` flip.",
        "",
        f"Scales: {blob['eval_scales']} (gates on 0 and 1; 0.5 diagnostic; -1 canary).",
    ]
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_REPO / "docs" / "unipolar-gan-sweep")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--arms", type=str, default="",
                   help="comma-separated subset of arms (default: all)")
    p.add_argument("--budgets", type=str, default="",
                   help="comma-separated subset of budgets (default: all)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    global ARMS, BUDGETS
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in want if a not in ARMS]
        if bad:
            raise ValueError(f"unknown arms: {bad} (have {sorted(ARMS)})")
        ARMS = {a: ARMS[a] for a in want}
    if args.budgets:
        BUDGETS = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    rows = run_grid(workers=args.workers)
    blob = write_scoreboard(rows, args.out)
    npass = sum(1 for s in blob["arms"].values() if s["pass_budgets"])
    print(f"arms with any PASS budget: {npass}/{len(blob['arms'])}", flush=True)
    for arm, s in blob["arms"].items():
        if s["pass_budgets"]:
            print(f"  PASS {arm}: budgets {s['pass_budgets']} earliest {s['earliest_pass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
