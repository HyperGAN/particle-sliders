"""Propose-only clone arms 4/5: anneal + g_interp_cap on the 2-D toy.

ParticleGAN-faithful directions the locked demo does NOT use:

- arm 4 (anneal): ``target_anneal='delayed'`` (hold the penalty center at
  c0 for 60% of the run, then ramp linearly to 0) or ``'linear'``. As the
  center slides to 0 the re-centered cap morphs into a zero-centered
  R1/R2-like penalty: a smooth handover from "D may not be flat" to "D
  should be flat", without switching arms mid-run.
- arm 5 (geometry): ``grad_arm='g_interp_cap'`` — the one-sided cap of
  ``b_cap``, enforced on real/fake interpolates instead of at the samples.
  In high-dimensional data the sample-point cap leaves D free to be
  arbitrarily steep *between* reals and fakes (where the fakes travel), so
  the interp cap is the cap's natural high-dim form.

``PROPOSE_ONLY = True``: nothing here changes a locked default. The demo
(``analysis/slider2d/run_lm_adv.py``), ``AdvConfig``, the Music trainer
argv (``ARM_B``), and the YuE2 locked recipe all stay sample-point
``b_cap`` / anneal ``none``. These cards are ablated *against* the locked
reference on CPU toy fixtures only — no Music weights, no audio, no GPU.

CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.run_lm_adv import DEFAULT_TEACHER, collect

PROPOSE_ONLY = True

# The locked reference is row 0 (not a proposal). Arms 4/5 are the rest.
# One clear winning card is preferred over a wide sweep; the table is tiny
# on purpose: delayed anneal x {sample cap, interp cap}.
CARDS: tuple[dict, ...] = (
    {"name": "locked_b_cap_none", "grad_arm": "b_cap", "target_anneal": "none"},
    {"name": "a4_bcap_delayed", "grad_arm": "b_cap", "target_anneal": "delayed"},
    {"name": "a5_ginterp_none", "grad_arm": "g_interp_cap", "target_anneal": "none"},
    {"name": "a45_ginterp_delayed", "grad_arm": "g_interp_cap", "target_anneal": "delayed"},
)

VERDICTS = ("KEEP", "HOLD", "DROP")


def _fmt(value, spec: str, empty: str = "N/A") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def _ok(value: bool | None) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "—"


def run_card(
    *,
    steps: int,
    exam_steps: int,
    seed: int,
    teacher: str,
    b_cap: float,
    kappa: float,
    fm_weight: float,
    baseline_steps: int,
    cover_weight: float,
    grad_arm: str,
    target_anneal: str,
) -> dict:
    """One propose card through the locked harness (same cells, same gates)."""
    return collect(
        steps=steps,
        exam_steps=exam_steps,
        seed=seed,
        teacher=teacher,
        b_cap=b_cap,
        kappa=kappa,
        fm_weight=fm_weight,
        baseline_steps=baseline_steps,
        cover_weight=cover_weight,
        grad_arm=grad_arm,
        target_anneal=target_anneal,
    )


def _slim(row: dict) -> dict:
    skip = {"log", "axis", "sings", "says"}
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def write_report(cards: list[dict], path: Path, verdict: str, winner: str) -> None:
    lines = [
        "# Clone arms 4/5 (propose-only): anneal + g_interp_cap vs locked",
        "",
        "CPU toy ablation only — leftover-gated caption poles plus a span/end",
        "cloud (hidden-state deltas, **not** rendered audio). This page does not",
        "claim Music 3 listen quality. The locked demo stays sample-point",
        "`b_cap` / anneal `none`; the Music trainer argv (`ARM_B`) and the YuE2",
        "locked recipe are untouched.",
        "",
        "## Cards",
        "",
        "- `locked_b_cap_none`: locked reference (not a proposal).",
        "- `a4_bcap_delayed`: arm 4 — sample-point cap + ParticleGAN `delayed`",
        "  anneal (center held 60%, then linear to 0).",
        "- `a5_ginterp_none`: arm 5 — interp-path cap (`g_interp_cap`), no anneal.",
        "- `a45_ginterp_delayed`: arms 4+5 — interp-path cap + delayed anneal",
        "  (closest to the ParticleGAN high-dim form with schedule).",
        "",
        "## Ablation table (same cells, same gates as the locked harness)",
        "",
        "| card | compiled | exam_score | field2d | sheet leftover | sheet gender | exam div/close/unused |",
        "|---|---|---|---|---|---|---|",
    ]
    for blob, card in cards:
        cells = blob["cells"]
        f2 = blob["field2d"]
        left = blob["sheet_leftover"]
        gender = blob["sheet_gender"]
        div = blob["exam_divergent"]
        close = blob["exam_close"]
        unused = blob["exam_unused_e"]
        lines.append(
            f"| `{card['name']}` ({card['grad_arm']}/{card['target_anneal']}) "
            f"| **{blob['compiled']}** "
            f"| {_fmt(blob.get('exam_score'), '.3f')} "
            f"| {_ok(f2.get('pass'))} "
            f"(slider {_fmt(f2.get('cos_slider_plus'), '+.3f')}, leak {_fmt(f2.get('leak_ratio'), '+.3f')}) "
            f"| {_ok(cells['sheet_leftover'])} "
            f"(leak {_fmt(left.get('leak_tok'), '+.3f')}, kept {_fmt(left.get('on_sheet_kept'), '.3f')}) "
            f"| {_ok(cells['sheet_gender'])} "
            f"(kept {_fmt(gender.get('on_sheet_kept'), '.3f')}) "
            f"| {_ok(cells['exam_divergent'])}/{_ok(cells['exam_close'])}/{_ok(cells['exam_unused_e'])} "
            f"(swing {_fmt(div.get('roll_swing_kept'), '.3f')}/{_fmt(close.get('roll_swing_kept'), '.3f')}) |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}** — winning card: `{winner}`.",
        "",
        "- KEEP: a propose card beats or ties locked on every cell with no new",
        "  failure mode (candidate to graduate toward a gated follow-up).",
        "- HOLD: mixed vs locked (wins some cells, loses or ties others) — keep",
        "  propose-only, needs a follow-up question answered first.",
        "- DROP: loses to locked outright — do not pursue.",
        "",
        "## Honesty notes",
        "",
        "- The `field2d` column FAILs identically for all four cards *including*",
        "  the locked reference: the harness runs the gated teacher on unpinned",
        "  pairs there (`with_attrs=False`), and unpinned poles copy even",
        "  leftover (leak ~1.3, predicted by `train_lm_adv`'s own docstring).",
        "  The pinned-pairs field2d gate still passes and is pinned by",
        "  `tests/test_lm_2d_adv.py::test_field2d_tracks_the_slider_without_gender_leak`;",
        "  the propose-only CPU tests re-check finiteness + reporting (not the",
        "  gate) on this config so the ablation comparison stays fair.",
        "- `linear` anneal is wired through `make_grad_regularizer` and",
        "  unit-tested (`tests/test_lm_adv_anneal.py`); the table uses `delayed`",
        "  because the 60%-hold schedule is the ParticleGAN-faithful one.",
        "- Honesty: toy hidden-state deltas only. No MiniMax weights, no audio",
        "  render, no GPU. Live default stays `--lm_target v9`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def judge(cards: list[dict]) -> tuple[str, str]:
    """KEEP/HOLD/DROP from the ablation table vs the locked reference."""
    ref = cards[0][0]
    ref_cells = ref["cells"]
    ref_score = ref.get("exam_score")
    best, best_key = None, None
    for blob, card in cards[1:]:
        cells = blob["cells"]
        wins = sum(1 for k in ref_cells if cells.get(k) and not ref_cells.get(k))
        losses = sum(1 for k in ref_cells if ref_cells.get(k) and not cells.get(k))
        score = blob.get("exam_score")
        gain = (score or 0.0) - (ref_score or 0.0)
        key = (wins - losses, gain)
        if best_key is None or key > best_key:
            best_key, best = key, (blob, card, wins, losses, gain)
    assert best is not None
    _blob, card, wins, losses, _gain = best
    if wins == 0 and losses == 0:
        # Exact tie vs locked on every compiled cell: nothing won, nothing
        # lost. Report the tie plainly instead of crowning an arbitrary card.
        return "HOLD", "none (tie vs locked)"
    if losses == 0 and wins > 0:
        return "KEEP", card["name"]
    if losses > 0 and wins > 0:
        return "HOLD", card["name"]
    if losses == 0:
        return "HOLD", card["name"]
    return "DROP", card["name"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO / "analysis" / "slider2d" / "anneal_ginterp_smoke.md")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--exam-steps", type=int, default=1200)
    parser.add_argument("--baseline-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--b-cap", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--fm-weight", type=float, default=0.0)
    parser.add_argument("--cover-weight", type=float, default=1.5)
    args = parser.parse_args(argv)

    cards: list[dict] = []
    for card in CARDS:
        blob = run_card(
            steps=args.steps,
            exam_steps=args.exam_steps,
            seed=args.seed,
            teacher=args.teacher,
            b_cap=args.b_cap,
            kappa=args.kappa,
            fm_weight=args.fm_weight,
            baseline_steps=args.baseline_steps,
            cover_weight=args.cover_weight,
            grad_arm=card["grad_arm"],
            target_anneal=card["target_anneal"],
        )
        cards.append((blob, card))
        print(
            f"{card['name']:22s} {blob['compiled']:20s} "
            f"exam={_fmt(blob.get('exam_score'), '.3f')} "
            f"f2d={_ok(blob['field2d'].get('pass'))} "
            f"left={_ok(blob['cells']['sheet_leftover'])} "
            f"div={_ok(blob['cells']['exam_divergent'])} "
            f"close={_ok(blob['cells']['exam_close'])}"
        )
    verdict, winner = judge(cards)
    print(f"verdict={verdict} winner={winner}")
    write_report(cards, args.out, verdict, winner)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        slim_cards = [
            {"card": card, "cfg": blob["cfg"], "cells": blob["cells"],
             "exam_score": blob.get("exam_score"), "compiled": blob["compiled"],
             "field2d": _slim(blob["field2d"]),
             "sheet_leftover": _slim(blob["sheet_leftover"]),
             "sheet_gender": _slim(blob["sheet_gender"]),
             "exam_divergent": _slim(blob["exam_divergent"]),
             "exam_close": _slim(blob["exam_close"]),
             "exam_unused_e": _slim(blob["exam_unused_e"])}
            for blob, card in cards
        ]
        args.metrics_out.write_text(
            json.dumps(
                {"propose_only": True, "verdict": verdict, "winner": winner,
                 "live_default_unchanged": True, "claims_music3_audio": False,
                 "cards": slim_cards},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
