"""Formulation leaderboard: bipolar and unipolar boards side by side.

Fire #90's ``FORMULATION_LEADERBOARD`` / ``cpu_alg_select`` harness is
box-local (absent on main), so this module ships the minimal harness that
can emit BOTH boards on main, reusing the scored surfaces that already
exist here instead of reinventing them:

- bipolar (``polarity="bi"``) reuses the pair-exam cell
  (``analysis.slider2d.exam``): both-pole continuation gates over
  ``delta(+1)`` and ``delta(-1)``, plus leftover leak where the field has
  an unused axis. Antipodal ``cos(+1, -1)`` and pair-odd cos are
  **logged, never gated** — the #22 result: the midpoint teacher locks
  ``cos(d+, a) = +1`` / ``cos(d+, d-) = -1`` while walking off the sheet,
  and the caption winners that pass both exam pairs measure
  ``collapse ~= 0``. Gating the bipolar hit on antipodal collapse would
  crown the known fail, so the hit is the audible continuation on both
  poles.
- unipolar (``polarity="uni"``) reuses the plus+neu exam
  (``analysis.slider2d.plus_neu_exam``): plus cover / plus off-caption
  (the plus leak) at scale 1, ``neu_hold`` at scale 0, and a half-scale
  (0.5) diagnostic. ``-1`` is an unscored canary. Antipodal collapse is
  never consulted.

Eval scales: unipolar reads ``0, 0.5, 1`` (gates on 0 and 1; 0.5 is a
logged interpolation diagnostic). Bipolar reads ``-1, +1`` (both gated
through the exam continuation). ``-1`` on the unipolar board is
diagnostic only.

This module does not change the live trainer default (``--lm_target v9``
/ ``--pole_mode hidden``), the ``ARM_B`` row, or the compiled bipolar
board (``analysis/slider2d/scoreboard.py`` — untouched; see
``docs/lm-2d-scoreboard.md``). No GPU, no Hub, no Music 3 weights.

CPU only.
"""

from __future__ import annotations

from analysis.slider2d.exam import (
    close_field,
    divergent_field,
    unused_e_field,
    exam_table,
)
from analysis.slider2d.plus_exam import (
    PLUS_COVER_MIN,
    PLUS_OFF_MAX,
    _continue,
    _off_share,
    _token_share,
    blend_toward_mid,
    plus_bags,
    plus_cover,
)
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_HOLD_MIN,
    PLUS_NEU_RECIPES,
    drift_from_neu,
    fit_plus_neu_exam,
    neu_bags,
    neu_hold,
    plus_neu_exam_table,
    plus_neu_rank,
)

POLARITY_UNI = "uni"
POLARITY_BI = "bi"

# Unipolar eval scales. Gates read 0 and 1; 0.5 is a logged diagnostic.
UNIPOLAR_EVAL_SCALES = (0.0, 0.5, 1.0)
# Bipolar eval scales. Both poles are gated through the exam continuation.
BIPOLAR_EVAL_SCALES = (-1.0, 1.0)

# Cells both boards are read on. ``unused_e`` is logged, never ranked.
REQUIRED_CELLS = ("divergent", "close")
ALL_CELLS = ("divergent", "close", "unused_e")

# Bipolar formulation arms: must exist in every pair-exam cell.
BI_FORMULATION_ARMS = (
    "faithful_raw",
    "faithful_sub_e_if_unused",
    "faithful_guard_e",
    "pair_odd_midpoint",
    "semantic_kl_null",
)

# Shared explainer rendered into both markdown boards.
POLARITY_NOTE = (
    "Bipolar vs unipolar: the bipolar board asks whether one residual "
    "serves a +/- pair — delta(+1) must sing the + pole AND delta(-1) "
    "must sing the - pole, on the same weights. Read it when the fader "
    "has two ends (a signed concept axis). The unipolar board asks "
    "whether scale +1 covers the + caption without leaking off-caption "
    "and whether scale 0 holds the neutral — delta(-1) is an unscored "
    "canary there. Read it when only the + end is trained (plus-only / "
    "UNI formulations). A method can top one board and fail the other: "
    "that split is the point, not a contradiction. Antipodal "
    "cos(+1, -1) is logged on both boards and gated on neither — "
    "perfect antipodal lock coexists with walking off the sheet (#22), "
    "so the gates stay audible (continuation / cover / hold)."
)


def _train_label(row: dict) -> str:
    if row.get("plus_neu"):
        return "plus+neu"
    if row.get("plus_only"):
        return "plus-only"
    return "bipolar ±"


def unipolar_hit(row: dict) -> bool:
    """Unipolar honesty gate: cover, plus leak, neu hold.

    Antipodal collapse, minus overlap and leak_frac are not inputs —
    passing a row with ``collapse`` set to anything (or missing) must
    not change the verdict.
    """
    return bool(
        float(row["cover"]) >= PLUS_COVER_MIN
        and float(row["off_caption"]) <= PLUS_OFF_MAX
        and float(row["neu_hold"]) >= PLUS_NEU_HOLD_MIN
    )


def bipolar_hit(row: dict) -> bool:
    """Bipolar honesty gate: the exam continuation passes on this cell.

    That already scores BOTH poles (delta(+1) against the + teacher
    rollout and delta(-1) against the - teacher rollout) plus leftover
    leak where the field has an unused axis. ``collapse`` (antipodal)
    and ``pair_odd_cos`` are accepted nowhere here — logged, never
    gated.
    """
    return bool(row["pass"])


def half_scale_diagnostic(
    *,
    recipe: dict,
    cell: str,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Score the fitted student at the 0.5 fader position (diagnostic only).

    Refits the same recipe (same capacity, same loss) and reads scale
    0.5 on the plus side: overlap with the + caption bag and blend
    toward mid. No gate reads these numbers; they show whether the
    halfway fader interpolates instead of jumping or collapsing.
    """
    ctor = {"divergent": divergent_field, "close": close_field,
            "unused_e": unused_e_field}[cell]
    field = ctor(seed=seed)
    residual = fit_plus_neu_exam(
        field,
        teacher=recipe["teacher"],
        leak_dir=field.declared_e(),
        even_scale=float(recipe.get("even_scale", 1.0)),
        plus_only=bool(recipe.get("plus_only", False)),
        plus_neu=bool(recipe.get("plus_neu", False)),
        steps=steps,
        seed=seed,
    )
    bags = plus_bags(field)
    half = None
    for prow in range(int(field.rows)):
        pos, neg, neu = field.poles(prow)
        mid = 0.5 * (pos + neg)
        student = neu + residual.delta(0.5)
        seqs = _continue(field, student, row=prow, sign=1.0)
        overlap = _token_share(seqs, bags["pos"])
        blend = blend_toward_mid(student, pos, mid, neg)
        cover = plus_cover(overlap, blend)
        half = cover if half is None else half + cover
    half_cover = (half / float(field.rows)) if field.rows else 0.0
    return {
        "scored": False,
        "scale": 0.5,
        "half_cover": float(half_cover),
    }


def collect_unipolar_board(*, steps: int = 400, seed: int = 0) -> dict:
    """Unipolar board: plus cover / plus leak / neu hold, polarity ``uni``."""
    table = plus_neu_exam_table(steps=steps, seed=seed)
    by_recipe = {c["name"]: c for c in PLUS_NEU_RECIPES}
    cells: dict[str, list[dict]] = {}
    for cell, rows in table.items():
        out = []
        for row in rows:
            half = half_scale_diagnostic(
                recipe=by_recipe[row["name"]], cell=cell,
                steps=steps, seed=seed,
            )
            hit = unipolar_hit(row)
            out.append(
                {
                    "name": row["name"],
                    "cell": cell,
                    "polarity": POLARITY_UNI,
                    "train": _train_label(row),
                    "cover": float(row["cover"]),
                    "off_caption": float(row["off_caption"]),
                    "neu_hold": float(row["neu_hold"]),
                    "overlap_pos": float(row["overlap_pos"]),
                    "overlap_neu": float(row["overlap_neu"]),
                    "hit": hit,
                    "eval_scales": list(UNIPOLAR_EVAL_SCALES),
                    "half_scale": half,
                    "canary": {
                        "scored": False,
                        "minus_landed": row["canary"]["minus_landed"],
                        "minus_overlap_neg": float(
                            row["canary"]["minus_overlap_neg"]),
                        "minus_off_caption": float(
                            row["canary"]["minus_off_caption"]),
                        "dangerous": bool(row["canary"]["dangerous"]),
                    },
                    "antipodal_consulted": False,
                }
            )
        cells[cell] = out
    rank = plus_neu_rank(table)
    for entry in rank:
        entry["polarity"] = POLARITY_UNI
    return {
        "polarity": POLARITY_UNI,
        "eval_scales": list(UNIPOLAR_EVAL_SCALES),
        "scored": ["cover", "off_caption", "neu_hold"],
        "not_scored": [
            "minus_continuation",
            "antipodal_collapse",
            "pair_odd_cos",
            "leak_frac",
            "exam_score",
        ],
        "minus_is_diagnostic_only": True,
        "gates": {
            "cover_min": PLUS_COVER_MIN,
            "off_caption_max": PLUS_OFF_MAX,
            "neu_hold_min": PLUS_NEU_HOLD_MIN,
        },
        "cells": cells,
        "rank": rank,
        "steps": steps,
        "seed": seed,
    }


def _bi_score(cell_rows: dict[str, dict]) -> float | None:
    """Sortable bipolar number: min(overlap, swing) on the required cells."""
    scores: list[float] = []
    for cell in REQUIRED_CELLS:
        row = cell_rows.get(cell)
        if row is None:
            continue
        for key in ("roll_overlap", "roll_swing_kept"):
            value = row.get(key)
            if value is not None:
                scores.append(float(value))
    if not scores:
        return None
    return min(scores)


def collect_bipolar_board(*, steps: int = 400, seed: int = 0) -> dict:
    """Bipolar board: both-pole continuation gates, polarity ``bi``."""
    table = exam_table(steps=steps, seed=seed)
    cells: dict[str, list[dict]] = {}
    for cell, rows in table.items():
        out = []
        for row in rows:
            if row["name"] not in BI_FORMULATION_ARMS:
                continue
            out.append(
                {
                    "name": row["name"],
                    "cell": cell,
                    "polarity": POLARITY_BI,
                    "teacher": row.get("teacher"),
                    "pass": bool(row["pass"]),
                    "hit": bipolar_hit(row),
                    "roll_overlap": row.get("roll_overlap"),
                    "roll_swing_kept": row.get("roll_swing_kept"),
                    "roll_off_corpus": row.get("roll_off_corpus"),
                    "roll_coherence": row.get("roll_coherence"),
                    "leak_tok": row.get("leak_tok"),
                    # delta(+1) / delta(-1) readouts: what each fader end
                    # sings (first draw), plus the antipodal diagnostic.
                    "sings_both_poles": row.get("sings"),
                    "antipodal_collapse": row.get("collapse"),
                    "antipodal_consulted": False,
                    "pair_odd_cos_logged": row.get("pair_odd_cos"),
                    "reason": row.get("reason"),
                }
            )
        cells[cell] = out
    by_name: dict[str, dict[str, dict]] = {}
    for cell, rows in cells.items():
        for row in rows:
            by_name.setdefault(row["name"], {})[cell] = row
    rank = []
    for name, per_cell in by_name.items():
        hits = [per_cell[c]["hit"] for c in REQUIRED_CELLS if c in per_cell]
        rank.append(
            {
                "name": name,
                "polarity": POLARITY_BI,
                "hit_divergent": per_cell.get("divergent", {}).get("hit"),
                "hit_close": per_cell.get("close", {}).get("hit"),
                "hits_required": all(hits) if hits else False,
                "bi_score": _bi_score(per_cell),
            }
        )
    rank.sort(
        key=lambda r: (
            0 if r["hits_required"] else 1,
            -(r["bi_score"] or 0.0),
            str(r["name"]),
        )
    )
    for i, row in enumerate(rank, start=1):
        row["rank"] = i
    return {
        "polarity": POLARITY_BI,
        "eval_scales": list(BIPOLAR_EVAL_SCALES),
        "scored": ["plus_continuation", "minus_continuation", "leak_where_unused"],
        "not_scored": ["antipodal_collapse", "pair_odd_cos", "pole_loss", "perc"],
        "antipodal_is_diagnostic_only": True,
        "gates": {"exam_continuation": "pass on the cell"},
        "cells": cells,
        "rank": rank,
        "steps": steps,
        "seed": seed,
    }


def collect_boards(*, steps: int = 400, seed: int = 0) -> dict:
    """Both boards, each row carrying its ``polarity`` column."""
    uni = collect_unipolar_board(steps=steps, seed=seed)
    bi = collect_bipolar_board(steps=steps, seed=seed)
    both = []
    for board in (uni, bi):
        for cell, rows in board["cells"].items():
            for row in rows:
                both.append({"board_cell": cell, **row})
    return {"uni": uni, "bi": bi, "rows": both, "steps": steps, "seed": seed}
