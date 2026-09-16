#!/usr/bin/env python3
"""Emit the formulation leaderboards: unipolar, bipolar, or both.

CPU only, no Hub, no GPU, no Music 3 weights. Does not change the live
trainer default (``--lm_target v9`` / ``--pole_mode hidden``) and does
not touch the compiled bipolar board (``docs/lm-2d-scoreboard.md``).

Writes ``docs/FORMULATION_LEADERBOARD_UNIPOLAR.md`` /
``docs/FORMULATION_LEADERBOARD_BIPOLAR.md`` (first-class siblings) plus
``metrics_*.json`` and plots under ``--out`` (default
``docs/formulation-leaderboard``). ``--polarity`` selects which board(s)
to regenerate; the combined ``metrics_both.json`` carries a
``polarity`` (``uni`` / ``bi``) column on every row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.formulation_leaderboard import (
    POLARITY_NOTE,
    collect_bipolar_board,
    collect_boards,
    collect_unipolar_board,
)
from analysis.slider2d.plus_exam import PLUS_COVER_MIN, PLUS_OFF_MAX
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN


DEFAULT_OUT = _REPO / "docs" / "formulation-leaderboard"
UNI_MD = _REPO / "docs" / "FORMULATION_LEADERBOARD_UNIPOLAR.md"
BI_MD = _REPO / "docs" / "FORMULATION_LEADERBOARD_BIPOLAR.md"


def _f(value, spec: str = ".3f", empty: str = "N/A") -> str:
    if value is None:
        return empty
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def plot_uni(board: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    marks = {"divergent": "o", "close": "s", "unused_e": "^"}
    colors = {
        "faithful_plus_neu": "#1e8449",
        "faithful_plus": "#7d3c98",
        "leftover_gate_bipolar": "#2471a3",
        "faithful_even_blend": "#b9770e",
        "pair_odd_midpoint": "#c0392b",
    }
    want = Rectangle(
        (PLUS_COVER_MIN, PLUS_NEU_HOLD_MIN),
        1.0 - PLUS_COVER_MIN,
        1.0 - PLUS_NEU_HOLD_MIN,
        facecolor="#d5f5e3",
        edgecolor="#1e8449",
        lw=1.0,
        alpha=0.55,
        zorder=1,
        label="want-box (cover + neu_hold)",
    )
    ax.add_patch(want)
    for cell, rows in board["cells"].items():
        if cell == "unused_e":
            continue
        for row in rows:
            ax.scatter(
                [row["cover"]],
                [row["neu_hold"]],
                c=colors.get(row["name"], "#7f8c8d"),
                marker=marks.get(cell, "o"),
                s=72,
                zorder=3,
                edgecolors="white",
                linewidths=0.7,
            )
            ax.annotate(
                f"{row['name']}\n{cell}",
                (row["cover"], row["neu_hold"]),
                fontsize=6.4,
                xytext=(5, 4),
                textcoords="offset points",
            )
    for name, color in colors.items():
        ax.scatter([], [], c=color, marker="o", s=58, label=name)
    ax.axvline(PLUS_COVER_MIN, color="#7f8c8d", ls=":", lw=0.9)
    ax.axhline(PLUS_NEU_HOLD_MIN, color="#7f8c8d", ls=":", lw=0.9)
    ax.set_xlabel("cover of the + state  [0, 1]  (higher is better)")
    ax.set_ylabel("neu_hold at scale 0  [0, 1]  (higher is better)")
    ax.set_title("Formulation leaderboard (UNIPOLAR) — cover vs neu_hold")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.4, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_bi(board: dict, path: Path) -> None:
    rows = [r for r in board["rank"] if r.get("bi_score") is not None]
    rows = sorted(rows, key=lambda r: float(r["bi_score"]))
    if not rows:
        return
    height = max(4.2, 0.6 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, height))
    y = list(range(len(rows)))
    colors = [
        "#1e8449" if r["hits_required"] else "#c0392b" for r in rows
    ]
    widths = [float(r["bi_score"]) for r in rows]
    bars = ax.barh(y, widths, color=colors, height=0.62, zorder=3)
    for bar, row in zip(bars, rows):
        ax.text(
            min(float(row["bi_score"]) + 0.012, 1.02),
            bar.get_y() + bar.get_height() / 2.0,
            f"{float(row['bi_score']):.3f}",
            va="center",
            ha="left",
            fontsize=7.4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax.set_xlim(0.0, 1.18)
    ax.set_xlabel("bi_score = min(overlap, swing) on divergent + close")
    ax.set_title("Formulation leaderboard (BIPOLAR) — both-pole ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_uni_markdown(board: dict, path: Path) -> None:
    lines = [
        "# Formulation leaderboard (UNIPOLAR)",
        "",
        "Generated by `analysis/slider2d/run_formulation_leaderboard.py "
        "--polarity uni`. CPU only, no Hub, no GPU, no Music 3 weights. "
        "Does not change the live trainer default (`--lm_target v9` / "
        "`--pole_mode hidden`). First-class sibling of "
        "[FORMULATION_LEADERBOARD_BIPOLAR.md](FORMULATION_LEADERBOARD_BIPOLAR.md); "
        "the compiled bipolar board lives at "
        "[lm-2d-scoreboard.md](lm-2d-scoreboard.md) and is not updated here.",
        "",
        "## What this board asks",
        "",
        "Plus-oriented fit and eval without requiring bipolar antipodal "
        "collapse. Three honesty gates, all on the + / neutral fader:",
        "",
        f"1. **plus cover** at scale 1: `>= {PLUS_COVER_MIN}`.",
        f"2. **plus leak** (off-caption on the + continuation) at scale 1: "
        f"`<= {PLUS_OFF_MAX}`.",
        f"3. **neu_hold** at scale 0: `>= {PLUS_NEU_HOLD_MIN}`.",
        "",
        "Eval scales are `0, 0.5, 1`. Gates read scales 0 and 1; scale 0.5 "
        "is a logged interpolation diagnostic (`half_cover`: does the "
        "halfway fader sit between neutral and + instead of jumping or "
        "collapsing). Scale `-1` is reported as an unscored canary "
        "(`dangerous` = landed on + or -1 off-caption above the 0.05 "
        "suggestion) — it is never a score. Antipodal `cos(+1, -1)` is "
        "never consulted.",
        "",
        "## Combined rank (divergent + close)",
        "",
        "In-box first (hit on both required pairs), then neu_hold, then "
        "cover, then off-caption. `unused_e` is logged below, not ranked.",
        "",
        "| rank | recipe | train | polarity | in-box | neu_hold | cover | "
        "off-caption | hit divergent | hit close |",
        "|---:|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in board["rank"]:
        lines.append(
            "| {rank} | `{name}` | {train} | {pol} | {box} | {hold} | "
            "{cover} | {off} | {hd} | {hc} |".format(
                rank=row["rank"],
                name=row["name"],
                train=row['train'],
                pol=row.get("polarity", "uni"),
                box="**yes**" if row["in_box"] else "—",
                hold=_f(row["neu_hold"]),
                cover=_f(row["cover"]),
                off=_f(row["off_caption"]),
                hd="**HIT**" if row["hit_divergent"] else "—",
                hc="**HIT**" if row["hit_close"] else "—",
            )
        )
    lines += ["", "## Per-cell board", ""]
    for cell, rows in board["cells"].items():
        lines += [
            f"### `{cell}`",
            "",
            "| recipe | train | cover | off-caption | neu_hold | "
            "half_cover@0.5 *(diag)* | -1 canary *(diag)* | hit |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
        for row in rows:
            can = row["canary"]
            lines.append(
                "| `{name}` | {train} | {cover} | {off} | {hold} | "
                "{half} | {land}/{canoff}{dang} | {hit} |".format(
                    name=row["name"],
                    train=row["train"],
                    cover=_f(row["cover"]),
                    off=_f(row["off_caption"]),
                    hold=_f(row["neu_hold"]),
                    half=_f(row["half_scale"]["half_cover"]),
                    land=can["minus_landed"],
                    canoff=_f(can["minus_off_caption"]),
                    dang=" **danger**" if can["dangerous"] else "",
                    hit="**HIT**" if row["hit"] else "—",
                )
            )
        lines.append("")
    lines += [
        "![cover vs neu_hold (unipolar)](formulation-leaderboard/uni-scale.png)",
        "",
        "## Bipolar vs unipolar: when to read which",
        "",
        POLARITY_NOTE,
        "",
        "## Related cells (not this scale)",
        "",
        "- [FORMULATION_LEADERBOARD_BIPOLAR.md]"
        "(FORMULATION_LEADERBOARD_BIPOLAR.md) — the bipolar sibling.",
        "- [lm-plus-neu-exam.md](lm-plus-neu-exam.md) — the plus+neu exam "
        "this board reuses.",
        "- [lm-plus-exam.md](lm-plus-exam.md) — plus-only cover / off-caption.",
        "- [lm-2d-scoreboard.md](lm-2d-scoreboard.md) — compiled bipolar board.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_formulation_leaderboard.py "
        "--polarity uni --out docs/formulation-leaderboard",
        "PYTHONPATH=. pytest tests/test_formulation_leaderboard.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights. Seed "
        f"`{board['seed']}`, `{board['steps']}` Adam steps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bi_markdown(board: dict, path: Path) -> None:
    lines = [
        "# Formulation leaderboard (BIPOLAR)",
        "",
        "Generated by `analysis/slider2d/run_formulation_leaderboard.py "
        "--polarity bi`. CPU only, no Hub, no GPU, no Music 3 weights. "
        "Does not change the live trainer default (`--lm_target v9` / "
        "`--pole_mode hidden`). First-class sibling of "
        "[FORMULATION_LEADERBOARD_UNIPOLAR.md]"
        "(FORMULATION_LEADERBOARD_UNIPOLAR.md); the compiled bipolar board "
        "lives at [lm-2d-scoreboard.md](lm-2d-scoreboard.md) and is not "
        "updated here.",
        "",
        "## What this board asks",
        "",
        "One residual serving a +/- pair: `delta(+1)` must sing the + pole "
        "AND `delta(-1)` must sing the - pole, on the same weights. The "
        "honesty gate per cell is the pair-exam continuation pass "
        "(overlap, position-wise agreement, off-caption, coherence, swing "
        "kept) plus leftover leak where the field has an unused axis. "
        "Eval scales are `-1, +1` — both gated.",
        "",
        "**Antipodal `cos(+1, -1)` is logged, never gated.** Gating on it "
        "would crown the known fail: the pair-odd midpoint locks "
        "`cos(d+, a) = +1` / `cos(d+, d-) = -1` while walking off the "
        "sheet, and the caption winners that pass both exam pairs measure "
        "`collapse ~= 0`. Same for pair-odd cos, pole loss and `p%` / `n%`: "
        "logged, never scored.",
        "",
        "## Combined rank (divergent + close)",
        "",
        "`bi_score = min(overlap, swing)` on the two required pairs. "
        "`unused_e` is logged below, not ranked.",
        "",
        "| rank | recipe | polarity | hits required | bi_score | "
        "hit divergent | hit close |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for row in board["rank"]:
        lines.append(
            "| {rank} | `{name}` | {pol} | {box} | {score} | {hd} | {hc} |".format(
                rank=row["rank"],
                name=row["name"],
                pol=row.get("polarity", "bi"),
                box="**yes**" if row["hits_required"] else "—",
                score=_f(row["bi_score"]),
                hd="**HIT**" if row["hit_divergent"] else "—",
                hc="**HIT**" if row["hit_close"] else "—",
            )
        )
    lines += ["", "## Per-cell board", ""]
    for cell, rows in board["cells"].items():
        lines += [
            f"### `{cell}`",
            "",
            "| recipe | overlap | swing_kept | off-corpus | coherence | "
            "leak_tok | antipodal *(diag)* | hit |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                "| `{name}` | {ov} | {sw} | {off} | {co} | {leak} | "
                "{anti} | {hit} |".format(
                    name=row["name"],
                    ov=_f(row["roll_overlap"]),
                    sw=_f(row["roll_swing_kept"]),
                    off=_f(row["roll_off_corpus"]),
                    co=_f(row["roll_coherence"]),
                    leak=_f(row["leak_tok"], "+.3f"),
                    anti=_f(row["antipodal_collapse"], "+.3f"),
                    hit="**HIT**" if row["hit"] else "—",
                )
            )
        lines.append("")
    lines += [
        "![both-pole ranking (bipolar)](formulation-leaderboard/bi-score.png)",
        "",
        "## Bipolar vs unipolar: when to read which",
        "",
        POLARITY_NOTE,
        "",
        "## Related cells (not this scale)",
        "",
        "- [FORMULATION_LEADERBOARD_UNIPOLAR.md]"
        "(FORMULATION_LEADERBOARD_UNIPOLAR.md) — the unipolar sibling.",
        "- [lm-pair-exam.md](lm-pair-exam.md) — the pair cells this board "
        "reuses.",
        "- [lm-2d-scoreboard.md](lm-2d-scoreboard.md) — compiled bipolar board.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_formulation_leaderboard.py "
        "--polarity bi --out docs/formulation-leaderboard",
        "PYTHONPATH=. pytest tests/test_formulation_leaderboard.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights. Seed "
        f"`{board['seed']}`, `{board['steps']}` Adam steps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(board: dict) -> dict:
    return json.loads(json.dumps(board, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--polarity",
        choices=("uni", "bi", "both"),
        default="both",
        help="which board(s) to (re)generate",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.polarity in ("uni", "both"):
        uni = collect_unipolar_board(steps=args.steps, seed=args.seed)
        (args.out / "metrics_uni.json").write_text(
            json.dumps(_jsonable(uni), indent=2) + "\n", encoding="utf-8"
        )
        plot_uni(uni, args.out / "uni-scale.png")
        write_uni_markdown(uni, UNI_MD)
        print(f"unipolar: wrote {UNI_MD} + metrics_uni.json")
    if args.polarity in ("bi", "both"):
        bi = collect_bipolar_board(steps=args.steps, seed=args.seed)
        (args.out / "metrics_bi.json").write_text(
            json.dumps(_jsonable(bi), indent=2) + "\n", encoding="utf-8"
        )
        plot_bi(bi, args.out / "bi-score.png")
        write_bi_markdown(bi, BI_MD)
        print(f"bipolar: wrote {BI_MD} + metrics_bi.json")
    if args.polarity == "both":
        both = collect_boards(steps=args.steps, seed=args.seed)
        (args.out / "metrics_both.json").write_text(
            json.dumps(_jsonable(both), indent=2) + "\n", encoding="utf-8"
        )
        print("both: wrote metrics_both.json (polarity column on every row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
