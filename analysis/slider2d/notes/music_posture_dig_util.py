"""Fire #23 — shared dig helper: surface music_close_posture_warn in runners.

Import from notes dig scripts::

    from analysis.slider2d.notes.music_posture_dig_util import (
        annotate_dig_rows,
        posture_block_for_md,
    )

Score rows already carry ``music_close_posture_warn`` via ``score_adv_field3d``
(Fire #22). This util only aggregates + formats for dig JSON/md.
"""
from __future__ import annotations

from typing import Any

from analysis.slider2d.field3d import summarize_music_close_posture


def annotate_dig_rows(rows: list[dict], *, payload: dict | None = None) -> dict:
    """Attach posture summary to a dig payload (mutates payload if given)."""
    summary = summarize_music_close_posture(rows)
    out = payload if payload is not None else {}
    out["music_close_posture"] = summary
    return out


def posture_block_for_md(summary: dict[str, Any]) -> list[str]:
    """Markdown lines for research notes."""
    lines = [
        "### Music close posture (Fire #22/#23)",
        f"- rows={summary.get('n_rows')} warn={summary.get('n_warn')} "
        f"clear={summary.get('n_clear')} missing_key={summary.get('n_missing_key')}",
    ]
    if summary.get("any_warn"):
        lines.append("- **any_warn=YES** — Music parts0 close knife posture present")
        for w in summary.get("warns") or []:
            lines.append(
                f"  - `{w.get('name')}` cell={w.get('cell')} "
                f"n={w.get('n_particles')} seed={w.get('seed')}"
            )
    else:
        lines.append("- any_warn=NO")
    return lines
