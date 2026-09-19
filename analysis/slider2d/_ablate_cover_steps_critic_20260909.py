"""Fire #2 dive: cover_weight x steps (gate fixed) + critic_hidden capacity."""
from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.slider2d.gan import default_cfg, score_adv_sheet
from analysis.slider2d.scoreboard import cell_works
from analysis.slider2d.sheet import leaky_field

SEED = 0
TEACHER = "faithful_guard_e"
B_CAP = 1.0
OUT = Path("analysis/slider2d/_ablation_cover_steps_critic_20260909.json")


def _sheet_row(*, cover: float, steps: int, critic_hidden: int, tag: str) -> dict:
    cfg = default_cfg(
        steps=steps,
        seed=SEED,
        b_cap=B_CAP,
        fm_weight=0.0,
        cover_weight=cover,
        critic_hidden=critic_hidden,
    )
    t0 = time.time()
    sheet = score_adv_sheet(leaky_field(), teacher=TEACHER, cfg=cfg)
    dt = time.time() - t0
    ok = cell_works(
        leak=sheet.get("leak_tok"),
        on_sheet_kept=sheet.get("on_sheet_kept"),
        off_sheet=sheet.get("garble"),
        argmax_on_sheet=sheet.get("argmax_on_sheet"),
        swing_kept=sheet.get("swing_kept"),
    )
    row = {
        "tag": tag,
        "teacher": TEACHER,
        "cover_weight": cover,
        "b_cap": B_CAP,
        "steps": steps,
        "critic_hidden": critic_hidden,
        "leak_tok": sheet.get("leak_tok"),
        "on_sheet_kept": sheet.get("on_sheet_kept"),
        "garble": sheet.get("garble"),
        "swing_kept": sheet.get("swing_kept"),
        "pass": bool(ok),
        "sec": round(dt, 2),
    }
    print(
        "tag=%-14s cover=%.1f steps=%d hid=%d leak=%s kept=%s swing=%s pass=%s (%.1fs)"
        % (
            tag,
            cover,
            steps,
            critic_hidden,
            row["leak_tok"],
            row["on_sheet_kept"],
            row["swing_kept"],
            ok,
            dt,
        ),
        flush=True,
    )
    return row


def main() -> None:
    rows = []
    # Primary thread: cover x steps with default critic (64)
    for steps in (800, 1200):
        for cover in (1.5, 2.0, 3.0):
            rows.append(
                _sheet_row(
                    cover=cover, steps=steps, critic_hidden=64, tag="cover_x_steps"
                )
            )
    # Critic capacity at fixed recipe-ish budget (800, cover 1.5)
    for hid in (32, 64, 128, 256):
        rows.append(
            _sheet_row(
                cover=1.5, steps=800, critic_hidden=hid, tag="critic_capacity"
            )
        )

    payload = {
        "sha_hint": "435e873",
        "note": "leftover gate fixed; dive cover schedule vs critic width",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
