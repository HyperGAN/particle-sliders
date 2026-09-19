#!/usr/bin/env python3
"""Fire #14: content↔leftover entanglement diagnostics mirroring Music span sampling.

Locked core: 1200 steps, cover_weight=1.5, teacher=faithful_guard_e, FM off,
n_particles≤12, particle_l2=0.02, b_cap=1.

Music LM analogy (adv.py sample_real_cloud):
  end_margin ≈ last-token / audio-start pooling mass on poles
  span_frac  ≈ lyric-span lerp window toward the pole

Ask: under entangled content↔ê Field3D geometry, which (span_frac, end_margin)
settings keep leftover gates seed-stable (≥6 seeds), and which create
knife-edge / false locks that would mis-transfer to Music lm_adv?
"""
from __future__ import annotations

import json
import sys
time_mod = __import__("time")
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from analysis.slider2d.field3d import Field3D, score_adv_field3d
from analysis.slider2d.gan import default_cfg

OUT = Path(__file__).resolve().parent / "span_entangle_diag_20260909.json"
MD = Path(__file__).resolve().parent / "span_entangle_diag_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"

SEEDS = [0, 1, 2, 3, 7, 42]
TEACHER = "faithful_guard_e"

# Entangled geometry (content ≈ leftover) — hardest for residual attribution.
def entangled_field(seed: int = 0) -> Field3D:
    return Field3D(
        kind="entangled_span_diag",
        rows=3,
        row_scales=(0.90, 1.0, 1.10),
        slider=1.0,
        content=0.70,
        leak=0.70,
        e_on_u=0.0,
        e_on_content=0.0,
        e_unused=1.0,
        seed=seed,
    )


# span_frac × end_margin grid (Music-transfer candidates)
# Default locked: span_frac=0.40, end_margin=0.60
SPAN_GRID = [
    {"span_frac": 0.20, "end_margin": 0.80, "tag": "pole_heavy"},
    {"span_frac": 0.40, "end_margin": 0.60, "tag": "locked_default"},
    {"span_frac": 0.60, "end_margin": 0.40, "tag": "span_heavy"},
    {"span_frac": 0.80, "end_margin": 0.20, "tag": "span_extreme"},
    {"span_frac": 0.40, "end_margin": 1.00, "tag": "end_only"},
    {"span_frac": 1.00, "end_margin": 0.00, "tag": "span_only"},
]


def locked_cfg(seed: int, span_frac: float, end_margin: float):
    return default_cfg(
        steps=1200,
        seed=seed,
        b_cap=1.0,
        cover_weight=1.5,
        fm_weight=0.0,
        n_particles=12,
        particle_l2=0.02,
        span_frac=span_frac,
        end_margin=end_margin,
    )


def summarize(rows):
    u = [r["u_kept"] for r in rows]
    c = [r["content_kept"] for r in rows]
    lk = [r["leak_ratio"] for r in rows]
    n_pass = sum(1 for r in rows if r["pass"])
    return {
        "n_pass": n_pass,
        "n_total": len(rows),
        "pass_rate": "%d/%d" % (n_pass, len(rows)),
        "u_kept_mean": sum(u) / len(u),
        "u_kept_span": max(u) - min(u),
        "content_kept_mean": sum(c) / len(c),
        "content_kept_span": max(c) - min(c),
        "leak_ratio_mean": sum(lk) / len(lk),
        "leak_ratio_max": max(lk),
        "knife_edge": bool(
            n_pass not in (0, len(rows))
            or (max(u) - min(u)) > 0.05
            or (max(c) - min(c)) > 0.05
        ),
    }


def write_md(blob):
    lines = [
        "# Span × entanglement diagnostics — 2026-09-09 (Fire #14)",
        "",
        "Geometry: Field3D entangled (content=0.70, leak=0.70, unused declared ê).",
        "Locked: 1200 / cover1.5 / faithful_guard_e / FM0 / n_particles=12 / l2=0.02 / b_cap=1.",
        "Sweep: span_frac × end_margin (Music lyric-span + last-token analogue).",
        "",
        "Seeds: %s" % (blob["seeds"],),
        "",
        "| tag | span_frac | end_margin | PASS | u_kept mean | u span | c_kept mean | leak max | knife |",
        "|:---|---:|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for g in blob["grid"]:
        s = g["summary"]
        lines.append(
            "| %s | %.2f | %.2f | %s | %.4f | %.4f | %.4f | %.4f | %s |"
            % (
                g["tag"],
                g["span_frac"],
                g["end_margin"],
                s["pass_rate"],
                s["u_kept_mean"],
                s["u_kept_span"],
                s["content_kept_mean"],
                s["leak_ratio_max"],
                "YES" if s["knife_edge"] else "no",
            )
        )
    lines += [
        "",
        "### Finding",
        "",
        "- %s" % blob["finding"],
        "- Verdict: `%s`" % blob["verdict"],
        "- Music transfer note: %s" % blob["music_note"],
        "- Wall: %.1fs" % blob["wall_s"],
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")


def append_log(blob):
    block = (
        "\n## Fire #14 — span×entangle diagnostics (Music span analogue) (2026-09-09)\n\n"
        "- Host: pop-os-cpu @ SHA `435e873`\n"
        "- Geometry: Field3D entangled c=0.70 e=0.70; seeds %s\n"
        "- Grid span_frac×end_margin @ locked recipe:\n" % (blob["seeds"],)
    )
    for g in blob["grid"]:
        s = g["summary"]
        block += (
            "  - `%s` (sf=%.2f em=%.2f): **%s** u=%.4f±%.4f c=%.4f leak_max=%.4f knife=%s\n"
            % (
                g["tag"],
                g["span_frac"],
                g["end_margin"],
                s["pass_rate"],
                s["u_kept_mean"],
                s["u_kept_span"],
                s["content_kept_mean"],
                s["leak_ratio_max"],
                s["knife_edge"],
            )
        )
    block += (
        "- Verdict: **%s** — %s\n"
        "- Music note: %s\n"
        "- Notes: `span_entangle_diag_20260909.{py,json,md}`\n"
        % (blob["verdict"], blob["finding"], blob["music_note"])
    )
    status = (
        "\n### STATUS (~Fire #14)\n"
        "- Best transferable span setting for Music lm_adv smoke: see verdict.\n"
        "- Current: `%s`\n" % blob["verdict"]
    )
    for path in (LOG, LOG_ROOT):
        if path.exists():
            path.write_text(path.read_text() + block + status)


def main():
    t0 = time_mod.time()
    print("Fire #14 span×entangle grid=%s seeds=%s" % ([g["tag"] for g in SPAN_GRID], SEEDS), flush=True)
    grid_out = []
    for g in SPAN_GRID:
        print(
            "--- tag=%s span_frac=%.2f end_margin=%.2f"
            % (g["tag"], g["span_frac"], g["end_margin"]),
            flush=True,
        )
        rows = []
        for seed in SEEDS:
            st = time_mod.time()
            cfg = locked_cfg(seed, g["span_frac"], g["end_margin"])
            row = score_adv_field3d(
                entangled_field(seed),
                teacher=TEACHER,
                cfg=cfg,
                name="span_%s_s%d" % (g["tag"], seed),
            )
            row["wall_s"] = round(time_mod.time() - st, 2)
            rows.append(row)
            print(
                "  seed=%2d pass=%s u=%.4f c=%.4f leak=%.4f (%.1fs)"
                % (seed, row["pass"], row["u_kept"], row["content_kept"], row["leak_ratio"], row["wall_s"]),
                flush=True,
            )
        summ = summarize(rows)
        grid_out.append(
            {
                "tag": g["tag"],
                "span_frac": g["span_frac"],
                "end_margin": g["end_margin"],
                "summary": summ,
                "detail": [
                    {
                        "seed": r["seed"],
                        "pass": r["pass"],
                        "u_kept": r["u_kept"],
                        "content_kept": r["content_kept"],
                        "leak_ratio": r["leak_ratio"],
                        "wall_s": r["wall_s"],
                    }
                    for r in rows
                ],
            }
        )
        print(
            "  >> %s u_mean=%.4f span=%.4f knife=%s"
            % (summ["pass_rate"], summ["u_kept_mean"], summ["u_kept_span"], summ["knife_edge"]),
            flush=True,
        )

    # Verdict logic
    locked = next(x for x in grid_out if x["tag"] == "locked_default")
    solid = [x for x in grid_out if x["summary"]["n_pass"] == x["summary"]["n_total"] and not x["summary"]["knife_edge"]]
    fragile = [x for x in grid_out if x["summary"]["knife_edge"] or x["summary"]["n_pass"] != x["summary"]["n_total"]]

    if locked["summary"]["n_pass"] == locked["summary"]["n_total"] and not locked["summary"]["knife_edge"]:
        if not fragile:
            verdict = "span_entangle_flat_keep_default"
            finding = (
                "All span/end settings 6/6 and non-knife under entangled Field3D. "
                "Keep locked span_frac=0.40 end_margin=0.60; no Music transfer pressure to retune."
            )
            music_note = (
                "Prefer default span/end on Music lm_adv smoke; do not chase span_extreme/end_only "
                "unless listen shows pole collapse or midpoint garble."
            )
        else:
            verdict = "span_entangle_default_solid_edges_fragile"
            finding = (
                "Locked default solid; fragile tags=%s. Those edges are false-lock risks for Music."
                % [x["tag"] for x in fragile]
            )
            music_note = (
                "Keep span_frac≈0.40 end_margin≈0.60 on Music; avoid end_only or span_only "
                "without multi-seed sheet+leftover margin."
            )
    else:
        verdict = "span_entangle_needs_attention"
        finding = "Locked default not fully solid under entanglement — inspect before Music transfer."
        music_note = "Re-check span/end on 2D sheet leftover first; do not smoke Music until default clears."

    blob = {
        "fire": 14,
        "date": "2026-09-09",
        "sha": "435e873",
        "seeds": SEEDS,
        "geometry": {"content": 0.70, "leak": 0.70, "rows": 3},
        "recipe_locked": {
            "steps": 1200,
            "cover_weight": 1.5,
            "teacher": TEACHER,
            "fm": 0.0,
            "n_particles": 12,
            "particle_l2": 0.02,
            "b_cap": 1.0,
        },
        "grid": grid_out,
        "verdict": verdict,
        "finding": finding,
        "music_note": music_note,
        "solid_tags": [x["tag"] for x in solid],
        "fragile_tags": [x["tag"] for x in fragile],
        "wall_s": round(time_mod.time() - t0, 1),
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    write_md(blob)
    append_log(blob)
    print("VERDICT=%s wall=%.1fs -> %s" % (verdict, blob["wall_s"], OUT), flush=True)


if __name__ == "__main__":
    main()
