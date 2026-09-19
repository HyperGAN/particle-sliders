#!/usr/bin/env python3
"""Fire #18: kill false lock — f3d_close @ n=1 cover 1.0 vs 1.5 (≥6 seeds).

Fire #17: f3d_close cover=1.0 n=1 → 5/6 knife (seed0 prim=0.20 leak=0.26).
Ask: does cover=1.5 restore 6/6? Is cover=1.0 Music lock unsafe for close cells?
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from analysis.slider2d.field3d import close_field3d, score_adv_field3d_exam
from analysis.slider2d.gan import default_cfg

OUT = Path(__file__).resolve().parent / "f3d_close_cover_knife_20260909.json"
MD = Path(__file__).resolve().parent / "f3d_close_cover_knife_20260909.md"
LOG = Path(__file__).resolve().parent / "research_log_20260909.md"
LOG_ROOT = Path(__file__).resolve().parents[1] / "research_log_20260909.md"
SEEDS = [0, 1, 2, 3, 7, 42]

def run(cover, seeds=SEEDS):
    rows = []
    for seed in seeds:
        st = time.time()
        cfg = default_cfg(steps=1200, seed=seed, b_cap=1.0, cover_weight=cover,
                          fm_weight=0.0, n_particles=1, particle_l2=0.02)
        row = score_adv_field3d_exam(close_field3d(seed=seed), teacher="faithful_guard_e", cfg=cfg)
        rows.append({
            "seed": seed,
            "pass": bool(row["exam_pass"]),
            "exam_score": float(row["exam_score"]),
            "u_kept": float(row["u_kept"]),
            "content_kept": float(row["content_kept"]),
            "leak_ratio": float(row["leak_ratio"]),
            "wall_s": round(time.time()-st, 2),
        })
        print("  cover=%.1f seed=%2d pass=%s exam=%.4f leak=%.4f (%.1fs)" % (
            cover, seed, rows[-1]["pass"], rows[-1]["exam_score"], rows[-1]["leak_ratio"], rows[-1]["wall_s"]), flush=True)
    scores = [r["exam_score"] for r in rows]
    n_pass = sum(1 for r in rows if r["pass"])
    return {
        "cover": cover,
        "pass_rate": "%d/%d" % (n_pass, len(rows)),
        "n_pass": n_pass,
        "exam_mean": sum(scores)/len(scores),
        "exam_span": max(scores)-min(scores),
        "leak_max": max(r["leak_ratio"] for r in rows),
        "knife_edge": n_pass not in (0, len(rows)) or (max(scores)-min(scores)) > 0.05,
        "detail": rows,
    }

def main():
    t0 = time.time()
    print("Fire #18 f3d_close cover knife n=1", flush=True)
    grid = []
    for cover in (1.0, 1.5):
        print("--- cover=%.1f" % cover, flush=True)
        grid.append(run(cover))
    c1 = grid[0]; c15 = grid[1]
    if c1["n_pass"] < 6 and c15["n_pass"] == 6:
        verdict = "cover1_knife_cover15_solid"
        finding = ("f3d_close @ n=1: cover=1.0 is knife/false-lock risk (%s); cover=1.5 restores 6/6. "
                   "Music pole_weight=1.0 may be thin on close/delivery — prefer 1.5 or multi-seed margin on close pairs."
                   % c1["pass_rate"])
        music = "Arm B: keep pole_weight=1 as start but seed-check close pairs; escalate pole_weight→1.5 if close fails."
    elif c1["n_pass"] == 6:
        verdict = "cover1_close_ok_on_rerun"
        finding = "cover=1.0 now 6/6 on rerun — Fire #17 seed0 may be flaky under CPU contention; still require ≥6 seeds."
        music = "pole_weight=1.0 OK if multi-seed close PASS with margin."
    else:
        verdict = "close_fragile_both"
        finding = "Both covers fragile on f3d_close @ n=1 — investigate."
        music = "Do not smoke Music close pairs until close cell solid."
    blob = {"fire": 18, "n_particles": 1, "grid": grid, "verdict": verdict,
            "finding": finding, "music_note": music, "wall_s": round(time.time()-t0,1)}
    OUT.write_text(json.dumps(blob, indent=2)+"\n")
    lines = ["# f3d_close cover knife @ n=1 — 2026-09-09 (Fire #18)", "",
             "| cover | PASS | exam mean | span | leak max | knife |",
             "|---:|:---:|---:|---:|---:|:---:|"]
    for g in grid:
        lines.append("| %.1f | %s | %.4f | %.4f | %.4f | %s |" % (
            g["cover"], g["pass_rate"], g["exam_mean"], g["exam_span"], g["leak_max"],
            "YES" if g["knife_edge"] else "no"))
    lines += ["", "### Finding", "", "- "+finding, "- Verdict: `%s`"%verdict, "- Music: "+music, ""]
    MD.write_text("\n".join(lines)+"\n")
    block = ("\n## Fire #18 — f3d_close cover knife n=1 (2026-09-09)\n\n"
             "- Verdict: **%s** — %s\n- Music: %s\n- Notes: `f3d_close_cover_knife_20260909.*`\n"
             % (verdict, finding, music))
    for p in (LOG, LOG_ROOT):
        if p.exists():
            p.write_text(p.read_text()+block)
    print("VERDICT=%s" % verdict, flush=True)

if __name__ == "__main__":
    main()
