# Arm B pole/cover analogue sweep (2026-09-09)

Recommendation **B** true #94 target: `faithful_guard_e` + leftover ON,
FM=0, b_cap=1, steps=1200, `n_particles=1` (Music `--parts 0` proxy).
Sweep toy `cover_weight` ∈ [0.5, 1.0, 1.5, 2.0] → Music modest `--pole_weight`.
Seeds: [0, 1, 2]. Wall: 533.8s.

| cover | geom | pass | primary_mean | primary_min | leak_abs_max | knife |
|---:|---|---|---:|---:|---:|---|
| 0.5 | sheet | 3/3 | 0.9170 | 0.9114 | 0.0074 | False |
| 0.5 | field3d | 3/3 | 0.9750 | 0.9611 | 0.0182 | False |
| 1.0 | sheet | 3/3 | 0.9337 | 0.9328 | 0.0036 | False |
| 1.0 | field3d | 3/3 | 0.9982 | 0.9956 | 0.0106 | False |
| 1.5 | sheet | 3/3 | 0.9353 | 0.9343 | 0.0030 | False |
| 1.5 | field3d | 3/3 | 0.9973 | 0.9925 | 0.0060 | False |
| 2.0 | sheet | 3/3 | 0.9367 | 0.9360 | 0.0018 | False |
| 2.0 | field3d | 3/3 | 0.9981 | 0.9969 | 0.0045 | False |

## Recommended Music pole/cover

- **cover_weight (toy)** = **1.0**
- **Music `--pole_weight`** = **1.0** (Arm B cover analogue)
- Rule: full sheet+Field3D pass, no knife_edge, sheet `primary_min` ≥ 0.92,
  prefer lowest modest cover; reject ≥2.0 when a lower comfortable cover exists.
- Note: lowest modest cover with full sheet+Field3D pass and sheet kept_min>=0.92; reject cover>=2.0 while lower comfortable exists (Fire #4 high-pin caution)
- Rejected cover=0.5: full-pass but sheet primary_min=0.9114 < comfort 0.92 (knife-adjacent to 0.90 gate)

### Pass rates at recommended

- sheet: **3/3** kept_mean=0.9337 kept_min=0.9328 leak_max=0.0036
- field3d: **3/3** u_kept_mean=0.9982 leak_max=0.0106

## False-lock reject

- cover=0.5: full-pass but knife-adjacent kept_min (0.9114) — not Music start.
- cover=2.0: also full-pass here with leftover ON, but rejected while cover=1.0
  already comfortable (Fire #4 high-pin / cover3 caution — do not chase higher pin).
- leftover_only (cover=0) known fail from dual_arm ablation — not re-swept.
- Arm T lyric+tx is listen-only; do not claim #94 from lyrichold alone.

## Paths

- JSON: `arm_b_pole_cover_sweep_20260909.json`
- Script: `arm_b_pole_cover_sweep_20260909.py`
- Arm B smoke: `music_arm_b_locked_smoke_20260909.sh` (default `POLE_WEIGHT=1`)
- Arm T smoke: `music_locked_recipe_smoke_20260909.sh`
- Strategy: `dual_arm_transfer_strategy_20260909.md`
