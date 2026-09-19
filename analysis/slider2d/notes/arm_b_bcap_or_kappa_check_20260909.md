# Arm B b_cap falsify check (2026-09-09)

Falsify attempt under Recommendation **B** posture: leftover ON
(`faithful_guard_e`), `cover_weight=1.0`, FM=0, `n_particles=1`
(Music `--parts 0` proxy), steps=1200. Axis: toy **`b_cap`** ∈ [0.5, 1.0, 2.0]
(Music `--adv_reg_coeff`; κ left at toy hard-coded margin=1 /
Music `--adv_reg_kappa=1`). Seeds: [0, 1, 2]. Wall: 2405.3s.

| b_cap | geom | pass | primary_mean | primary_min | leak_abs_max | knife |
|---:|---|---|---:|---:|---:|---|
| 0.5 | sheet | 3/3 | 0.9343 | 0.9334 | 0.0038 | False |
| 0.5 | field3d | 3/3 | 0.9985 | 0.9979 | 0.0098 | False |
| 1.0 | sheet | 3/3 | 0.9337 | 0.9328 | 0.0036 | False |
| 1.0 | field3d | 3/3 | 0.9982 | 0.9956 | 0.0106 | False |
| 2.0 | sheet | 3/3 | 0.9345 | 0.9335 | 0.0040 | False |
| 2.0 | field3d | 3/3 | 0.9964 | 0.9946 | 0.0073 | False |

## Verdict

- **CONFIRMED** — b_cap=1 remains best under Arm B posture (cover=1.0, leftover ON, FM0, n=1).
- Best / locked pick: **b_cap = 1.0** (Music `--adv_reg_coeff=1.0`,
  `--adv_reg_kappa=1.0`)
- Comfortable b_caps: [1.0, 0.5, 2.0]

## Read

- If CONFIRMED: recipe card keeps c=κ=1; no revise.
- If REVISED: update `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md` + Arm B smoke.
- κ not swept on toy (single coeff API); Music split remains c=κ paired at 1
  unless a later Music-only kappa smoke says otherwise.


## Flatness

All three b_caps are full-pass + comfortable (sheet kept_min ≥ 0.92, no knife).
Δ kept / leak vs b_cap=1 is noise-scale. Do **not** revise toward 0.5 or 2.0 —
prefer canonical locked **1** (Music c=κ=1). κ not independently swept on toy.

## Paths

- JSON: `arm_b_bcap_or_kappa_check_20260909.json`
- Script: `arm_b_bcap_or_kappa_check_20260909.py`
- Recipe: `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md`
- Arm B smoke: `music_arm_b_locked_smoke_20260909.sh`
- Prior cover sweep: `arm_b_pole_cover_sweep_20260909.md`
