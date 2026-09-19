# Hard multipair @ n=1 cover=1.0 — 2026-09-09 (Fire #17)

Music-transfer posture: n_particles=1, cover=1.0, faithful_guard_e, FM0, b_cap=1.
Maps to Arm B smoke: `music_arm_b_locked_smoke_20260909.sh` (POLE_WEIGHT=1).

| cell | PASS | prim mean | span | leak max | knife |
|:---|:---:|---:|---:|---:|:---:|
| sheet_leftover | 6/6 | 0.9334 | 0.0018 | 0.0036 | no |
| sheet_gender | 6/6 | 0.9962 | 0.0018 | 0.0040 | no |
| exam_divergent | 6/6 | 1.0000 | 0.0000 | 0.0000 | no |
| exam_close | 6/6 | 1.0000 | 0.0000 | 0.0000 | no |
| exam_unused_e | 6/6 | 0.9922 | 0.0208 | 0.0034 | no |
| f3d_divergent | 6/6 | 0.9333 | 0.0134 | 0.0139 | no |
| f3d_close | 5/6 | 0.8433 | 0.7906 | 0.2645 | YES |
| f3d_unused_e | 6/6 | 0.9199 | 0.0069 | 0.0097 | no |
| f3d_leftover | 6/6 | 0.9930 | 0.0263 | 0.0106 | no |
| f3d_mismatch_cross_declare | 0/6 | -0.0530 | 0.0365 | 0.6924 | no |

### Finding

- Fails under cover=1.0/n=1: ['f3d_close'] — may need cover=1.5 on those cells.
- Verdict: `n1_cover1_partial`
- Wall: 2462.9s

## Music transfer caveat (cover=1.0)

- Sheet + most Field3D PairField/leftover cells: **solid 6/6** at n=1 cover=1.0.
- **`f3d_close` knife 5/6** (seed0 collapse prim=0.20 leak=0.26) — close/delivery geometry
  is the weak cell under Music-locked cover=1.0.
- mismatch_cross_declare solid 0/6 (YAML hygiene; expected).
- Follow-up Fire #18: cover 1.0 vs 1.5 on f3d_close only.
- Arm B smoke may keep `POLE_WEIGHT=1` but **must multi-seed close pairs**; escalate to 1.5 if close fails.
