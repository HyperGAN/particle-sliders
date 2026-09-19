# Fire #19 Music→toy dig — 2026-09-09

Host: box-cpu @ `435e873` (+ local field3d.py). Wall 356.3s. CPU only.

## Recipe

- Locked: 1200 / cover=1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1
- Music-posture: n_particles=1 (parts0 proxy), cover∈{1.0,1.5}
- Skip 800×cover3.0

## Fixes this fire

- `Field3D.odd()` now uses `row_amps` when set (required for M1/M2/cross-axis)
- `test_leftover_basis_orthonormalish` (was misnamed poles check)

## Pass grid

| cell | cover | n | PASS | primary mean | leak max | fail seeds |
|---|---:|---:|:---:|---:|---:|---|
| leftover_n12_c1.5 | 1.5 | 12 | 6/6 | 0.9920 | 0.0004 | [] |
| close_n12_c1.5 | 1.5 | 12 | 6/6 | 0.9819 | 0.0129 | [] |
| close_n1_c1.5 | 1.5 | 1 | 5/6 | 0.8911 | 0.0537 | [0] |
| close_n1_c1.0 | 1.0 | 1 | 5/6 | 0.8605 | 0.2024 | [0] |
| lyric_span_entangle_n12_c1.5 | 1.5 | 12 | 0/6 | 0.9903 | 0.1923 | [0, 1, 2, 3, 7, 42] |
| lyric_span_entangle_n1_c1.0 | 1.0 | 1 | 0/6 | 0.9911 | 0.2076 | [0, 1, 2, 3, 7, 42] |
| cross_axis_rows_n1_c1.0 | 1.0 | 1 | 0/6 | 0.5742 | 0.4636 | [0, 1, 2, 3, 7, 42] |
| dual_arm_L_guard_c1.5_n1 | 1.5 | 1 | 3/3 | 0.9947 | 0.0089 | [] |
| dual_arm_listen_faithful_c1.5_n1 | 1.5 | 1 | 0/3 | 0.9923 | 0.4565 | [0, 1, 2] |
| dual_arm_gate_only_guard_c0_n1 | 0.0 | 1 | 0/3 | 0.1499 | 0.1419 | [0, 1, 2] |

## Verdict

- **lyric_span_entangle bites?** YES (fail mode: `pass_multi_row` False, rows_covered=0/5 despite high exam_score/u_kept on row0) — locked 0/6 (mean prim 0.9903, leak_max 0.1923); music 0/6
- **f3d_close seed0-only knife @ n=1?** YES — c1.0 5/6 fail=[0]; c1.5 5/6 fail=[0]
- **leftover/close @ n=12 regression?** NO — leftover 6/6, close 6/6
- **cross_axis Music posture:** 0/6
- **dual_arm:** L 3/3; listen(expect FAIL) 0/3; gate-only(expect FAIL) 0/3
- **Recipe change?** NO (default)
- **Ping user?** YES

JSON: `music_toy_dig_fire19_20260909.json`  Catalog: `music_to_toy_stressor_catalog_20260909.md`
