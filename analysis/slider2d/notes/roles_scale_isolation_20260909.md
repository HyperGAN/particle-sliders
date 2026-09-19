# roles/scale isolation vs cross_axis — 2026-09-09

Wall 391.9s. Soften ladder: homo stagger → roles split → full cross_axis.

| cell | PASS | multi | exam | rows | leak | fail |
|---|:---:|:---:|---:|---:|---:|---|
| `leftover_n12` | 3/3 | 3/3 | 0.9304 | 1.0 | 0.0004 | [] |
| `scale_stagger_homo_n12` | 0/6 | 0/6 | 0.9434 | 3.0 | 0.0021 | [0, 1, 2, 3, 7, 42] |
| `roles_split_n12` | 0/6 | 0/6 | 0.6266 | 0.0 | 0.0067 | [0, 1, 2, 3, 7, 42] |
| `cross_axis_rows_n12` | 0/6 | 0/6 | 0.5848 | 0.0 | 0.3839 | [0, 1, 2, 3, 7, 42] |
| `blend_mix0.0_n12` | 3/3 | 3/3 | 0.9908 | 4.0 | 0.0005 | [] |
| `blend_mix0.25_n12` | 3/3 | 3/3 | 0.9009 | 4.0 | 0.0005 | [] |
| `blend_mix0.5_n12` | 0/3 | 0/3 | 0.8152 | 0.0 | 0.0017 | [0, 1, 2] |
| `blend_mix0.75_n12` | 0/3 | 0/3 | 0.7295 | 0.0 | 0.0013 | [0, 1, 2] |
| `blend_mix1.0_n12` | 0/3 | 0/3 | 0.641 | 0.0 | 0.0087 | [0, 1, 2] |
| `roles_split_n1_c1.0` | 0/6 | 0/6 | 0.6417 | 0.0 | 0.0243 | [0, 1, 2, 3, 7, 42] |
| `scale_stagger_n1_c1.0` | 0/6 | 0/6 | 0.9434 | 3.0 | 0.0086 | [0, 1, 2, 3, 7, 42] |

JSON: `roles_scale_isolation_20260909.json`


## Takeaway

- Soften ladder: **all-u (mix0–0.25) PASS** → **mix≥0.5 FAIL** (rows_cov drops 4→0).
- Cliff between mild content-lerp and roles_split / cross_axis hard bites.
- `scale_stagger_homo` still BITES (rows≈3) without axis mix — multi-row scale DoF alone.
- Recipe change: **NO**. Shared residual limit; per-row band is separate ADOPT (sibling).
