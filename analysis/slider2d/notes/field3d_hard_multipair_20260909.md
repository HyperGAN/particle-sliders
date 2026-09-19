# Field3D HARD multi-pair / cross-axis — 2026-09-09 (Fire #13)

Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,
fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.

Ask: more pairs + mismatched leftover/content + wide rows + ≥6 seeds.

Seeds: [0, 1, 2, 3, 7, 42]

## Per-cell summary

| cell | PASS | primary mean | primary span | leak abs max | knife_edge |
|:---|:---:|---:|---:|---:|:---:|
| sheet_leftover | 6/6 | 0.9302 | 0.0016 | 0.0001 | no |
| sheet_gender | 6/6 | 0.9955 | 0.0010 | 0.0003 | no |
| exam_divergent | 6/6 | 1.0000 | 0.0000 | 0.0000 | no |
| exam_close | 6/6 | 1.0000 | 0.0000 | 0.0000 | no |
| exam_unused_e | 6/6 | 0.9870 | 0.0312 | 0.0006 | no |
| f3d_divergent | 6/6 | 0.9216 | 0.0066 | 0.0011 | no |
| f3d_close | 6/6 | 0.9828 | 0.0094 | 0.0097 | no |
| f3d_unused_e | 6/6 | 0.9203 | 0.0008 | 0.0017 | no |
| f3d_mismatch_content_heavy | 0/6 | 0.9989 | 0.0004 | 0.0682 | no |
| f3d_mismatch_leak_heavy | 0/6 | 0.9998 | 0.0004 | 1.1867 | no |
| f3d_mismatch_cross_declare | 0/6 | -0.0436 | 0.0045 | 0.6965 | no |
| f3d_entangled_wide_rows | 0/6 | 0.8396 | 0.0025 | 0.0048 | no |
| f3d_leftover | 6/6 | 0.9917 | 0.0015 | 0.0009 | no |

### Finding

- Locked recipe fails on: f3d_mismatch_content_heavy, f3d_mismatch_leak_heavy, f3d_mismatch_cross_declare, f3d_entangled_wide_rows. Document as transfer risk; do not weaken recipe without multi-seed margin.
- Verdict: `hard_multipair_partial`
- Wall: 776.2s

## Transfer interpretation (dual-arm / post CLI gaps)

Solid PASS (6/6, no knife): sheet_leftover/gender, exam_*, f3d PairField
(divergent/close/unused_e), f3d_leftover — **these transfer** as leftover-gate
arm content (`faithful_guard_e` + mlp, `--parts 0`).

Solid FAIL (0/6, not knife-edge — good, not false locks):
- `f3d_mismatch_content_heavy`: declared ê points at û while content dominates → leak gate fails (~0.06)
- `f3d_mismatch_leak_heavy`: unused ê amplitude ≫ û → leak_ratio ~1.18
- `f3d_mismatch_cross_declare`: declared_e names content while a has unused ê → exam_score negative (guard refuse / attribution break)
- `f3d_entangled_wide_rows`: 5 wide row_scales + equal c/e → exam_score ~0.84 < cont floor

**Music lesson:** locked recipe does **not** rescue wrong `leak_*` YAML or
extreme leftover/content mismatch. Port PairField cells, not pathological
declare. Particles did not cause these fails (locked n=12 throughout).
Do not map n_particles → `--parts`.
