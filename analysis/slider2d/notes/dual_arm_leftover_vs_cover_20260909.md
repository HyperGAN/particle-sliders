# Dual-arm leftover-only vs cover-only (2026-09-09)

CPU ablation for Music dual-arm transfer. FM=0, b_cap=1, steps=1200,
n_particles=1 (Music `--parts 0` proxy — toy `n=0` breaks `ParticlePrior.sample`).
Seeds: [0, 1, 2]. Wall: 230.9s.

| cell | geom | pass | primary_mean | primary_min | leak_abs_max |
|---|---|---|---:|---:|---:|
| leftover_only | sheet | 0/3 | 0.4815 | 0.2362 | 0.0598 |
| leftover_only | field3d | 0/3 | 0.5690 | 0.5116 | 0.1160 |
| cover_only | sheet | 0/3 | 0.9910 | 0.9906 | 0.2315 |
| cover_only | field3d | 0/3 | 0.9974 | 0.9950 | 0.4565 |
| locked | sheet | 3/3 | 0.9353 | 0.9343 | 0.0030 |
| locked | field3d | 3/3 | 0.9973 | 0.9925 | 0.0060 |
| neither | sheet | 0/3 | 0.7075 | 0.6994 | 0.5921 |
| neither | field3d | 0/3 | 0.5640 | 0.5285 | 0.7339 |

## Finding

- **leftover_only** (guard, cover=0): leak held (sheet abs≤0.06) but kept fails
  (sheet mean≈0.48). Gate alone undershoots poles.
- **cover_only** (faithful, cover=1.5): kept≈0.99 but leak fails (sheet≈0.23,
  Field3D≈0.46). Cover without gate copies ê — Music lyric/pole pin risk
  without leftover teacher.
- **locked** (guard+cover1.5): **3/3 PASS** sheet+Field3D (kept≈0.935 /
  u_kept≈0.997, leak≈0). Both knobs required.
- **neither**: collapse (kept mid, leak high).

## Music dual-arm read

True #94 needs **both** leftover gate and a cover/pole pin. A Music arm that
is only lyric+tx+lyrichold (no `faithful_guard_e`) is structurally closer to
**cover_only**. A Music arm that is only guard+mlp with pole_weight=0 is
**leftover_only**. Neither alone transfers #94.

JSON: `dual_arm_leftover_vs_cover_20260909.json`
Script: `dual_arm_leftover_vs_cover_20260909.py`
