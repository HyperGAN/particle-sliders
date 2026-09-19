# Particle-count vs residual cover (2026-09-09 fire #5)

SHA `435e873` pop-os `/ml2/music/sliders-conceptmod` CPU. GPUs left alone
(GPU0 ~30 GB / GPU1 ~13 GB reserved, util low; Music/H3/nano-work-server untouched).

Locked recipe: steps=1200, cover_weight=1.5, teacher=`faithful_guard_e`,
b_cap=1.0, fm=0, seed=0. Sweep `n_particles` ∈ {4, 8, 12, 24} (default 12).

## leftover sheet

| n_particles | kept | leak | garble | swing | covered | residual_norm | w_odd | w_even | pass |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|
| 4 | 0.9354 | +0.0004 | 0.0008 | 1.0992 | yes | 1.6537 | 1.1126 | 1.2234 | PASS |
| 8 | 0.9326 | -0.0000 | 0.0008 | 1.0934 | yes | 1.6412 | 1.1063 | 1.2123 | PASS |
| 12 | 0.9295 | +0.0001 | 0.0009 | 1.0874 | yes | 1.6275 | 1.1014 | 1.1981 | PASS |
| 24 | 0.9259 | -0.0005 | 0.0010 | 1.0807 | yes | 1.6117 | 1.0980 | 1.1798 | PASS |

Summary: **4/4 PASS**, kept mean `0.9309`, span `0.0095` (min `0.9259`, max `0.9354`).
Default-12 kept `0.9295` matches fire #2/#3 lock (~0.929–0.930).

## Takeaway

- **PASS/FAIL is nullspace** at the locked recipe: every n_particles ∈ {4,8,12,24}
  clears the leftover gate (kept ≫ 0.90, leak ≈ 0, covered=yes).
- **Monotonic residual steal is real.** Raising n_particles 4→24 drops kept
  0.935→0.926 (−0.009) and residual_norm 1.654→1.612 (−0.042); w_even falls
  more than w_odd. Matches `gan_bcap_findings.md`: particles must not steal the
  mode from the shared residual — cover_weight 1.5 holds the lock, but more
  particles still nibble residual magnitude.
- **Do not raise n_particles above 12.** Prefer default 12 (or 4–8 if shaving
  residual steal); 24 is strictly worse kept with no leak benefit.
- Mapping note for Music `lm_adv` (doc only): particle count is a weak dial once
  cover+steps lock the sheet; if residual undershoots, cut particles / raise
  particle_l2 before touching cover_weight.

Script/JSON: `analysis/slider2d/notes/particles_vs_cover_20260909.{py,json,md}`.
