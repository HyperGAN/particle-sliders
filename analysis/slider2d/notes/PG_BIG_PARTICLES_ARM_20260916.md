# CLONE ARM 1/5 — `pg_big_particles` vs locked_shared

Date: 2026-09-16. Arm: `pg_big_particles` (propose_only). Only delta vs
`locked_cfg`: `n_particles` 12 → 64 (feasible_max 256). All other knobs
byte-identical to `analysis/slider2d/locked_baseline_defaults.py`
(`assert_only_delta` in `tests/test_pg_big_particles.py` proves it).
Production argv (`run_lm_adv.py`) and the live trainer default
(`--lm_target v9`) are untouched.

## Measured (CPU, torch 2.2.1+cpu, seed 0)

Field2D polarity, 200 steps:

| cfg | wall | slider cos | leak ratio | ±1 cos | leak_frac | pass |
|---|---|---|---|---|---|---|
| locked (n=12) | 1.1s | +1.000 | −0.001 | −1.000 | −1.000 | True |
| arm (n=64) | 0.8s | +1.000 | +0.001 | −1.000 | −1.000 | True |
| arm-max (n=256) | 0.8s | +1.000 | −0.007 | −1.000 | −1.000 | True |

Sheet leftover (`leaky_field`), 1200 steps:

| cfg | wall | leak_tok | on_sheet_kept | garble | swing_kept | pass |
|---|---|---|---|---|---|---|
| locked (n=12) | 5.3s | −0.0003 | 0.930 | 0.001 | 1.089 | True |
| arm (n=64) | 5.0s | −0.0001 | 0.914 | 0.001 | 1.057 | True |

Gaussian smoke, 8 modes / 1200 steps / seed 1234:

| cfg | wall | modes | hq | grad_peak_med |
|---|---|---|---|---|
| locked (n=16) | 4.6s | 8/8 | 1.000 | 0.92 |
| arm (n=64) | 4.3s | 8/8 | 1.000 | 0.87 |

## Verdict vs locked_shared: HOLD

- **KEEP** locked_shared as production. The arm moves no gate: same
  pass/fail everywhere, kept/hq within noise (±0.02), D-steepness bound
  intact (peak 0.87 vs 0.92, both ≤ 2.0).
- **HOLD** `pg_big_particles` as propose_only. It is cheap and harmless
  (wall time flat: particle ops are trivial next to the critic
  grad-penalty on 2-D toys), so it stays available if a high-dim cell
  ever needs cloud scale — but nothing here justifies flipping n=12.
- **DROP** the hypothesis that particle count is the binding formulation
  gap on 2-D toys. On dim=2 the scored object is the shared residual and
  particles are a jitter prior by design (pinned by particle_l2=0.02 +
  VICReg); 12 → 256 changes neither gates nor cost. The remaining gap is
  update budget / data geometry (see gap note), not cloud scale.

No Music GPU used. No win claimed: the arm ties locked, which is the
honest result for ARM 1/5.
