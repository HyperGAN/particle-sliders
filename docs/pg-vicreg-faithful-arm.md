# CLONE ARM 2/5 — VICReg-faithful propose-only arm (`pg_vicreg_faithful`)

Upstream: `particlegan 0.2.0` `particlegan/vicreg_loss.py`
(`VICRegLikeLoss` + `ParticleRegularizer`): **weight 1, `target_std=1`,
var+cov only** (no invariance/sim term). Demo (`analysis/slider2d/adv.py`
`vicreg_loss`, `AdvConfig.vicreg_weight=0.05`): sim+var+cov, `std_target=0.05`,
squared variance hinge, per-term weights 10/10/1, noise augmentation.

Arm lives in `analysis/slider2d/pg_vicreg_faithful.py` (`PROPOSE_ONLY = True`):
`vicreg_faithful_loss` + `pg_vicreg_faithful_cfg` (locked recipe, only
`vicreg_weight` → 1.0). `fit_adv` / `train_lm_adv` accept an optional
`vicreg_fn` (default `None` = locked demo path, byte-identical). Locked
defaults and Music trainer argv are untouched; CPU tests in
`tests/test_pg_vicreg_faithful.py`. No Music GPU.

## Formulation delta (demo → this arm)

| Line | Upstream | Demo (locked) | This arm |
|---|---|---|---|
| weight | 1 (`Recipe.prior_reg`) | 0.05 | **1.0** |
| `target_std` | 1.0 | 0.05 | **1.0** |
| invariance/sim | absent | MSE(z, z+noise), w=10 | **dropped** |
| variance | `mean(relu(t − sqrt(var_unbiased + eps)))`, eps=1e-4 | `mean(relu(t − std_biased)²)`, no eps | **upstream** |
| covariance | off-diag² sum / dim, `(N−1)` norm | same value | **same** |
| `< 2` rows | silent 0 | raises | **raises** (intentional drift, fail-closed) |

## Remaining drift vs upstream (closest the toy allows)

1. **Target rows**: upstream regularizes a sampled prior batch; the toy has no
   sampled batch at the G step, so both recipes apply VICReg to the
   concatenated prior tables (`prior_p` + `prior_m` particles). Same call
   site for locked and arm — relative comparison is apples-to-apples.
2. **`< 2`-row behavior**: upstream returns a silent 0; the arm raises like
   the demo (BUG HUNT B fail-closed — a 1-particle prior would otherwise lose
   spread pressure silently).
3. **Covariance spelling**: `off.pow(2).sum() − diag.pow(2).sum()) / d` instead
   of upstream's flatten trick — algebraically identical, verified equal to
   1e-5 on synthetic batches in tests.
4. **Scale context**: upstream `target_std=1` spreads a `z_dim=4` / 20k-particle
   prior; the toy spreads 2×12 particles around O(1) poles. Whether std=1 is
   the right pressure at toy scale is unproven (hence HOLD, not KEEP).
5. **Everything else locked**: `b_cap` coeff/kappa, cover, `particle_l2`,
   span/end cloud, LR/schedule, EMA, β — shared with the locked recipe.

## Disposition (KEEP / HOLD / DROP)

| Knob | Verdict | Note |
|---|---|---|
| `b_cap` κ=1 | KEEP | locked GradRegularizer path, reused unchanged |
| Music extras (cover, span/end cloud, `particle_l2`) | KEEP | arm keeps them; only VICReg moves |
| `vicreg_weight` 0.05 → 1.0 | HOLD | needs multi-seed GPU before any flip |
| formulation sim+var+cov²-hinge → var+cov linear-hinge | HOLD | same gate as weight |
| `std_target` 0.05 → 1.0 | HOLD | toy-scale fit unproven |
| Music trainer argv (`ARM_B` vicreg 0 at `--parts 0`) | DROP | weight 1.0 refused under `--adv_preset arm_b` / `--require_arm_b` by design |
| live defaults (v9 / hidden / none) | DROP | untouched |

## CPU smoke vs locked_shared (seed 0)

`tests/test_pg_vicreg_faithful.py` asserts finiteness + locked-shape on both
paths (smoke, not gates). Quoted runs:

- Field2D polarity, 200 steps — locked: cos_slider 1.0, leak −0.0005,
  collapse −1.0; arm: cos_slider 0.9986, leak 0.0525, collapse −0.9999.
- Sheet `fit_adv`, 200 steps — both finite, `b_cap` 1.0 on both arms
  (coverage False at this budget for both; budget too small to cover).
- Sheet `fit_adv`, **1200 steps** (arm only, one-off, seed 0) — leftover:
  err 0.106/0.113, covered=True all-rows (worst 0.185); gender: err
  0.051/0.081, covered=True all-rows (worst 0.169). Weight-1/std-1/var+cov
  does not break pole coverage on either sheet cell at this budget/seed.
  Multi-seed + exam rollout still required before any HOLD → KEEP.
