# ParticleGAN clone salvage — one thin propose_only PR (consolidated)

Base: `be5e0e4` (#110: bug hunts A + B + D + C + E reconciled).
Supersedes HOLD arms #111 #112 #113 #114 #115. `PROPOSE_ONLY=True`,
`MERGE_TO_TRAINER=False`. CPU-only, no Music GPU, no audio claims.

## Verdict

- **Locked: KEEP.** b_cap coeff=1.0, kappa=1.0, arm=b_cap, norm=l2, lazy=1,
  anneal=none + cover_weight=1.5, teacher=faithful_guard_e, FM off,
  n_particles=12, particle_l2=0.02, vicreg_weight=0.05, shared LR 5e-3
  (d/prior mults 1.0). `AdvConfig()` defaults, Music `ARM_B` argv, and the
  live `--lm_target v9` default are untouched (pinned by
  `tests/test_pg_clone_consolidated.py`).
- **All five arms: HOLD** (not KEEP, not DROP as directions) — each ties or
  tracks locked on CPU toy gates with no instability to justify DROP, but
  none has the multi-seed / full-gate / GPU-transfer evidence to justify
  KEEP.

## Salvage map (what was kept vs dropped)

| arm | source PR | salvaged (thin) | dropped (deliberately) |
|---|---|---|---|
| 1/5 big particles | #111 | `locked_baseline_defaults.py` single-source (`LOCKED`, `locked_cfg`, `arm_cfg`, `assert_only_delta`, `PRODUCTION_ARGV_DEFAULTS`, `build_parser` pin) | n-particles as a living recipe: no trainer row, no CLI flag, no default change; n=64 stays a propose-only delta (HOLD tie: field2d/sheet/gaussian tie, wall time flat) |
| 3/5 2x LR | #112 | `d_lr_mult` / `prior_lr_mult` on `AdvConfig` (default 1.0), `toy_lr_triplet()`, per-party LR wiring in `fit_adv` / `train_lm_adv`, opt-in `--lr` / `--d-lr-mult` / `--prior-lr-mult` CLI, `pg_2x_lr_cfg()` card in `pg_clone_propose.py` | 10x prior (HOLD as separate variable: particles must stay a jitter prior under the cover pin); LR-clone runs never overwrite locked findings pages |
| 2/5 VicReg faithful | #113 | `vicreg_faithful_loss` (linear-hinge var + cov, no sim, 1e-5 vs upstream) + optional `vicreg_fn` on `fit_adv` / `train_lm_adv` (`None` = locked path byte-identical) + `pg_vicreg_faithful_cfg()` (weight 1.0) | separate `pg_vicreg_faithful.py` / `docs/pg-vicreg-faithful-arm.md` full tree; folded into `pg_clone_propose.py` + this note |
| 4/5 anneal + g_interp | #115 | opt-in `--grad-arm {b_cap,g_interp_cap}` / `--target-anneal {none,linear,delayed}` CLI + threading through `collect()` (defaults locked); anneal-center + interp-geometry pins in the consolidated test | 4-card `propose_anneal.py` harness + `anneal_ginterp_smoke.md` full tree (redundant: HOLD 4-way tie vs locked documented here; math already in vendored `GradRegularizer`) |
| 5/5 full clone | #114 | thin `pg_full_clone_cfg()` (existing knobs only: beta2 0.999, n=32, vicreg 1.0, particle_l2 0.0) + `PG_FULL_CLONE_LEDGER` gap checklist in `pg_clone_propose.py` | new `AdvConfig` fields (`vicreg_*`, `ema_scope`, `delay_frac`, `recipe`), `effective_delay` / `ema_param_groups` / `vicreg_loss_for_cfg` wiring, `run_pg_full_clone.py`, trainer `--adv_preset pg_full_clone` row, `docs/pg-full-clone*.md` full tree; NOT the default, never satisfies `--require_arm_b` (no trainer change at all) |

## Formulation gap checklist (remaining drifts vs ParticleGAN proper)

1. Particle scale: 20k movable prior vs 12/side toy (32/side in full-clone preset; PARTIAL).
2. Update budget: ~7k updates vs 1200 toy steps (HOLD).
3. Data geometry: real modes vs leftover-gated caption poles + span/end cloud (Music HOLD).
4. Mode pin: `cover_weight=1.5` has no ParticleGAN analogue (Music HOLD).
5. Critic capacity: full critic vs Fourier-2 MLP toy width (HOLD).
6. LR values/hold: G 6e-4 / x1.5 D / x10 prior + 60% hold vs shared 5e-3 + delay 80 (HOLD; per-party wiring salvaged, values not flipped).
7. EMA scope: G+particles vs residual-only (HOLD; not wired — gap).
8. VICReg inner scale: 4-D particle scale vs 2-D fixture scale (HOLD).
9. Teacher: modes/spans vs faithful_guard_e (Music HOLD).
10. Trainer transfer: hidden-space particles + VICReg weight 1.0 on Music GPU is unproven (no trainer row added).

## Files in this PR (thin surface)

- `analysis/slider2d/adv.py`: +`d_lr_mult` / `prior_lr_mult` (1.0) + `toy_lr_triplet()`.
- `analysis/slider2d/gan.py`: per-party LR groups + `vicreg_fn=None` on `fit_adv` / `train_lm_adv`.
- `analysis/slider2d/locked_baseline_defaults.py` (new): single-source LOCKED.
- `analysis/slider2d/pg_clone_propose.py` (new): `PROPOSE_ONLY` / `MERGE_TO_TRAINER=False`, faithful loss, 2xLR / vicreg / full-clone presets, ledger, disposition.
- `analysis/slider2d/run_lm_adv.py`: `build_parser()` + 5 opt-in flags (locked defaults) + no-overwrite guard for arm runs.
- `analysis/slider2d/notes/PG_CLONE_SALVAGE_20260916.md` (this note).
- `tests/test_pg_clone_consolidated.py` (new): locked pins + salvaged-hook tests, CPU-only.

## CPU evidence (thin smokes in the consolidated test)

- Locked pins: `AdvConfig()` == `LOCKED` == `default_cfg()` == production argv; live stays `--lm_target v9`, `ARM_B` row untouched.
- 2xLR: field2d 250-step smoke locked vs arm both finite (arm tracks/ahead, b_cap norms O(1)).
- VicReg: formulation vs inline upstream transcription to 1e-5; `None` path byte-identical; sheet/arm smoke finite.
- Anneal/g_interp: linear/delayed center math, `total_steps` required, `g_interp_cap` closed form + interp-vs-sample geometry, per-card finite smoke.
- Full clone: preset values + ledger coverage + locked-default pins; thin smoke finite.
