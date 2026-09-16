# CLONE 5/5 — pg_full_clone: full recipe port (propose-only)

One coherent closest-ParticleGAN clone toy arm. Closes the five intentional
remaining drifts vs ParticleGAN (particle count, VICReg, β2 0.99 vs 0.999,
absolute LR delay vs 60% hold, residual-only EMA) as far as CPU toys allow,
in a single preset — `analysis/slider2d/adv.py::pg_full_clone_cfg()`.

Propose-only: this never flips the Music trainer production argv, never
touches `AdvConfig()` defaults, and never replaces `locked_shared` (the
`ARM_B` row). The Music trainer carries a matching propose-only preset
(`--adv_preset pg_full_clone`: same scaffolding, `--parts 8`,
`--vicreg_weight 1.0`) that is refused by `--require_arm_b` by
construction. No Music GPU train in this port; no rendered-audio claim.

## Recipe ledger (per-knob match/drift)

| knob | ParticleGAN | pg_full_clone | locked toy | status |
|---|---|---|---|---|
| GAN loss | RpGAN pair logistic | RpGAN pair logistic | RpGAN pair logistic | MATCH |
| b_cap / κ / norm | 1 / 1 / L2 | 1 / 1 / L2 | 1 / 1 / L2 | MATCH |
| critic | Fourier-2 MLP | Fourier-2 MLP (toy width) | Fourier-2 MLP (toy width) | HOLD |
| particles | 20k prior | 32/side (64 total) | 12/side (24 total) | PARTIAL |
| VICReg weight/form | 1, var+cov | 1, var+cov (sim 0) | 0.05, sim+var+cov | MATCH |
| VICReg inner scale | 4-D particle scale | 2-D fixture scale (var 10/cov 1/std 0.05) | 2-D fixture scale | HOLD |
| Adam β2 | 0.999 | 0.999 | 0.99 | MATCH |
| LR hold | 60% hold + cosine | delay_frac 0.6 | absolute delay 80 | MATCH |
| LR values | 6e-4 G, ×1.5 D, ×10 prior | shared 5e-3 | shared 5e-3 | HOLD |
| EMA | G+particles, 0.995 | g_all, 0.995 | residual-only, 0.995 | MATCH |
| κ anneal | static cap in b_cap=1 run | none (static) | none (static) | HOLD |
| feature matching | off | off (0.0) | off (0.0) | MATCH |
| particle L2 | none | 0.0 | 0.02 | MATCH |
| teacher | modes / spans | faithful_guard_e | faithful_guard_e | HOLD |
| span/end cloud | lyric-span + last-token | span 0.40 / end 0.60 | span 0.40 / end 0.60 | HOLD |
| cover pin | none | 1.5 (sheet/exam width) | 1.5 (sheet/exam width) | HOLD |

## KEEP / HOLD / DROP

KEEP (already ParticleGAN-faithful, unchanged): RpGAN pair logistic, b_cap
coeff 1 / κ 1 / L2 through the vendored `GradRegularizer`, FM off,
leftover-gated `faithful_guard_e` teacher, live `--lm_target v9` default,
locked_shared `ARM_B` row byte-identical.

HOLD (kept deliberately, documented above): shared-LR value (no per-group
6e-4/×1.5/×10 ratios on the toy), toy critic width, fixture VICReg inner
scaling, static κ (delayed center-anneal exists in `GradRegularizer` but is
enabled in neither toy recipe), particle-count magnitude, cover pin and
span/end cloud (Music adaptations: pure RpGAN undershoots sheet/exam width
on CPU — the 2-D HQ analogue).

DROP (removed vs the locked toy to match upstream): sim-invariance VICReg
term (`vicreg_sim_weight` 10 → 0), particle L2 (0.02 → 0), absolute LR
delay 80 (→ 60% hold), β2 0.99 (→ 0.999), residual-only EMA (→ G+particles).

## Gate/smoke vs locked_shared (CPU, seed 0, 1200 steps)

```
PYTHONPATH=. python analysis/slider2d/run_pg_full_clone.py --steps 1200 --seed 0
PYTHONPATH=. pytest tests/test_pg_full_clone.py tests/test_lm_2d_adv.py tests/test_music_arm_b.py tests/test_music_arm_b_gates.py -q
```

Measured (`docs/pg-full-clone-metrics.json`):

| cell | locked_shared | pg_full_clone |
|---|---|---|
| field2d | PASS (slider 1.000, leak 0.000, ±1 −1.000) | PASS (slider 1.000, leak −0.001, ±1 −1.000) |
| sheet leftover | PASS (leak −0.000, kept 0.930) | PASS (leak 0.000, kept 0.927) |
| exam divergent | PASS (overlap 1.000, swing 1.000) | PASS (overlap 1.000, swing 1.000) |

The clone reaches parity with the locked row on every cell — closing the
five drifts regresses nothing at the pinned budget. (At 300 steps both rows
fail the sheet-kept lock together at ~0.55: under-budget, not a recipe
difference.)

## Gap checklist (what stays open)

- [ ] 20k-particle prior (clone: 32/side — direction closed, magnitude CPU-bound)
- [ ] Per-group LRs (6e-4 G / ×1.5 D / ×10 prior; trainer has no knobs for these)
- [ ] Critic width match at scale (toy 64-hidden / 16-bank held)
- [ ] Delayed κ-anneal evaluation (static in both toy recipes)
- [ ] Music GPU transfer: hidden-space particles (`--parts 8`) + VICReg 1.0,
  eval scales `-1, 0, 0.5, 1`, multi-seed — the trainer preset declares the
  row but proves nothing until a GPU run
- [ ] Any win claim still needs multi-seed GPU + the locked_shared gate
  (`--require_arm_b` stays the `ARM_B` row: parts 0, vicreg 0)
