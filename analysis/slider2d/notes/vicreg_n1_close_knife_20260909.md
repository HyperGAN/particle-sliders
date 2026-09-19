# vicreg @ n_particles=1 vs close seed0 knife — 2026-09-09

Host: box-cpu. Evidence from `harden_music_toy_bites_20260909` (+ `vic0_safety`).
Complements Fire #21 (n≥2 harden). **No locked-recipe change.**

## Finding

Default `vicreg_weight=0.05` with **`n_particles=1`** destabilizes close-family
fits on seed0: residual captures content (`content_kept≈0.97`) but **undershoots û**
(`u_kept≈0.38`, on_u≈0.046 vs a_u=0.12). Multi-row coverage still 3/3.

Setting **`vicreg_weight=0`** at n=1 recovers:

| cell | vic=0.05 | vic=0.0 |
|---|:---:|:---:|
| close n1 c1.5 | 5/6 fail=[0] | **6/6** |
| close_live n1 | 5/6 fail=[0] | **6/6** |
| leftover n12 | 6/6 | 6/6 (flat) |
| leftover n1 | 3/3 | 3/3 |
| divergent n12 | 3/3 | 3/3 |

## Why (portable)

`vicreg_loss` targets per-dim std across the particle batch. With **one** particle,
std is ill-posed / near-zero → the VICReg term fights the cover/û pin and can
trap the shared residual in a content-only basin (seed-dependent via critic/init RNG).

This is **not** a 3D-only hack: same AdvConfig/ParticlePrior on any dim.

## Relation to Fire #21 n≥2

Both fix the knife:

1. **n≥2** — VICReg becomes well-posed; particles diversify (Fire #21 primary harden).
2. **vic=0 @ n=1** — removes ill-posed term under parts0 proxy.

Prefer documenting **both** as Music parts0 postures. Do **not** silently set
global `vicreg_weight=0` (still useful at n=12). Candidate: `if n_particles<=1: vicreg=0`
in Music-posture harness only — needs Music multi-seed confirm before adopt.

## Rejected as seed0@n=1 fix

particle_l2 ∈[0,0.2], span/end/cloud/jitter, lr, steps1600, cover2.0 — all still FAIL.

JSON parent: `harden_music_toy_bites_20260909.json` → `verdict` + `vic0_safety`.
