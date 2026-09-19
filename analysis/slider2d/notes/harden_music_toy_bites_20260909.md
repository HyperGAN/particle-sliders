# Harden Music→toy bites — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Primary wall **1461.3s** (+ vic0 safety ~1134s). CPU only. No Music train.

## Context

- **Fire #20:** `lyric_span_entangle` HARD BOUNDARY — shared AdvResidual cannot
  cover heterogeneous `row_amps` (rows_covered=0/5 despite high row0 u_kept).
  Do not chase cover/n/steps/eoc. Do not weaken cell.
- **cross_axis_rows:** hard boundary (document; do not soften).
- **This fire:** pivot harden dig to `close` / `close_live_noise` seed0 knife @ n=1.

## Diagnosis (close seed0)

- Geometry identical across seeds (`slider=0.12`, content=1.0, scales fixed).
- seed0 @ n=1: `u_kept≈0.38` (û undershoot: on_u≈0.046 vs a_u=0.12);
  `content_kept≈0.97`; multi-row **3/3**; fail = `pass_u` / leftover gate.
- seed1+: û recovers (`u_kept≈1.05`, exam≈0.99).
- **Cause:** single-particle init/basin under default `vicreg_weight=0.05`, not geom or steps.

## Rejected n=1 knobs (seed0 still FAIL, exam≈0.35–0.48)

| family | probes | result |
|---|---|---|
| `particle_l2` | 0.0 … 0.2 | all FAIL |
| sampling | span 0.2/0.6, end 0.3/0.8, cloud 0.01/0.06, jitter 0.02/0.05 | all FAIL |
| lr / steps / cover | lr 2e-3 / 1e-2, steps 1600, cover 2.0 | all FAIL |
| `vicreg=0.15` | worse (exam negative, leak≈0.94) | FAIL |

## What recovers

| posture | PASS | notes |
|---|:---:|---|
| `n_particles≥2` @ locked else | **6/6** | floor is **2** (Fire #21 confirmed; not only ≥4); n=2/3/4 all 6/6 |
| `vicreg_weight=0` @ n=1 | **6/6** close; **6/6** live | portable loss knob; seed0 û recovers |
| more steps / l2 / sampling @ n=1 | no | basin, not budget |

## vic=0 safety (do NOT flip locked default yet)

| cell | vic=0.05 | vic=0.0 |
|---|:---:|:---:|
| leftover n12 c1.5 | 6/6 exam 0.9303 | 6/6 exam 0.9303 |
| leftover n1 c1.0 | 3/3 | 3/3 |
| divergent n12 | 3/3 | 3/3 |
| close_live n1 | **5/6** fail=[0] | **6/6** |
| close n1 | **5/6** fail=[0] | **6/6** |

Leftover flat; close-family knife heals. Still **prefer documenting** over recipe change:
VICReg @ n=1 is ill-posed (std over one particle), so Music parts0 proxy should either
disable VICReg **or** use n≥2 — not a silent global default flip without Music multi-seed.

## Verdict

- **lyric_span / cross_axis:** HARD BOUNDARY (confirm).
- **close / live n=1 knife:** seed0-only û undershoot.
- **Recipe change?** **NO** (locked stays 1200/c1.5/guard/FM0/n≤12/l2=0.02/vic=0.05/b_cap=1).
- **Music parts0 proxy rule (adopt as scoring posture):**
  1. **Multi-seed gate** for close / close_live_noise under n=1, AND/OR
  2. **`n_particles ≥ 2`** floor for close-family cells, AND/OR
  3. **`vicreg_weight=0` when n_particles=1** (candidate portable harden; needs Music confirm).
- **Rejected:** 800×cover3.0; lyric_span cover/n/steps/eoc chase; weaken cross_axis cell;
  particle_l2 / sampling / lr / steps as close-seed0 fix.

JSON: `harden_music_toy_bites_20260909.json` (includes `vic0_safety`).

## Fire #21 cross-ref

`close_harden_fire21_20260909`: close n2 c1.0/c1.5 **6/6**; live n1→n2 recovers.
Same n≥2 harden path. This harden-bites fire adds **vic=0@n=1** candidate + rejected knobs.
