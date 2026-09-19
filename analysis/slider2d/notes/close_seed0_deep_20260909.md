# close_seed0_deep — f3d_close / close_live_noise seed0 knife (2026-09-09)

Host: box-cpu @ `435e87363bf1`. Wall **2175.1s**. CPU only. No Music train.
Folds **Fire #21** (`close_harden_fire21_20260909`): close knife clears at **n≥2**.

## Decision (TL;DR)

**`mandatory_multi_seed_gate_PLUS_n_ge_2_harden_PLUS_propose_vic0_when_n1`**

MANDATORY multi-seed gate for close/close_live_noise under Music parts0 (n=1). Operational harden (Fire #21 + this dig): n_particles>=2 clears knife (close+live 6/6). PROPOSE (not silent): vicreg_weight=0 when n_particles==1 (21/21 heal; leftover n12 flat 0.9303→0.9303). Do NOT silently flip locked defaults. Rejected: l2/b_cap/cover/jitter/cloud as seed0@n=1 fix; 800×cover3.0.

### Rationale

- seed0 fail is û undershoot (exam/pass_u), not leak or multi-row — init/basin under n=1
- seeds0..20 still knife: close 20/21 fail=[0]; live 20/21 fail=[0]
- no seed0@n=1 recovery from particle_l2 / b_cap / cover / jitter / cloud
- n_particles>=2 recovers 6/6 close+live under locked else
- vicreg=0 @ n=1 heals 21/21 close+live with leftover n12 flat — propose conditional Music-posture (vic=0 when n=1), NOT silent global default flip
- Fire #21 fold: close n2 c1.0/c1.5 = 6/6; live n1→n2 recovers — n>=2 is the operational harden (tighter than prior n>=4 note)
- cover=1.0 seeds0..20: 19/21 fail=[0, 4] (not seed0-only) — strengthens multi-seed gate

- **Silent recipe change?** NO (locked stays 1200/c1.5/guard/FM0/n≤12/l2=0.02/vic=0.05/b_cap=1)
- **Propose portable conditional?** YES — `vicreg_weight=0` when `n_particles==1` (not applied)
- **Leftover regression under vic=0?** NO (n12: 6/6@0.9303 → 6/6@0.9303)

## Fire #21 fold (n≥2 harden)

| cell | PASS |
|---|:---:|
| close n1 c1.0 / c1.5 | 5/6 / 5/6 (seed0 knife) |
| close n2 c1.0 / c1.5 | 6/6 / 6/6 |
| live n1 → n2 | 3/4 → 4/4 |
| this dig n_floor close∧live | **2** |

Prior note said n≥4; **Fire #21 + this dig tighten to n≥2**.

## 1) Failure mode (seed0 vs seed1, close c1.5 n=1)

**Classification: `exam_u_undershoot_init_basin`**

| seed | pass | u_kept | on_u | content | leak | pass_u | pass_c | pass_leak | swing | multi |
|---:|:---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 0 | False | 0.3807 | 0.0457 | 0.9658 | 0.0344 | False | True | True | True | True |
| 1 | True | 1.0485 | 0.1258 | 0.9924 | 0.0092 | True | True | True | True | True |

Not leak (c1.5 leak≪0.20). Not multi-row (3/3). Exam fails via `pass_u` / leftover gate because û undershoots (on_u≈0.046 vs a_u≈0.12). Content+swing OK → **init/basin** under single particle + default VICReg.

## 2) Seeds 0..20 @ n=1

| cell | PASS | mean exam | fail seeds | knife |
|---|:---:|---:|---|:---:|
| close c1.5 | 20/21 | 0.9615 | [0] | True |
| close c1.0 | 19/21 | 0.9144 | [0, 4] | True |
| live c1.5 | 20/21 | 0.9688 | [0] | True |
| close c1.5 **vic=0** | 21/21 | 0.9872 | [] | False |
| live c1.5 **vic=0** | 21/21 | 0.9897 | [] | False |

**Note:** cover=1.0 fail=[0,4] → not seed0-only across 0..20; multi-seed gate is mandatory.

## 3) n ∈ {1,2,3,4} (std seeds) — confirms Fire #21

| n | close c1.5 | live c1.5 |
|---:|:---:|:---:|
| 1 | 5/6 | 5/6 |
| 2 | 6/6 | 6/6 |
| 3 | 6/6 | 6/6 |
| 4 | 6/6 | 6/6 |

**n_floor for 6/6 close∧live = 2** (Fire #21 aligned)

## 4) seed0@n=1 knob sweeps (recover?)

| family | any recover? |
|---|:---:|
| particle_l2 | False |
| b_cap | False |
| cover (incl 3.0) | False |
| particle_jitter | False |
| cloud_std | False |
| vicreg=0 | True (full: 21/21 / 21/21) |

Recovered seed0-only cells: `[]`

## 5) Proposal (evidence-gated; **not applied**)

**PROPOSE** portable conditional (do not silently change `default_cfg`):

```python
# Music parts0 / n_particles==1 posture only
if cfg.n_particles == 1:
    cfg = replace(cfg, vicreg_weight=0.0)  # VICReg std ill-posed at n=1
```

Evidence:
- close n1 vic0 seeds0..20: **21/21** (default was 20/21)
- live n1 vic0 seeds0..20: **21/21** (default was 20/21)
- leftover n12: vic0.05 6/6@0.9303 vs vic0 6/6@0.9303 (flat; no regression)
- Still require **multi-seed gate** + prefer **n≥2** operational harden (Fire #21)

## Rejected

- 800×cover3.0 false lock
- raise particle_l2 / b_cap / cover / jitter / cloud as seed0@n=1 fix
- silent global vicreg default flip without Music multi-seed
- weaken close_live_noise cell

JSON: `close_seed0_deep_20260909.json`
Fire #21: `close_harden_fire21_20260909.{json,md}`
