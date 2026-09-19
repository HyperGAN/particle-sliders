# LEADER CARD — Music→toy / slider2d — 2026-09-10

**Question for future fires:** solved, or what’s leading?
**Answer:** **Not solved under locked shared.** Leading stack = locked defaults (unchanged) + analysis ADOPTs (portable / per-row) + propose-only Music YAML/posture. **merge=NO. recipe_change=NO. No Music GPU.**

Host: box-cpu @ `435e87363bf1`. Living pointers → scoreboard + digs below.
Do **not** invent new M-cells; do **not** weaken HARD_BITEs; leave nano-work-server / ComfyUI / `run_server.py` alone.

---

## 1) Locked defaults (shared AdvResidual) — **UNCHANGED**

| knob | value | note |
|---|---|---|
| steps | **1200** | reject 800×cover3.0 |
| cover_weight | **1.5** | Music posture proxy often 1.0 |
| teacher | **`faithful_guard_e`** | leftover-gated |
| FM | **0** | flat |
| n_particles | **≤12** (default 12) | Music parts0 proxy = n=1 |
| particle_l2 | **0.02** | |
| vicreg_weight | **0.05** | Music posture may override @ n=1 only |
| b_cap | **1** | |
| student | **shared AdvResidual** | not per-row / not lowrank in trainer |

**merge_to_trainer = NO.** Locked shared stays the regression default.

---

## 2) What’s leading (analysis ADOPTs) — harness only

| ADOPT | clears / wins | still fails | scope |
|---|---|---|---|
| **`lowrank_k3` + Reoc0 combo** | DoF∪{M14,M21} + CTRL leftover/close/unused_e; M20 stock still HARD_BITE | (none in union under combo; stock declare cells stay stressors) | `ADOPT_analysis_lowrank_reoc0_combo` — analysis/propose-only; merge=NO |
| **`lowrank_k3` portable student** (preferred; ~55 params @M1) | DoF: M1/M2/M16/M17 + deepen M24/M27/M29; CTRL leftover/close/unused_e | declare-lie / ê-floor **M14/M21** (ideal split) | `ADOPT_analysis_portable_student` + deepen_ok |
| **`row_cond`** (heavier portable twin) | same ideal split as lowrank_k3 | M14/M21 | analysis only; prefer lowrank_k3 |
| **per-row AdvResidual `w≤0.3`** | multi-span DoF (M1/M2/M16/M17/M24/M27/M29); couple band [0,0.3] | M14/M20/M21/M28/M30/M31 | heavier cousin of lowrank; merge=NO |

Evidence: `lowrank_reoc0_combo_20260910`, `portable_student_dof_20260910`, `portable_student_lowrank_deepen_20260910`, `declare_e_hygiene_rule_20260910`, `per_row_residual_{explore,deepen}_20260909`, falsify batch4 / M29.

**Ideal portable split (keep):** DoF✓ ∧ declare✗ ∧ CTRL✓ — do not “fix” declare by overpowering heads.

---

## 3) Propose-only Music (YAML / scoring posture) — **not** locked flips

| ADOPT | rule | effect | stay bites |
|---|---|---|---|
| **Reoc0 `e_on_content=0`** | declared_ê must not tilt onto content | under *shared rewrite*: clears **M14 + M21**; Ru(e_on_u=0 alone) clears **neither** (û projected out of guard) | **M20 stock** HARD_BITE (amp-lie); cells stay violation stressors |
| **R0 pure unused** (stronger twin) | e_on_u=0 ∧ e_on_content=0 | also clears M14+M21 | same — propose-only |
| **M20 honest hygiene (`Reoc0`)** | clears M20 under shared+lowrank as **nature-change** (eoc=0 only; soft>0 fails); R0/AmpAlign stronger twins; collateral M26+M28 nature-clear (refuse driver=hot eoc); stock HARD_BITEs | soft caps / gate-raise / Ru | `ADOPT_propose_M20_Reoc0_same_as_DigB` — merge=NO |
| **Close Music posture** | **n≥2** and/or **vic0@n1** + **multi-seed** | heals close-family seed0 knife (M3/M11/M13/M18); Fire #38 reconfirm leftover/unused flat-enough | do **not** flip AdvConfig.vicreg=0.05 |

Evidence: `declare_e_hygiene_rule_20260910`, `vicreg0_n1_propose_stress_20260909`, `close_seed0_deep_20260909`, `core_trio_music_adopt_fire38_20260910`.

```python
# Music-posture scoring harness ONLY — locked AdvConfig.vicreg_weight stays 0.05
if n_particles <= 1:
    cfg = replace(cfg, vicreg_weight=0.0)
```

Gate-raise on `EXAM_LEAK_LOCK` to clear M21 = **rejected** (cheat, not hygiene).

---

## 4) Hard boundaries still open (do not chase with locked recipe)

| family | cells | driver | locked knobs recover? |
|---|---|---|---|
| **shared DoF / multi-row** | M1, M2, M16, M17, M24, M27, M29 | mean-δ / scale / roles / cascade | **NO** (need portable/per-row analysis) |
| **declare-lie content_deleted** | M14, M20, M26 (+ M6 raw) | declared ê ≠ exam leak axis | **NO** (YAML/teacher; Reoc0 propose-only) |
| **ê-floor / eoc admit** | M21, M30, M31 | `e_on_content` tilt → lr floor ≈0.205 | **NO** (Reoc0 propose-only; keep cells) |
| **eoc refuse** | M28 | guard refuse → raw leak | **NO** |
| **close n=1 seed0 knife** | M3/M11/M13/M18 (+ Music M22 soft) | exam_u undershoot basin | posture ADOPT only |
| **dual-arm encoding** | M4/M7/M8 | both arms required | N/A |


**Soft→hard multipair (Fires #32–#37 + lowrank recheck):** see `SOFT_MULTIPAIR_CONTINUUM_20260910.md` — shared DoF-first cliffs M22→{M16≤0.0, M17≤0.15, M2≤0.25} / M23→{M2≤0.15, M17≤0.35, M16≤0.40}; **lowrank_k3 expands soft bands to full continuum** on M22→M16 / M23→M2 / M22→M2 (last_pass=1.0, no bite — `lowrank_soft_continuum_20260910`); Music n1+vic0 seed-knifes M22 starts under shared.

CTRL **leftover / close@n12 / unused_e** must stay green under locked shared.

---

## 5) Scoreboard + dig pointers (fold here first)

| doc | role |
|---|---|
| `MUSIC_TO_TOY_SCOREBOARD_20260909.md` | living cell table + locked reminder + knives |
| `music_to_toy_stressor_catalog_20260909.md` | catalog / natures |
| `research_log_20260909.md` | fire chronology |
| `portable_student_dof_20260910.{md,json}` | Fire #39 ideal portable |
| `portable_student_lowrank_deepen_20260910.{md,json}` | lowrank deepen OK |
| `declare_e_hygiene_rule_20260910.{md,json}` | Reoc0 propose-only |
| `lowrank_reoc0_combo_20260910.{md,json}` | **combo** lowrank_k3+Reoc0 |
| `m20_honest_hygiene_20260910.{md,json}` | **M20** smallest Reoc0 hygiene |
| `core_trio_music_adopt_fire38_20260910.{md,json}` | Music ADOPT reconfirm |
| `vicreg0_n1_propose_stress_20260909.*` / `close_seed0_deep_20260909.*` | close posture |
| **`SOFT_MULTIPAIR_CONTINUUM_20260910.md`** | **Fires #32–#37 soft→hard multipair continuum card (portable)** |
| `lowrank_soft_continuum_20260910.{md,json}` | **lowrank vs shared soft continuum shifts** |
| cliff digs `m22_*` / `m23_*` / `highd_*` / `sheet_*` / `exam_leak_*` | soft-band cliffs (analysis harness) |

---

## 5b) Fire #40 refine — leak_max honesty (2026-09-11)

Prior soft-continuum / portable “clears M2” claims used **row0-only** `pass_leak`.
**ADOPT_analysis_lowrank_leakmax_honesty:** analysis clears/soft-bands must also require
`leak_max_rows ≤ EXAM_LEAK_LOCK (0.20)`.

| claim | after leak_max gate |
|---|---|
| lowrank clears **M16** / scale continuum | **still YES** (honest) |
| lowrank clears **M2** / cross_axis | **NO** (DoF✓ coverage, leak_max≈2.57 → honest 0/3) |
| soft continuum “full soft under lowrank” | **only M22→M16**; →M2 expands but bites later via leak_max |

Evidence: `lowrank_leak_honesty_fire40_20260911.{md,json}`. merge=NO.

## 6) Status one-liner

> **Leading, not solved:** locked shared remains the green CTRL + HARD_BITE regression floor; **lowrank_k3+Reoc0 combo** leads analysis union (DoF∪{M14,M21}+CTRL); **lowrank_k3 alone** expands soft→hard DoF continuum to full soft (M22→M16/M23→M2/M22→M2); **M20 stock still HARD_BITE** — smallest honest clear is same **Reoc0** nature-change (eoc cliff: only exact 0; AmpAlign/R0 stronger; collateral M26+M28 nature-clear); Music close uses **n≥2 ∨ vic0@n1 + multi-seed**. Nothing merges to trainer.

*Append new ADOPTs / cliff knifes below; bump date in filename on material regime change.*
