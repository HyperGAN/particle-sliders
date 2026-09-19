# Soft→hard multipair continuum card — Fires #32–#37 — 2026-09-10

Host: box-cpu @ `435e87363bf1`. **Docs synthesize only** (no new dig). CPU only. **No Music GPU.**
Locked shared AdvResidual **unchanged**. merge=**NO**. recipe_change=**NO**. No new CELLS.

Portable fold of soft→hard lerp cliffs from Music soft starts
(`M22` = `stagger_mild_cross`, `M23` = `multipair_corr_seed`) into hard DoF targets
(`M2` cross_axis, `M17` roles_split, `M16` scale_stagger_homo).

Evidence digs: `m22_soft_multipair_cliff_20260910`, `m23_corr_multipair_cliff_20260910`,
`m22_to_m17_roles_cliff_20260910`, `m23_to_m17_roles_cliff_20260910`,
`m22_to_m16_scale_cliff_20260910`, `m23_to_m16_scale_cliff_20260910`.

---

## TLDR continuum table (locked n=12)

| path | Fire | last_pass t | first_bite t | first_all_fail t | first_bite | leak later? | Music n1+vic0 |
|---|:---:|---:|---:|---:|---|---|---|
| **M22→M2** | #32 | **0.25** | **0.40** | 0.40 | exam_dof (lr≈0.001) | yes ≥0.60 (lr≈0.26→0.38) | seed-knife @0 (1/3); pass@0.25; fail@0.40 |
| **M23→M2** | #33 | **0.15** | **0.25** | 0.25 | exam_dof (lr≈0.018) | yes ≥0.50 (lr≈0.22→0.38) | **more tolerant**: pass@0.25 → fail@0.40 |
| **M22→M17** | #34 | **0.15** | **0.25** | 0.25 | exam_dof (lr≈0.0005) | no (lr≲0.004 @1.0) | knifes earlier (1/3 @0; never all-pass) |
| **M23→M17** | #35 | **0.35** | **0.40** | 0.40 | exam_dof (lr≈0.019) | no (lr falls →0.004) | **matches** locked 0.35→0.40 |
| **M22→M16** | #36 | **0.00** | **0.10** (2/3) | **0.20** | exam_dof (lr≈0.0005) | no (lr≲0.001) | knifes earlier (2/3 @0; all-fail ≥0.10) |
| **M23→M16** | #37 | **0.40** | **0.50** (1/3) | **0.65** | exam_dof (lr≈0.012) | no (lr≲0.02) | **left-shifts ~0.10**: pass@0.35 → fail@0.40 → all-fail@0.50 |

All six: CTRL leftover/close **3/3**; soft stock **3/3**; hard stock **0/3**. Analysis harness only.

---

## Soft bands (analysis harness — do not merge)

Conservative **all-pass** soft band under locked n=12 (stop before first_bite):

| start → target | soft band (all-pass) | mixed / first bite | hard all-fail |
|---|---|---|---|
| M22 → M16 scale | **t ≤ 0.00** (stock mild only) | 0.10–0.15 | ≥0.20 |
| M22 → M17 roles | **t ≤ 0.15** | — | ≥0.25 |
| M22 → M2 cross | **t ≤ 0.25** | — | ≥0.40 |
| M23 → M2 cross | **t ≤ 0.15** | — | ≥0.25 |
| M23 → M17 roles | **t ≤ 0.35** | — | ≥0.40 |
| M23 → M16 scale | **t ≤ 0.40** | 0.50 | ≥0.65 |

**Rank by how early locked bites (strictest → softest):**
`M22→M16` ≪ `M22→M17` ≈ `M23→M2` < `M22→M2` ≈ `M23→M17` < `M23→M16`.

---

## DoF-first vs leak-later

```
every path:  first_bite_mode = exam_dof
only M*→M2:  leak blowup AFTER DoF bite (M22→M2 ≥0.60; M23→M2 ≥0.50)
M*→M17/M16:  leak stays tiny / modest all the way to hard stock
```

Implication for Music caption drift: **roles/scale specialization alone is enough to kill
locked shared** — you do not need full cross_axis leak geometry. Soft-band knifes should
gate on exam/multi-row DoF, not on leak_ratio thresholds.

---

## Music n1+vic0 shifts (propose-only posture)

| pattern | paths | shift vs locked |
|---|---|---|
| **M22 seed-knife at mild** | M22→M2/M17/M16 | Music fails (or partial) at t=0 even when locked passes — known close/M22 family; needs **n≥2 ∨ multi-seed** (vic0@n1 is the propose-only twin) |
| **Music more tolerant** | M23→M2 | right-shift: locked fails @0.25, Music holds to 0.25 and fails @0.40 |
| **Music matches** | M23→M17 | same cliff 0.35→0.40 |
| **Music knifes earlier** | M23→M16 | left-shift ~0.10 (0.35→0.40 vs locked 0.40→0.50) |

Rule of thumb: **mild-stagger (M22) start always Music-seed-knifes**; **corr (M23) start is
Music-stable on roles, Music-tolerant on cross_axis, but Music-earlier on scale**.

Do **not** flip locked `AdvConfig.vicreg_weight=0.05`. Posture stays propose-only YAML/scoring.

---

## Geometry read (why start > target)

1. **Mild stagger (M22)** already sits partway toward roles/scale specialization → cliffs arrive
   early, especially **scale** (essentially only stock mild is all-pass).
2. **Corr seed (M23)** sits farther from roles/scale → soft bands stretch (roles ≤0.35, scale ≤0.40)
   but is **closer to full cross_axis bite** under locked n=12 (≤0.15) than mild→cross (≤0.25).
3. From the *same* M23 start: tolerance order is **scale > roles > cross_axis**
   (0.40 / 0.35 / 0.15 last_pass). From M22: **cross > roles > scale** (0.25 / 0.15 / 0.00).
4. Path **start** dominates cliff location more than the hard target label — fold soft-band
   policy by Music basin (mild vs corr), not by hard-cell name alone.

---

## Portable ADOPT (analysis only)

| ADOPT | scope |
|---|---|
| Soft multipair continuum card | this file — fold cliffs before inventing new lerps |
| Soft-band gates above | analysis harness sweeps / Music caption-drift sims |
| Music n≥2 ∨ vic0@n1 + multi-seed | close/M22 family (already on LEADER_CARD) |
| DoF-first scoring for soft→hard | gate on exam/rows before leak_ratio |
| lowrank_k3 dissolves DoF continuum cliffs | analysis portable — soft bands expand to t≤1.0 on key paths |

**Not ADOPTed / rejected:** any locked recipe change; trainer merge; new M-cells; Music GPU;
raising leak gates to “fix” DoF bites.

---

## lowrank_k3 vs shared (Fire lowrank_soft_continuum — 2026-09-10)

Dig: `lowrank_soft_continuum_20260910.{md,json,py}` wall=1672s. Portable harness;
**Reoc0 not used** (DoF-only). Locked n=12. merge=**NO**.

| path | shared last_pass | shared first_bite | shared all_fail | lowrank last_pass | lowrank first_bite | lowrank all_fail | soft expands? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **M22→M16** | 0.00 | 0.10 | 0.15 | **1.00** | — | — | **YES** |
| **M23→M2** | 0.15 | 0.25 | 0.25 | **1.00** | — | — | **YES** |
| **M22→M2** | 0.25 | 0.40 | 0.40 | **1.00** | — | — | **YES** |

**Answer: YES — soft bands expand.** Under `lowrank_k3`, all three paths are
**full-continuum soft** (last_pass=1.0, no first_bite) — cliffs do not arrive;
hard stock DoF clears (M16/M2 3/3) so the soft→hard lerp never knifes.
Shared arm reconfirms card cliffs (exam_dof bites; M22→M16 all_fail 0.15 here vs
card 0.20 — same first_bite 0.10). CTRL leftover/close 3/3 both arms.

Implication: soft-band policy under analysis portable student is **not** the shared
cliff table — lowrank dissolves DoF continuum cliffs. Declare-lie (M14/M20/M21)
still needs Reoc0 (propose-only); do not merge lowrank to trainer.

---

## Pointers

| doc | role |
|---|---|
| `LEADER_CARD_20260910.md` | living “what’s leading” |
| `MUSIC_TO_TOY_SCOREBOARD_20260909.md` | cell table + Fire #32–#37 folds |
| `research_log_20260909.md` | fire chronology |
| per-fire `m22_*` / `m23_*` `.{md,json,py}` | raw sweeps |
| `lowrank_soft_continuum_20260910.{md,json,py}` | **lowrank_k3 vs shared continuum shift** |

*Shared cliffs CTRL-anchored; lowrank_k3 recheck shows full soft-band expansion on key paths.*


---

## Fire #40 addendum — leak_max honesty (2026-09-11)

Re-score of lowrank soft dig under `leak_max_rows ≤ 0.20` (row0-only `pass_leak` is insufficient):

| path | lowrank honest last_pass | first_bite | full soft? |
|---|---:|---:|:---:|
| M22→M16 | 1.00 | — | YES |
| M23→M2 | 0.40 | 0.50 (leak_max) | NO |
| M22→M2 | 0.50 | 0.60 (leak_max) | NO |

M2_stock lowrank full 3/3 is a **false clear** (leak_max≈2.57). See `lowrank_leak_honesty_fire40_20260911.md`.
