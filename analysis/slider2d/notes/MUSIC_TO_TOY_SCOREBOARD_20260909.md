# Music → toy living scoreboard — 2026-09-09

**Host:** box-cpu @ `435e87363bf1`. CPU harness only.
**No Music GPU train.** Leave nano-work-server, ComfyUI :18888, `/ml2/music/run_server.py` alone.
**Recipe change:** **NO** (locked shared AdvResidual stays default).

Synthesizes: catalog + batch2/3 + Fire #20/#21 + harden bites + per_row explore (+ deepen in flight).

---

## Locked recipe reminder

| knob | locked value | notes |
|---|---|---|
| steps | **1200** | reject 800×cover3.0 (Fire #4 false lock) |
| cover_weight | **1.5** | Music-posture proxy often uses 1.0 |
| teacher | **`faithful_guard_e`** | leftover-gated |
| FM | **0** | FM-on flat (Fire #6); temptation flat batch2 |
| n_particles | **≤12** | default 12; Music parts0 proxy = **n=1** |
| particle_l2 | **0.02** | Fire #7 / batch2 extremes flat |
| vicreg_weight | **0.05** | locked default keep; Music-posture harness **ADOPT** `vic=0` when n=1 (stress 20260909) |
| b_cap | **1** | |

**Music-posture variant (scoring only):** `n_particles=1` (proxies `--parts 0`), `cover_weight=1.0`, same guard/FM0/b_cap/l2.

---

## Hard boundaries (do not chase with locked recipe)

| ID / cell | Fail mode | Driver | Recipe knobs recover? |
|---|---|---|---|
| **M1** `lyric_span_entangle` | multi_row 0/5 (Fire #20) | hetero `row_amps` → shared δ = mean(a_r) DoF floor; `e_on_content` worsens leak | **NO** (n/cover/steps/e_unused) |
| **M2** `cross_axis_rows` | 0/6 | heterogeneous R³ mixes | **NO** |
| **M21** `hold_e_lyric_mix` | leak_ratio **floor 0.20503** (emp≈0.21), **0/6** shared+per_row | **teacher `declared_e` vs exam `leak_e` axis mismatch** (`eoc·ĉ+e_unused·ê` tilt); per-row cannot clear; train-pure-ê clears (ablation only) | **NO** (n/cover/steps/vic/false-lock/per-row) |
| **M14/M20/M26** declare-lie | content_deleted_under_declare_lie | declared ê ≠ exam leak axis (û / content / split); guard deletes content; n/vic/per_row NO; eou≥0.15 bites; faithful no-guard PASSes | **NO** |
| **M4/M7/M8** dual-arm | Arm T alone / Arm L cover=0 fail | tx∩guard SystemExit; both arms required | N/A (encoding) |

Shared AdvResidual + multi-span / e_on_content pool mix = **Music hard boundary**. Document; do not weaken cells; do not revise locked recipe.

---

## Knives (close n=1 → n≥2 harden)

| cell | n=1 Music posture | harden | complementary |
|---|---|---|---|
| **M3** `f3d_close` | seed0-only **5/6** (c1.0+c1.5) | **n≥2 → 6/6** (Fire #21) | **vic=0 @ n=1 → 21/21** (ADOPT Music posture) |
| **M11** `close_live_noise` | seed0 **3/4→5/6** family | **n≥2 → 4/4** | vic=0@n=1 ADOPT |
| **M13** `tiny_slider_dom` | music n1 **5/6** seed0 | **n≥2 → 6/6** (M20 transfer) | **vic=0 @ n=1 → 6/6** (ADOPT Music posture) |
| **M18** `close_with_leak` | music n1 **5/6** seed0 | **n≥2 → 6/6** | **vic=0 @ n=1 → 6/6** |
| **M22** `stagger_mild_cross` | music **4/6** fail={2,7}; locked 6/6 | soft multipair knife @ parts0 | — |

**Music parts0 rule (scoring posture, not recipe flip):**
1. multi-seed gate for close-family under n=1, **and/or**
2. **`n_particles ≥ 2`** for close-family, **and/or**
3. **`vicreg_weight=0` when n=1** — **ADOPT** Music-posture harness only (`vicreg0_n1_propose_stress_20260909`; locked default stays 0.05).

Rejected as seed0@n=1 fix: particle_l2, span/end/cloud/jitter, lr, steps1600, cover2.0.

---

## Hard bites kept as regression (negative controls / Music wins)

Keep these failing under locked shared recipe — they encode Music bugs or declare lies.

| ID | Cell | locked | music | Why keep |
|---|---|---|---|---|
| **M14** | `e_on_u_declare_lie` | **0/6** | **0/6** | YAML û-lie; **content_deleted_under_declare_lie** (exam≈0.18=content_kept, u≈0.99); n/vic/per-row NO; faithful-no-guard PASSes; keep vs M20 |
| **M16** | `scale_stagger_homo` | **0/6** | **0/6** | **multi_row_scale_dof_partial**; per_row w0 clears; n/vic NO; scale cliff 0.4→0.6 |
| **M17** | `roles_split_proxy` | **0/6** | **0/6** | **multi_row_axis_roles_zero**; per_row clears; blend ≤0.35 PASS / 0.4 knife / ≥0.5 FAIL |
| **M20** | `amp_lie_leftover_declare` | **0/6** | **0/6** | content-axis amp lie; same nature **content_deleted_under_declare_lie** (exam≈-0.08); n/vic/per-row NO; **must stay bite** |
| **M21** | `hold_e_lyric_mix` | **0/6** | **0/6** | analytic floor lr=**0.20503**; teacher declared_e≠exam leak_e; per-row cannot clear |
| **M24** | `content_leak_flip_rows` | **0/6** | **0/6** | mid-caption content↔leak flip (exam≈0.64, leak tiny) |
| **M26** | `declare_split_three` | **0/6** | **0/6** | **content_deleted_under_declare_lie** (exam≈0.09); same family as M14/M20; n/vic/per_row NO |
| **M27** | `scale_descent_homo` | **0/6** | **0/6** | descending scales / traj-outro reverse (multi_row) |
| **M6** | `cross_axis_mismatch_declare` / axis_leak | expected FAIL | — | declare/leak negative controls |
| CTRL | `leftover_field3d` | **must PASS** | — | regression green (3/3–6/6) |

**Wins (Music bug now fails in-toy):** M1, M2, M14, M16, M17, M20, M21, M24, M26, M27 (+ dual-arm arms alone).

---

## Promising non-defaults — merge = **NO**

### Per-row / multi-residual (`per_row_residual_explore_20260909`)

| finding | result |
|---|---|
| Geometry | shared mean-δ covers **0/5** M1 rows (MSE floor); DoF limit not optimization |
| Per-row lyric M1 | **6/6** multi + bite (teacher-aligned) |
| Controls under per-row | leftover **3/3**, close **3/3**, unused_e **3/3** |
| Curriculum / weighted cover / scaled shared | still **0/3** on M1 amp-*mix* (scaled helps homo-scale only) |
| Soft couple | w∈{0,0.1,0.3} ok; w=1.0 collapses to shared floor |
| Head fragmentation | unconstrained `head_min_cos`≈0.56; couple w0.3 → ≈0.62 |

Deepen: couple robust band **w∈[0.0, 0.3]** (w0.7 collapses); per-row also saves `cross_axis_rows`; Music-posture n=1 lyric clears; amp_lie expected_fail.

**ADOPT (analysis-only):** per-row AdvResidual band **w≤0.3** for digs that need multi-span DoF — **not** a trainer merge; does **not** clear M14/M20 declare-lies or M21 eoc leak.

**Label:** `NON_DEFAULT_explore / deepen / falsify_A` — analysis-only.
**merge_to_trainer = NO.** Locked shared AdvResidual stays.
**Next (GPU later):** multi-seed Music smoke with per-row student — not this box job.

Deepen in flight: couple band + M21/cross/M20 shared-vs-per_row + Music n=1 lyric.

---

## Encoding gaps (cannot fully put in-toy)

| gap | why |
|---|---|
| **`tx` arch** | `faithful_guard_e` + `--adv_arch tx` → SystemExit; toy MLP-only, no SpanTransformerD |
| **M25** `lyric_neu_heavy_gate` | **NO_BITE 6/6** both — heavy lyric neu ≠ hold-ê mix; neu magnitude alone does not break leftover gate |
| **Lyric token span alignment** | no token positions in Field3D |
| **`--parts` ↔ `n_particles`** | different objects; only proxy via n=1 |
| **`lyrichold_weight` ≡ `cover_weight`** | unvalidated; keep separate (M10) |
| **LoRA / live Music listen metrics** | CPU toy residual scores only |
| **GPU Music UNI failure modes** | approximate via geometry only |

---

## Scoreboard snapshot (locked shared, unless noted)

| ID | Verdict under locked | Harden / note |
|---|---|---|
| M1 lyric_span | HARD BOUNDARY 0/6 | per-row NON_DEFAULT clears; merge=NO |
| M2 cross_axis | HARD BOUNDARY 0/6 | — |
| M3/M11 close | KNIFE @ n=1 seed0 | n≥2 / vic0@n=1 |
| M13 tiny_slider | KNIFE music n1; locked 6/6 | n≥2 / vic0@n1 clear |
| M14 e_on_u lie | HARD_BITE 0/6 | keep (û-declare; ≠ parts0 knife) |
| M15 prefix_shared | NO_BITE 6/6 both | pos control |
| M16 scale_stagger | HARD_BITE 0/6 | DoF scale; per_row w0 clears; cliff span 0.4→0.6 |
| M17 roles_split | HARD_BITE 0/6 | axis-roles DoF; per_row clears; mix ≤0.35/0.4/≥0.5 |
| M18 close_with_leak | KNIFE music n1; locked 6/6 | n≥2 / vic0@n1 clear |
| M19 grit_content | NO_BITE 6/6 both | grit content-dom OK |
| M20 amp_lie | HARD_BITE 0/6 | keep (overpowered-head falsifier) |
| M21 hold_e mix | HARD BOUNDARY 0/6 | driver e_on_content>0 |
| M22 stagger_mild | NO locked; KNIFE music | — |
| M23 multipair_corr | NO_BITE 6/6 | — |
| M24 content↔leak flip | HARD_BITE 0/6 | — |
| M25 lyric_neu heavy | NO_BITE 6/6 | encoding gap |
| M26 declare_split | HARD_BITE 0/6 | content_deleted family; see declare_lie_family |
| M27 scale_descent | HARD_BITE 0/6 | — |
| M28 guard_refuse | HARD_BITE 0/6 | refuse→raw leak; per-row NO |
| M29 content_cascade | HARD_BITE 0/6 | DoF; **per-row clears** merge=NO |
| M30 eoc_threshold | HARD_BITE 0/6 | ê-floor edge; per-row 1/6 knife |
| M31 leftover_hot_eoc | HARD_BITE 0/6 | admit hot eoc; per-row NO |
| leftover CTRL | PASS | must stay |

---



## Close seed0 deep (thread C, completed sibling) — 2026-09-09

From `close_seed0_deep_20260909` wall=2175s (beyond Fire #21):

- Fail mode: **exam_u_undershoot_init_basin** (û undershoot; not leak/multi)
- seeds0..20 @ n=1: close c1.5 **20/21** fail=[0]; live **20/21** fail=[0]; c1.0 also [0,4]
- **n_floor=2** for close∧live 6/6
- seed0@n=1 knobs (l2/b_cap/cover/jitter/cloud): **no recovery**
- **vic=0 @ n=1**: close+live **21/21**; leftover n12 flat 0.9303
- Decision: **multi_seed_gate_PLUS_ADOPT_vic0_when_n1** — Music-posture harness ADOPT after stress (`vicreg0_n1_propose_stress`); locked default unchanged



## Folded: couple band + Music-posture vic0@n1 ADOPT (2026-09-09)

### (1) Per-row couple robust band — `NON_DEFAULT` / merge=NO

From `per_row_residual_deepen_20260909` (wall≈2138s):

| finding | result |
|---|---|
| Couple robust band | **w ∈ [0.0, 0.3]** (full 6/6 multi+bite+exam; head_min_cos≈0.56–0.62) |
| Soft edge | w=0.5 smoke knife; **w≥0.7 collapses** toward shared floor |
| Saves beyond M1 lyric | **`cross_axis_rows`** YES (DoF); M21/M20 NO |
| Music-posture n=1 + per-row lyric | clears 6/6 |
| Controls under per-row | leftover 3/3, close 3/3 |

**merge_to_trainer = NO.** Locked shared AdvResidual stays default.

### (2) ADOPT Music-posture `vicreg_weight=0` when `n_particles<=1` — not locked defaults

From `vicreg0_n1_propose_stress_20260909` + `close_seed0_deep_20260909`:

```python
# Music-posture scoring harness ONLY — AdvConfig.vicreg_weight stays 0.05
if n_particles <= 1:
    cfg = replace(cfg, vicreg_weight=0.0)
```

| gate | result |
|---|---|
| close/live Music c1.0 under vic0 | **21/21** / **21/21** |
| leftover n12 flat | 0.9303→0.9303 |
| M20/M21/M24 HARD_BITEs | **stay 0/6** (no false fix) |
| Fire #21 n≥2 + multi-seed | remain primary harden |

**ADOPT** for Music parts0 posture harness only. **Do NOT** flip locked `vicreg_weight=0.05`.


## Next digs (priority)

1. **A DONE:** per-row clears M24/M27, not M20; M21 uncleared (leak/eoc) — see deep thread A below.
2. **B OPEN:** Field2D / sheet analogues of M24/M27 if still missing.
3. **C DONE:** `close_seed0_deep` + **stress ADOPT** `vicreg0_n1_propose_stress` (see section below).

---

*Living doc — append deep-thread results below / in research_log.*

## Deep thread A result (2026-09-09)

**Script:** `per_row_falsify_m21_m24_m27_20260909` wall=1578.8s

| mid | shared pass | per_row pass | clears? |
|---|---|---|---|
| M21 | 0/3 | 0/6 | no |
| M24 | 0/3 | 6/6 | YES |
| M27 | 0/3 | 6/6 | YES |
| M20 | 0/3 | 0/6 | no |

**Verdict:** YES — per-row clears ['M24', 'M27'] without clearing M20 (amp-lie still bites); uncleared=['M21']; CTRL ok; merge=NO
**merge=NO.** Locked shared recipe unchanged.

**Interpretation:**
- **M24 / M27:** shared DoF / scale-coverage bites — per-row heads recover (exam≈0.98).
- **M21:** not a DoF bite — multi covers 4/4 under shared+per_row but **leak≈0.21** (`e_on_content` pool); per-row does not heal gate leak.
- **M20:** amp-lie YAML still 0/6 under per-row (leak≈0.38) — **not overpowered**; declare lies need target/YAML fix, not more heads.
- Deepen sibling: couple robust band **w∈[0.0, 0.3]**; per-row also saves `cross_axis_rows`; Music n=1 lyric clears; amp_lie expected_fail.

---

## vicreg0 @ n=1 propose stress (2026-09-09)

**Recommend: ADOPT** for Music-posture n=1 harness only — **not** a locked AdvConfig flip.

Evidence (`vicreg0_n1_propose_stress_20260909.{md,py,json}`, wall≈19.5min):

| gate | result |
|---|---|
| Music c1.0 close/live under vic0 | **21/21** / **21/21** (default knife 19/21, 20/21) |
| c1.5 propose replicate | prior 21/21 + reconfirm 7/7 (seed0 knife→heal) |
| leftover n12 flat | 0.9303→0.9303; Music n1 leftover 6/6 |
| dual-arm pos / arms alone | 3/3 PASS / 0/3 FAIL under vic0 |
| M22 stagger_mild | 4/6→6/6 soft false-lock risk (document side-effect) |
| M23 multipair | 6/6 flat |
| M20/M21/M24 | **0/6** stay HARD_BITE (no YAML-lie fix) |

```python
# Music-posture harness only — locked AdvConfig.vicreg_weight stays 0.05
if n_particles <= 1:
    cfg = replace(cfg, vicreg_weight=0.0)
```

Fire #21 n≥2 + multi-seed gate remain. No Music train. Defaults unchanged.


## Folded: M13–M19 catalog + roles/scale cliff (2026-09-09)

From `music_toy_new_stressors_m13_20260909` + `roles_scale_isolation_20260909` + M20 transfer + WRAP:

| ID | Cell | locked n12 | music n1 | Status |
|---|---|---|---|---|
| M13 | `tiny_slider_dom` | 6/6 | 5/6 seed0 knife | n2/vic0 → 6/6 |
| M14 | `e_on_u_declare_lie` | **0/6** | **0/6** | **BITES** YAML û-lie; n2 still 0/6 |
| M15 | `prefix_shared_proxy` | 6/6 | 6/6 | positive / prefix hold OK |
| M16 | `scale_stagger_homo` | **0/6** | **0/6** | **BITES** mild multi-row (rows≈3) |
| M17 | `roles_split_proxy` | **0/6** | **0/6** | **BITES** soft cross-axis (rows≈0) |
| M18 | `close_with_leak` | 6/6 | 5/6 seed0 knife | n2/vic0 → 6/6 |
| M19 | `grit_content_dom` | 6/6 | 6/6 | grit content-dom OK |

**roles/scale cliff:** blend_mix **≤0.25 PASS** (rows=4) → **≥0.5 FAIL** (rows=0). Soft→hard boundary into M17/M2.

### ADOPTs (Music scoring posture — locked defaults unchanged)

1. **Close n≥2 harden** for close-family (M3/M11/M13/M18) under Music parts0 proxy.
2. **Multi-seed gate** @ n=1 close-family (do not raise pole on seed0 alone).
3. **`vicreg_weight=0` when n≤1** — Music-posture harness **ADOPT** only; AdvConfig default stays 0.05 @ n=12.
4. **Per-row AdvResidual band w≤0.3** — **analysis-only** ADOPT for multi-span digs; merge_to_trainer=NO; does not clear declare-lies.

Recipe change: **NO**.



## Mechanistic: M21 e_on_content leak irreducible (2026-09-09)

**Script:** `m21_eoc_leak_irreducible_20260909` wall=737.5s

**Closed form:** `declared_e = eoc·ĉ + e_unused·ê` tilts ê̂; `faithful_sub_e` leaves
teacher â with ê component. At M21 defaults â_e≈-0.20503,
**leak_ratio floor≈0.20503** (>0.20). Exam scores pure `leak_e()`;
perfect cover reproduces the floor.

| probe | result |
|---|---|
| analytic↔teacher | match=True |
| shared locked | pass_leak=0/6 mean_lr=0.210619 ≈floor |
| per-row w0 / w0.3 | pass=0/6 / 0/3 — multi OK, **leak gate fails** |
| train leak_dir=pure ê | pass_leak=6/6 **CLEARS** (diagnostic only) |
| leftover / M20 | ok=True / still_bites=True |

**Why per-row does not clear M21:** homogeneous rows + identical tilted teacher —
not a DoF / mean-δ bite (unlike M1/M24/M27).

**merge=NO.** Do not strip eoc from cell. Recipe change=NO.

---

## Fire #24 fold (2026-09-09)

Folded M21 irreducible mechanism into hard-boundary / hard-bites rows above
(analytic floor **0.20503**; declared_e vs leak_e axis mismatch; per-row cannot clear).
Full dig: `m21_eoc_leak_irreducible_20260909`; fold note: `m21_fold_fire24_20260909.md`.
Retrofit: `retrofit_annotate_dig_rows_fire24_20260909.py` (+ annotate wire in close digs).
**recipe_change=NO; merge=NO.**

## Dig: M14 fail-mode vs M20 (2026-09-09)

Source: `m14_failmode_20260909.{md,json}` wall=2288.4s.

| finding | result |
|---|---|
| M14 nature | **content_deleted_under_declare_lie** — exam≈0.18 equals content_kept; u≈0.99 / swing≈0.85 / multi OK |
| M20 nature | **same** — content_kept≈-0.08, leak≈0.38 |
| Clearance | n=1 / n=2 / vic0@n1 / per-row **all fail to clear** M14 and M20 |
| Teacher | `faithful` (no guard) **3/3 PASS** both — bite needs `faithful_guard_e` |
| e_on_u cliff | eou=0 PASS → **eou≥0.3 BITES** |
| Keep | both HARD_BITEs (overpowered-head / YAML-lie falsifiers); recipe_change=NO |



## Batch4 M28–M31 (2026-09-09)

Wall=1428.2s. New cells from M21 mech + Music gaps. Recipe change=NO.

| ID | locked | music | note |
|---|---|---|---|
| M28 `guard_refuse_hot_eoc` | 0/6 HARD_BITE | 0/6 HARD_BITE | |
| M29 `content_cascade_rows` | 0/6 HARD_BITE | 0/6 HARD_BITE | |
| M30 `eoc_threshold_edge` | 0/6 HARD_BITE | 0/6 HARD_BITE | |
| M31 `leftover_hot_eoc_declare` | 0/6 HARD_BITE | 0/6 HARD_BITE | |
| CTRL leftover | 3/3 | — | must PASS |


## Session fold (box dig) — 2026-09-09 evening

### M14 label (confirm)
**M14 `e_on_u_declare_lie` = `content_deleted_under_declare_lie`** (M20 family).
exam≈0.18 equals content_kept; u≈0.99; n/vic/per-row do not clear. Keep as HARD_BITE vs M20.

### M21 mechanism (DONE)
`m21_eoc_leak_irreducible_20260909`: teacher-axis geometry floor — `declared_e` tilt → â_e≠0 → lr≈0.205.
Per-row cannot clear (homogeneous + same tilted teacher). train-pure-ê clears (diagnostic only).
Two eoc fail modes: **admit→ê-floor** (M21/M30/M31) vs **refuse→raw-pole leak** (M28).

### Batch4 M28–M31 (DONE, wall=1428s)
All **HARD_BITE 0/6** locked+music; leftover 3/3; vic0@n1 still fails M28/M30.

| ID | fail mode | family |
|---|---|---|
| M28 | leak≈0.66 guard refuse | eoc refuse (new) |
| M29 | multi 2/4, leak tiny | DoF / cascade (per-row candidate) |
| M30 | leak≈0.21 ê-floor edge | M21 floor sibling |
| M31 | leak≈0.21 + exam≪0 | admitting hot eoc on leftover amps |

**Recipe change=NO. merge=NO.** Leak-gate kinds patched for M28–M31.

### Still locked
Couple w∈[0,0.3] NON_DEFAULT; Music-posture **ADOPT** vic=0@n≤1 (not AdvConfig default).
Close: n≥2 + multi-seed + propose vic0@n1.


## Fire #25 — Per-row falsify M29 (2026-09-09)

**Script:** `per_row_falsify_m29_fire25_20260909` wall=1691.2s couple_w=0.0

| mid | shared pass | per_row pass | clears? |
|---|---|---|---|
| M29 | 0/6 | 6/6 | YES |
| M20 | 0/3 | 0/3 | no (keep-bite) |
| M28 | 0/3 | 0/3 | no (keep-bite) |
| M30 | 0/3 | 1/3 | no (keep-bite) |

**Verdict:** YES — per-row clears M29 content_cascade without clearing M20/M28/M30; CTRL ok; merge=NO
**merge=NO.** Locked shared recipe unchanged.

**Nuance:** M30 per_row smoke **1/3** (seed 1 cleared; seeds 0,2 still bite; leak_max≈0.2045 near ê-floor). Not a full clear — keep HARD_BITE / ê-floor family.



## Per-row falsify batch4 (2026-09-09)

**Script:** `per_row_falsify_batch4_20260909` wall=1786.7s

| mid | shared | per_row | clears? |
|---|---|---|---|
| M28 | 0/3 | 0/6 | False |
| M29 | 0/3 | 6/6 | True |
| M30 | 0/3 | 1/6 | False |
| M31 | 0/3 | 0/6 | False |

**Verdict:** YES — per-row clears M29 (DoF) without clearing M28/M30/M31 (eoc/teacher); CTRL ok; merge=NO
**merge=NO.** M14 remains content_deleted_under_declare_lie (M20 family).


## M30 per-row knife (2026-09-09)

Wall=2000.6s. pass_seeds=[1] (1/6 class). Couple w∈{0,0.1,0.3} no 6/6 clear.
Keep M30 HARD_BITE. Near analytic floor noise — not a recipe lever.


## Dig: M16/M17 mechanistic (2026-09-09)

Source: `m16_m17_mech_20260909.{md,json}` wall=4331.1s.

| finding | result |
|---|---|
| M16 nature | **multi_row_scale_dof_partial** |
| M17 nature | **multi_row_axis_roles_zero** |
| M2 / M27 | multi_row_plus_leak_cross_axis / multi_row_scale_dof_partial |
| M16 clearance | {'n1_music': False, 'n2': False, 'vic0_n1': False, 'per_row_w0': True, 'per_row_w0.3': False} |
| M17 clearance | {'n1_music': False, 'n2': False, 'vic0_n1': False, 'per_row_w0': True, 'per_row_w0.3': True} |
| scale-span cliff | first all-fail = **0.6** |
| blend cliff | lastPASS=**0.35** → firstFAIL=**0.4** |
| Keep | both HARD_BITEs; recipe_change=NO |

## M16/M17 mech (Fire #26b, 2026-09-09)

Wall=4331s. Natures: M16/M27=`multi_row_scale_dof_partial`; M17=`multi_row_axis_roles_zero`; M2=`multi_row_plus_leak_cross_axis`.
Per_row w=0 clears M16/M17/M2/M27 (6/6); n/vic do not. Scale cliff span=0.6; blend cliff 0.35→0.4.
Keep HARD_BITEs under locked shared. recipe_change=NO. merge=NO.


## Declare-lie family theory + neg-control suite (2026-09-09)

Source: `declare_lie_family_20260909` wall=1243.4s. **suite_pass=True**.

| member | nature |
|---|---|
| M14 | content_deleted_under_declare_lie |
| M20 | content_deleted_under_declare_lie |
| M26 | content_deleted_under_declare_lie |
| M6 | declare_lie_raw_leak_blowup |

n/vic/per_row clear any? **False**. Neg controls (eou0 / faithful / leftover / M16 DoF contrast): **{'leftover': True, 'm14_eou0': True, 'm14_faithful': True, 'm20_faithful': True, 'm26_faithful': True, 'm16_shared_bites': True, 'm16_per_row_clears': True}**.
Keep HARD_BITEs. Recipe change=NO.


## Declare-lie / eoc-floor family theory (2026-09-09)

Source: `declare_lie_eoc_family_theory_20260909` wall=827.8s. **suite_pass=True**.

| branch | members | cliff |
|---|---|---|
| A content_deleted_under_declare_lie | M14, M20, M26 | eou 0→≥0.15; faithful-no-guard PASS |
| B ê-floor admit | M21, M30, M31 | ana 0.32Y/0.33N; emp@0.32 bites; valley 0.55–0.65 |
| C guard-refuse raw-pole | M28 | refuse eoc≥0.68; lr≈0.66 |

n/vic/per_row clear any? **NO**. Keep HARD_BITEs. Recipe change=NO. merge=NO.
One-page map: `declare_lie_eoc_family_theory_20260909.md`.

## Valley emp + sheet/highd A/B/C ports (2026-09-09)

Source: `valley_emp_sheet_abc_ports_20260909` wall=565.9s.

| finding | result |
|---|---|
| analytic valley emp | `valley_emp_BITES_mixed_mech` eoc∈[0.55, 0.58, 0.6, 0.62, 0.65] |
| sheet A/C declare | û-lie / hot-sheet **BITES** leak_tok |
| sheet B tilt | often PASS (no Field3D ê-floor exam) |
| sheet DoF M16/24/27/29 | **NO_BITE** encoding gap (prefer Field3D) |
| highd A synonym | **BITES** leftover_leak under hold-λ |
| recipe / merge | **NO** / **NO** |

**B OPEN (scoreboard) update:** Field2D/sheet DoF analogues exist as `CELLS_SHEET_DOF` but are NO_BITE under sheet verdicts; real DoF bites stay Field3D.
