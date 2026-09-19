# Music → toy stressor catalog — 2026-09-09

**Host target:** pop-os `/ml2/music/sliders-conceptmod` (CPU harness only).
**No Music GPU train.** Leave nano-work-server, ComfyUI :18888, `/ml2/music/run_server.py` alone.

**Locked toy recipe:** steps=1200, `cover_weight=1.5`, `teacher=faithful_guard_e`,
FM=0, `n_particles≤12`, `particle_l2=0.02`, `b_cap=1`. Reject 800×cover3.0 (Fire #4 false lock).

**Music-posture variant:** `n_particles=1` (proxies `--parts 0`; toy `n=0` breaks ParticlePrior),
`cover_weight=1.0` (modest Music pole pin), same guard/FM0/b_cap.
**ADOPT (scoring harness only):** `vicreg_weight=0` when `n≤1`; close-family prefer **n≥2** + multi-seed gate.
**ADOPT (analysis-only):** per-row band **w≤0.3** — merge_to_trainer=NO.

Evidence mined from: `train_lm_slider_music3.py` gates (tx∩lyric), `lm_adv.py` residual/RpGAN core,
`analysis/slider2d/notes/` transfer / dual-arm / Fire #13 / research_log, dual-arm ablation.

---

## Catalog rows

| ID | Music symptom | Why toy doesn't hit it yet | Proposed Field2D/Field3D/exam cell (concrete) | Expected fail under locked recipe? |
|---|---|---|---|---|
| M1 | Span-entangled leftover: lyric spans share residual ê with content (hold-ê / lyric pool mixes leak into content dirs) | Homogeneous `leftover_field3d` / scalar amps keep e unused (`e_unused=1`); Fire #13 `cross_axis_span_sample` exists but lacks **e_on_content** entanglement | **`lyric_span_entangle`**: Field3D rows=5, `row_amps=((1.05,0.55,0.35),(0.60,1.10,0.45),(0.75,0.50,0.90),(1.10,0.85,0.55),(0.90,0.70,0.65))`, `row_scales=(0.75..1.35)`, `e_on_content=0.45`, `e_unused=0.7`; exam score | **YES / HARD BOUNDARY** — Fire #19/#20: locked 0/6 via `pass_multi_row` (rows_cov 0/5) despite high row0 exam; hetero `row_amps` alone enough; e_on_content worsens leak; cover/n/steps harden probes fail — shared residual ≠ multi-span Music |
| M2 | Multi-span lyrics = heterogeneous per-row axis mix (each caption row lives on different R³ mix) | Pre–Fire #13 Field3D only scalar×`row_scales` (same axis family); now `cross_axis_rows` exists | **`cross_axis_rows`**: `row_amps=((1.2,0.15,0.1),(0.35,1.4,0.1),(0.4,0.2,1.1),(0.9,0.7,0.65))` exam | **YES / hard boundary** — Fire #13: 0/6 u≈0.58; dig whether n=1 vs n=12 changes anything |
| M3 | Close-pair multi-seed knife (delivery/close axes flake by seed) | Locked sheet leftover is seed-stable 6/6; `f3d_close` only flakes at Music posture | **`f3d_close` knife**: `close_field3d` @ n=1 cover∈{1.0,1.5} seeds {0,1,2,3,7,42}; optional live-like amp jitter ±2% | **Knife @ n=1** — Fire #17–#21: 5/6 seed0-only; **harden n≥2** (Fire #21: n2 c1.0 6/6; n2 c1.0+c1.5 both 6/6; live n1 3/4→n2 4/4) — tighter than prior n≥4 note. Do not change locked n≤12 recipe; avoid Music parts0-alone on close. **vic=0@n=1 also 6/6** (harden bites; candidate, no default flip). **Fire #22:** `field3d.music_close_posture_warn` / `score_adv_field3d` now emits Music-posture warn at n=1 close |
| M4 | Dual-arm argv fork: leftover `#94` cannot share argv with listen/tx | Toy has no `adv_arch` / lyric-TX; single teacher path hides the fork | **Separate exam cells that must both pass for “full transfer” claim:** (L) leftover geometry + `faithful_guard_e` + cover; (T) same geom + `teacher=faithful` (no guard) = listen/cover_only | **Arm T alone FAIL** (leak); Arm L alone with cover=0 FAIL (undershoot). Both required. Cite `tx ∩ guard` SystemExit |
| M5 | `--parts 0` Music posture (no latent ParticlePrior) | Default toy uses n=12; hiding parts0 fragility | Force **`n_particles=1`** on Music-posture cells; document Δ vs n=12 cover=1.5 | Often **PASS still** on Arm A leftovers (Fire #15/#17 n≈n12); knives appear on close/cross-axis |
| M6 | Bad declared leak YAML / amplitude lie | Easy to miss if only happy-path cells run | Keep **`cross_axis_mismatch_declare`** / `axis_leak_primary` as **negative controls** | **YES expected FAIL** (leak≈2+ / declare lie) — not a recipe bug |
| M7 | Cover-only listen arm leaks ê (lyric+tx+lyrichold, no leftover gate) | Sheet lock always pairs guard∧cover | **`arm_listen_cover_only`**: leftover geom, teacher=`faithful`, cover=1.5, n=1 | **YES** — dual_arm ablation sheet leak≈0.23 / f3d≈0.46 |
| M8 | Gate-only (pole/cover=0) undershoots | Same — locked always cover=1.5 | **`arm_leftover_gate_cover0`**: guard, cover=0, n=1 | **YES** — sheet kept≈0.48 / f3d≈0.57 |
| M9 | False lock short×high-pin (800×cover3.0) | Tempting “faster” Music schedule | Reject cell — do not adopt | PASS can be false lock (Fire #4 1/6) |
| M10 | `lyrichold_weight` ≠ `cover_weight` / pole | Name collision in notes | Doc only — do **not** equate in toy | N/A (encoding gap) |
| M11 | Live amp/span noise on close pairs (Music caption jitter) | Deterministic `close_field3d` | **`close_live_noise`**: close geom + seed-tied content/leak jitter + optional span scale noise; multi-seed | Expect **more knives** than clean close; Music-posture n=1 cover=1.0 worst |
| M12 | `tx` arch requires lyric recipe (`PLUS_NEU_LYRIC_RECIPES`) | Toy MLP-only; cannot encode SpanTransformerD / position-aligned lyric tokens | Doc + dual-arm cells only; Option C propose-only | Cannot fully encode in toy |

---

## Hard reject / encoding gaps (cannot fully put in-toy)

1. **`faithful_guard_e` + `--adv_arch tx` → SystemExit** — recipe-set gate, not “has span”. Toy has no TX critic.
2. **Lyric token span alignment / SpanTransformerD** — no token positions in Field3D.
3. **`--parts` ↔ `n_particles`** — different objects; only proxy via n=1.
4. **`lyrichold_weight` ≡ `cover_weight`** — unvalidated; keep separate.
5. **LoRA / live Music listen metrics** — CPU toy residual scores only.
6. **GPU Music train failure modes** that need real UNI captions — approximate via geometry only.

---

## Dual-arm cite (tx ∩ guard)

```text
# train_lm_slider_music3.py (~L206, ~L2170–2176, ~L2182–2186 on laptop @ 435e873+local)
PLUS_NEU_LYRIC_RECIPES = frozenset({"faithful_plus_neu_lyric"})
if adv_arch == "tx" and recipe not in PLUS_NEU_LYRIC_RECIPES: raise SystemExit(...)
```

| arm | `--lm_target` | `--adv_arch` | toy cell |
|---|---|---|---|
| L leftover/#94 | `faithful_guard_e` | `mlp` | guard + cover |
| T listen | `faithful_plus_neu_lyric` | `tx` | cover_only / no guard |

---

## New stressors shipped this job

| Script / hook | Music ID | Cell |
|---|---|---|
| `music_to_toy_stressors_run_20260909.py` | M1, M4, M7/M8, M3/M11, M5 | runnable CPU exam grid |
| `field3d_music_stressor_hooks.py` | M1, M11 | scaffold ctors for `field3d.py` |
| `cross_axis_*` (Fire #13, may already be in field3d) | M2, M6 | extend/harden if present |

## Success criteria mapping

- Catalog = this file.
- ≥2 runnable Music-derived stressors with numbers after run.
- Call out wins: “this Music bug now fails in-toy”.


---

## Fire #20 harden verdict (2026-09-09)

`lyric_span_entangle` under locked 1200+c1.5 n12: **0/6** (multi_row). Isolation:

- hetero amps + e_on_content=0 → still multi_row fail (rows_cov≈1/5), leak tiny
- homo amps + e_on_content=0 → rows_cov≈3/5 (row_scales stagger), still fail
- n=4 / cover=2.0 / steps=1600 / e_unused=1: no recovery
- leftover single-row regression: still PASS

**Document as Music multi-span shared-residual hard boundary.** Do not revise locked recipe. Optional future: per-row residual research (out of recipe scope).


## M13–M19 new stressors (2026-09-09) — RESULTS

| ID | Cell | locked n12 | music n1 | Status / notes |
|---|---|---|---|---|
| M13 | `tiny_slider_dom` | **6/6** | **5/6 KNIFE** seed0 | harder û; **n≥2 / vic0@n1 → 6/6** (Fire #21 transfer) |
| M14 | `e_on_u_declare_lie` | **0/6 BITES** | **0/6 BITES** | YAML û-lie; **content_deleted** (exam≈0.18=content_kept); n/vic/per-row NO; vs M20 same nature; see `m14_failmode_20260909` |
| M15 | `prefix_shared_proxy` | **6/6** | **6/6** | positive control / prefix hold OK |
| M16 | `scale_stagger_homo` | **0/6 BITES** | **0/6 BITES** | **multi_row_scale_dof_partial** rows≈3; n/vic NO; per_row w0 **6/6**; w0.3 knife 2/3; scale cliff span≤0.4 PASS / ≥0.6 FAIL; see `m16_m17_mech_20260909` |
| M17 | `roles_split_proxy` | **0/6 BITES** | **0/6 BITES** | **multi_row_axis_roles_zero** rows≈0; n/vic NO; per_row w0/w0.3 **YES**; blend cliff lastPASS=0.35 / knife@0.4 / hardFAIL≥0.5; soft→M2; see `m16_m17_mech_20260909` |
| M18 | `close_with_leak` | **6/6** | **5/6 KNIFE** seed0 | close+ê; **n≥2 / vic0@n1 → 6/6** |
| M19 | `grit_content_dom` | **6/6** | **6/6** | grit/content-dom survival OK |

See `music_toy_new_stressors_m13_20260909.md`, `roles_scale_isolation_20260909.md`, `catalog_m13_m19_results_20260909.md`.

### ADOPTs (Music posture — not locked defaults)

- Close-family **n≥2** harden + multi-seed gate @ n=1
- **`vicreg_weight=0` when n≤1** Music-posture harness only
- Per-row AdvResidual **w≤0.3** analysis-only (merge=NO; clears M1/M16/M17/M24/M27/M2/M29 DoF; **not** M14/M20/M21/M28/M30/M31 declare-lie/eoc; M16 prefers w=0 — w0.3 knife 2/3)

## Fire #21 close harden (2026-09-09)

`f3d_close` / `close_live_noise` under locked 1200 steps, cover∈{1.0,1.5}, `faithful_guard_e`:

- n=1 Music posture: **seed0-only knife** — close 5/6 (c1.0+c1.5), live default 3/4 fail=[0]
- **Harden path n≥2**: close n2 c1.0 **6/6**, c1.5 **6/6**; live n2 **4/4**
- Prior “n≥4” note was conservative — **n≥2 is enough** in-toy
- leftover n12 c1.5 still **6/6** (no regression). Recipe unchanged (n≤12 lock stands).
- **Fire #22:** `field3d` now emits Music-posture warn at n=1 close (`music_close_posture_warn` / score key); recipe unchanged.
- **Fire #23:** dig util `music_posture_dig_util.annotate_dig_rows` + pytest Music→toy asserts; `summarize_music_close_posture`; recipe unchanged.
- **Fire #24:** fold M21 irreducible (floor 0.20503 / declared_e≠leak_e) into scoreboard+catalog; retrofit `annotate_dig_rows` via `retrofit_annotate_dig_rows_fire24_20260909.py`; recipe unchanged.



## Batch2 new stressors (2026-09-09)

| ID | Music symptom | Cell | Notes |
|---|---|---|---|
| M20 | Declared leak YAML / content-axis amplitude lie (leftover geom) | `amp_lie_leftover_declare` | Distinct from e_on_u_declare_lie / cross_axis_mismatch_declare; **HARD_BITE 0/6** both postures |
| M21 | Content↔leftover mix ≈ hold-ê lyric pool | `hold_e_lyric_mix` | **HARD_BITE 0/6** both; **irreducible floor lr=0.20503** (declared_e tilt vs exam pure leak_e); per-row cannot clear; train-pure-ê clears (ablation); see `m21_eoc_leak_irreducible_20260909` / Fire #24 fold |
| M22 | Mild multipair (between homo and full cross_axis) | `stagger_mild_cross` | Soft axis wander + stagger; locked NO; music **KNIFE 4/6** |
| M23 | Multi-pair R³ correlated seeds | `multipair_corr_seed` | Seed-tied amp correlation; **NO_BITE 6/6** both |
| FM | Batch FM on vs off under close | recipe probe | Keep FM0 |
| l2 | particle_l2 extremes @ Music n=1 | recipe probe | Keep l2=0.02 |

See `music_to_toy_new_stressors_batch2_20260909.md` bite table.


## Batch3 new stressors (2026-09-09)

| ID | Music symptom | Cell | Notes |
|---|---|---|---|
| M24 | Mid-caption content↔leak attribute flip (û stays primary) | `content_leak_flip_rows` | Not roles_split / cross_axis / stagger_mild |
| M25 | Lyric-token neu heavy vs leftover unused-ê gate | `lyric_neu_heavy_gate` | Not prefix_shared / hold_e_lyric_mix |
| M26 | Ambiguous declared ê split û/content/unused | `declare_split_three` | **content_deleted_under_declare_lie** (same family as M14/M20); n/vic/per_row NO; see `declare_lie_family_20260909` |
| M27 | Descending span scales (traj/outro reverse) | `scale_descent_homo` | Mirror of M16 ascending |
| M21d | hold_e_lyric_mix portable-knob deepen | probes | **recipe hard boundary**; driver=`e_on_content>0` → declared_e tilt; analytic floor 0.20503; per-row NO; see batch3 + `m21_eoc_leak_irreducible_20260909` |

See `music_to_toy_batch3_20260909.md` bite table.


## Batch4 M28–M31 (2026-09-09) — RESULTS

| ID | Cell | locked | music | notes |
|---|---|---|---|---|
| M28 | `guard_refuse_hot_eoc` | **0/6** HARD_BITE | **0/6** HARD_BITE | exam_l=1.000 leak_m=0.660 |
| M29 | `content_cascade_rows` | **0/6** HARD_BITE | **0/6** HARD_BITE | exam_l=0.881 leak_m=0.010 |
| M30 | `eoc_threshold_edge` | **0/6** HARD_BITE | **0/6** HARD_BITE | exam_l=0.710 leak_m=0.214 |
| M31 | `leftover_hot_eoc_declare` | **0/6** HARD_BITE | **0/6** HARD_BITE | exam_l=-0.155 leak_m=0.215 |
| CTRL | leftover | **3/3** | — | must stay green |

From M21 mech dig: M28 = guard-refuse band; M30 = ê-floor edge; M31 = admitting hot eoc on leftover amps. Recipe change=NO.


## Fire #25 note — M29 per-row clearance (2026-09-09)

- `content_cascade_rows` (M29): shared HARD_BITE; **per-row CLEARS** (6/6) at couple_w=0.0 — DoF cascade like M24/M27.
- Keep-biting held: M20 declare-lie, M28 guard-refuse, M30 ê-floor.
- recipe_change=NO; merge_to_trainer=NO.

## Fire #26 — M30 per-row knife (2026-09-09)

- M30 `eoc_threshold_edge` per_row is a **1/6 knife** (seed1 only; lr 0.1992 vs fails 0.2016–0.2053).
- Analytic floor at eoc=0.33 is lr≈0.2016 (gate 0.20); couple cannot clear.
- Keep HARD_BITE; do not chase with recipe/couple/per-row. Same ê-floor family as M21.


## M16/M17 mechanistic (2026-09-09)

Source: `m16_m17_mech_20260909` wall=4331s.

| mid | nature | n/vic clear? | per_row clear? |
|---|---|---|---|
| M16 | multi_row_scale_dof_partial | NO | w0 YES 6/6; w0.3 knife 2/3 |
| M17 | multi_row_axis_roles_zero | NO | w0 YES; w0.3 YES |
| M2 | multi_row_plus_leak_cross_axis | — | YES (DoF) |
| M27 | multi_row_scale_dof_partial | — | YES |

Scale-span cliff: ≤0.4 PASS → ≥0.6 FAIL. Blend cliff: ≤0.35 PASS → 0.4 knife → ≥0.5 hard FAIL.
**DoF family** (not declare-lie). Recipe change=NO. Keep HARD_BITEs under locked shared.

## Fire #26b — M16/M17 mech (2026-09-09)

- M16/M27: **scale DoF** (multi_row_scale_dof_partial); per_row w0 clears; n/vic no; scale cliff at span **0.6**
- M17: **axis-roles** (multi_row_axis_roles_zero); blend cliff **0.35 PASS / 0.4 FAIL**; per_row clears
- M2: still hard under locked shared; per_row clears (axis-mix DoF) — consistent with couple-band ADOPT
- Keep as HARD_BITEs under locked shared; recipe_change=NO


## Declare-lie family (2026-09-09)

Unified note: `declare_lie_family_20260909`. M14/M20 = content_deleted_under_declare_lie; M26=content_deleted_under_declare_lie; suite_pass=True. n/vic/per_row NO. Fix=YAML/target not recipe.


**Declare-lie cliff:** M14 `e_on_u=0` PASS → **`e_on_u≥0.15` BITES** (suite 20260909; sharper than prior ≥0.3).


## Declare-lie / eoc-floor family (2026-09-09)

Unified theory: `declare_lie_eoc_family_theory_20260909`. A=M14/M20/M26 content_deleted; B=M21/M30/M31 ê-floor admit; C=M28 refuse-raw. Cliffs: eou≥0.15; emp floor@0.32 / ana@0.33; refuse_on=0.68. suite_pass=True. Fix=YAML/declared_ê not recipe.
