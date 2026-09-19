# Session findings — box dig evening 2026-09-09

Pop-os sync succeeded earlier (92 paths). These notes are for later delta-sync.
**No Music GPU train. Locked recipe unchanged. merge per-row = NO.**

## Completed this session

### Folds (scoreboard)
1. Per-row couple band **w∈[0.0, 0.3]**; saves `cross_axis_rows`; merge=NO
2. **ADOPT** Music-posture `vicreg=0` when `n<=1` — not locked AdvConfig
3. **M14** = `content_deleted_under_declare_lie` (M20 family)

### Digs
| dig | wall | finding |
|---|---|---|
| `m21_eoc_leak_irreducible_20260909` | ~738s | Teacher-axis floor: tilted `declared_e` → lr≈0.205; per-row cannot clear; train-pure-ê clears (diag). Two modes: admit→ê-floor / refuse→raw leak |
| `music_to_toy_batch4_20260909` | 1428s | **M28–M31 all HARD_BITE 0/6** both postures; leftover 3/3; vic0 still fails M28/M30 |
| `per_row_falsify_batch4_20260909` | 1787s | **M29 clears 6/6**; M28/M30/M31 stay; CTRL ok; merge=NO (aligns Fire #25) |
| `m30_per_row_knife_20260909` | 2000.6s | **Fire #26**: seed1-only 1/6; lr≈0.199–0.205 near analytic floor; couple w∈{0,0.1,0.3} no clear; keep HARD_BITE |

### Batch4 cells (new in field3d + leak-gate kinds)
- M28 `guard_refuse_hot_eoc` — refuse band leak≈0.66
- M29 `content_cascade_rows` — DoF multi 2/4; **per-row YES**
- M30 `eoc_threshold_edge` — ê-floor edge eoc=0.33
- M31 `leftover_hot_eoc_declare` — admit hot eoc on leftover amps

### Pytest
- `test_field3d` + `test_music_to_toy_stressors`: **22 passed**


| `m16_m17_mech_20260909` | 4331s | **Fire #26b**: M16/M27 scale-DoF; M17/M2 axis-mix; per_row clears all; n/vic no; cliffs span0.6 / blend0.35→0.4 |

### Pytest Fire #26
- box-cpu @ 435e873: **70 passed** (~680s under dig contention). pop-os SSH down.

## Paths
- Scoreboard: `analysis/slider2d/notes/MUSIC_TO_TOY_SCOREBOARD_20260909.md`
- Research log: `analysis/slider2d/notes/research_log_20260909.md`
- M21: `.../m21_eoc_leak_irreducible_20260909.{py,md,json}`
- Batch4: `.../music_to_toy_batch4_20260909.{py,md,json}`
- Falsify: `.../per_row_falsify_batch4_20260909.{py,md,json}`


### M30 knife (complete)
- pass_seeds=[1]; couple no 6/6; keep HARD_BITE. Paths: `m30_per_row_knife_20260909.*`

### Declare-lie / eoc-floor family theory (DONE)
- `declare_lie_eoc_family_theory_20260909.{py,md,json}` wall=827.8s suite_pass=True
- Branches: A=M14/M20/M26 content_deleted; B=M21/M30/M31 ê-floor admit; C=M28 refuse-raw
- Cliffs: eou 0→≥0.15; emp floor@0.32 / ana@0.33; refuse≥0.68; M31 in catalog
- Companion: `declare_lie_family_20260909` neg-control suite_pass=True (M14/M20/M26 natures)
- Recipe=NO merge=NO No Music train

### Fire #27 (routine 22:02 MDT)
- pop-os pytest **70 passed** / 134.68s @ 435e873
- Declare-lie + eoc-floor family theory folded (suite_pass=True both digs)
- Cliffs: eou≥0.15 BITES; eoc floor emp@0.32 / ana@0.33; refuse≥0.68
- recipe_change=NO

### Valley emp + sheet/highd A/B/C ports
- `valley_emp_sheet_abc_ports_20260909` wall=565.9s valley=`valley_emp_BITES_mixed_mech`; sheet A/C bites; DoF sheet NO_BITE; highd A bites
- recipe=NO merge=NO No Music train
