# 2D adversarial slider research log — 2026-09-09

Box-cpu fires against mikkel main `435e873` (Port ParticleGAN RpGAN + b_cap into the 2D slider example #94).
Locked recipe: 1200 steps + cover_weight 1.5, leftover-gated (`faithful_guard_e`), FM off, n_particles≤12, b_cap=1. Do NOT adopt 800×cover3.0 (false lock).

## Fire #6 — FM-on-normalized vs FM-off (2026-09-09)

- Host: box-cpu @ SHA `435e873`
- Tests: `pytest tests/test_lm_2d_adv.py -q` → **10 passed in 50.89s** (wall 51.6s)
- Grid seed=0: baseline kept=0.9306; baseline_fm0 kept=0.9306 PASS, fm_norm_0.1 kept=0.9298 PASS, fm_norm_0.5 kept=0.9305 PASS, fm_norm_1.0 kept=0.9295 PASS, fm_raw_0.5_control kept=0.9282 PASS
- Multi-seed: w=0.0 6/6; w=0.1 6/6; w=0.5 6/6; w=1.0 6/6
- Raw FM control (w=0.5, normalize=False): kept 0.9282 PASS but worse pole_rel_err / sheet_dir vs off — still document as known-bad uncapped path; do not enable.
- Verdict: **keep_fm_off_flat** — FM-on PASSes but is flat vs off (mean 0.9304 vs 0.9303); no recipe change.
- Next: particle_l2 micro-sweep at high n, or multi-pair/cross-axis stress, or exam_score at locked recipe (prefer particle_l2 micro-sweep @ n=12 first)
- Notes: `fm_normalized_vs_off_20260909.{py,json,md}`

## Fire #7 — particle_l2 micro-sweep (2026-09-09)

- Host: pop-os-cpu @ SHA `435e873`
- Seed0 grid l2∈[0.0, 0.005, 0.01, 0.02, 0.05, 0.1]: 0.0→0.9288P, 0.005→0.9302P, 0.01→0.9303P, 0.02→0.9295P, 0.05→0.9302P, 0.1→0.9308P
- Multi-seed l2=0.0: 6/6 kept mean 0.9303 span 0.0025
- Multi-seed l2=0.02: 6/6 kept mean 0.9302 span 0.0016
- Multi-seed l2=0.05: 6/6 kept mean 0.9306 span 0.0019
- Multi-seed l2=0.1: 6/6 kept mean 0.9306 span 0.0010
- Verdict: **keep_default_0.02** — default l2=0.02 seed-stable 6/6 mean kept 0.9302 span 0.0016; l2=0 ablation 6/6 mean kept 0.9303 (Δ vs 0.02 = +0.0000); best multi-seed l2=0.1 6/6 mean 0.9306 (Δ=+0.0004); flat vs default (Δkept ≪ 0.01) — no recipe change; seed0 grid: 0.0->0.9288P, 0.005->0.9302P, 0.01->0.9303P, 0.02->0.9295P, 0.05->0.9302P, 0.1->0.9308P
- Next: multi-pair/cross-axis stress, or exam_score at locked recipe, or LR/b_cap micro-sweep that preserves seed stability
- Notes: `particle_l2_sweep_20260909.{{py,json,md}}`

## Fire #19 — Music→toy dig (lyric_span_entangle + f3d_close knife) (2026-09-09)

- Host: box-cpu @ SHA `435e873` (+ local field3d.py row_amps fix)
- Tests: see parent report (pytest after odd()/basis rename)
- Dig: `music_toy_dig_fire19_20260909.{py,json,md}` wall=356.3s
- Catalog synced: `music_to_toy_stressor_catalog_20260909.md`
- lyric_span_entangle: locked 0/6 mean_prim=0.9903 leak_max=0.1923; music 0/6 (fail=pass_multi_row 0/5 rows)
- f3d_close n=1: c1.0 5/6 fail=[0]; c1.5 5/6 fail=[0]
- leftover/close n=12: 6/6 / 6/6
- cross_axis n=1 c1.0: 0/6; dual L/listen/gate: 3/3/0/3/0/3
- Verdict: lyric_bite=YES; seed0_knife=YES; regression=NO; recipe_change=NO; ping_user=YES
- No Music GPU train; servers untouched.

## Fire — Music→toy stressors (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1` (laptop/pop-os hop DOWN)
- Catalog: `music_to_toy_stressor_catalog_20260909.md`
- Dig: `dig_close_cross_music_stress_20260909.{py,json,md}` wall=351.5s
- Run: `music_to_toy_stressors_run_20260909.{py,json,md}` wall=1034.5s
- field3d: Music ctors + CELLS_3D registry (lyric_span_entangle / close_live_noise / dual_arm_leftover_geom)
- Wins: ['lyric_span_entangle/c1.5_n12: 0/6 — M1 span-entangled leftover bites', 'lyric_span_entangle/c1.5_n1: 0/6 — M1 span-entangled leftover bites', 'lyric_span_entangle/c1.0_n12: 0/6 — M1 span-entangled leftover bites', 'lyric_span_entangle/c1.0_n1: 0/6 — M1 span-entangled leftover bites', 'dual_arm/leftover_only_guard_cover0_n1: 0/3 — M4/M7/M8 dual-arm arm must not pass alone', 'dual_arm/listen_cover_only_faithful_c1.5_n1: 0/3 — M4/M7/M8 dual-arm arm must not pass alone', 'dual_arm/locked_guard_cover1.5_n1: 3/3 — gate∧cover required (positive control)', 'close_live_noise/c1.5_n1: 5/6 — M3/M11 close+live noise knife']
- No Music GPU train; servers untouched.

## Fire #20 — lyric_span_entangle harden (2026-09-09)

- Host: box-cpu @ SHA `435e873` (pop-os hop DOWN)
- Tests: pytest test_lm_2d_adv + test_lm_highd_leftover + test_field3d → **59 passed** (~121s)
- Dig: `lyric_span_harden_fire20_20260909.{py,json,md}` wall=300.0s
- locked M1: 0/6 multi=0/6 exam=0.9903 leak_max=0.1923
- hetero eoc0→0.45: 0/3 → 0/3; homo eoc0/0.45: 0/3/0/3
- Verdict: hard_boundary=True; eoc_soft=False; cross_axis_driver=False; any_harden=False; recipe_change=NO; ping_user=YES
- Next: if hard_boundary, document Music multi-span shared-residual limit in catalog; optional scaffold per-row residual research (do NOT adopt without multi-seed Music).

## Fire — harden Music→toy bites (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1` (laptop offline; no SSH)
- Notes: `harden_music_toy_bites_20260909.{py,json,md}` wall=1461.3s
- lyric_span / cross_axis: HARD BOUNDARY (confirm; no recipe chase) — Fire #20 stands
- close n=1 c1.5: 5/6 fail=[0]; live: 5/6 fail=[0]
- seed0 @ n=1 û-undershoot (u≈0.38); content+multi OK — init/basin
- seed0 n=1 probe recovers: ['close_s0_n1_vic0']; n_floor_6of6=2; any_n1_6of6=True
- leftover regression: False
- Verdict: recipe_change=NO; Music rule=multi-seed close / n≥4 for close-family under parts0
- No Music GPU train; servers untouched.

## Fire — harden bites follow-up: vic=0 safety + rules (2026-09-09)

- Host: box-cpu @ `435e87363bf1`. pytest related: **62 passed** (~545s).
- vic=0 @ n=1: close 6/6, live 6/6; leftover n12 6/6 flat vs vic0.05; divergent 3/3 both.
- n_particles floor for close 6/6: **2** (n=2/3/4 all green).
- Rejected as seed0@n=1 fix: particle_l2, span/end/cloud/jitter, lr, steps1600, cover2.0.
- **Recipe change=NO.** Music parts0 rule: multi-seed close OR n≥2 OR vic=0@n=1 (candidate).
- Notes updated: `harden_music_toy_bites_20260909.md` (+ `vic0_safety` in json).

## Note — vicreg_n1_close_knife (2026-09-09)

- Doc: `vicreg_n1_close_knife_20260909.md`
- Complements Fire #21 n≥2: vic=0@n=1 also clears close/live seed0 knife; leftover flat.
- Portable theory: VICReg std ill-posed at n=1 → content-only basin.
- Recipe change still NO; Music-posture candidate only.

## Fire #21 — close / live_noise harden (2026-09-09)

- Host: box-cpu @ SHA `435e873` (pop-os hop DOWN)
- Dig: `close_harden_fire21_20260909.{py,json,md}` (+21a/b/d) wall_d=219.6s
- Tests: pytest test_lm_2d_adv + test_lm_highd_leftover + test_field3d + test_music_to_toy_stressors → **64 passed** (~823s, CPU-contended)
- leftover n12 c1.5: 6/6
- close n1: c1.0 5/6 c1.5 5/6 (seed0-only knife)
- close n2: c1.0 6/6 c1.5 6/6 (harden_n_ge_2=True)
- live: n1 3/4 n2 4/4 (seeds [0, 1, 2, 42])
- Verdict: recipe_change=NO; Music close harden **n≥2** (tighter than prior n≥4); ping_user=YES
- Next: catalog M3 update; optional field3d close Music-posture warning

## Fire — Music→toy new stressors batch2 (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Notes: `music_to_toy_new_stressors_batch2_20260909.{py,json,md}` wall=659.4s
- New CELLS_3D: amp_lie_leftover_declare (M20), hold_e_lyric_mix (M21), stagger_mild_cross (M22), multipair_corr_seed (M23)
- Bites: ['amp_lie_leftover_declare/locked: 0/6 HARD_BITE', 'amp_lie_leftover_declare/music: 0/6 HARD_BITE', 'hold_e_lyric_mix/locked: 0/6 HARD_BITE', 'hold_e_lyric_mix/music: 0/6 HARD_BITE', 'stagger_mild_cross/music: 4/6 KNIFE', 'close_fm0.0_music/music: 2/3 KNIFE', 'close_fm0.5_music/music: 2/3 KNIFE', 'close_n1_l2_0.0/music: 2/3 KNIFE', 'close_n1_l2_0.02/music: 2/3 KNIFE', 'close_n1_l2_0.2/music: 2/3 KNIFE']
- No-bites: ['stagger_mild_cross/locked: 6/6', 'multipair_corr_seed/locked: 6/6', 'multipair_corr_seed/music: 6/6', 'close_fm0.0_locked/locked: 3/3', 'close_fm0.5_locked/locked: 3/3', 'leftover_n1_l2_0.0/music: 3/3', 'leftover_n1_l2_0.02/music: 3/3', 'leftover_n1_l2_0.2/music: 3/3', 'leftover_regression/locked: 3/3']
- Verdict: recipe_change=NO; leftover_regression=3/3; ping_user=YES
- No Music GPU train; servers untouched.

## Align — Fire #21 n≥2 into harden bites (2026-09-09)

- Fire #21 confirmed close/live harden at **n_particles≥2** (c1.0+c1.5 6/6).
- Updated `harden_music_toy_bites_20260909.{md,json}` music_rule to n≥2 (was still saying ≥4 in places).
- vic=0@n=1 remains complementary candidate (see `vicreg_n1_close_knife_20260909.md`).
- M13 grid in flight: M13 tiny_slider music n1 = KNIFE seed0; M14 e_on_u_declare_lie locked = BITES 0/6.

## Mid-flight — M13–M19 grid (2026-09-09 ~18:55 MDT)

- Still running on box-cpu (~20 min in). Partial:
  - M13 tiny_slider locked n12: **6/6**; music n1: **5/6 KNIFE** fail=[0] (Fire #21 family)
  - M14 e_on_u_declare_lie: **0/6 BITES** locked + music (YAML û-lie — new win)
  - M15 prefix_shared locked n12: **6/6** so far
- Fire #21 n≥2 incorporated into harden bites notes.

## Fire — per-row / multi-residual explore (NON-DEFAULT) (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1` (laptop offline; no Cursor cloud; no Music GPU)
- Dig: `per_row_residual_explore_20260909.{py,json,md}` wall=875.3s
- Geometry: M1 mean-δ covers 0/5 (floor); shared lyric multi=0/3
- Per-row lyric: pass=6/6 multi=6/6 bite=6/6
- Controls under per-row: leftover=3/3, close=3/3, unused_e=3/3
- Curriculum/weighted/scaled shared: multi 0/3/0/3/0/3 (expect still dead on M1 amp mix)
- Verdict: YES — per-row residual recovers multi-span (teacher-aligned exam) without breaking leftover/close smoke controls
- recipe_change=NO; merge_to_trainer=NO; locked defaults untouched
- Caveat: unconstrained head_min_cos≈0.56; couple w0.3 still 3/3 multi; w1.0 kills multi 0/3; scaled_shared fixes homo-scale only

## Fire — Music→toy batch3 (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Notes: `music_to_toy_batch3_20260909.{py,json,md}` wall=951.4s
- New CELLS_3D: content_leak_flip_rows (M24), lyric_neu_heavy_gate (M25), declare_split_three (M26), scale_descent_homo (M27)
- Geom bites: ['content_leak_flip_rows/locked: 0/6 HARD_BITE', 'content_leak_flip_rows/music: 0/6 HARD_BITE', 'declare_split_three/locked: 0/6 HARD_BITE', 'declare_split_three/music: 0/6 HARD_BITE', 'scale_descent_homo/locked: 0/6 HARD_BITE', 'scale_descent_homo/music: 0/6 HARD_BITE']
- M21 deepen recipe_hard_boundary=True (n/cover/steps/vic/false-lock never recover); driver=e_on_content>0 (eoc0 geom ablation clears — keep cell eoc); still_biting_probes=['M21/baseline_locked: 0/3 HARD_BITE', 'M21/baseline_music: 0/3 HARD_BITE', 'M21/n1_c1.5: 0/3 HARD_BITE', 'M21/n2_c1.5: 0/3 HARD_BITE', 'M21/n4_c1.5: 0/3 HARD_BITE', 'M21/n12_c1.5: 0/3 HARD_BITE', 'M21/n12_c1.0: 0/3 HARD_BITE', 'M21/n12_c2.0: 0/3 HARD_BITE']
- leftover_regression=3/3; recipe_change=NO; ping_user=YES
- No Music GPU train; servers untouched.

## Fire — close_seed0_deep (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Dig: `close_seed0_deep_20260909.{py,json,md}` wall=2175.1s
- **Folds Fire #21**: close knife clears at **n≥2** (c1.0+c1.5 6/6; live n1→n2 recovers)
- fail_mode: **exam_u_undershoot_init_basin** (û undershoot; not leak)
- seeds0..20 n=1: close_c1.5 20/21 fail=[0]; close_c1.0 19/21 fail=[0, 4] (NOT seed0-only); live_c1.5 20/21 fail=[0]
- n_floor close∧live 6/6: **2**; s0 knobs l2/bcap/cover/jitter/cloud all fail
- vic0@n=1 seeds0..20: close 21/21 live 21/21; leftover n12 flat 0.9303→0.9303
- Decision: **mandatory_multi_seed_gate_PLUS_n_ge_2_harden_PLUS_propose_vic0_when_n1**; recipe_change_silent=NO; propose_vic0_when_n1=True
- Music rule: MANDATORY multi-seed gate for close/close_live_noise under Music parts0 (n=1). Operational harden (Fire #21 + this dig): n_particles>=2 clears knife (close+live 6/6). PROPOSE (not silent): vicreg_weight=0 when n_particles==1 (21/21 heal; leftover n12 flat 0.9303→0.9303). Do NOT silently flip locked defaults. Rejected: l2/b_cap/cover/jitter/cloud as seed0@n=1 fix; 800×cover3.0.


## Fire #22 — Music-posture warn helper for close cells (2026-09-09)

- Host: box-cpu @ SHA `435e873` (pop-os/laptop SSH DOWN; no CloudAgent; no Music GPU)
- Tests: pytest test_lm_2d_adv + test_lm_highd_leftover + test_field3d + test_music_to_toy_stressors → **67 passed in 154.07s** (wall ~155s; log `/workspace/2d-adv-pytest-fire22.log`)
- Scaffold: `music_close_posture_warn(cfg_or_n_particles, cell_kind=...) -> Optional[str]` in `field3d.py`
- Warns when cell ∈ {close, close_live_noise, tiny_slider_dom, close_with_leak} AND n_particles==1 (Music --parts 0 proxy)
- Message cites harden: n_particles>=2 OR multi-seed gate; optional candidate vicreg_weight=0@n=1 (no silent default flip)
- Call site: `score_adv_field3d` emits `music_close_posture_warn` key (None when safe); locked recipe untouched
- Tests: `test_music_close_posture_warn_n1_close_family`, `test_score_adv_field3d_emits_music_close_posture_warn_key`
- Verdict: recipe_change=NO; ping_user=NO — documents already-known Fire #21 knife as an explicit testable warn; no new bite/hard boundary
- Notes: `music_close_posture_warn_fire22_20260909.{md,json}`
- Next: optional wire warn into Music→toy stressor runners; keep mandatory multi-seed gate for Music parts0 close

## Fire — Music→toy M13–M19 new stressors (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Notes: `music_toy_new_stressors_m13_20260909.{py,json,md}` wall=3108.2s
- Registered CELLS_3D: ['tiny_slider_dom', 'e_on_u_declare_lie', 'prefix_shared_proxy', 'scale_stagger_homo', 'roles_split_proxy', 'close_with_leak', 'grit_content_dom']
- Wins: ['M13_tiny_slider_dom_music_n1_c1.0: 5/6 KNIFE fail=[0]', 'M14_e_on_u_declare_lie_locked_n12_c1.5: 0/6 BITES', 'M14_e_on_u_declare_lie_music_n1_c1.0: 0/6 BITES', 'M16_scale_stagger_homo_locked_n12_c1.5: 0/6 BITES', 'M16_scale_stagger_homo_music_n1_c1.0: 0/6 BITES', 'M17_roles_split_proxy_locked_n12_c1.5: 0/6 BITES', 'M17_roles_split_proxy_music_n1_c1.0: 0/6 BITES', 'M18_close_with_leak_music_n1_c1.0: 5/6 KNIFE fail=[0]', 'probe_close_n1_end0: 5/6 KNIFE fail=[0]', 'probe_close_n1_span1: 2/3 KNIFE fail=[0]']
- Recipe change: NO. No Music GPU train.

## Fire — per-row residual deepen (NON-DEFAULT) (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1` (no Music GPU; locked recipe UNCHANGED)
- Dig: `per_row_residual_deepen_20260909.{py,json,md}` wall=2138.4s
- Couple band: w∈[0.0, 0.3] robust=[{'w': 0.0, 'pass': '6/6', 'multi': '6/6', 'bite': '6/6', 'mean_head_min_cos': 0.5571, 'min_head_min_cos': 0.5547, 'mean_exam': 0.9894, 'leak_max': 0.132}, {'w': 0.1, 'pass': '6/6', 'multi': '6/6', 'bite': '6/6', 'mean_head_min_cos': 0.581, 'min_head_min_cos': 0.5771, 'mean_exam': 0.9985, 'leak_max': 0.1116}, {'w': 0.3, 'pass': '6/6', 'multi': '6/6', 'bite': '6/6', 'mean_head_min_cos': 0.6186, 'min_head_min_cos': 0.613, 'mean_exam': 0.9928, 'leak_max': 0.1757}]
- Other bites saved by per-row: ['cross_axis_rows']; still_fail=['hold_e_lyric_mix', 'amp_lie_leftover_declare']; amp_lie_expected_fail=True
- Music-posture n=1 c1.0 per-row lyric clears=True
- Controls leftover/close: 3/3 / 3/3 (ok=True)
- Verdict: YES — couple robust band w∈[0.0, 0.3] clears lyric multi; per_row saves ['cross_axis_rows']; amp_lie expected_fail=True; music_posture_clears=True; controls_ok
- recipe_change=NO; merge_to_trainer=NO; Music mapping sketch in md (propose-only)

## Fire — per-row falsify A (M21/M24/M27 vs M20) (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Dig: `per_row_falsify_m21_m24_m27_20260909.{py,json,md}` wall=1578.8s
- Scoreboard: `MUSIC_TO_TOY_SCOREBOARD_20260909.md`
- cleared_geom=['M24', 'M27']; uncleared=['M21']; m20_still_bites=True; overpowered=False
- Verdict: YES — per-row clears ['M24', 'M27'] without clearing M20 (amp-lie still bites); uncleared=['M21']; CTRL ok; merge=NO
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- No Music GPU train; servers untouched.

## Note — vicreg0_n1_propose_stress (2026-09-09)

- Doc: `vicreg0_n1_propose_stress_20260909.md` / `.json` / `.py`
- **Recommend: ADOPT** for Music-posture n=1 only (locked defaults unchanged).
- Replicate c1.5: close 6/7→7/7; live 6/7→7/7
- Music c1.0: close 19/21→21/21; live 20/21→21/21
- leftover n12: 6/6@0.9303 vs 6/6@0.9303 flat=True
- dual_ok=True; M22 4/6→6/6 false_lock_risk=True; M23 6/6
- bites_ok=True: M20_amp_lie 0/6, M21_hold_e 0/6, M24_content_leak_flip 0/6
- reject_reasons=[]; warnings=['M22 knife clears under vic0 (4/6→6/6) — soft false-lock risk; document as posture side-effect, not hard bite']
- wall=1168.8s sha=435e87363bf1

## Fire #23 — Wire music_close_posture_warn into dig runners (2026-09-09)

- Host: box-cpu @ `435e873` (pop-os SSH up this fire; suite also run there); no CloudAgent; no Music GPU
- Tests (box): pytest test_lm_2d_adv + test_lm_highd_leftover + test_field3d + test_music_to_toy_stressors → **69 passed in 139.19s** (log `/workspace/2d-adv-pytest-fire23-box.log`)
- Tests (pop-os after sync): same suite → **69 passed in 142.30s** (log `/tmp/2d-adv-pytest-fire23b-popos.log`); pre-sync baseline was 57 passed (older field3d, no music_to_toy). SHA `435e873`. nano-work-server left running.
- Scaffold: `summarize_music_close_posture(rows)` in `field3d.py`; dig util `notes/music_posture_dig_util.py` (`annotate_dig_rows` / `posture_block_for_md`)
- Smoke dig: `wire_music_posture_warn_fire23_20260909.{json,md}` — close-family@n=1 warn 12/12; clear @n=2 + leftover/lyric 24/24; missing_key=0
- Pytest: Music→toy stressors now assert warn @ close_live_noise n=1 and clear @ n=2 / lyric / dual; `test_summarize_music_close_posture_aggregates_dig_rows`
- Verdict: recipe_change=NO; ping_user=NO — wires Fire #22 warn into dig aggregation + smoke tests; no new bite/hard boundary
- Next: retrofit older dig scripts with `annotate_dig_rows`; sync field3d+tests to pop-os (`SYNC_TO_POPOS.md`)

## Fire — M20 tiny/closeleak n≥2 transfer (2026-09-09)

- Notes: `music_toy_m20_sheet_span_20260909.{py,json,md}` wall=1065.1s
- Wins: ['M13_tiny_n1_c1.0: 5/6', 'M18_n1_c1.0: 5/6', 'M14_n12: 0/6', 'M14_n2: 0/6']
- Fire #21 n≥2 transfer; recipe_change=NO.

## Note — fold couple band + vic0@n1 ADOPT into scoreboard (2026-09-09)

- Scoreboard `MUSIC_TO_TOY_SCOREBOARD_20260909.md` now has explicit fold section:
  (1) per-row couple robust **w∈[0.0, 0.3]**; saves `cross_axis_rows`; merge=NO
  (2) **ADOPT** Music-posture `vicreg=0` when n<=1 — locked defaults unchanged
- Next dig in flight: M21 eoc leak irreducible mechanistic

## Fire — roles/scale isolation (2026-09-09)

- Notes: `roles_scale_isolation_20260909.{py,json,md}` wall=391.9s
- Soften ladder vs M2 hard boundary; recipe_change=NO.

## WRAP — executor harden/M13 thread (2026-09-09)

- See `WRAP_executor_harden_m13_20260909.md` (full). Stopping after 2h+.
- Delivered: harden bites + vic0 propose; M13–M19 cells+grid; M20 n≥2 transfer; roles blend cliff mix0.25→0.5.
- pytest field3d+music: 19 passed. Recipe change=NO. No Music train.

## Fire — M21 eoc leak irreducible (mechanistic) (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Dig: `m21_eoc_leak_irreducible_20260909.{py,json,md}` wall=737.5s
- Analytic floor @ M21 defaults: leak_ratio=0.20503 (â_e=-0.20503)
- analytic↔teacher match=True; residual≈floor=True
- shared declared: pass_leak=0/6 mean_lr=0.210619
- per_row w0: pass=0/6 leak_max=0.207 (still fails leak gate)
- train pure ê: pass_leak=6/6 (clears — teacher-tilt diagnostic)
- leftover_ok=True; m20_still_bites=True
- Verdict: YES — M21 leak is teacher-axis geometry floor (declared_e tilt), not shared-DoF / optimization; per-row cannot clear; train-pure-ê clears
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- No Music GPU train; servers untouched.

## Fire #24 — Fold M21 irreducible + annotate retrofit (2026-09-09)

- Host: box-cpu; no CloudAgent; no Music GPU
- Docs: scoreboard M21 hard-boundary + hard-bites rows cite floor **0.20503** / declared_e≠leak_e / per-row cannot clear; catalog M21/M21d + Fire #24 bullet; fold note `m21_fold_fire24_20260909.md`
- Retrofit smoke: `retrofit_annotate_dig_rows_fire24_20260909` PASS (util path; wall≈0.015s)
- Scripts wired (minimal): dig_close_cross_music_stress, close_harden_fire21, harden_music_toy_bites
- Inventory: 15 close-ish scripts still lack annotate (listed in fold note)
- Optional assert: M21 locked recipe **0/2** still_bites=True (lr≈[0.2096, 0.2109]); leftover CTRL pass; wall=183.62s
- Verdict: recipe_change=NO; merge=NO; ping_user=YES_mechanistic_M21

## Mid-flight — batch4 + box-only (2026-09-09 ~20:23 MDT)

- Laptop hop DOWN; stay box-cpu; sync deferred.
- M21 mech dig complete (see prior entry).
- Batch4 running: M28 locked+music **0/6 HARD_BITE** leak≈0.66; M29 locked failing multi 2/4.
- Leak-gate kinds patched for M28–M31 (else close-path false-pass).
- Interim: `batch4_interim_20260909.md`

## Fire — M14 fail-mode dig vs M20 (2026-09-09)

- Host: box-cpu @ `435e87363bf1`
- Dig: `m14_failmode_20260909.{py,json,md}` wall=2288.4s
- M14 nature: **content_deleted_under_declare_lie** (exam=0.1817, u=0.9878, swing=0.8462, gates={'cont': 6, 'swing': 0, 'leftover': 6, 'multi': 0})
- M20 nature: **content_deleted_under_declare_lie** (exam=-0.0826, leak=0.3841, swing=0.8724, gates={'cont': 6, 'swing': 0, 'leftover': 6, 'multi': 0})
- Clearance M14: n1=False n2=False vic0=False per_row=False → any_clear=False
- Clearance M20: n2=False vic0=False per_row=False
- Verdict: M14 nature=content_deleted_under_declare_lie; vs M20 nature=content_deleted_under_declare_lie; n/vic/per_row clear M14? NO (n1=False, n2=False, vic0=False, per_row=False); M20 also uncleared by n/vic/per_row (n2=False, vic0=False, per_row=False); keep both HARD_BITEs; recipe_change=NO
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- DoF family: M16/M27=scale; M17/M2=axis-mix; per_row clears all four; n/vic do not
- Keep HARD_BITEs under locked shared; analysis-only per_row w≤0.3 ADOPT (already)
- No Music GPU train.

## Fire — Music→toy batch4 (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Notes: `music_to_toy_batch4_20260909.{py,json,md}` wall=1428.2s
- New CELLS_3D: ['guard_refuse_hot_eoc', 'content_cascade_rows', 'eoc_threshold_edge', 'leftover_hot_eoc_declare']
- Guard contrast: M21 admit=True lr=0.205; M28 admit=False lr=0.65
- Bites: ['M28_guard_refuse_hot_eoc/locked: 0/6 HARD_BITE', 'M28_guard_refuse_hot_eoc/music: 0/6 HARD_BITE', 'M29_content_cascade_rows/locked: 0/6 HARD_BITE', 'M29_content_cascade_rows/music: 0/6 HARD_BITE', 'M30_eoc_threshold_edge/locked: 0/6 HARD_BITE', 'M30_eoc_threshold_edge/music: 0/6 HARD_BITE', 'M31_leftover_hot_eoc_declare/locked: 0/6 HARD_BITE', 'M31_leftover_hot_eoc_declare/music: 0/6 HARD_BITE']
- Knives: []
- No-bites: []
- leftover_regression=3/3; recipe_change=NO; ping_user=YES
- No Music GPU train; servers untouched.

## Note — M14 = content_deleted_under_declare_lie (M20 family) (2026-09-09)

- Confirmed fold: M14 fail nature is **content_deleted_under_declare_lie** (exam≈content_kept≈0.18), same family as M20 amp_lie.
- Scoreboard + catalog already carry the label; session fold reaffirms.
- Batch4 complete: M28–M31 all HARD_BITE 0/6 both postures; leftover 3/3; recipe_change=NO.
- Next: per-row falsify batch4 — expect M29 clears (DoF), M28/M30/M31 stay biting (eoc/teacher).

## Fire #25 — per-row falsify M29 (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Dig: `per_row_falsify_m29_fire25_20260909.{py,json,md}` wall=1691.2s couple_w=0.0
- M29 shared=0/6 per_row=6/6 clears=True
- Keep-biting: M20_still=True M28_still=True M30_still=True; CTRL ok=True
- Verdict: YES — per-row clears M29 content_cascade without clearing M20/M28/M30; CTRL ok; merge=NO
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- No Music GPU train; servers untouched.

## Fire — per-row falsify batch4 (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Dig: `per_row_falsify_batch4_20260909.{py,json,md}` wall=1786.7s
- cleared=['M29']; uncleared=['M28', 'M30', 'M31']; overpowered=False; CTRL ok=True
- Verdict: YES — per-row clears M29 (DoF) without clearing M28/M30/M31 (eoc/teacher); CTRL ok; merge=NO
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- M14 note: content_deleted_under_declare_lie (M20 family)
- No Music GPU train; servers untouched.

## Fire #26 — M30 per-row knife (2026-09-09)

- Host: box-cpu @ SHA `435e87363bf1`
- Dig: `m30_per_row_knife_20260909.{py,json,md}` wall=2000.6s
- pass_seeds=[1]; fail_seeds=[0, 2, 3, 7, 42]
- couple: ['1/6', '0/6', '0/6']; still no 6/6
- Verdict: M30 per-row knife = seeds [1] (leak near floor); couple does not clear 6/6; eoc sweep confirms threshold; NOT a recipe fix — keep HARD_BITE
- recipe_change=NO; merge=NO; ping_user=YES
- Pop-os sync OK earlier; box dig continues. No Music GPU.
- Host note: laptop/pop-os SSH DOWN this fire (Tailscale 100.90.104.57 + LAN 192.168.1.90 timed out); box-cpu only
- Analytic: eoc=0.33 → lr≈0.2016 (just over 0.20 gate); seed1 empirical 0.1992 clears by noise
- eoc≤0.30 per_row clears 3/3; eoc≥0.34 never clears — threshold confirmed
- Tests (box): pytest test_lm_2d_adv + test_lm_highd_leftover + test_field3d + test_music_to_toy_stressors → **70 passed in 679.63s** (log `/workspace/2d-adv-pytest-fire26-20260910T033807Z.log`; slow from dig CPU contention)

## Fire #26b — M16/M17 mechanistic dig (2026-09-09)

- Host: box-cpu @ `435e87363bf1`
- Dig: `m16_m17_mech_20260909.{py,json,md}` wall=4331.1s
- Natures: M16=multi_row_scale_dof_partial; M17=multi_row_axis_roles_zero; M2=multi_row_plus_leak_cross_axis; M27=multi_row_scale_dof_partial
- Clearance M16: {'n1_music': False, 'n2': False, 'vic0_n1': False, 'per_row_w0': True, 'per_row_w0.3': False} → any=True
- Clearance M17: {'n1_music': False, 'n2': False, 'vic0_n1': False, 'per_row_w0': True, 'per_row_w0.3': True} → any=True
- Scale cliff span=0.6; blend lastPASS=0.35 firstFAIL=0.4
- M27_per_row=True; M2_per_row=True
- Verdict: DoF families confirmed — per_row clears M16/M17/M2/M27; n/vic do not; keep HARD_BITEs under locked shared
- DoF: M16/M27=scale; M17/M2=axis-mix; analysis-only per_row w≤0.3 ADOPT (already)
- recipe_change=NO; merge_to_trainer=NO; ping_user=YES
- No Music GPU train.

## Fire — declare-lie family unified theory + neg-control suite (2026-09-09)

- Host: box-cpu @ `435e87363bf1`
- Dig: `declare_lie_family_20260909.{py,json,md}` wall=1243.4s
- Natures: M14=content_deleted_under_declare_lie; M20=content_deleted_under_declare_lie; M26=content_deleted_under_declare_lie; M6=declare_lie_raw_leak_blowup
- Clearance any_declare=False: {'M14': {'n2': False, 'vic0': False, 'per_row': False}, 'M20': {'n2': False, 'vic0': False, 'per_row': False}, 'M26': {'n2': False, 'vic0': False, 'per_row': False}}
- neg_ok={'leftover': True, 'm14_eou0': True, 'm14_faithful': True, 'm20_faithful': True, 'm26_faithful': True, 'm16_shared_bites': True, 'm16_per_row_clears': True}; suite_pass=True
- Verdict: family M14/M20 natures=content_deleted_under_declare_lie/content_deleted_under_declare_lie same_content_delete=True; M26=content_deleted_under_declare_lie; M6=declare_lie_raw_leak_blowup; clearance n/vic/per_row any_declare=False ({'M14': {'n2': False, 'vic0': False, 'per_row': False}, 'M20': {'n2': False, 'vic0': False, 'per_row': False}, 'M26': {'n2': False, 'vic0': False, 'per_row': False}}); neg_ok={'leftover': True, 'm14_eou0': True, 'm14_faithful': True, 'm20_faithful': True, 'm26_faithful': True, 'm16_shared_bites': True, 'm16_per_row_clears': True}; suite_pass=True; keep HARD_BITEs; recipe_change=NO
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.

## Note — declare-lie suite fold (2026-09-09 evening)

- **suite_pass=True** (`declare_lie_family_20260909` wall=1243.4s).
- M14/M20/M26 all **content_deleted_under_declare_lie**; M6 = raw_leak_blowup sibling.
- eou cliff sharpened: **≥0.15 BITES** (was documented ≥0.3).
- Neg controls all green; M16 DoF contrast holds (shared bites / per_row clears).
- Taxonomy folded into theory note + scoreboard hard-boundary row.
- recipe_change=NO; no Music train; no production code touched (pytest N/A this dig).

## Fire — declare-lie / eoc-floor family unified theory (2026-09-09)

- Host: box-cpu @ `435e87363bf1`
- Dig: `declare_lie_eoc_family_theory_20260909.{py,json,md}` wall=827.8s
- Branches: A=content_deleted(M14/M20/M26); B=ê-floor-admit(M21/M30/M31); C=refuse-raw(M28)
- Cliffs: eou 0→≥0.3; eoc floor_on=0.33 refuse_on=0.68; M30 0.32Y/0.33N
- suite_pass=True; ctrl_ok=True
- Verdict: family map: A=content_deleted(M14/M20/M26) B=ê-floor-admit(M21/M30/M31) C=refuse-raw(M28); cliffs eou 0→≥0.3; eoc floor_on=0.33 refuse_on=0.68; suite_pass=True; n/vic/per_row clear? NO; recipe_change=NO; merge=NO
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.

## Note — cliff refinements from family theory dig (2026-09-09)

- M14 eou cliff refined: **eou=0 PASS / eou≥0.15 BITES** (was reported ≥0.3).
- M21-pool emp floor: shared **0/3 @ eoc=0.32** (analytic still pass lr=0.1996) — residual overshoot.
- Refuse onset refined: **eoc≥0.68** (teach_lr→0.65); prior ~0.7.
- Analytic valley eoc∈[0.55,0.65] admit+ana_lr<0.20 before refuse (teacher-only).

## Fire #27 — pop-os pytest + declare-lie/eoc family fold (2026-09-09 ~22:12 MDT)

- Host tests: **pop-os** via laptop LAN `192.168.1.90` @ SHA `435e873` (mikkel main #94)
- Pytest: `tests/test_lm_2d_adv.py` + `test_lm_highd_leftover.py` + `test_field3d.py` + `test_music_to_toy_stressors.py` → **70 passed in 134.68s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); nano-work-server left alone; log `/tmp/2d-adv-pytest-fire27-popos-20260910T041028Z.log`
- Research (box-cpu, completed this fire window):
  - `declare_lie_family_20260909` wall=1243.4s — suite_pass=True; M14/M20/M26=`content_deleted_under_declare_lie`; M6 raw leak blowup; n/vic/per_row clear=NO; eou cliff **≥0.15 BITES**
  - `declare_lie_eoc_family_theory_20260909` wall=827.8s — A/B/C map; ê-floor onset eoc≥0.33 (emp@0.32); refuse onset eoc≥0.68; analytic floor 0.20503
- recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (unified family map + cliff sharpenings)

## Fire — valley emp + sheet/highd A/B/C ports (2026-09-09)

- Host: box-cpu @ `435e87363bf1`
- Dig: `valley_emp_sheet_abc_ports_20260909.{py,json,md}` wall=565.9s
- Valley emp eoc∈[0.55, 0.58, 0.6, 0.62, 0.65]: `valley_emp_BITES_mixed_mech` (ana_pass=True; emp_all_fail=True)
- Sheet: A_on_u/C_hot BITES leak_tok; B_tilt often PASS; DoF CELLS_SHEET_DOF NO_BITE
- Highd: synonym/hot_content BITES leftover_leak (hold-λ A port)
- recipe_change=NO; merge=NO; No Music GPU train.

## Note — valley emp mechanism refine (2026-09-09)

- Relabel: `valley_emp_BITES_mixed_mech` (was overshoot-only).
- eoc=0.55 leak overshoot; eoc∈[0.58,0.65] content/exam collapse with leak under gate; refuse@0.68 raw leak≈0.66.
- Analytic valley remains teacher-only — emp never clears.

## Fire — DoF family unified cliffs (2026-09-09)

- Host: box-cpu @ `435e87363bf1`
- Dig: `dof_family_cliffs_20260909.{py,json,md}` wall=1335.4s
- Synthesized: m16_m17_mech + falsify M24/M27/M29 + roles_scale; filled M24 flip / M29 ascent / M27 descent cliffs
- Cliffs: M16_span≥0.6; M17_blend≥0.5; M24_flip firstFAIL=0.5; M29_ascent firstFAIL=0.6; M27_descent firstFAIL=0.4
- Sheet analogues: `CELLS_SHEET_DOF` (M16/M27 faithful; M24/M29 weak; M2/M17 Field3D-native)
- per_row w≤0.3 clears band: M24=True M29=True; CTRL ok=True; sheet_ok=True
- Verdict: DoF family unified: scale(M16/M27) axis-mix(M17/M2) flip(M24) cascade(M29); cliffs M16_span≥0.6 M17_blend≥0.5 M24_flip firstFAIL=0.5 M29_ascent firstFAIL=0.6 M27_descent firstFAIL=0.4; shared fails / per_row w≤0.3 clears all six; n/vic NO; sheet analogues M16/M27 faithful + M24/M29 weak; M2/M17 Field3D-native; ctrl_ok=True sheet_ok=True; recipe_change=NO merge=NO
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.

## Fire #28 — Field2D/exam A/B/C ports (2026-09-09)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN → **70 passed in 138.79s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); log `/tmp/2d-adv-pytest-fire28-popos-20260910T044216Z.log`; nano-work-server left alone
- Also box-cpu pytest earlier this fire: **70 passed in 492.77s** (`/workspace/2d-adv-pytest-fire28-box-20260910T043052Z.log`)
- Note: `/ml2/sliders/sliders-conceptmod` still stale @ `337d2b8` (#93) — use **music** checkout for adv suite
- Dig (box): `field2d_exam_abc_ports_20260909.{py,json,md}` wall=7.8s
- **Finding:** exam.pass **NO_BITE** for A/B/C; unused_e A_on_u/A_on_sheet **leak_tok=0.2277** (= sheet A); CTRL leak_tok=0
- divergent yaml ê also NO_BITE on exam.pass (no unused → leak_tok None)
- Encoding gap: exam gates omit leak — prefer sheet/Field3D for declare-lie family bites
- recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (encoding-gap finding)

## Fire #29 — exam leak-aware gate cliff (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `exam_leak_gate_cliff_20260910.{py,json,md}` wall=2.5s
- Cliff unused_e: max τ CTRL∧A_bites=0.22; CTRL∧A_all_fail=0.22; @EXAM_LEAK_LOCK={'ctrl_ok': True, 'A_bites': True, 'A_all_fail': True, 'B_bites': False, 'C_bites': False, 'ctrl_pass': '3/3', 'A_rows': {'A_on_u': '0/3', 'A_on_sheet': '0/3'}, 'B_rows': {'B_tilt_mild': '3/3', 'B_tilt_mid': '3/3', 'B_tilt_hot': '3/3'}, 'C_rows': {'C_hot_u': '3/3', 'C_hot_unused_flip': '3/3'}}
- Live @0.20: {'CTRL_leftover_e': {'base': '3/3', 'leak_aware': '3/3', 'runs': [{'seed': 0, 'base_pass': True, 'leak_tok': 0.0, 'leak_aware_pass': True}, {'seed': 1, 'base_pass': True, 'leak_tok': 0.0, 'leak_aware_pass': True}, {'seed': 2, 'base_pass': True, 'leak_tok': 0.0, 'leak_aware_pass': True}]}, 'A_on_u': {'base': '3/3', 'leak_aware': '0/3', 'runs': [{'seed': 0, 'base_pass': True, 'leak_tok': 0.2277, 'leak_aware_pass': False}, {'seed': 1, 'base_pass': True, 'leak_tok': 0.2277, 'leak_aware_pass': False}, {'seed': 2, 'base_pass': True, 'leak_tok': 0.2277, 'leak_aware_pass': False}]}}
- adopt_exam_harness=True (analysis-only; production exam_verdicts untouched)
- divergent leak gate no-op (has_unused=False)
- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN → **65 passed, 1 skipped in 109.51s** (`CUDA_VISIBLE_DEVICES=""`, `/home/mikkel/anaconda3/envs/minimax-music3`); log `/tmp/2d-adv-pytest-fire29-popos-20260910T065904Z.log`; nano-work-server left alone
- Skip note: `test_music_to_toy_stressors` had hardcoded `/workspace/...` REPO → module-skip on pop-os (1 skipped); patched to `Path(__file__).parents[3]` on box notes (analysis harness only). Prior 70-pass counts included those 3 when run on box.
- Also box-cpu: **68 passed in 95.18s** (`/workspace/2d-adv-pytest-fire29-box-20260910T070354Z.log`); music_to_toy recheck 3 passed after REPO patch
- ping_user=YES (exam leak gate ADOPT for unused_e branch A; B/C still sheet/Field3D; divergent still gap)
- recipe_change=NO; merge=NO; No Music GPU

## Fire #30 — sheet B_tilt leak_tok cliff (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `sheet_b_tilt_leak_cliff_20260910.{py,json,md}` wall=7.2s
- Cliffs: {"last_sheet_pass_alpha": 0.75, "first_sheet_fail_alpha": 0.8, "last_aware_pass_alpha": 0.75, "first_aware_fail_alpha": 0.8, "first_mean_leak_gt_lock_alpha": 0.8}
- B-family leak-aware knife αs: []
- Verdict: b_knife_nonempty=False; adopt_sheet_leak_aware=False; unifies with Fire #29 exam A analysis gate if knife nonempty; recipe_change=NO; merge=NO
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.

## Fire #30 — host tests + sheet B→C cliff fold (2026-09-10 ~02:40 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 128.42s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); log `/tmp/2d-adv-pytest-fire30-popos-20260910T0836Z.log`; nano-work-server left alone
- Dig (box-cpu): `sheet_b_tilt_leak_cliff_20260910` wall=7.2s
- Cliffs: last_sheet_pass α=**0.75** (leak≈0.1974); first_sheet_fail α=**0.80** (leak≈0.2096); coincides with `EXAM_LEAK_LOCK=0.20`
- B-family leak-aware knife: **empty** (sheet.pass and aware@0.20 fail together) → **adopt_sheet_leak_aware=False**
- Nuance: valley `B_tilt_hot` PASS used non-simplex `unit(0.85ŝ+0.30ê)`; simplex α=0.85 is already refuse/fail
- Fire #29 exam unused_e A gate remains the place leak_aware adds bite beyond base pass; sheet native pass already tracks the lock on this axis
- recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (cliff + negative sheet leak-aware ADOPT)


## Fire #31 — highd content↔leftover leftover_leak cliff (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `highd_content_leftover_cliff_20260910.{py,json,md}` wall=10.9s
- Cliffs: {"last_pass_beta": 0.2, "first_fail_beta": 0.35, "last_aware_pass_beta": 0.2, "first_aware_fail_beta": 0.35, "first_mean_leak_gt_lock_beta": 0.35, "knife_betas_pass_green_aware_fail": []}
- Anchors: CTRL_leftover_only=0/3 leak=0.504, A_synonym_content=0/3 leak=4.7744, A_hot_content=0/3 leak=5.215
- Verdict: knife_nonempty=False; first_fail_β=0.35; first_aware_fail_β=0.35; first_leak_cross_β=0.35; adopt_highd_extra_leak_aware=False; recipe_change=NO; merge=NO
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.

## Fire #31 — host tests + highd β cliff fold (2026-09-10 ~04:34 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 120.30s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); log `/tmp/2d-adv-pytest-fire31-popos-20260910T1034Z.log`; nano-work-server left alone
- Dig (box-cpu): `highd_content_leftover_cliff_20260910` wall≈12.3s
- Cliffs: last_pass β=**0.20** (leak≈0.1459); first_fail β=**0.35** (leak≈0.2176); coincides with `LEAK_LOCK=0.20`; knife empty
- Nuance: β=0.45 leak-green but slider-axis fail (content-pin); `CTRL_leftover_only` 0/3 matches valley (not regression)
- adopt_highd_extra_leak_aware=False; recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (cliff + post-cliff slider-only fail)

## Fire #32 — M22 soft→M2 multipair lerp cliff (2026-09-10 ~06:51 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 128.19s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); log `/tmp/2d-adv-pytest-fire32-popos-20260910T1245Z.log`; nano-work-server left alone
- Dig (box-cpu): `m22_soft_multipair_cliff_20260910` wall=291.6s
- Lerp t of row_amps/scales M22→M2 under locked 1200+c1.5+faithful_guard_e
- Cliffs: last_locked_pass **t=0.25**; first_locked_fail **t=0.40** (0/3, lr≈0.001 exam/DoF); leak rises later t≥0.60
- Music n1+vic0: t=0 **1/3** (seeds 0,1 — known M22 knife); t=0.25 **3/3**; same hard cliff at 0.40
- CTRL leftover/close 3/3; M22 stock 3/3; M2 stock 0/3
- adopt=analysis_harness only (Music seed-knife at mild; locked soft band to t≤0.25); recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (soft multipair continuum cliff + early DoF-vs-late leak split)

## Fire #33 — M23 corr→M2 multipair cliff (2026-09-10 ~08:51 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **65 passed, 1 skipped** in 109.32s then music_to_toy **3 passed in 9.33s** after harness REPO fix (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); skip was hardcoded `/workspace` in `test_music_to_toy_stressors` + `field3d_music_stressor_hooks` (Fire #29 gap drifted back on music checkout). Patched both to `Path(__file__).parents[3]` on pop-os + box notes. Effective **68** for this fire's suite. nano-work-server left alone; log `/tmp/2d-adv-pytest-fire33-popos-20260910T1443Z.log`
- Dig (box-cpu): `m23_corr_multipair_cliff_20260910` wall=330.2s @ `435e87363bf1`
- Lerp M23 `multipair_corr_seed` (seed0) → M2 `cross_axis_rows` under locked 1200+c1.5+faithful_guard_e
- Cliffs: last_locked_pass **t=0.15**; first_locked_fail **t=0.25** (0/3, lr≈0.018 exam/DoF); leak rises later t≥0.50
- vs Fire #32 M22→M2: **earlier_than_m22** (M22 last_pass 0.25 / first_fail 0.40)
- Music n1+vic0: last_pass **t=0.25**, first_fail **t=0.40** — Music **more tolerant** than locked here (opposite of M22 seed-knife)
- CTRL leftover/close 3/3; M23 stock 3/3; M2 stock 0/3; first_bite_mode=exam_dof
- adopt=analysis_harness only; recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (corr multipair bites earlier than stagger; Music posture right-shifts cliff)


## Fire #34 — M22→M17 roles intermediate cliff (2026-09-10 ~10:48 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop Tailscale → **70 passed in 134.59s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); suite=`test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`; log `/tmp/2d-adv-pytest-fire34-popos-20260910T1653Z.log`; nano-work-server left alone. First collect attempt ERROR'd on duplicate `test_music_to_toy_stressors` basename (tests/ + notes/) — reran tests/ only.
- Dig (box-cpu): `m22_to_m17_roles_cliff_20260910` wall=344.3s @ `435e87363bf1`
- Lerp M22 `stagger_mild_cross` → M17 `roles_split_proxy` under locked 1200+c1.5+faithful_guard_e
- Cliffs: last_locked_pass **t=0.15**; first_locked_fail **t=0.25** (0/3, lr≈0.0005 exam/DoF); leak stays tiny to t=1 (lr≈0.004)
- vs Fire #32 M22→M2: **earlier_than_m22_to_m2** (M22→M2 was 0.25→0.40)
- Music n1+vic0: **knifes earlier** — stock mild t=0 only **1/3** (fail seeds 0,1 — known M22 knife); never all-pass on sampled t
- CTRL leftover/close 3/3; M22 stock 3/3; M17 stock 0/3; first_bite_mode=exam_dof
- adopt=analysis_harness only (soft band to roles ≤t≈0.15; Music needs n≥2/vic0 gate even at mild); recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (roles path bites earlier than full cross_axis; Music seed-knife at mild)

## Fire #35 — M23 corr→M17 roles cliff (2026-09-10 ~13:02 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop Tailscale → **70 passed in 124.45s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); suite=`test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`; log `/tmp/2d-adv-pytest-fire35-popos-20260910T1902Z.log`; nano-work-server left alone. LAN `192.168.1.90` timed out from box+laptop.
- Dig (box-cpu): `m23_to_m17_roles_cliff_20260910` wall=368.0s @ `435e87363bf1`
- Lerp M23 `multipair_corr_seed` (seed0) → M17 `roles_split_proxy` under locked 1200+c1.5+faithful_guard_e
- Cliffs: last_locked_pass **t=0.35**; first_locked_fail **t=0.40** (0/3, lr≈0.019 exam/DoF)
- vs Fire #34 M22→M17: **later_than_m22_to_m17** (was 0.15→0.25)
- vs Fire #33 M23→M2: **later_than_m23_to_m2** (was 0.15→0.25) — roles path from corr is *more* tolerant than cross_axis path
- Music n1+vic0: same cliff 0.35→0.40 (no seed-knife; Music matches locked)
- CTRL leftover/close 3/3; M23 stock 3/3; M17 stock 0/3; first_bite_mode=exam_dof
- adopt=analysis_harness only (corr→roles soft band ≤t≈0.35); recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (triangle complete; corr→roles right-shifted vs mild→roles and vs corr→M2)

## Fire #36 — M22→M16 scale intermediate cliff (2026-09-10 ~14:47 MDT)

- Host tests: **box-cpu** `/workspace/sliders-conceptmod` @ `435e873` → **70 passed in 108.98s** (`CUDA_VISIBLE_DEVICES=""`, `/workspace/.venv-2d`); suite=`test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`; log `/tmp/2d-adv-pytest-fire36-box-20260910.log`. No registered user machines this run (laptop/pop-os offline). nano-work-server left alone.
- Dig (box-cpu): `m22_to_m16_scale_cliff_20260910` wall=400.3s @ `435e87363bf1`
- Lerp M22 `stagger_mild_cross` (4-row resampled→5) → M16 `scale_stagger_homo` under locked 1200+c1.5+faithful_guard_e
- Cliffs: last_locked_pass **t=0.0**; first_locked_bite **t=0.10** (2/3, seed1); first_all_fail **t=0.20** (0/3, lr≈0.0005 exam/DoF); leak stays tiny to t=1 (lr≲0.001)
- vs Fire #34 M22→M17: **earlier_than_m22_to_m17** (was 0.15→0.25) — scale/DoF path bites sooner than roles
- vs Fire #32 M22→M2: **earlier_than_m22_to_m2** (was 0.25→0.40)
- Music n1+vic0: **knifes earlier** — resampled mild t=0 only **2/3** (fail seed 0); never all-pass on sampled t; all-fail from t≥0.10
- CTRL leftover/close 3/3; M22 stock 3/3; M16 stock 0/3; first_bite_mode=exam_dof
- Note: t=0 lerp uses 5-row resample of M22 (not identical 4-row stock); soft band toward M16 is essentially only stock mild — any nonzero scale blend bites by t≈0.10
- adopt=analysis_harness only (scale soft band ≤t≈0.0 for all-pass; mixed 0.10–0.15); recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (scale path bites earlier than roles and cross_axis; Music knife at mild)

## Fire #37 — M23→M16 scale triangle cliff (2026-09-10 ~16:48 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 127.11s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); suite=`test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`; log `/tmp/2d-adv-pytest-fire37-popos-20260910.log`; nano-work-server left alone
- Dig (box-cpu): `m23_to_m16_scale_cliff_20260910` wall=469.0s @ `435e87363bf1`
- Lerp M23 `multipair_corr_seed` (seed0, 4-row resampled→5) → M16 `scale_stagger_homo` under locked 1200+c1.5+faithful_guard_e
- Cliffs: last_locked_pass **t=0.40**; first_locked_bite **t=0.50** (1/3, seeds 0,1); first_all_fail **t=0.65** (0/3, lr≈0.009 exam/DoF); leak stays modest (lr≲0.02)
- vs Fire #36 M22→M16: **later_than_m22_to_m16** (was 0.0 / bite 0.10 / all_fail 0.20) — corr soft start far more tolerant toward scale than mild stagger
- vs Fire #35 M23→M17: **later_than_m23_to_m17** (was 0.35→0.40) — from corr, scale path more tolerant than roles
- vs Fire #33 M23→M2: **later_than_m23_to_m2** (was 0.15→0.25) — scale more tolerant than full cross_axis
- Music n1+vic0: **knifes earlier** — last_pass **t=0.35**, first_fail **t=0.40**, all_fail **t=0.50** (Music left-shifts ~0.10 vs locked)
- CTRL leftover/close 3/3; M23 stock 3/3; M16 stock 0/3; first_bite_mode=exam_dof
- adopt=analysis_harness only (corr→scale soft band ≤t≈0.40; mixed 0.50); recipe_change=NO; merge=NO; No Music GPU; No CloudAgent
- ping_user=YES (scale triangle complete; corr→scale right-shifted vs mild→scale and vs corr→roles/M2; Music knife)

## Fire #38 — core leftover/close/unused_e + Music ADOPT reconfirm (2026-09-10 ~18:42 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 139.96s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`); suite=`test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`; log `/tmp/2d-adv-pytest-fire38-popos-20260910.log`; nano-work-server left alone
- Dig (box-cpu): `core_trio_music_adopt_fire38_20260910` wall=265.0s @ `435e87363bf1`
- Anti-slop core trio only (no new M-cells, no cliff lerps)
- leftover/close/unused_e n12: **6/6 / 6/6 / 6/6** (unused max_leak=0.0005 ≪ EXAM_LEAK_LOCK 0.20)
- close n1 seed0 knife **reconfirmed** 5/6 fail=[0]; n2 clears 6/6; vic0@n1 heals 6/6
- leftover flat under vic0 (Δu≈+0.0003); unused_e mean_u 0.9836→0.9887 (Δ=+0.0051 improvement, still 6/6) — not a regression
- Portable: Music ADOPTs (vic0@n1; n≥2 close) do not disturb leftover CTRL; unused_e stays leak-green
- recipe_change=NO; merge=NO; No Music GPU; No CloudAgent; ping_user=NO (green + no locked-story change)

## Fire #39 — portable student DoF (lowrank_k3 / row_cond ideal) (2026-09-10 ~20:41 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 137.95s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3` via `/home/mikkel/anaconda3`); suite=`test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`; nano-work-server left alone. Box-cpu corroboration **70 passed in 439.55s** (`/workspace/.venv-2d`; log `/workspace/2d-adv-pytest-fire39-box-20260911.log`).
- Dig (box-cpu): `portable_student_dof_20260910` wall=1876.3s @ `435e87363bf1`
- Question: single portable student (not full per-row heads) ≈ DoF wins under locked recipe **and still fail** declare-lie/ê-floor?
- Ideal split (DoF✓ ∧ declare✗ ∧ CTRL✓): **`lowrank_k3`** (params@M1=55) and **`row_cond`** (76). Preferred: **lowrank_k3** (k=3 matches R³ PCA floor; lighter).
- Phase1 M1: lowrank_k3 3/3, row_cond 3/3, per_row_w0.3 3/3; softmix_k3/shared/scaled/lowrank_k≤2 fail M1.
- Phase2: lowrank_k3 clears M16/M17/M2; M14/M21 stay 0/3 all forms; CTRL leftover+close 3/3 all forms.
- Nuance: oracle `per_row_w0.3` M16 knife 2/3 this run → dof_all=False; portable lowrank beat full heads on M16 stability here.
- adopt=`ADOPT_analysis_portable_student` (analysis harness only); recipe_change=**NO**; merge=**NO**; No Music GPU; No CloudAgent
- ping_user=**YES** (first portable form with ideal DoF/declare/CTRL split; Music propose-only rank-k=3 residual bank)

## Fire #39 deepen — lowrank_k3 stress (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `portable_student_lowrank_deepen_20260910.{py,json,md}` wall=636.1s
- Stress: M24/M27/M29 + leftover/close/unused_e + close n1 knife; init/k/couple
- DoF extra: M24=True M27=True M29=True (shared bite=True)
- CTRL n12: leftover=3/3 close=3/3 unused=3/3 leak_max=0.0024
- close n1=6/6 fail=[] seed0_knife=False; n2=6/6
- Init: zero_dead=True default=True large=True; k4 M1=True M24=True; couple0.3 M1=True M24=True
- still_ideal=True; any_fail=[]
- Verdict: **ADOPT_analysis_lowrank_k3_deepen_ok** — lowrank_k3 clears M24/M27/M29 DoF + CTRL leftover/close/unused_e; zero-init dead zone reconfirmed; no CTRL regression. merge=NO.
- recipe_change=NO; merge_to_trainer=NO; No Music GPU train.

## Fire — Dig B declare_ê YAML hygiene (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `declare_e_hygiene_rule_20260910.{py,json,md}` wall=266.4s
- Smallest rule (propose-only): **Reoc0_eoc0** — e_on_content=0 (declared_ê must not tilt onto content; û component is orthogonalized out of guard subtract, so content-tilt is the bite driver under faithful_guard_e).
- Stronger twin: R0 pure unused (e_on_u=0∧e_on_content=0) also clears both
- Cells cleared under locked shared + hygiene rewrite: ['M14', 'M21']
- Nuance: Ru(e_on_u=0 alone) clears NEITHER — prior eou cliff conflated eou with e_on_content; û is projected out of guard subtract
- Falsifiers: amp_lie_stock_bites=True; gate_cheat_rejected=True; amp_rewrite=nature_change_not_gate_cheat
- Verdict: **ADOPT_propose_Music_YAML_hygiene** — Smallest rule Reoc0 (e_on_content=0) clears ['M14', 'M21'] under locked shared rewrite; R0 (pure unused) is stronger narrative twin also clears both; Ru(e_on_u=0 alone) clears NEITHER (refines prior eou cliff — bite is content-tilt, û is projected out); Rsoft≤0.20 clears M21 only. Amp-lie stock HARD_BITE; gate-raise cheat rejected. Propose-only Music YAML/teacher — merge=NO.
- recipe_change=NO; merge_to_trainer=NO; No Music GPU; cells stay HARD_BITEs; lowrank_k3 ADOPT stays; merge=NO.


## Fire — living LEADER_CARD fold (2026-09-10 ~21:25 MDT)

- Host: box-cpu @ `435e87363bf1`. Docs-only; **No Music GPU**; no new cells.
- Wrote: `analysis/slider2d/notes/LEADER_CARD_20260910.md` — answers “solved or what’s leading” for future fires.
- Folded: locked defaults unchanged; analysis ADOPTs (`lowrank_k3` preferred portable + deepen; per-row w≤0.3 heavier cousin); propose-only Music (Reoc0 e_on_content=0 hygiene; n≥2 / vic0@n1 / multi-seed close); hard boundaries still open (DoF / declare-lie / ê-floor / refuse); scoreboard + dig pointers.
- Verdict: **not solved under locked shared**; leading stack documented; recipe_change=NO; merge=NO.
- Optional: delta-sync LEADER_CARD (+ research_log append) to pop-os via laptop hop if connected.


## Fire — soft multipair continuum card (#32–#37 fold) (2026-09-10 ~21:27 MDT)

- Host: box-cpu @ `435e87363bf1`. Docs-only synthesize; **No Music GPU**; no new CELLS; locked shared unchanged; merge=NO.
- Wrote: `analysis/slider2d/notes/SOFT_MULTIPAIR_CONTINUUM_20260910.md` — portable card folding Fires #32–#37.
- Paths: M22→M2, M23→M2, M22→M17, M23→M17, M22→M16, M23→M16.
- TLDR locked last_pass→first_bite: M22→M2 0.25→0.40; M23→M2 0.15→0.25; M22→M17 0.15→0.25; M23→M17 0.35→0.40; M22→M16 0.00→0.10 (all-fail 0.20); M23→M16 0.40→0.50 (all-fail 0.65).
- Soft bands (all-pass): M22→{M16≤0.0, M17≤0.15, M2≤0.25}; M23→{M2≤0.15, M17≤0.35, M16≤0.40}.
- DoF-first vs leak-later: all six first_bite=exam_dof; only →M2 paths leak later (≥0.50/0.60).
- Music n1+vic0: M22 starts seed-knife at mild; M23→M2 Music more tolerant; M23→M17 matches; M23→M16 Music left-shifts ~0.10.
- Pointed `LEADER_CARD_20260910.md` §5 + soft-band note at continuum card.
- No gap-recheck py (six cliffs already CTRL-anchored + stock-biting).
- recipe_change=NO; merge=NO; No Music GPU; No CloudAgent.
- Delta-sync to pop-os via laptop if hop UP.

## Fire — lowrank_k3 + Reoc0 combo (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `lowrank_reoc0_combo_20260910.{py,json,md}` wall=728.2s
- Question: combo clear DoF∪{M14,M21} + CTRL green + M20 stock bite?
- combo_ok=True; better_than_alone=True
- alone: lowrank DoF=True declare=False; Reoc0 DoF=False declare=True
- M20 stock bites=True; gate_cheat_rejected=True
- Verdict: **ADOPT_analysis_lowrank_reoc0_combo** — lowrank_k3 + Reoc0 together clear DoF∪{M14,M21} with CTRL green and M20 stock still HARD_BITE; each alone stays specialized (DoF vs declare); gate-raise cheat rejected. Analysis/propose-only — merge=NO.
- recipe_change=NO; merge_to_trainer=NO; No Music GPU; No new CELLS.

## Fire — M20 honest hygiene dig (2026-09-10)

- Host: box-cpu @ `435e87363bf1`
- Dig: `m20_honest_hygiene_20260910.{py,json,md}` wall=667.1s
- Question: smallest honest declare rewrite clearing M20 under shared/lowrank_k3 w/o gate-cheat?
- Smallest: **Reoc0_eoc0** — e_on_content=0 (declared_ê must not tilt onto content)
- Clears shared=True lowrank=True; soft020 fails; Ru fails; eoc cliff only exact 0
- Stronger twins: R0 + AmpAlign (declared onto leak_e) also clear, not smaller
- Collateral: M26+M28 nature-clear under Reoc0 (declare-lie / refuse=hot eoc); stock still HARD_BITEs; soft no false-pass; CTRL green
- Falsifiers: stock_bites=True; gate_cheat_rej=True; CTRL=True
- Verdict: **ADOPT_propose_M20_Reoc0_same_as_DigB** — Smallest honest M20 clear = Reoc0 (e_on_content=0) under shared AND lowrank_k3; same Dig B YAML hygiene — nature-change not gate-cheat. R0/AmpAlign also clear (declared ê onto leak_e) but not smaller. Soft eoc≤0.20 and Ru fail (eoc cliff: only exact 0 clears). Stock M20 stays HARD_BITE; CTRL leftover/close green. Collateral: Reoc0 also nature-clears M26 (declare-lie) and M28 (refuse driver=hot eoc) — not student/gate false-passes; stock cells remain violation stressors.
- recipe_change=NO; merge_to_trainer=NO; No Music GPU; No new CELLS; merge=NO.

## Fire — lowrank_k3 soft→hard continuum shift (2026-09-10 ~22:30 MDT)

- Host: box-cpu @ `435e87363bf1`
- Dig: `lowrank_soft_continuum_20260910.{py,json,md}` wall=1672.0s
- Question: does **lowrank_k3** (no Reoc0 needed for DoF) push soft→hard multipair cliffs later vs shared @ locked n=12?
- Paths: M22→M16 (earliest), M23→M2, M22→M2
- Shift table (last_pass / first_bite / all_fail):
  - M22→M16: shared 0.00→0.10→0.15 → lowrank **1.00 / — / —** (Δlp=+1.00)
  - M23→M2: shared 0.15→0.25→0.25 → lowrank **1.00 / — / —** (Δlp=+0.85)
  - M22→M2: shared 0.25→0.40→0.40 → lowrank **1.00 / — / —** (Δlp=+0.75)
- Soft bands expand? **YES** — all three paths **full-continuum soft** under lowrank_k3
- Anchors: CTRL leftover/close 3/3 both; soft M22/M23 3/3 both; hard M16/M2 shared 0/3, lowrank 3/3
- Updated: `SOFT_MULTIPAIR_CONTINUUM_20260910.md` + `LEADER_CARD_20260910.md`
- Verdict: **ADOPT_analysis_lowrank_soft_continuum** — lowrank_k3 dissolves DoF soft→hard cliffs (analysis only); Reoc0 still required for declare-lie; merge=NO
- recipe_change=NO; merge_to_trainer=NO; No Music GPU; No new CELLS


## Fire #40 — lowrank leak-max honesty (2026-09-11 ~00:30 MDT)

- Host tests: **pop-os** `/ml2/music/sliders-conceptmod` @ `435e873` via laptop LAN `192.168.1.90` → **70 passed in 117.24s** (`CUDA_VISIBLE_DEVICES=""`, conda `minimax-music3`; suite=`tests/test_lm_2d_adv`+`test_lm_highd_leftover`+`test_field3d`+`test_music_to_toy_stressors`); nano-work-server left alone (262 MiB).
- Dig (box-cpu reanalysis): `lowrank_leak_honesty_fire40_20260911.{py,json,md}` — re-score `lowrank_soft_continuum_20260910` with `leak_max_rows≤0.20`
- Root cause: `score_adv_field3d` / portable `leftover_ok` gate **row0-only** `pass_leak`; non-row0 leak ignored
- M2_stock lowrank: full **3/3** but leak_max≈**2.57** → honest **0/3** (false clear). M16_stock lowrank stays honest green. CTRL leftover/close honest green.
- Continuum under honest gate: M22→M16 still full soft (1.00); M23→M2 soft expands to last_pass **0.40** / bite **0.50 leak_max** (not full); M22→M2 last_pass **0.50** / bite **0.60 leak_max**
- Prior ADOPT_analysis_lowrank_soft_continuum “full continuum soft” **refined** → **ADOPT_analysis_lowrank_leakmax_honesty**; Fire #39 “clears M2” = DoF-only, leak_max✗
- recipe_change=NO; merge=NO; No Music GPU; No CloudAgent; No new CELLS
- ping_user=**YES** (false-clear on M2 / soft-continuum ADOPT refine)
