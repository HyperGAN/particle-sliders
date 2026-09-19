# Research log 2026-09-09 — leftover gate vs cover_weight / b_cap

Standing 2D adversarial loop fire #1. Repo at `435e873` (PR #94 RpGAN+b_cap).
CPU only; no Music LM GPU trains. nano-work-server and existing GPU jobs left alone.

## Tests

```text
PYTHONPATH=. python -m pytest tests/test_lm_2d_adv.py -q
# 10 passed in 55.73s  (conda env minimax-music3)
```

## What I tried

1. Read `analysis/slider2d/gan_bcap_findings.md` (recipe: teacher=`faithful_guard_e`,
   b_cap=1.0, cover_weight=1.5, steps=1200 → compiled **works**, exam_score 0.968).
2. Tiny CPU ablation on **sheet leftover** only (`leaky_field`), steps=400:
   teacher ∈ {faithful_guard_e, faithful} × cover_weight ∈ {0, 1.0, 1.5} at b_cap=1;
   then b_cap ∈ {0.5, 1.0, 2.0} at guard + cover=1.5.
   Script: `analysis/slider2d/_ablate_leftover_cover_bcap.py`
   JSON: `analysis/slider2d/_ablation_leftover_cover_bcap_20260909.json`
3. Short full collect: `run_lm_adv.py --steps 400 --exam-steps 400 --baseline-steps 200`
   → out `/tmp/lm-2d-adv-tiny-20260909` (not overwriting docs/).

## Numbers (sheet leftover, steps=400)

| teacher | cover | b_cap | leak_tok | on_sheet_kept | pass |
|---|---:|---:|---:|---:|---|
| faithful_guard_e | 0.0 | 1.0 | -0.013 | 0.272 | FAIL |
| faithful_guard_e | 1.0 | 1.0 | ~0.000 | 0.631 | FAIL |
| faithful_guard_e | 1.5 | 1.0 | ~0.000 | 0.643 | FAIL |
| faithful | 0.0 | 1.0 | -0.315 | 0.248 | FAIL |
| faithful | 1.0 | 1.0 | **+0.483** | 0.706 | FAIL |
| faithful | 1.5 | 1.0 | **+0.476** | 0.721 | FAIL |
| faithful_guard_e | 1.5 | 0.5 | ~0.000 | 0.639 | FAIL |
| faithful_guard_e | 1.5 | 2.0 | ~0.000 | 0.644 | FAIL |

Gates: leak ≤ 0.2, on-sheet kept ≥ 0.9. Full 400-step collect:
`works-on-some-pairs`, exam=0.925, left=FAIL, div/close=PASS, field2d PASS
(slider cos +1.000, leak +0.001).

## Finding

- **Leftover gate dominates leak.** Raw `faithful` reals + any cover_weight copy unused ê
  (leak ≈ 0.48 ≫ 0.2). `faithful_guard_e` holds leak ≈ 0 at the same cover/b_cap.
- **cover_weight dominates on-sheet kept** under the gate: 0 → 0.27, 1.0 → 0.63,
  1.5 → 0.64 at 400 steps. Still short of the 0.90 lock (matches findings: need
  ~1200 + cover 1.5 to clear).
- **b_cap is flat at this budget** (0.5 / 1.0 / 2.0 → kept ≈ 0.64). Sensitivity to
  b_cap likely shows up only once cover has parked the residual near the poles.
- Next loop candidate: cover_weight ∈ {1.5, 2.0, 3.0} at steps 800–1200 with gate
  fixed, *or* a cover schedule; do not drop the leftover gate to chase kept.

---

## Fire #2 (09:15 MT) — cover × steps + critic capacity

Tests: `10 passed in 58.63s` (minimax-music3). Repo still `435e873` music checkout
(ahead of origin; not force-synced). CPU only; GPUs left alone (GPU0 ~30 GB / GPU1 ~13 GB
idle util).

Script/JSON: `_ablate_cover_steps_critic_20260909.py` /
`_ablation_cover_steps_critic_20260909.json`.

### cover × steps (teacher=`faithful_guard_e`, b_cap=1, hid=64)

| cover | steps | kept | leak | swing | pass |
|---:|---:|---:|---:|---:|:---:|
| 1.5 | 800 | 0.882 | ~0 | 0.991 | FAIL |
| 2.0 | 800 | 0.890 | ~0 | 1.008 | FAIL |
| 3.0 | 800 | 0.899 | ~0 | 1.026 | FAIL (≈ lock) |
| 1.5 | 1200 | 0.929 | ~0 | 1.087 | **PASS** |
| 2.0 | 1200 | 0.933 | ~0 | 1.095 | **PASS** |
| 3.0 | 1200 | 0.936 | ~0 | 1.100 | **PASS** |

### critic_hidden @ cover=1.5, steps=800

| hid | kept | pass |
|---:|---:|:---:|
| 32 | 0.879 | FAIL |
| 64 | 0.882 | FAIL |
| 128 | 0.883 | FAIL |
| 256 | 0.883 | FAIL |

### Finding

- **Steps dominate the sheet lock.** At 800, raising cover 1.5→3.0 only lifts kept
  0.882→0.899 (still shy of 0.90). At 1200, cover 1.5 already PASSes (0.929); 2.0/3.0
  add ~0.004–0.007 kept — diminishing returns.
- **Critic capacity is nullspace at this budget.** hid 32→256 does not move kept
  (~0.88). Do not chase wider D; park residual with steps (≥1200) + cover≈1.5.
- Mapping note for Music `lm_adv` (doc only): prefer longer train / cover pin over
  bigger Fourier critic when sheet residual undershoots.

Next loop candidate: seed stability at 1200+cover1.5 (seeds 0/1/2), *or* a cover
warmup schedule that front-loads cover then anneals (see if 800 can clear).

---

## Fire #4 (10:07 MT) — seed-check knife-edge 800×cover3.0

Tests: `10 passed in 91.35s` (minimax-music3). Repo still `435e873` music checkout
(ahead/behind origin; not force-synced). CPU only; GPUs busy (Music trains + Comfy +
nano-work-server left alone).

Script/JSON/MD: `analysis/slider2d/notes/seed_check_800_c3_20260909.{py,json,md}`.

### flat_800_c3.0 leftover sheet across seeds

| seed | kept | leak | swing | pass |
|---:|---:|---:|---:|:---:|
| 0 | 0.8990 | ~0 | 1.026 | FAIL |
| 1 | 0.8998 | ~0 | 1.026 | FAIL |
| 2 | 0.8991 | ~0 | 1.025 | FAIL |
| 3 | 0.8992 | ~0 | 1.025 | FAIL |
| 7 | 0.9000 | ~0 | 1.027 | **PASS** |
| 42 | 0.8998 | ~0 | 1.026 | FAIL |

Summary: **1/6 PASS**, kept mean `0.8995`, span `0.0010` (all seeds hug 0.90).

### Finding

- Fire #3 seed0 `flat_800_c3.0` kept **0.9008 PASS** was a **false / noisy lock**.
  Re-run seed0 → 0.8990 FAIL; only seed 7 clears the gate by ~1e-6.
- **Do not adopt 800×cover3.0.** Keep locked recipe **`1200+cover1.5`** (fire #3:
  6/6 PASS, kept mean 0.9303, span 0.0017).
- Mapping note for Music `lm_adv` (doc only): short-budget high-cover that barely
  clears a sheet lock on one seed is not a recipe — require multi-seed PASS with
  margin (prefer kept ≫ 0.90).

Next loop candidate: **particle-count vs residual cover** at locked 1200+c1.5
(`n_particles` ∈ {4,8,12,24} default 12), *or* FM-on-normalized smoke vs FM-off.

---

## Fire #5 (10:39 MT) — particle-count vs residual cover @1200+c1.5

Tests: `10 passed in 74.10s` (minimax-music3). Repo still `435e873` music checkout
(ahead/behind origin; not force-synced). CPU only; GPUs left alone (~30/13 GB
reserved, low util; nano-work-server untouched).

Script/JSON/MD: `analysis/slider2d/notes/particles_vs_cover_20260909.{py,json,md}`.

### n_particles sweep (teacher=`faithful_guard_e`, cover=1.5, steps=1200, seed=0)

| n_particles | kept | leak | covered | residual_norm | pass |
|---:|---:|---:|:---:|---:|:---:|
| 4 | 0.9354 | ~0 | yes | 1.654 | **PASS** |
| 8 | 0.9326 | ~0 | yes | 1.641 | **PASS** |
| 12 (default) | 0.9295 | ~0 | yes | 1.628 | **PASS** |
| 24 | 0.9259 | ~0 | yes | 1.612 | **PASS** |

Summary: **4/4 PASS**, kept mean `0.9309`, span `0.0095`.

### Finding

- **PASS/FAIL nullspace** at the locked recipe — every particle count clears
  leftover (kept ≥ 0.926 ≫ 0.90). Not a new lock cell.
- **Actionable residual-steal trend:** more particles → lower kept and lower
  residual_norm (even more than odd). Confirm the findings warning that the
  ParticlePrior must stay a jitter prior. Prefer **n_particles ≤ 12** (default
  fine; 4–8 slightly better kept); do not raise to 24.
- Mapping note for Music `lm_adv` (doc only): if sheet residual undershoots with
  cover pinned, try fewer particles / higher particle_l2 before raising cover.

Next loop candidate: **FM-on-normalized smoke vs FM-off** at locked 1200+c1.5
(n_particles=12), *or* particle_l2 ∈ {0.01, 0.02, 0.05} at n_particles=24 to see
if L2 restores residual magnitude without cutting count.

---

## Fire #8 (multi-pair / cross-axis) — see notes/research_log + notes/multipair_cross_axis_20260909.md

Verdict: `portable_recipe_solid`. Compiled WORKS 6/6. exam_score mean=0.9928753894151736. Flukes=none.

## Fire #10 — highd flake + Field3D leftover smoke (2D→3D Fire #9) (2026-09-09)

- Host: pop-os-cpu @ SHA `435e873`
- Tests: `pytest tests/test_lm_2d_adv.py -q` → **10 passed in 45.84s**; `tests/test_lm_highd_leftover.py` → **38 passed in 62.91s**
- Highd flake fix: `test_window_mean_is_the_stable_read_when_the_reply_is_not_a_mirror` — finals-span floor **0.05 → 0.03** (option a). Probe on bent field steps (200,400,800): windows span ≈0.009 (<0.02 with margin); finals span ≈0.040 (flake vs 0.05). Window-stability assert unchanged. Comment documents why.
- Scaffold: `analysis/slider2d/field3d.py` — R³ leftover (û, ĉ content/intended-off-û, ê leftover) + lyric → dim=4; poles `h± = neu ± a`; `score_adv_field3d` reuses `fit_adv` (minimal gan.py isinstance hooks).
- Locked smoke seeds {0,1,2,3,7,42}: **6/6 PASS** in 37.4s. u_kept mean≈0.9917 span≈0.0015; content_kept mean≈0.9948; leak_ratio max≈0.0009.
- Verdict: **field3d_leftover_transfers** — locked 1200+c1.5 / faithful_guard_e / b_cap=1 / particles≤12 transfers to R³ leftover.
- Next: harden Field3D gates / exam-port rollout on same geometry, or stress content vs leftover entanglement.
- Notes: `field3d_leftover_smoke_20260909.{py,json,md}`

## Fire #11 — Field3D content↔leftover entanglement / exam-port (2026-09-09)

- Host: pop-os-cpu @ SHA `435e873`
- Tests: `pytest tests/test_lm_2d_adv.py -q` → **10 passed in 46.63s**; `tests/test_lm_highd_leftover.py` → **38 passed in 62.96s`
- Families × seeds {0,1,2,3,7,42} @ locked 1200+c1.5 / faithful_guard_e:
  - `baseline` (c=0.55 e=0.45): **6/6** u_kept mean=0.9917 content_kept=0.9948 leak_max=0.0009
  - `close_like` (c=0.90 e=0.10): **6/6** u_kept mean=0.9929 content_kept=0.9937 leak_max=0.0009
  - `unused_e_like` (c=0.10 e=0.90): **6/6** u_kept mean=0.9919 content_kept=1.0016 leak_max=0.0003
  - `entangled` (c=0.70 e=0.70): **6/6** u_kept mean=0.9925 content_kept=0.9947 leak_max=0.0004
  - `content_zero` (c=0.00 e=0.45): **6/6** u_kept mean=0.9920 content_kept=n/a leak_max=0.0002
  - `leak_zero` (c=0.55 e=0.00): **6/6** u_kept mean=0.9917 content_kept=0.9948 leak_max=0.0009
- Verdict: **field3d_exam_entangle_solid** — Locked leftover gating stays seed-stable across exam-port amplitude mixes (baseline/close/unused_e/entangled + content_zero/leak_zero). No false locks.
- Next: port a true PairField-style exam cell into R3 or multi-row Field3D
- Notes: `field3d_exam_entangle_20260909.{py,json,md}`

## Sprint STATUS (15:03 MT) — pytest + transfer scaffold

- Host: pop-os-cpu @ SHA `435e873`; CUDA_VISIBLE_DEVICES=""; GPU Music/Comfy/nano-work left alone.
- Pytest: `tests/test_lm_2d_adv.py` + `tests/test_lm_highd_leftover.py` → **48 passed in 179s**.
- New: `tests/test_field3d.py` → **8 passed** (Field3D dim/basis/declared_e/cells/teacher/fit_adv short).
- Transfer writeup: `analysis/slider2d/notes/music_lm_transfer_map_20260909.md` + `field3d_README_20260909.md`.
- Portable recipe **unchanged**: 1200 / cover1.5 / faithful_guard_e / FM off / n_particles≤12 / particle_l2=0.02 / b_cap=1.
- In flight: Fire #13 hard multipair (`field3d_hard_multipair_20260909.py`); then Fire #14 span×entangle.
- Best transferable so far: leftover gate + steps≥1200 + cover≈1.5 + particles as jitter prior; Field3D PairField cells already solid (Fire #12).


## Fire #13 — HARD multi-pair / cross-axis (2026-09-09)

- Host: pop-os-cpu @ SHA `435e873`
- Tests: pytest 2d_adv+highd already green this sprint (48 passed)
- Cells: 13 × seeds [0, 1, 2, 3, 7, 42] @ locked 1200+c1.5 / faithful_guard_e
  - `sheet_leftover`: **6/6** primary mean=0.9302 span=0.0016 leak_max=0.0001 knife=False
  - `sheet_gender`: **6/6** primary mean=0.9955 span=0.0010 leak_max=0.0003 knife=False
  - `exam_divergent`: **6/6** primary mean=1.0000 span=0.0000 leak_max=0.0000 knife=False
  - `exam_close`: **6/6** primary mean=1.0000 span=0.0000 leak_max=0.0000 knife=False
  - `exam_unused_e`: **6/6** primary mean=0.9870 span=0.0312 leak_max=0.0006 knife=False
  - `f3d_divergent`: **6/6** primary mean=0.9216 span=0.0066 leak_max=0.0011 knife=False
  - `f3d_close`: **6/6** primary mean=0.9828 span=0.0094 leak_max=0.0097 knife=False
  - `f3d_unused_e`: **6/6** primary mean=0.9203 span=0.0008 leak_max=0.0017 knife=False
  - `f3d_mismatch_content_heavy`: **0/6** primary mean=0.9989 span=0.0004 leak_max=0.0682 knife=False
  - `f3d_mismatch_leak_heavy`: **0/6** primary mean=0.9998 span=0.0004 leak_max=1.1867 knife=False
  - `f3d_mismatch_cross_declare`: **0/6** primary mean=-0.0436 span=0.0045 leak_max=0.6965 knife=False
  - `f3d_entangled_wide_rows`: **0/6** primary mean=0.8396 span=0.0025 leak_max=0.0048 knife=False
  - `f3d_leftover`: **6/6** primary mean=0.9917 span=0.0015 leak_max=0.0009 knife=False
- Verdict: **hard_multipair_partial** — Locked recipe fails on: f3d_mismatch_content_heavy, f3d_mismatch_leak_heavy, f3d_mismatch_cross_declare, f3d_entangled_wide_rows. Document as transfer risk; do not weaken recipe without multi-seed margin.
- Notes: `field3d_hard_multipair_20260909.{py,json,md}`

### STATUS (~Fire #13)
- Best transferable: locked recipe survives harder mismatch suite iff all cells 6/6 and no knife_edge; else document the failing cell.
- Current verdict: `hard_multipair_partial`

## Fire #13 DONE — HARD multi-pair results (2026-09-09 15:19 MT)

- Verdict: **hard_multipair_partial** wall=776s
- PASS 6/6: sheet_leftover, sheet_gender, exam_divergent/close/unused_e, f3d_divergent/close/unused_e, f3d_leftover
- FAIL 0/6 (solid, not knife): f3d_mismatch_content_heavy, f3d_mismatch_leak_heavy, f3d_mismatch_cross_declare, f3d_entangled_wide_rows
- Transfer: PairField-port cells OK; extreme declare/amplitude mismatch NOT rescued by recipe
- Dual-arm bake-in: these PASS cells are leftover-gate arm candidates; do not map particles→`--parts`
- Notes: `field3d_hard_multipair_20260909.{py,json,md}`

### STATUS (~Fire #13 complete)
- Best transferable: locked recipe on sheet/exam/Field3D PairField + leftover; refuse pathological leak YAML
- Next: Fire #15 dual_arm_transfer (n=1 vs n=12, cover sweep, leftover vs listen arms)

## TRANSFER parallel — dual-arm strategy (2026-09-09 ~15:26 MT)

- Host: pop-os laptop-SSH @ SHA `435e873`. **No Music LM GPU train.** Servers left alone
  (nano-work-server / ComfyUI :18888 / run_server.py still up).
- Hard reject cite: `train_lm_slider_music3.py` L2170–2176
  `raise SystemExit("--adv_arch tx needs the lyric span...")` when
  `adv_arch=="tx"` and recipe not in `PLUS_NEU_LYRIC_RECIPES={faithful_plus_neu_lyric}`
  (L206). Twin gate for `--txfm_weight` at L2182–2186.
- CPU ablation Music-transferable subset (`n_particles=1` ≈ `--parts 0`; toy n=0
  breaks ParticlePrior.sample): FM=0 b_cap=1 steps=1200 seeds{0,1,2} sheet+Field3D
  - leftover_only (guard,cover0): sheet 0/3 kept≈0.48 leak≤0.06
  - cover_only (faithful,cover1.5): sheet 0/3 kept≈0.99 **leak≈0.23** (f3d leak≈0.46)
  - **locked** (guard+cover1.5): **3/3** sheet kept≈0.935 / f3d u_kept≈0.997
  - neither: fail both
- **Recommendation: B** (true #94 = guard+mlp+modest pole_weight+FM0+b_cap1+parts0)
  with **A** as operational dual smoke (Arm T lyric+tx listen-only; knobs/diagnostics
  carry, not LoRA). **C** propose-only (decouple arch↔recipe); Ada merges.
- Reject: tx∩guard argv; parts↔particles; invent cover_weight; lyrichold≡cover;
  Arm T alone as #94; pole_weight=0 as done; 800×c3; FM-on; GPU train; implement C.
- Notes: `dual_arm_transfer_strategy_20260909.md`,
  `dual_arm_leftover_vs_cover_20260909.{py,json,md}`

### STATUS (~Fire #15 mid — Arm A n=1 complete)

- **A_n1_minimal_jitter** (Music-honest, no ParticlePrior): **6/6 all Arm A cells**
  - sheet_leftover kept mean=0.9351 (better than n=12 locked 0.9302)
  - f3d_leftover / unused_e / divergent + exam unused/divergent all PASS
- Transfer claim strengthening: leftover-gate arm survives without toy ParticlePrior → keep Music `--parts 0`
- Still running: A_n12 control, A cover0 / b_cap edges, Arm B cover sweep
- Dual-arm constraints baked; nano-work/Comfy/run_server untouched

### STATUS (strategy → Recommendation B)

- Locked Music #94 target: **faithful_guard_e + mlp + modest pole_weight + FM0 + b_cap1 + parts0**
- Arm T lyric+tx = diagnostic only; cover_only alone leaks (parallel 0/3 vs 3/3 locked)
- Fire #15 still running (Arm A n=1 already **6/6 all cells** — ParticlePrior not required)
- Next queued: Fire #16 `recB_leftover_and_cover_n1_20260909.py` — cover sweep ∧ gate @ n=1 + cover_only negative
- Recipe portable core **unchanged**; n_particles=1 is Music-transfer posture, not a recipe change for toy defaults when n=12 still fine

### STATUS (Fire #15 — cover0 vs cover1.5 @ n=1 = Rec B)

- `A_n1_minimal_jitter` (gate+cover1.5, n=1): **6/6 all cells** (sheet kept 0.935)
- `A_n12_locked_toy`: **6/6** (sheet 0.930) — Δ vs n=1 negligible; ParticlePrior not required
- `A_n1_cover0` (gate, cover=0, n=1): **0/6 FAIL** sheet mean kept=0.424 knife — **cover required with gate**
- Matches Recommendation B: leftover∧cover together; cover_only/gate-only insufficient
- Still running b_cap edges + Arm B; Fire #16 queued for finer cover sweep + cover_only neg

### STATUS (Music cover lock → **1.0**)

- Parallel sweep: cover 1.0 → sheet+Field3D 3/3; 0.5 knife; 2.0 unnecessary
- Arm B smoke: `music_arm_b_locked_smoke_20260909.sh` (`POLE_WEIGHT=1`)
- Toy default 1.5 still OK historically; **Music-transfer posture = cover/pole 1.0**
- Next: Fire #17 hard multipair @ n=1 cover=1.0; Fire #16 cover sweep prioritizes 1.0
- Fire #15 still running (cover0 fail already confirms gate needs cover)

### STATUS (Fire #17 mid — cover=1.0 n=1)

- sheet/exam cells so far: **6/6** (sheet_leftover kept mean=0.9334)
- f3d_divergent: **6/6** mean=0.9333
- **f3d_close seed0 FAIL** prim=0.1995 leak=0.2645 (258s) — seed1 PASS 0.990 — possible knife at cover=1.0 on close cell
- If f3d_close ends knife_edge: Music cover=1.0 may need margin check on close/delivery pairs; do not silently claim 3/3 without this cell
- CPU contention: parallel arm_b_bcap_check + pytest also running

## Fire #17 — hard multipair n=1 cover=1.0 (2026-09-09)

- Music-transfer: cover/pole **1.0**, n_particles=1, FM0, b_cap1, guard_e
- Verdict: **n1_cover1_partial** — Fails under cover=1.0/n=1: ['f3d_close'] — may need cover=1.5 on those cells.
- Notes: `hard_multipair_n1_cover1_20260909.{py,json,md}`

### STATUS (~Fire #17)
- Locked Music cover=1.0 under n=1 multipair: `n1_cover1_partial`

## Fire #18 — f3d_close cover knife n=1 (2026-09-09)

- Verdict: **close_fragile_both** — Both covers fragile on f3d_close @ n=1 — investigate.
- Music: Do not smoke Music close pairs until close cell solid.
- Notes: `f3d_close_cover_knife_20260909.*`

## Fire #18 DONE + 2h sprint WRAP (2026-09-09 ~16:28 MT)

- Fire #18 `f3d_close` cover knife @ n=1: cover1.0 **5/6**, cover1.5 **5/6** — both fail **seed0 only**; verdict `close_fragile_both`
- Fire #17: `n1_cover1_partial` — sheet/exam/f3d leftover+divergent+unused_e 6/6; f3d_close knife; mismatch 0/6
- **Music recipe LOCKED:** `MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md` — guard_e+mlp+pole=1+FM0+b_cap1+parts0
- Wrap: `analysis/slider2d/notes/sprint_2h_wrap_20260909.md`
- Portable toy recipe unchanged (1200/c1.5/…); Music-transfer posture pole=1 / parts=0 / n=1 proxy
- Stop clean. No Music GPU train. Servers left alone.

### STATUS (sprint end)
- Best transferable: locked Arm B card above; Field3D PairField leftover cells + sheet @ n=1 cover=1.0
- Caveat: f3d_close seed0 flake — multi-seed close on Music; don’t raise pole on seed0 alone
- Next: VERIFY_ONLY Arm B smoke when ready; train only on Mikkel go + idle GPU≠0
