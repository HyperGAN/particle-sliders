# Music LM transfer map — locked 2D / Field3D → `lm_adv` (2026-09-09)

Portable recipe (unchanged after Fire #1–#14 unless noted):

| knob (2D / Field3D) | locked value | do NOT adopt |
|---|---|---|
| steps | **1200** | 800×cover3.0 (false lock, Fire #4) |
| cover_weight | **1.0 Music-transfer** (toy default 1.5 still OK) | 0.5 knife; 2.0 unnecessary |
| teacher | **faithful_guard_e** | raw `faithful` (leak≈0.48) |
| FM | **off** (`fm_weight=0`) | raw FM (uncapped by b_cap) |
| n_particles | **≤12** (default 12) | 24 (steals residual) |
| particle_l2 | **0.02** | flat vs 0–0.1; keep default |
| b_cap | **1.0** | flat at short budget; keep 1 |
| span_frac / end_margin | **0.40 / 0.60** | end_only / span_only without multi-seed margin |

## Knob → Music LM surface

| 2D / Field3D knob | Music LM analogue | Where | What to try on smoke (hypotheses) |
|---|---|---|---|
| `faithful_guard_e` | `--lm_target faithful_guard_e` + declared `leak_*` / `slider_*` | `train_lm_slider_music3.py`, `slider_targets.lm_faithful_guard_e` | Keep as leftover teacher. Refuse path must fire when ê restates axis (energy-v4 / divergent). |
| leftover / content entanglement | yaml `leak_positive/negative` vs content in poles; exam cells divergent/close/unused_e | prompts YAML + pair exam | Port Field3D mismatch cells as Music pair-exam rows before trusting a single gender smoke. |
| `cover_weight` | mode-pin MSE onto leftover-gated teacher centers (2D HQ analogue); Music often uses pole MSE / HQ terms, not this exact name | 2D: `AdvConfig.cover_weight`; Music: pole supervision / gan cover analogues | If sheet residual undershoots on Music listen proxies, prefer longer steps + modest cover/pin before raising critic width. |
| `b_cap` | `CAP_COEFF=1.0`, `CAP_KAPPA=1.0` one-sided sample-point cap | `conceptmod/textsliders/lm_adv.py` | Keep c=k=1. Do not swap to R1/R2. |
| `n_particles` / `particle_l2` | `lm_particles` row miner (adaptive prompt rows) ≠ ParticlePrior jitter | `lm_particles.py`; 2D ParticlePrior is the jitter prior | Keep miner as coverage prior; do not let particles/miners eat the shared LoRA residual. Prefer fewer/stronger L2 if residual undershoots. |
| `fm_weight` / `fm_normalize` | gan_v2 `fm_weight`, bounded FM; historical smoke used FM=1 | `gan_v2/engine.py`, `docs/lm-gan-bcap.md` | 2D: keep FM off. Music: if re-enabling FM, prefer normalized/capped FM and multi-seed listen — raw FM uncapped by b_cap was rejected in 2D. |
| `span_frac` + `end_margin` | lyric-span gather + last-token / audio-start channels (`SpanTransformerD`, `_assert_lyric_span`, gan_v2 span critic) | `lm_adv.SpanTransformerD`, `gan_v2/data.py` | Default 0.40/0.60 maps to “mass near pole + some span”. Avoid pole-only or span-only without leftover+sheet multi-seed PASS. |
| multi-row `row_scales` | multi-prompt / multi-song batch diversity | prompts file rows | Field3D 5-row wide scales stress coverage; Music smoke should include ≥3 mismatched songs, not one prompt. |
| seed stability ≥6 | train seeds / eval seeds | launcher `--seed` / eval | Require 6/6 PASS with kept margin ≫ lock (Fire #4 lesson). |

## What transferred (toy → claim)

1. **Leftover gate dominates leak** — `faithful_guard_e` holds leak≈0; raw faithful copies ê.
2. **Steps dominate sheet lock** — 1200+c1.5 clears; 800+c3.0 is knife-edge false lock.
3. **Particles are a jitter prior** — raising n_particles lowers kept; ≤12 preferred.
4. **FM flat / raw uncapped** — keep FM off in 2D; Music FM is a separate listening trade.
5. **Field3D R³ leftover** — same recipe 6/6 on leftover smoke, exam-port amplitude mixes, PairField cells (Fire #10–#12).
6. **Harder mismatch / span sweeps** — see Fire #13 / #14 note files for latest PASS/FAIL.

## What failed / false locks killed

- **800×cover3.0**: 1/6 seed PASS, kept hugs 0.90 → rejected.
- Critic width 32→256: nullspace at 800 steps.
- b_cap 0.5/1/2 at 400 steps: flat (sensitivity needs cover parked first).
- Raw FM: worse pole_rel_err / sheet_dir even when PASS.

## Recommended next Music LM smoke (hypotheses only)

GPU0 busy (~30GB Music + Comfy); GPU1 free but **no smoke run this sprint** (CPU transfer depth preferred).

1. **Primary:** `--lm_target faithful_guard_e` with declared leak axis, `b_cap` c=k=1, adv on, FM off or capped-normalized only, steps ≥ smoke baseline (prefer longer over high cover).
2. **Exam rows first:** include divergent (ê restates axis), close (small û), unused_e (true leftover) — mirror Field3D PairField cells — before claiming WORKS.
3. **If residual undershoots:** lower miner aggression / particle count analogue + pin cover; do **not** jump to 800-short + high cover.
4. **Seed bar:** ≥6 seeds, require margin past gates (not 1e-6 over lock).
5. **Span:** keep SpanTransformer / lyric-span+last-token; do not ablate to last-token-only without leftover multi-seed check.

## Paths

- Recipe findings: `analysis/slider2d/gan_bcap_findings.md`, `docs/lm-2d-adv.md`
- Field3D scaffold: `analysis/slider2d/field3d.py`
- Tests: `tests/test_field3d.py`, `tests/test_lm_2d_adv.py`, `tests/test_lm_highd_leftover.py`
- Sprint notes: `analysis/slider2d/notes/*_20260909.md`
- Research log: `analysis/slider2d/research_log_20260909.md` + `analysis/slider2d/notes/research_log_20260909.md`


## Music adv CLI restore (same day)

See `music_adv_cli_restore_20260909.md`. Live `train_lm_slider_music3.py` again
exposes ParticleGAN knobs. Defaults aligned with locked #94:

| Music flag | default | 2D / Field3D map |
|---|---|---|
| `--fm_weight` | **0.0** | `fm_weight=0` |
| `--adv_weight` | 0.0 (set **1** to enable) | fit_adv / RpGAN on |
| `--adv_reg_coeff` / `--adv_reg_kappa` | **1.0 / 1.0** | `b_cap=1` |
| `--parts` | miner count | `n_particles` (≤12) |
| `--lm_target faithful_guard_e` | opt-in (default v9) | leftover teacher |
| `--pole_weight` / `--lyrichold_weight` | cover analogues | `cover_weight` |

Suggested Music smoke (hypothesis only):

```bash
# FM_WEIGHT=0 override; adv opt-in; b_cap c=κ=1; parts≤12
--lm_target faithful_guard_e --adv_weight 1 --fm_weight 0 \
  --adv_reg_coeff 1 --adv_reg_kappa 1 --parts 0 \
  --gan_beta1 0
```


## Dual-arm constraint (post CLI gaps) — bake into experiments

Source: `transfer_gaps_post_cli_20260909.md`, smoke `music_locked_recipe_smoke_20260909.sh`.

1. **TX × leftover teacher incompatible in one argv.**
   - Listen arm: `--lm_target faithful_plus_neu_lyric --adv_arch tx`
   - Leftover arm: `--lm_target faithful_guard_e --adv_arch mlp` + YAML `leak_*`
   Run **sequentially**, not combined.
2. **`--parts 0` always for #94.** Toy `n_particles` / ParticlePrior does **not** exist on Music. Do not map.
3. **Cover analogue** = `--lyrichold_weight` / `--pole_weight` — **unvalidated** on listen metrics.
4. Knobs that still transfer without ParticlePrior: `faithful_guard_e`, `b_cap` c=κ=1, `fm_weight=0`, steps/budget, leftover exam cells (divergent/close/unused_e).


## Strategy lock: Recommendation B

**True Music #94 target:** `faithful_guard_e + mlp + modest pole_weight + FM0 + b_cap1 + parts0`.

- Leftover gate **and** modest cover/pole together (single-arm cover_only leaks).
- Arm T (lyric+tx) = diagnostic only, not the #94 path.
- Toy stress under `n_particles=1` (no ParticlePrior port).
- See `strategy_recB_locked_20260909.md`, Fire #16 `recB_leftover_and_cover_n1_20260909.*`.

## Cover lock update (same day)

Music-transferable cover/pole_weight = **1.0** (not 1.5). Sweep: 1.0 → sheet+Field3D 3/3;
0.5 knife; 2.0 unnecessary. Arm B smoke: `music_arm_b_locked_smoke_20260909.sh`.
Toy AdvConfig may keep default 1.5; transfer posture uses 1.0 under n_particles=1.
