# Dist-match vs locked 2D RpGAN + b_cap (#94)

Host: pop-os CPU via Tailscale (`mikkel@100.90.104.57`). Checkout
`/ml2/music/sliders-conceptmod` at SHA **`435e873`**
(`Port ParticleGAN RpGAN + b_cap into the 2D slider example (#94)`).
`CUDA_VISIBLE_DEVICES=""`; nano-work-server left alone on GPU 0.
No Music-LM GPU trains.

## What the “distribution-matching” recipe actually is

On Music LM / gan_bcap, the earlier “worked kinda” distribution-matching
slider path is **batch feature matching on critic features**, not a
separate Wasserstein/OT trainer:

| piece | path / flag |
| --- | --- |
| Live helper | `conceptmod/textsliders/lm_adv.py` → `feature_matching_loss` = `MSE(mean(feat_fake), mean(feat_real))` |
| Trainer | `train_lm_slider_music3.py` → `matching_loss(...)` / `--fm_weight` / `fm_mode` ∈ {batch, paired} (`matched_weight_batch_mean` vs `paired_teacher_features`) |
| 2D harness analogue | `analysis/slider2d/adv.py` → `feature_match_loss` + `AdvConfig.fm_weight` / `fm_normalize` |
| Docs | `analysis/gan_bcap/paired_fm_findings.md`, `docs/hub-formulation-uni16-gan.md`, `analysis/slider2d/gan_bcap_findings.md` (FM off by default) |

Key loss terms (Music LM / 2D):

- **RpGAN** logistic pairing (`rp_g_loss` / `rp_d_loss`)
- **b_cap** one-sided `relu(||∇D||−1)²` on real+fake
- **FM** batch-mean (or paired) MSE on D features — *this* is the dist-match term
- **cover_weight** MSE mode-pin on the shared residual (2D HQ analogue)
- optional VICReg / particle L2 on the ParticlePrior

A *different* research fork (`analysis/gan_bcap/objective_20260905/`) tried
**energy distance / RBF MMD** as a *replacement* for RpGAN+b_cap. Unit tests
pass (`tests/test_distribution_objective.py`, `tests/test_conditional_energy.py`),
but the audit’s own verdict is **no replacement accepted**. That fork is
**not** the locked #94 recipe and is not what this head-to-head scores.

Locked #94 2D recipe (from `gan_bcap_findings.md` / Fire notes):
`teacher=faithful_guard_e`, `b_cap=1`, **`fm_weight=0`**, `cover_weight=1.5`,
`steps=1200`, `n_particles≤12`.

## Tests run

| command | result |
| --- | --- |
| `pytest tests/test_lm_2d_adv.py -q` | **10 passed** in 46.69s |
| `pytest tests/test_distribution_objective.py tests/test_conditional_energy.py -q` | **16 passed** in 1.42s |
| `analysis/slider2d/notes/distmatch_vs_rpgan_bcap_20260909.py` | head-to-head below |

## Head-to-head (seed=0, steps=1200, cover=1.5, b_cap=1, teacher=`faithful_guard_e`)

| arm | family | compiled | exam_score | leftover kept | leftover leak | leftover PASS | gender PASS | exam D/C/U | field2d |
| --- | --- | --- | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| `locked_rpgan_bcap_fm0` | #94 locked | **works** | 1.000 | 0.9295 | +0.0001 | PASS | PASS | P/P/P | PASS |
| `distmatch_fm_norm_1` | FM-on (norm) | **works** | 1.000 | 0.9294 | +0.0005 | PASS | PASS | P/P/P | PASS |
| `distmatch_fm_raw_0p5` | raw FM control | **works** | 1.000 | 0.9278 | +0.0005 | PASS | PASS | P/P/P | PASS |
| supervised `faithful_raw` | MSE poles | — | — | 0.9926 | **+0.2277** | **FAIL** | — | exam_div PASS | — |
| supervised `pair_odd` | MSE midpoint | — | — | **0.3672** | +0.2277 | **FAIL** | — | exam_close PASS | — |

Gates: leak ≤ 0.2, on-sheet kept ≥ 0.9, garble ≤ 0.05, swing ≥ 0.6.
`exam_score = min(overlap, swing)` on divergent+close only.

Raw numbers: `distmatch_vs_rpgan_bcap_20260909.json`. Prior FM-only sheet
sweep (Fire #6) already showed multi-seed flatness (`fm_normalized_vs_off_20260909.*`
on the shared box checkout).

## Verdict

**RELATED / SAME family — agreement on 2D gates; FM is a no-op knob here.**

1. Normalized batch FM (`fm_weight=1`) on the same RpGAN+b_cap+cover harness
   is **not in disagreement** with locked #94: same compiled `works`, same
   `exam_score=1.0`, leftover kept Δ ≈ −0.0001. Matches Fire #6
   `keep_fm_off_flat`.
2. Locked #94 deliberately sets **FM off** because (a) it is flat vs off on
   these fixtures, and (b) *raw* FM is uncapped by `b_cap` and was the Music
   failure mode (`architecture_review_20260905.md` / `failure_cause_findings`).
3. Supervised matching (`faithful_raw` / `pair_odd`) is a **different**
   objective family: passes exam pairs but **fails** the leftover sheet
   (leak ≈ 0.228 / kept ≈ 0.37). The GAN leftover gate is what #94 adds.
4. Energy-distance / MMD (`objective_20260905`) is a **rejected alternative**
   to RpGAN+b_cap — complementary research, not the locked recipe.

**Do not treat “distribution matching” as a rival 2D lock.** On these
fixtures it is the optional FM term inside the same ParticleGAN game;
keep `fm_weight=0`.
