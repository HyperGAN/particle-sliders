# Transfer gaps post adv-CLI restore (2026-09-09)

After restoring `--adv_*` / `--fm_weight` on live `train_lm_slider_music3.py`
(`music_adv_cli_restore_20260909.md`), argparse no longer blocks #94 transfer.
Remaining mismatches are semantic / wiring, not missing flags.

Skim: `conceptmod/textsliders/lm_adv.py`, `lm_particles.py`,
`train_lm_slider_music3.py` vs `analysis/slider2d/adv.py` + `gan.py`.

## Status of previously blocking gap

| item | status |
|---|---|
| live music3 missing `--adv_weight` / `--fm_weight` / `--adv_reg_*` | **fixed** (defaults fm=0, adv=0 opt-in, coeff/kappa=1) |
| smoke `scripts/train_lm_gan_bcap.sh` FM_WEIGHT default **1** | still disagrees with locked FM=0 — override in `music_locked_recipe_smoke_20260909.sh` |
| `gan_v2.Recipe.fm_weight=1` | unchanged research arm; not the music3 #94 path |

## Remaining mismatches (post-CLI)

### 1. Teacher × critic architecture coupling (blocker for naive copy)

| locked toy | Music live |
|---|---|
| `teacher=faithful_guard_e` + MLP critic | `--lm_target faithful_guard_e` + `--adv_arch mlp` OK |
| (toy has no span-TX) | `--adv_arch tx` **SystemExit** unless recipe ∈ `{faithful_plus_neu_lyric}` |

**Gap:** Cannot run leftover-gated #94 teacher with the listening-preferred TX
smoke card in one argv. Transfer needs **two arms**:

1. **Lyric listen smoke (tx):** `--lm_target faithful_plus_neu_lyric --adv_arch tx`
   (matches `train_lm_gan_bcap.sh` + FM=0). Does **not** exercise leftover ê gate.
2. **Leftover gate smoke (mlp):** `--lm_target faithful_guard_e --adv_arch mlp`
   + YAML `leak_*` declared. Exercises the locked teacher; no span cloud.

### 2. `cover_weight` naming / analogue

| toy | Music |
|---|---|
| `AdvConfig.cover_weight=1.5` — MSE pin `neu±δ → pole` on shared residual | **no** `--cover_weight` |
| | `--pole_weight` (hidden MSE to teacher poles; smoke often **0** for all-GAN) |
| | `--lyrichold_weight` (UNI lyric-span hold; restore note suggests **1** for cover analogue) |

**Gap:** Locked cover=1.5 does not map 1:1. Primary smoke uses
`pole_weight=0` + `lyrichold_weight=1`. Leftover mlp arm should prefer
modest `--pole_weight` (not lyric hold). Do not invent a Music
`cover_weight` flag without a design pass.

### 3. ParticlePrior vs `--parts` (still the #1 dial trap)

| toy | Music |
|---|---|
| `n_particles≤12` `ParticlePrior` — latent jitter around residual | `--parts` / `ParticleBatch` — **row miner** affinity table |
| `particle_l2=0.02` L2 on particle coords | `parts_vic` KL balance on mined row distribution |

**Gap:** Keep `--parts 0` for #94. Raising parts does not restore toy
particle_l2. Music has **no** latent ParticlePrior around the LoRA residual.

### 4. b_cap API split + LR schedule

| toy | Music |
|---|---|
| single `b_cap=1.0` (coeff; margin hard-coded ≈1 in `cap_penalty`) | `--adv_reg_coeff` + `--adv_reg_kappa` (both default **1.0**; `CAP_COEFF`/`CAP_KAPPA`) |
| `delayed_cosine` LR (delay=80, min_ratio=0.05) default in `AdvConfig` | smoke / locked script: `--gan_lr_schedule constant` |
| AdvConfig `lr=5e-3` on residual | LoRA `--lr 5e-4`; D lr = 1.5× via `D_LR_MULT` / `--adv_lr` |

**Gap:** Coeff/kappa aligned when both set to 1. Schedule differs (constant vs
delayed cosine). Transfer **ratios**, not toy step count or toy residual LR.

### 5. Feature matching normalize

| toy | Music `lm_adv.feature_matching_loss` |
|---|---|
| `fm_normalize=True` (L2-normalize pooled features before MSE) | **raw** batch-mean MSE; **no** normalize flag |
| locked `fm_weight=0` | locked `--fm_weight 0` |

**Gap:** If anyone re-enables FM on Music, raw path is uncapped by b_cap
(arch review). Keep FM off for #94; do not port toy normalize=True as a
silent default until a Music normalize path exists.

### 6. Leftover gate name / prerequisites

| toy | Music |
|---|---|
| `faithful_guard_e` always available on leaky sheet field | `--lm_target faithful_guard_e` requires declared `slider_dir` + `leak_*` (else refuse / degrade) |
| gate in teacher construction | same name on CLI; plus many UNI/plus variants |

**Gap:** Name matches. Live YAML must declare leak pairs for leftover axes.
Clean gender UNI smoke does **not** need the gate (use lyric UNI + tx).

### 7. Critic width / real cloud

| toy | Music |
|---|---|
| Fourier-2 MLP `critic_hidden=64`; real cloud = span/end lerp on poles | mlp `adv_hidden=256` **or** `SpanTransformerD` over lyric tokens + `mean_last` |
| flat critic width ablations (Fire #2) | TX width/layers/heads separate knobs |

**Gap:** Do not chase toy critic_hidden=64 on Music. Prefer tx+mean_last for
lyric recipes; mlp for leftover-only.

### 8. Step budget

Toy locked **1200** updates ≠ Music listen smoke **120**. Transfer the
loss/geometry recipe; re-tune Music length with listen + multi-seed. Reject
800×cover3.0-style short×high-cover false locks.

## What is now transfer-ready

Smoke file (not executed GPU):
`analysis/slider2d/notes/music_locked_recipe_smoke_20260909.sh`

- `FM_WEIGHT=0` `ADV_WEIGHT=1` `--adv_reg_coeff 1` `--adv_reg_kappa 1`
- lyric smoke teacher + `--adv_arch tx` (bcap.sh shape)
- `parts=0`, `gan_beta1=0`, `lyrichold_weight=1`
- commented leftover `faithful_guard_e` + mlp arm

## Remaining blockers for live Music transfer of #94

1. **GPU schedule:** free device ≠ GPU0 (nano-work-server) / ComfyUI :18888 /
   `/ml2/music/run_server.py` — operator picks idle GPU.
2. **Dual-arm truth:** one run cannot be both `faithful_guard_e` and `tx`;
   decide listen (lyric+tx) vs leftover-gate (guard+mlp) first, or run both.
3. **Cover analogue unvalidated on Music** after CLI restore (lyrichold=1 vs
   pole_weight modest) — needs listen/metrics, not more argparse.
4. **No Music latent ParticlePrior** — if residual undershoots, raise
   pole/lyric pin or cut elsewhere; do not turn on `--parts`.
5. Optional: Music FM normalize helper still missing (irrelevant while FM=0).

## Non-goals

No Music LM GPU train this track. Servers left alone.
