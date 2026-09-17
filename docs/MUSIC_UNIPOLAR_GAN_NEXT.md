# Music / YuE2 unipolar GAN — what already exists and what to try next

Correction of the previous handoff (`MUSIC_UNIPOLAR_PARTICLEGAN_IMPLEMENT.md`):
that doc told the Music implementer to **invent a new unipolar propose_only path**
as if unipolar GAN training did not exist yet. It does. This doc assumes the
production unipolar YuE2 GAN train already exists on `main` and tells you
**what to run and what to tweak next**. Do not build a third unipolar trainer.

Tip context: `main` at `94fc990` (plus `3e5a3a0`, `6b0fd12`). The UniPG
schedule/particle findings (`c9_g4x`, `g_interp_cap` / prior-freeze) live on
**side-branch commits, not on `main`** — they are propose-only experiments to
run explicitly, never silent default flips.

---

## 0) Copy-paste prompt for the Music / YuE2 runner

```text
You run Music/YuE2 unipolar GAN sliders on mikkel/sliders-conceptmod,
main at 94fc990 (unipolar GAN already exists — do NOT invent a new
unipolar path). Stop ComfyUI / Music studio first if they hold the GPU.
Use CUDA_VISIBLE_DEVICES=N with --device cuda:0.

MISSION
Re-run the EXISTING production unipolar YuE2 GAN (recipe
unipolar-rpgan-bcap-yue2-v3, --recipe unipolar_gan) and its opt-in
+/0 sibling (rpgan-bcap-plus-neu-yue2-v1, --recipe gan_plus_neu),
then run ONE explicit propose-only ablation: the UniPG-C c9_g4x
schedule (4x LR, D still 1.5x G) against a matched production-LR
control. No supervised MSE on G. Keep bipolar Arm B /
locked_shared / live --lm_target intact.

ENTRYPOINTS (already on main, nothing to invent)
- Trainer: conceptmod/textsliders/train_lora_yue2_arm_b.py
  --recipe unipolar_gan   -> conceptmod/textsliders/yue2_arm_b.py
  --recipe gan_plus_neu   -> conceptmod/textsliders/yue2_gan_plus_neu.py
                           (shared update: conceptmod/textsliders/unipolar_gan.py)
- Campaign (train + held-out renders + dashboard):
  scripts/train_yue2_arm_b_campaign.py --recipe {unipolar_gan,gan_plus_neu}
- Prompts: conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml (4 train rows)
  Eval:    conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml (2 held-out rows,
           seeds 1709/2903, scales 0/0.5/1 + positive-caption reference)

RULES
- Do NOT flip trainer defaults (RECIPE dicts, ARM_B, --lm_target v9).
  Any c9_g4x run must be an explicit propose-only edit/argv that you
  revert or gate behind a flag, and you must print the full argv + LR
  table before spending GPU.
- G stays GAN-only (paired Rp logistic). Fail-closed if cover MSE /
  positive MSE / ending / FM / lyric-hold / plan / zero-anchor / VICReg
  appear on G. YuE2 has NO particle prior under G (prompt-state loss),
  so there is nothing to freeze unless you deliberately add one.
- Report: argv table, seeds, sample grid (0/0.5/1 + -1 canary),
  cover/leak/neu_hold (or Music equivalents), GPU hours, what
  transferred from toys and what didn't.
```

## One-liner for Slack

> YuE2 unipolar GAN already exists (`train_lora_yue2_arm_b.py --recipe
> unipolar_gan`, `unipolar-rpgan-bcap-yue2-v3`: Rp + b_cap k=1, G 5e-4 / D
> 7.5e-4, constant, mlp-256, raw-positive teacher, 600 steps). Audit:
> FAIL @ 600/1200, PASS @ 3400 on toys. Next: explicit propose-only
> **c9_g4x** ablation (G 2e-3 / D 3e-3, D still 1.5x G) vs matched
> control. No MSE shortcut; don't touch bipolar Arm B / live defaults.

---

## 1) The production unipolar YuE2 GAN that already ran (recipe identity)

Primary recipe — `conceptmod/textsliders/yue2_arm_b.py`:

| Knob | Value |
|---|---|
| RECIPE name | `unipolar-rpgan-bcap-yue2-v3` |
| Polarity / teacher | unipolar; raw positive caption via shared `lm_faithful_plus_neu` (no negative caption, no negative train branch) |
| G objective | GAN-only paired Rp logistic (`rp_g_loss`), averaged over 4 rows. `adv_weight=1`, `end/pole/cover/lyrichold/plan/anchor_weight=0`, FM off |
| Grad spine | `b_cap`, coeff 1, kappa 1, norm l2, exact autograd every D update, fixed teacher-RMS coords, `critic.net` (see `docs/yue2-arm-b-verification.md`) |
| Critic | MLP, 2 layers x 256 (`critic_hidden=256, critic_layers=2, adv_in='scaled'`), music-start hidden delta only |
| Optimizer | G AdamW lr **5e-4** (wd 1e-6), D Adam lr **7.5e-4** (**D = 1.5x G**), betas **(0, 0.999)** |
| Schedule | **constant** (no cosine, no EMA) |
| Clipping | generator grad value clip 1 |
| Scales | trained `[1]`; scale 0 = exact base by adapter construction (`zero_behavior='exact_base_by_adapter_scale'`), needs no learned anchor |
| Batch / adapter | `adv_batch=4` distinct rows, balanced shuffled passes; rank/alpha 8, AR q/k/v/o only, base frozen |
| Particles | **none** — prompt-state loss, no ParticlePrior, no VICReg (`--parts 0` equivalent; Music `ARM_B` posture). Nothing to freeze. |
| Sampling | none during training (no audio sampling, no ending margins, no history preroll) |
| Prompt policy | `balanced_shuffled_passes`, `history_policy='none_prompt_states_only'` |
| Default budget | `--steps 600` |

Opt-in +/0 sibling — `conceptmod/textsliders/yue2_gan_plus_neu.py`
(RECIPE `rpgan-bcap-plus-neu-yue2-v1`, shared update in
`conceptmod/textsliders/unipolar_gan.py`): trains scales **[0, 1]** with a
scale-conditioned critic (2x256 LeakyReLU MLP, positive-teacher-RMS
calibrated), G/D lr **5e-4 equal**, betas **(0, 0.99)**,
**delayed_cosine delay 80, floor 0.05**. GAN-only, no MSE/FM/particles/EMA/
clipping. Native LoRA LR 5e-4 is an explicit transfer setting vs the toy's
direct-residual LR 5e-3 (different units). Passes the CPU unipolar toy gates
at 400 updates on seeds 0/1/7 (`tests/test_unipolar_gan.py`); see
`docs/unipolar-gan-plus-neu.md`. Board status: propose-only
(`MERGE_TO_TRAINER=False`).

YAML / prompts:

| File | Role |
|---|---|
| `conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml` | 4 sound-only train rows (`neutral`/`positive`/`lyrics`), `recommended_range: [0, 1]` |
| `conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml` | 2 held-out rows, disjoint lyrics |
| `conceptmod/textsliders/data/config-yue2.yaml` / `config-yue2-female-uni16.yaml` | historical UNI16 configs (`recipe: uni16`, `lr: 5e-4`, 600 steps) — **not** the unipolar-GAN path; do not confuse them |

`train_lora_yue2_arm_b.py` argv (both unipolar recipes; historical filename,
checkpoint recipe is what matters):

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --recipe unipolar_gan \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --save_dir models/metal-yue2-unipolar-gan --steps 600
# sibling:
#   --recipe gan_plus_neu --save_dir models/metal-yue2-gan-plus-neu-600-<date>
```

Full campaign (train + renders + listening page, 2 held-out prompts x seeds
1709/2903 at 0/0.5/1 + positive-caption reference, 16 clips):

```bash
.venv-yue2/bin/python scripts/train_yue2_arm_b_campaign.py --recipe unipolar_gan \
  --save_dir models/metal-yue2-unipolar-gan --steps 600 --gpu 1 \
  --output_dir eval/listen/yue2-metal-unipolar-gan
```

Resume is exact (adapter + critic + both optimizers + row sampler + RNG);
source/prompt/model mismatch rejects resume. `--until N` caps a preflight.

## 2) The audit: production uni GAN fails toys @ 600/1200, passes @ 3400

Source of truth: `docs/yue2-gan-toy-audit.md` (+ `docs/yue2-gan-toy-audit.json`),
harness `analysis/slider2d/yue2_gan_exam.py` (calls the **actual**
`yue2_arm_b.build_game/update`; frozen backend swapped for `PairField`,
student = toy shared odd/even residual, zero exact like multiplier LoRA),
`tests/test_yue2_gan_exam.py`. Gates per required cell
(**divergent + close**), seeds **0/1/7**: cover >= 0.85, off-caption <= 0.05,
neu_hold >= 0.85. Scale -1 canary only; antipodal cos never a gate.

| Updates | Divergent cover | Close cover | neu_hold | Both cells pass? |
|---:|---:|---:|---:|---|
| 600 | 0.482–0.484 (leak 0.042–0.083, fails 2/3 seeds) | 0.585–0.587 | 1.000 | **No** |
| 1200 | 0.658–0.661 | 0.931–0.933 | 1.000 | **No** (divergent fails every seed) |
| 3400 | 0.931–0.932 | 0.933–0.935 | 1.000 | **Yes** (all seeds) |

Notes: supervised `faithful_plus_neu` control passes at 400 on every
cell/seed (it is MSE — **not** a license to add MSE to the GAN). Toy losses
stay finite (peak G < 3.4); the live metal run spiked (G 621.4 @ update 357,
cos 0.303 then -0.172). Finite toys + a passing 3400 endpoint do **not**
establish native stability. Longer music runs are not an established fix for
the spike.

Reproduce (exits 1 unless **all** requested budgets pass):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m analysis.slider2d.yue2_gan_exam \
  --steps 600 1200 3400 --seeds 0 1 7 --out /tmp/yue2-gan-exam.json
PYTHONPATH=. pytest -q tests/test_yue2_gan_exam.py tests/test_yue2_arm_b.py \
  tests/test_formulation_leaderboard.py tests/test_lm_plus_neu_exam.py
```

## 3) What to change next (from UniPG findings — tweaks, not invention)

### 3a) Primary: c9_g4x schedule ablation (Music already has no hungry prior)

UniPG-C (`15fe481`, `docs/unipg-c-scoreboard.md` — **side branch, not on
`main`**) swept 9 propose-only schedule/optimizer arms over the production
YuE2 unipolar loop. Verdicts (both cells, all seeds 0/1/7):

- **`c9_g4x` PASS @ 600/1200/3400**: 4x G LR keeping D 1.5x G
  (**G 2e-3 / D 3e-3**, else production: constant, beta2 0.999, no EMA,
  thick MLP critic). Follow-up rung after the clean `c6_g2x` win.
- `c6_g2x` PASS @ 1200/3400 (2x: G 1e-3 / D 1.5e-3).
- beta2-0.99 / EMA-on / cosine-delay / shared-LR (D 1x) all track production
  (FAIL @ 600/1200, PASS @ 3400).
- Full locked-toy clone with Fourier-2 critic **FAIL even @ 3400** — keep the
  **thick MLP** critic; do not switch critics.

Mechanics warning: `train_lora_yue2_arm_b.py` takes **no `--lr` flags** —
`g_lr`/`d_lr` are hardcoded in `yue2_arm_b.RECIPE`. So a c9_g4x music run
requires an explicit propose-only edit (revert afterwards, or gate behind a
flag/env override), e.g.:

```python
# propose-only c9_g4x patch to yue2_arm_b.RECIPE — DO NOT COMMIT as default:
RECIPE = dict(RECIPE, g_lr=2e-3, d_lr=3e-3)  # 4x base, D still 1.5x G
```

Run it against a matched production-LR control at the same step budget and
seed, same prompts/save layout, then the campaign eval. Print the full argv +
LR table before spend.

### 3b) Only if a learned ParticlePrior is under G: freeze/slow prior, g_interp_cap

UniPG-A/B/D/E agree: with a learned particle branch under G (or prior LR ~
residual LR), particles **steal the pole** (cover stuck ~0.5–0.7 while
`(|mu_prior|)` absorbs +1). Fixes that unlocked PASS @ 1200: freeze prior
(prior LR = 0) or slow it (~0.01x–0.1x residual), and/or prefer
`g_interp_cap` over sample-point `b_cap`; best early matrix recipe
`frozen_gi` = frozen jitter prior + `g_interp_cap`, VICReg/L2 off
(`bf0f030`, `docs/UNIPOLAR_GAN_SWEEP_UNIPG_A.md`, side branch).

**This does not apply to the default YuE2 path**: `yue2_arm_b` /
`yue2_gan_plus_neu` have no ParticlePrior, no VICReg, no `--parts` knob at
all (Music `--parts 0` equivalent by construction). Treat 3b as a warning:
do not reintroduce a hungry learned prior without the freeze, and only reach
for `g_interp_cap` if you deliberately turn particles back on.

## 4) Keep intact unless explicitly training uni

- Music bipolar `ARM_B` (`conceptmod/textsliders/train_lm_slider_music3.py`,
  `docs/music-arm-b-gates.md`): Rp + b_cap k=1, FM=0, mlp, `--parts 0`,
  `pole/cover_weight=1`, `vicreg_weight=0`, eval scales -1/0/0.5/1.
- Live trainer defaults: `--lm_target v9` / `--pole_mode hidden` /
  `--adv_preset none` — Arm B stays explicit, never silent.
- `locked_shared` / `AdvConfig()` fail-closed pin (#118).
- Unipolar eval scales: **0 / 0.5 / 1** (+ positive-caption reference);
  scale **-1 canary only**, antipodal cos never a gate.
- Bipolar run stopped at update 142 is preserved; unipolar resumes with a
  different recipe identity reject cross-resume.

## 5) Re-run / ablation one-liners + success criteria

Control (production LR) + ablation (c9_g4x), same budget/seed/prompts:

```bash
# A: matched control — production LR, 600 steps, seed 7
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --recipe unipolar_gan \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --save_dir models/metal-yue2-uni-control-600 --steps 600 --seed 7
# B: propose-only c9_g4x — same command + explicit RECIPE patch
#    (g_lr=2e-3, d_lr=3e-3, D still 1.5x G), different save_dir, same seed
```

Fallback ladder if the smoke fails cover early: `c6_g2x` (2x) → production
LR at longer budget (1200/3400 toy-calibrated) → and **only if parts > 0**:
prior freeze/slow + `g_interp_cap`. Never enable cover/positive MSE to clear
gates.

Success:

- Both runs documented: argv + LR table, seed(s), sample grid
  (0/0.5/1 + -1 canary), cover/leak/neu_hold or Music equivalents,
  GPU hours.
- Preferred: c9_g4x-shaped run clearly beats the matched production-LR
  control at the same step budget (the toy effect: PASS @ 600 vs FAIL).
- Bipolar Arm B path untouched (no default flip; `git diff` shows only the
  propose-only patch + this doc or the new run artifacts).
- Short note: what transferred from toys, what didn't (esp. native stability
  vs toy finiteness, and the step-357-class spike watch).

Read first: `docs/yue2-gan-toy-audit.md`, `docs/unipolar-gan-plus-neu.md`,
`docs/yue2-arm-b-verification.md`, `docs/yue2-slider.md`,
`docs/FORMULATION_LEADERBOARD_UNIPOLAR.md`,
`docs/music-arm-b-gates.md`; UniPG-C/A scoreboards on their side branches
(`15fe481`, `bf0f030`) until merged.
