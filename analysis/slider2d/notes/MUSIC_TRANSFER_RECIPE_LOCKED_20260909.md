# Music #94 transfer recipe — LOCKED (2026-09-09)

Host: pop-os `/ml2/music/sliders-conceptmod` @ SHA `435e873` (+ local adv CLI restore).
**CPU transfer research only. NO Music LM GPU train.** Leave nano-work-server,
ComfyUI :18888, `/ml2/music/run_server.py` alone.

Definitive portable card from today’s dual-arm / Arm B / gaps / CLI restore notes.
**Recommendation B** is the true #94 → Music target. Arm T is diagnostic only.

---

## 1. One-block copy-paste — Arm B argv

Pole **1.0**, FM **0**, parts **0**, `faithful_guard_e` + **mlp**, b_cap **c=κ=1**.

```bash
# VERIFY_ONLY by default — do NOT train until Mikkel says go.
# Idle free GPU only (never GPU0 / nano-work / ComfyUI :18888 / run_server.py).
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
cd "$ROOT" && export PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface

# Preferred wrapper (parse_args only unless RUN_TRAIN=1):
#   bash analysis/slider2d/notes/music_arm_b_locked_smoke_20260909.sh <idle_gpu≠0> locked94-armB

"$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
  --name locked94-armB-fm0-bcap1-guard-mlp \
  --prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml \
  --save_dir models/gan-bcap-repair/locked94-armB-fm0-bcap1-guard-mlp \
  --lm_target faithful_guard_e \
  --pole_mode hidden --rank 8 --alpha 8 --lr 5e-4 \
  --steps 120 --seed 7 --no-early_stop --endreg_weight 1 \
  --save_every 0 --device 0 \
  --adv_arch mlp --adv_in scaled \
  --adv_weight 1 --fm_weight 0 --fm_mode batch \
  --pole_weight 1 --lyrichold_weight 0 \
  --adv_reg_coeff 1 --adv_reg_kappa 1 --adv_batch 4 \
  --gan_beta1 0 --gan_lr_schedule constant --grad_account \
  --parts 0
```

| knob | value | why |
|---|---|---|
| `--lm_target` | `faithful_guard_e` | leftover ê gate (required) |
| `--adv_arch` | `mlp` | tx∩guard → SystemExit ~L2171 |
| `--fm_weight` | `0` | locked #94 FM off |
| `--adv_weight` | `1` | opt-in GAN |
| `--pole_weight` | `1.0` | cover analogue; CPU Arm B sweep |
| `--lyrichold_weight` | `0` | leftover arm (not listen) |
| `--adv_reg_coeff` / `--adv_reg_kappa` | `1` / `1` | b_cap |
| `--parts` | `0` | no Music ParticlePrior |
| `--gan_beta1` | `0` | locked |

YAML must declare `leak_*` (and slider axis) for `faithful_guard_e`. Re-tune
`--steps` with listen; do **not** copy toy 1200 by name.

---

## 2. Explicit non-transfer list

Do **not** carry these from 2D/Field3D into Music #94:

1. **`n_particles` / `particle_l2` ↔ `--parts`** — different objects (latent
   ParticlePrior vs row miner). Always `--parts 0`.
2. **`tx ∩ guard`** — `--adv_arch tx` + `faithful_guard_e` hard `SystemExit`
   (`PLUS_NEU_LYRIC_RECIPES` only). One argv cannot be both leftover and listen.
3. **`cover=1.5` from locked 2D-with-particles as Music start** — Arm B n=1
   sweep recommends **cover/pole = 1.0** (1.5 also full-pass; prefer lowest
   modest). Do not invent Music `--cover_weight`.
4. **FM on / raw FM** — locked off; Music FM lacks normalize; uncapped by b_cap.
5. **800 × cover3.0 / short×high-pin** — Fire #4 false lock; prefer longer
   budget over high pin.

Also reject: Arm T alone as #94; `pole_weight=0` as “done”; claiming
`lyrichold_weight ≡ cover_weight`; implementing arch↔recipe decouple without Ada.

---

## 3. Arm T — diagnostic only

```text
faithful_plus_neu_lyric + --adv_arch tx + --lyrichold_weight 1
+ FM0 + b_cap c=κ=1 + parts0
```

- Wrapper: `analysis/slider2d/notes/music_locked_recipe_smoke_20260909.sh`
- **Listen / span / lyrichold smoke only** — not the leftover #94 transfer.
- Ablation: cover_only (no guard) leaks (sheet leak≈0.23); do not merge LoRA
  with Arm B; knobs/diagnostics may carry, weights do not.

---

## 4. Preconditions

| gate | rule |
|---|---|
| GPU | Need **idle GPU ≠ GPU0** if servers hold GPU0 (nano-work-server). Never touch ComfyUI :18888 or `/ml2/music/run_server.py`. |
| Train | **VERIFY_ONLY** (`parse_args` / smoke default) **until Mikkel says go**. Set `RUN_TRAIN=1` only then, on a free device. |
| YAML | `faithful_guard_e` needs declared `leak_*` in prompts file. |
| CLI | Live music3 must expose `--adv_*` / `--fm_weight` (restored; see CLI note). |

---

## 5. Links — smoke scripts + sweep evidence

| artifact | role |
|---|---|
| `music_arm_b_locked_smoke_20260909.sh` | **Arm B** VERIFY_ONLY wrapper (pole=1 default) |
| `music_locked_recipe_smoke_20260909.sh` | **Arm T** listen-only smoke |
| `dual_arm_transfer_strategy_20260909.md` | Rec B primary; A companion; C propose-only |
| `dual_arm_leftover_vs_cover_20260909.{py,json,md}` | gate∧cover both required (3/3 locked vs 0/3 single-arm) |
| `arm_b_pole_cover_sweep_20260909.{py,json,md}` | cover∈{0.5..2.0} → **pole_weight=1.0** |
| `transfer_gaps_post_cli_20260909.md` | remaining semantic mismatches |
| `music_adv_cli_restore_20260909.md` | adv/FM flags restored on music3 |
| `strategy_recB_locked_20260909.md` | one-liner lock card |
| `arm_b_bcap_or_kappa_check_20260909.{py,json,md}` | falsify: b_cap∈{0.5,1,2} @ cover=1.0 n=1 |
| Cite | `train_lm_slider_music3.py` L206, L2170–2176, L2182–2186 |

### Toy → Music map (portable core)

| 2D / Field3D | Music Arm B |
|---|---|
| `teacher=faithful_guard_e` | `--lm_target faithful_guard_e` + leak_* YAML |
| MLP critic | `--adv_arch mlp` |
| `cover_weight=1.0` (n=1 Arm B) | `--pole_weight 1` |
| `fm_weight=0` | `--fm_weight 0` |
| `b_cap=1` | `--adv_reg_coeff 1 --adv_reg_kappa 1` |
| `n_particles=1` proxy | `--parts 0` |
| steps=1200 | re-tune with listen (smoke often 120) |

### One-liner

```text
Music #94 = faithful_guard_e + mlp + pole_weight=1 + FM0 + b_cap(c=κ=1) + parts0
            [VERIFY_ONLY; Arm T lyric+tx = listen diagnostic only]
```

---

## Falsify stamp (same day)

`arm_b_bcap_or_kappa_check_20260909` — b_cap ∈ {0.5, 1.0, 2.0} @ cover=1.0 /
leftover ON / FM0 / n_particles=1 / seeds{0,1,2}: **CONFIRMED** c=κ=1 still best
(all three comfortable; keep canonical 1). No recipe revise.
