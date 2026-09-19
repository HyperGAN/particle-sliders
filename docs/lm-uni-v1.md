# Uni-v1: plus-only sliders, raw h+, no leftover-gate

Plus-only. One concept per yaml. Teacher is raw h+ (the + caption).
Do not leftover-gate. Do not teach −1.

This is the #45 plus-only exam card
([docs/lm-plus-exam.md](lm-plus-exam.md)) on **uni-v1** prompts, not
leftover-gate, not even-blend, not pair-odd, not v4. Default stays
`--lm_target v9` / `--pole_mode hidden`. The bipolar board is not
updated here.

No Hub, no GPU, no Music 3 weights in this page. Train on the box
that has the weights.

## Why uni-v1 is not v4

v4 poles are two songs (genre / BPM / mix / kit move with the axis).
Uni-v1 pins those and writes only the + concept. There is no opposite
pole: `negative := target` (yaml shape / canary). `minus_label` is
`Off`. Omit `leak_*` — `faithful_plus` leftover-gates h+ if `leak_*`
is declared and unused; uni-v1 must be raw h+.

| axis | + concept (one place) |
|---|---|
| energy | loud / slammed (Global Metadata) |
| gender | woman / female (Vocal Details; no attributes) |
| tempo | fast and the BPM number (Global Metadata) |
| distortion | distorted tone (Global Metadata) |
| breath | audible inhales (Vocal Details) |
| rhyme | dense rhyme (Vocal Details) |
| triphop | dusty / vinyl / hazy (Global Metadata) |
| live | bleed / crowd / human time (Global Metadata) |

Prompts: `conceptmod/textsliders/data/prompts-<axis>-uni-v1.yaml`.

## Train card (opt-in)

`--lm_target faithful_plus --pole_mode hidden`. Teacher is leftover-gated
`h+` (raw pos when leftover ê is unused or undeclared). Uni-v1 declares
no ê, so the teacher is the + caption. Student +1 fits that + state.
No pair-odd, no `h0 ± a`, no minus MSE. Inference may still expose a
−1 fader; that fader is not trained. Sidecar records `lm_target` and
`plus_only`.

```bash
CUDA_VISIBLE_DEVICES=N python conceptmod/textsliders/train_lm_slider_music3.py \
  --name energy-lm-uni-v1 \
  --prompts_file conceptmod/textsliders/data/prompts-energy-uni-v1.yaml \
  --lm_target faithful_plus --pole_mode hidden \
  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \
  --no-early_stop --endreg_weight 1.0 --device 0
```

Same flags for gender, tempo, distortion, breath, rhyme, triphop, live
(`--name <axis>-lm-uni-v1`, matching `prompts-<axis>-uni-v1.yaml`).
Do not pass v4 files.

## Both GPUs, sample in between, into uni-v1

Train one axis, sample it, then the next. Two cards in parallel:

```bash
./scripts/train_uni_v1_lm.sh 0 energy gender tempo distortion
./scripts/train_uni_v1_lm.sh 1 breath rhyme triphop live
```

Weights: `models/<axis>-lm-uni-v1/`. Listens:
`eval/listen/uni-v1/<axis>-lm-uni-v1/` (scales 0, 1, 2 plus the + REF).
−1 is an unconstrained canary, not a pole.

## Related

- [lm-plus-exam.md](lm-plus-exam.md) — plus-only cover / off-caption scale.
- [lm-pair-exam.md](lm-pair-exam.md) — bipolar continuation gates (not this).
- [lm-2d-scoreboard.md](lm-2d-scoreboard.md) — compiled bipolar board (not this).
