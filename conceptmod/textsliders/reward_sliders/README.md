# Reward-derived Music 3 sliders

This research package implements the [CE handoff](../../../docs/reward-model-sliders-handoff.md).
The running investigation and its exact artifacts are in
[reward_sliders_20260907](../../../analysis/reward_sliders_20260907/README.md).

`experiment.py` freezes the 8/4/4 family split, four matched seeds, model/source/
style/scorer hashes, reward protocol and render budget before collection. It
captures two intermediate residual layers plus the complete conditional and
unconditional prompt/feedback embeddings. The installed pipeline performs
semantic sampling, residual-code decoding and acoustic synthesis unchanged.

The installed semantic loop uses the final prefill hidden state for a discarded
warmup decode. We exclude that position and the entire prompt, then steer every
feedback position on both branches. Each 20-second primary screen uses the first
500 nominal frame positions. Acoustic windows use 200 semantic frames and overlap
by 100; output waveform crops are recorded in `CaptureSpec`. These are pooled
window associations, not token-level reward labels.

Directions average high-minus-low residuals *within exact prompt/style cells*,
then average families equally. Centering and residual-norm calibration use only
training families. Development selects layer 11/23 and coefficient 0.01/0.03,
with Off eligible. The causal test compares the frozen positive teacher with Off,
its reverse and a predeclared unit random vector on four test families/four seeds.
The initial gate requires +0.02 mean CE against both Off and random, more than
half of seed pairs won against Off, and no unresolved arm failures.

`data.py` constructs the generation-span residuals on one saved history per row:

```text
real = H_activation_teacher(history) - H_frozen_base_and_style(history)
fake = H_base_and_style_and_reward_lora(history) - H_frozen_base_and_style(history)
```

The two CFG branches become separate same-geometry rows. The frozen pilot student
uses 128 generation positions, stride-four critic features and the full preceding
prompt. Prompt/lyric positions never become the reward target. The bounded
`gan_v2` baseline objective supplies RpGAN, b_cap, batch-mean feature matching and
the neutral EOS-margin safeguard; no extra KL or supervised distillation loss is
silently added. Homogeneous four-row family batches give equal family exposure
while avoiding repeated style merges during one optimizer update.

`train.py` requires the exact passing causal result. It exports live rank-8,
alpha-8 attention LoRA weights with the existing 144-projection, 432-tensor host
key contract. Full critic/optimizer/sampler/RNG states are separate. Checkpoints
at 300/600/660 are budgeted candidates, not preselected optima.

`evaluate.py` selects checkpoint and effective multiplier on development CE,
measures prompt drift and generation timing, tests direct added strength and
fixed-total-energy composition with reduced-style controls, and freezes twelve
fresh families/two seeds for at least 48 full songs. Full-song CE weights every
nonoverlapping ten-second window by duration, including the actual short tail.
PQ/PC/CU, fixed-caption CLAP style/voice scores, full-song ASR phrase diagnostics,
levels, silence, clipping, duration and output diversity remain visible.
Preservation tolerances are frozen from baseline variability before transfer.
Bootstrap samples are whole families. A broad candidate requires positive CE
with a 95% interval above zero and passing preservation diagnostics.

`interfaces.py` exposes `RewardScorer`, oriented reward fitting, candidate
selection and paired statistics for other scalar rewards. Lower-is-better
rewards reverse the ranking before fitting; they require new directions, causal
tests and student weights. Existing CE tensors are never relabeled as another
reward or as universal quality.

Run in the existing environment, with the studio restricted to GPU 0:

```bash
cd /ml2/music/sliders-conceptmod
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/ml2/music/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/ml2/music/sliders-conceptmod
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m conceptmod.textsliders.reward_sliders.campaign \
  --run-dir /ml2/music/sliders-conceptmod/analysis/reward_sliders_20260907
```

The durable campaign runs each GPU stage in a child process to release models
between stages. A failed causal probe stops before LoRA training. Interrupted
observations remain failed, without rerolling seeds; changed frozen sources
require a recorded amendment. The production generator and registry are not
modified by the research package. All authored fixture and style-checkpoint
prompt provenance must pass name validation.
