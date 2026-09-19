#!/usr/bin/env bash
# Redo every v6 LM and TF half. Final checkpoint only. If collapse >= 0, retry
# with the next seed (7, 17, 27).
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

echo "[v6-all] LM then TF, collapse<0 or retry seed"

(
  train_lm 0 gender-lm-v6 conceptmod/textsliders/data/prompts-gender-v4.yaml
  train_lm 0 triphop-lm-v6 conceptmod/textsliders/data/prompts-triphop-v4.yaml
  train_lm 0 distortion-lm-v6 conceptmod/textsliders/data/prompts-distortion-v4.yaml
  train_lm 0 live-lm-v6 conceptmod/textsliders/data/prompts-live-v4.yaml
  train_lm 0 rhyme-lm-v6 conceptmod/textsliders/data/prompts-rhyme-v4.yaml
  train_lm 0 space-lm-v6 conceptmod/textsliders/data/prompts-space.yaml
  train_tf 0 energy-tf-v6 conceptmod/textsliders/data/prompts-energy.yaml
  train_tf 0 distortion-tf-v6 conceptmod/textsliders/data/prompts-distortion.yaml
  train_tf 0 space-tf-v6 conceptmod/textsliders/data/prompts-space.yaml
  train_tf 0 gender-tf-v6 conceptmod/textsliders/data/prompts-gender-tf.yaml --plus_label Female --minus_label Male
  train_tf 0 live-tf-v6 conceptmod/textsliders/data/prompts-live-tf.yaml
) > models/v6-all-gpu0.log 2>&1 &
pid0=$!

(
  train_lm 1 rapslow-lm-v6 conceptmod/textsliders/data/prompts-rapslow-v4.yaml
  train_lm 1 energy-lm-v6 conceptmod/textsliders/data/prompts-energy-v4.yaml
  train_lm 1 tempo-lm-v6 conceptmod/textsliders/data/prompts-tempo-v4.yaml
  train_lm 1 breath-lm-v6 conceptmod/textsliders/data/prompts-breath-v4.yaml
  train_tf 1 tempo-tf-v6 conceptmod/textsliders/data/prompts-tempo.yaml
  train_tf 1 triphop-tf-v6 conceptmod/textsliders/data/prompts-triphop-v3-single.yaml
  train_tf 1 dust-tf-v6 conceptmod/textsliders/data/prompts-cand-dust-v1.yaml
  train_tf 1 rapslow-tf-v6 conceptmod/textsliders/data/prompts-rapslow-tf.yaml
  train_tf 1 breath-tf-v6 conceptmod/textsliders/data/prompts-breath-tf.yaml
  train_tf 1 rhyme-tf-v6 conceptmod/textsliders/data/prompts-rhyme-tf.yaml
) > models/v6-all-gpu1.log 2>&1 &
pid1=$!

wait "$pid0" "$pid1"
echo "[v6-all] finished"
