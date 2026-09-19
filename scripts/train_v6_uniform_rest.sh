#!/usr/bin/env bash
# After the in-flight LM v6 catalog trains finish: train space-lm-v6 and every TF half.
# Waiter lives on disk so pgrep does not match this shell.
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

echo "[uniform-rest] waiting for in-flight LM python jobs..."
while pgrep -f 'minimax-music3/bin/python -u conceptmod/textsliders/train_lm_slider_music3.py' >/dev/null; do
  sleep 30
done
echo "[uniform-rest] starting space-lm + 11 TF halves"

(
  train_lm 0 space-lm-v6 conceptmod/textsliders/data/prompts-space.yaml
  train_tf 0 energy-tf-v6 conceptmod/textsliders/data/prompts-energy.yaml
  train_tf 0 distortion-tf-v6 conceptmod/textsliders/data/prompts-distortion.yaml
  train_tf 0 space-tf-v6 conceptmod/textsliders/data/prompts-space.yaml
  train_tf 0 gender-tf-v6 conceptmod/textsliders/data/prompts-gender-tf.yaml --plus_label Female --minus_label Male
  train_tf 0 live-tf-v6 conceptmod/textsliders/data/prompts-live-tf.yaml
) > models/v6-uniform-gpu0.log 2>&1 &
pid0=$!

(
  train_tf 1 tempo-tf-v6 conceptmod/textsliders/data/prompts-tempo.yaml
  train_tf 1 triphop-tf-v6 conceptmod/textsliders/data/prompts-triphop-v3-single.yaml
  train_tf 1 dust-tf-v6 conceptmod/textsliders/data/prompts-cand-dust-v1.yaml
  train_tf 1 rapslow-tf-v6 conceptmod/textsliders/data/prompts-rapslow-tf.yaml
  train_tf 1 breath-tf-v6 conceptmod/textsliders/data/prompts-breath-tf.yaml
  train_tf 1 rhyme-tf-v6 conceptmod/textsliders/data/prompts-rhyme-tf.yaml
) > models/v6-uniform-gpu1.log 2>&1 &
pid1=$!

wait "$pid0" "$pid1"
echo "[uniform-rest] finished"
