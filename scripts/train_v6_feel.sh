#!/usr/bin/env bash
# Song-feel sliders on one GPU (desk should sit on the other).
# Skips a half whose last sidecar already has collapse < 0.
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

GPU="${1:-1}"
echo "[v6-feel] LM then TF on GPU ${GPU}"

train_lm "$GPU" sexy-lm-v6 conceptmod/textsliders/data/prompts-sexy-v4.yaml
train_lm "$GPU" tender-lm-v6 conceptmod/textsliders/data/prompts-tender-v4.yaml
train_lm "$GPU" grit-lm-v6 conceptmod/textsliders/data/prompts-grit-v4.yaml
train_lm "$GPU" hurt-lm-v6 conceptmod/textsliders/data/prompts-hurt-v4.yaml
train_lm "$GPU" joy-lm-v6 conceptmod/textsliders/data/prompts-joy-v4.yaml
train_lm "$GPU" yearn-lm-v6 conceptmod/textsliders/data/prompts-yearn-v4.yaml

train_tf "$GPU" sexy-tf-v6 conceptmod/textsliders/data/prompts-sexy-tf.yaml
train_tf "$GPU" tender-tf-v6 conceptmod/textsliders/data/prompts-tender-tf.yaml
train_tf "$GPU" grit-tf-v6 conceptmod/textsliders/data/prompts-grit-tf.yaml
train_tf "$GPU" hurt-tf-v6 conceptmod/textsliders/data/prompts-hurt-tf.yaml
train_tf "$GPU" joy-tf-v6 conceptmod/textsliders/data/prompts-joy-tf.yaml
train_tf "$GPU" yearn-tf-v6 conceptmod/textsliders/data/prompts-yearn-tf.yaml

echo "[v6-feel] done"
