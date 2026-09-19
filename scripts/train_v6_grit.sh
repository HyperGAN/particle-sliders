#!/usr/bin/env bash
# Grit only. Prefer train_v6_feel.sh for the full song-feel queue.
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
echo "[v6-grit] GPU ${GPU}"
train_lm "$GPU" grit-lm-v6 conceptmod/textsliders/data/prompts-grit-v4.yaml
train_tf "$GPU" grit-tf-v6 conceptmod/textsliders/data/prompts-grit-tf.yaml
echo "[v6-grit] done"
