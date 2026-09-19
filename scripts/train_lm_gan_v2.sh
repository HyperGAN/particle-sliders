#!/usr/bin/env bash
# Versioned research trainer. The catalog remains a separate, paused campaign.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MUSIC3_PYTHON:-/home/mikkel/anaconda3/envs/minimax-music3/bin/python}"
NAME="${1:?Supply a new run name}"
ARM="${ARM:-fm_capped}"
case "$ARM" in
  baseline|fm|fm_normalized|fm_capped|decay)
    DEFAULT_PROMPTS="$ROOT/conceptmod/textsliders/data/prompts-gender-uni-v2.yaml"
    DEFAULT_EMA=0 ;;
  *) DEFAULT_PROMPTS="$ROOT/analysis/gan_bcap/v2_20260905/fixtures/train.yaml"
     DEFAULT_EMA=1 ;;
esac
PROMPTS="${2:-$DEFAULT_PROMPTS}"
EMA_ARGS=()
case "${USE_EMA:-$DEFAULT_EMA}" in
  0) EMA_ARGS=(--no-ema) ;;
  1) ;;
  *) echo 'USE_EMA must be 0 or 1' >&2; exit 2 ;;
esac
SOURCE="${SOURCE_STATE:-$ROOT/models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_state.pt}"
ORIGIN="${SCHEDULE_ORIGIN:-600}"
HORIZON="${SCHEDULE_HORIZON:-660}"
UNTIL="${UNTIL:-$HORIZON}"
SOURCE_ARGS=(--source-state "$SOURCE")
if [[ -n "${RESUME_STATE:-}" ]]; then SOURCE_ARGS=(--resume "$RESUME_STATE"); fi
export CUDA_VISIBLE_DEVICES="${MUSIC_GAN_GPU:-0}" HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
exec "$PY" -u -m conceptmod.textsliders.gan_v2.train \
  --prompts "$PROMPTS" --run-dir "$ROOT/models/gan-v2/$NAME" --arm "$ARM" \
  --schedule-origin "$ORIGIN" --schedule-horizon "$HORIZON" --until "$UNTIL" \
  --save-every "${SAVE_EVERY:-30}" --ema-every "${EMA_EVERY:-15}" \
  --diagnostics-every "${DIAGNOSTICS_EVERY:-15}" \
  "${EMA_ARGS[@]}" "${SOURCE_ARGS[@]}"
