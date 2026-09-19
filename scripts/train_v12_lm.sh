#!/usr/bin/env bash
# Every catalog LM axis as --lm_target symmetric (rapslow-style pair-odd).
# Names are <axis>-lm-v12. Samples at eval/listen/v12/<axis>-lm-v12/.
#
#   ./scripts/train_v12_lm.sh 0 energy tempo breath sexy grit yearn hurt
#   ./scripts/train_v12_lm.sh 1 triphop distortion rhyme tender joy live
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

prompts_for() {
  local axis="$1"
  local v7="conceptmod/textsliders/data/prompts-${axis}-v7.yaml"
  local v6="conceptmod/textsliders/data/prompts-${axis}-v6.yaml"
  if [ -f "$v7" ]; then
    echo "$v7"
  elif [ -f "$v6" ]; then
    echo "$v6"
  else
    return 1
  fi
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-v12"
  local prompts out logs
  prompts="$(prompts_for "$axis")" || {
    echo "[v12] no pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v12/${name}"
  logs="eval/listen/v12/logs"
  mkdir -p "$logs" "$out" "models/${name}"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v12] SKIP train ${name} (weights exist)"
  else
    echo "[v12] ${name} GPU${gpu} prompts=$prompts lm_target=symmetric"
    if ! train_lm_symmetric "$gpu" "$name" "$prompts"; then
      echo "[v12] train failed ${name}"
      return 1
    fi
  fi
  echo "[v12] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v12] GATE OK ${name}"
  else
    echo "[v12] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v12] sample failed ${name}"
    return 1
  fi
  echo "[v12] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
