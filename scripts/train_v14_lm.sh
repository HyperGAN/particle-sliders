#!/usr/bin/env bash
# Every catalog LM axis as --lm_target v9 on the v4 pair file
# (3fa7bdb: pair-odd + hold ê_⊥û). Names are <axis>-lm-v14.
# Samples at eval/listen/v14/<axis>-lm-v14/.
#
#   ./scripts/train_v14_lm.sh 0 energy tempo breath sexy grit yearn hurt
#   ./scripts/train_v14_lm.sh 1 gender rapslow triphop distortion rhyme tender joy live
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
  local v4="conceptmod/textsliders/data/prompts-${axis}-v4.yaml"
  if [ -f "$v4" ]; then
    echo "$v4"
  else
    return 1
  fi
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-v14"
  local prompts out logs
  prompts="$(prompts_for "$axis")" || {
    echo "[v14] no v4 pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v14/${name}"
  logs="eval/listen/v14/logs"
  mkdir -p "$logs" "$out" "models/${name}"
  echo "[v14] flags ${name} GPU${gpu}"
  echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
  echo "    --name ${name} --prompts_file ${prompts} \\"
  echo "    --save_dir $ROOT/models/${name} \\"
  echo "    --lm_target v9 --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --device 0"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v14] SKIP train ${name} (weights exist)"
  else
    echo "[v14] ${name} GPU${gpu} prompts=$prompts lm_target=v9 hold=ê_⊥û"
    if ! train_lm_v9 "$gpu" "$name" "$prompts"; then
      echo "[v14] train failed ${name}"
      return 1
    fi
  fi
  echo "[v14] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v14] GATE OK ${name}"
  else
    echo "[v14] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v14] sample failed ${name}"
    return 1
  fi
  echo "[v14] done ${name} -> $out"
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
