#!/usr/bin/env bash
# Train LM sliders with a *bare* --lm_target v9 run (projected-odd + hold +
# endreg, κ=0). Do not pass Hub leash flags. After each train, sample
# -2,-1,0,1,2 on the same GPU so two cards can train in parallel.
#
#   ./scripts/train_v11_lm.sh 0 gender distortion tempo
#   ./scripts/train_v11_lm.sh 1 rapslow triphop grit
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
  local name="${axis}-lm-v1"
  local prompts out logs
  prompts="$(prompts_for "$axis")" || {
    echo "[v1-lm] no pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v1/${name}"
  logs="eval/listen/v1/logs"
  mkdir -p "$logs" "$out" "models/${name}"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v1-lm] SKIP train ${name} (weights exist)"
  else
    echo "[v1-lm] ${name} GPU${gpu} prompts=$prompts"
    if ! train_lm_v9 "$gpu" "$name" "$prompts"; then
      echo "[v1-lm] train failed ${name}"
      return 1
    fi
  fi
  echo "[v1-lm] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v1-lm] GATE OK ${name}"
  else
    echo "[v1-lm] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v1-lm] sample failed ${name}"
    return 1
  fi
  echo "[v1-lm] done ${name} -> $out"
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
