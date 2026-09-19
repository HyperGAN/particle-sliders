#!/usr/bin/env bash
# All LM v4 axes, both poles: --lm_target faithful --pole_mode hidden.
# Skips energy-lm-uni-v1 (already faithful_plus_neu). Do not pass v4 rewrite.
# After each train: generate_listen -2,-1,0,1,2 into eval/listen/uni-v1/.
#
#   ./scripts/train_uni_v1_bipolar.sh 0 gender tempo distortion breath rhyme triphop live
#   ./scripts/train_uni_v1_bipolar.sh 1 rapslow grit hurt joy sexy tender yearn
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CATALOG="gender tempo distortion breath rhyme triphop live rapslow grit hurt joy sexy tender yearn"

prompts_for() {
  local axis="$1"
  local v4="conceptmod/textsliders/data/prompts-${axis}-v4.yaml"
  if [ -f "$v4" ]; then
    echo "$v4"
  else
    return 1
  fi
}

train_one() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful pole_mode=hidden endreg=1.0"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful --pole_mode hidden \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \
      --no-early_stop --endreg_weight 1.0 \
      --save_every 0 --device 0 \
      > "$save/${name}_train.log" 2>&1; then
    echo "[gpu${gpu}] TRAIN OK ${name}"
    return 0
  fi
  echo "[gpu${gpu}] TRAIN FAIL ${name}"
  tail -40 "$save/${name}_train.log" || true
  return 1
}

assert_sidecar() {
  local sidecar="$1" name="$2"
  "$PY" - "$sidecar" "$name" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
d = json.loads(open(path).read())
target = d.get("lm_target")
pole = d.get("pole_mode")
plus_only = d.get("plus_only")
if target != "faithful":
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar lm_target={target!r} want faithful")
if pole != "hidden":
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar pole_mode={pole!r}")
if plus_only is True:
    raise SystemExit(f"[uni-v1] FAIL {name}: plus_only set; this run is both poles")
print(f"[uni-v1] sidecar {name} lm_target={target} pole_mode={pole}")
PY
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-uni-v1"
  if [ "$axis" = energy ]; then
    echo "[uni-v1] SKIP energy (existing faithful_plus_neu listen)"
    return 0
  fi
  local prompts out sidecar
  prompts="$(prompts_for "$axis")" || {
    echo "[uni-v1] no v4 file for $axis" >&2
    return 1
  }
  out="eval/listen/uni-v1/${name}"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "eval/listen/uni-v1/logs" "$out" "models/${name}"
  echo "[uni-v1] ${name} GPU${gpu} prompts=$prompts lm_target=faithful pole_mode=hidden"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[uni-v1] SKIP train ${name} (weights exist)"
  else
    if ! train_one "$gpu" "$name" "$prompts"; then
      return 1
    fi
  fi
  if [ -f "$sidecar" ]; then
    if ! assert_sidecar "$ROOT/$sidecar" "$name"; then
      echo "[uni-v1] refusing leftover recipe weights for ${name} — delete and retry" >&2
      return 1
    fi
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[uni-v1] sample failed ${name}"
    return 1
  fi
  echo "[uni-v1] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful --pole_mode hidden (both poles)" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
