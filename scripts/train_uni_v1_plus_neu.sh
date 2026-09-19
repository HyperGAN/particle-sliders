#!/usr/bin/env bash
# UNI card matching energy-lm-uni-v1: --lm_target faithful_plus_neu --pole_mode hidden
# Teacher is raw h+; student +1 fits h+, scale 0 fits h0; no minus MSE.
# v4 yamls (same as energy). Skip energy (already done).
# After each train: generate_listen -2,-1,0,1,2 into eval/listen/uni-v1/.
#
#   ./scripts/train_uni_v1_plus_neu.sh 0 gender tempo distortion breath rhyme triphop live
#   ./scripts/train_uni_v1_plus_neu.sh 1 rapslow grit hurt joy sexy tender yearn
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
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_plus_neu pole_mode=hidden endreg=1.0"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_plus_neu --pole_mode hidden \
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
plus_neu = d.get("plus_neu")
if target != "faithful_plus_neu":
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar lm_target={target!r} want faithful_plus_neu")
if pole != "hidden":
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar pole_mode={pole!r}")
if plus_neu is not True:
    raise SystemExit(f"[uni-v1] FAIL {name}: plus_neu={plus_neu!r}")
print(f"[uni-v1] sidecar {name} lm_target={target} pole_mode={pole} plus_neu={plus_neu}")
PY
}

sample_both() {
  local gpu="$1" name="$2" prompts="$3" out_dir="$4"
  local save="models/${name}"
  local weights
  weights="$(lm_weights "$save" "$name")"
  [ -n "${weights:-}" ] && [ -f "$weights" ] || {
    echo "[gpu${gpu}] no weights for ${name} in $save"
    return 1
  }
  wait_vram "$gpu"
  echo "[gpu${gpu}] SAMPLE ${name} -2,-1,0,1,2 -> $out_dir"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/generate_listen.py \
    --weights "$weights" \
    --prompts_file "$prompts" \
    --name "$name" --kind lm \
    --out_dir "$out_dir" \
    --scales=-2,-1,0,1,2 --duration 20 --seed 7 --device 0
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-uni-v1"
  if [ "$axis" = energy ]; then
    echo "[uni-v1] SKIP energy (already faithful_plus_neu)"
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
  echo "[uni-v1] ${name} GPU${gpu} prompts=$prompts lm_target=faithful_plus_neu pole_mode=hidden"
  if [ -f "$sidecar" ]; then
    if assert_sidecar "$ROOT/$sidecar" "$name"; then
      echo "[uni-v1] SKIP train ${name} (faithful_plus_neu weights exist)"
    else
      echo "[uni-v1] FAIL ${name}: leftover non-UNI weights; not skipping" >&2
      return 1
    fi
  else
    if ! train_one "$gpu" "$name" "$prompts"; then
      return 1
    fi
    if [ -f "$sidecar" ] && ! assert_sidecar "$ROOT/$sidecar" "$name"; then
      return 1
    fi
  fi
  if ! sample_both "$gpu" "$name" "$prompts" "$out"; then
    echo "[uni-v1] sample failed ${name}"
    return 1
  fi
  echo "[uni-v1] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful_plus_neu --pole_mode hidden (UNI, energy card)" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
