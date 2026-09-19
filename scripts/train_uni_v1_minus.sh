#!/usr/bin/env bash
# UNI LoRA for the OLD minus pole of each v4 axis.
# prompts-<axis>-minus-uni-v1.yaml, --lm_target faithful_plus_neu
# +1 fits the old minus caption; scale 0 fits h0; no minus MSE.
# Separate names/samples from the plus UNI LoRAs.
#
#   ./scripts/train_uni_v1_minus.sh 0 energy gender tempo distortion breath rhyme triphop live
#   ./scripts/train_uni_v1_minus.sh 1 rapslow grit hurt joy sexy tender yearn
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

# axis -> train name (plus_label of the minus yaml)
declare -A MINUS_NAME=(
  [energy]=energy-quiet-lm-uni-v1
  [gender]=gender-male-lm-uni-v1
  [tempo]=tempo-slow-lm-uni-v1
  [distortion]=distortion-clean-lm-uni-v1
  [breath]=breath-clean-lm-uni-v1
  [rhyme]=rhyme-prose-lm-uni-v1
  [triphop]=triphop-pop-lm-uni-v1
  [live]=live-studio-lm-uni-v1
  [rapslow]=rapslow-slow-lm-uni-v1
  [grit]=grit-smooth-lm-uni-v1
  [hurt]=hurt-numb-lm-uni-v1
  [joy]=joy-somber-lm-uni-v1
  [sexy]=sexy-plain-lm-uni-v1
  [tender]=tender-fierce-lm-uni-v1
  [yearn]=yearn-settled-lm-uni-v1
)

prompts_for() {
  local axis="$1"
  local f="conceptmod/textsliders/data/prompts-${axis}-minus-uni-v1.yaml"
  if [ -f "$f" ]; then
    echo "$f"
  else
    return 1
  fi
}

train_one() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_plus_neu pole_mode=hidden (minus UNI)"
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
if d.get("lm_target") != "faithful_plus_neu":
    raise SystemExit(f"[minus-uni] FAIL {name}: lm_target={d.get('lm_target')!r}")
if d.get("pole_mode") != "hidden":
    raise SystemExit(f"[minus-uni] FAIL {name}: pole_mode={d.get('pole_mode')!r}")
if d.get("plus_neu") is not True:
    raise SystemExit(f"[minus-uni] FAIL {name}: plus_neu={d.get('plus_neu')!r}")
print(f"[minus-uni] sidecar {name} lm_target=faithful_plus_neu plus_neu=True plus_label={d.get('plus_label')}")
PY
}

sample_plus() {
  local gpu="$1" name="$2" prompts="$3" out_dir="$4"
  local save="models/${name}"
  local weights
  weights="$(lm_weights "$save" "$name")"
  [ -n "${weights:-}" ] && [ -f "$weights" ] || {
    echo "[gpu${gpu}] no weights for ${name}"
    return 1
  }
  wait_vram "$gpu"
  echo "[gpu${gpu}] SAMPLE ${name} 0,1,2 -> $out_dir"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/generate_listen.py \
    --weights "$weights" \
    --prompts_file "$prompts" \
    --name "$name" --kind lm \
    --out_dir "$out_dir" \
    --scales=0,1,2 --duration 20 --seed 7 --device 0
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${MINUS_NAME[$axis]:-}"
  if [ -z "$name" ]; then
    echo "[minus-uni] unknown axis $axis" >&2
    return 1
  fi
  local prompts out sidecar
  prompts="$(prompts_for "$axis")" || {
    echo "[minus-uni] no minus-uni yaml for $axis" >&2
    return 1
  }
  out="eval/listen/uni-v1/${name}"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "eval/listen/uni-v1/logs" "$out" "models/${name}"
  echo "[minus-uni] ${name} GPU${gpu} prompts=$prompts"
  if [ -f "$sidecar" ]; then
    if assert_sidecar "$ROOT/$sidecar" "$name"; then
      echo "[minus-uni] SKIP train ${name}"
    else
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
  if ! sample_plus "$gpu" "$name" "$prompts" "$out"; then
    return 1
  fi
  echo "[minus-uni] done ${name} -> $out"
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
