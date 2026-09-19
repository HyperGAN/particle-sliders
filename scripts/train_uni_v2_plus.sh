#!/usr/bin/env bash
# Uni-v2 plus UNI: same-room prompts-<axis>-uni-v1.yaml (never v4, never minus).
# --lm_target faithful_plus_neu --pole_mode hidden
# Teacher is raw h+; student +1 fits h+, scale 0 fits h0; no minus MSE.
# Names are <axis>-lm-uni-v2. Samples at eval/listen/uni-v2/<axis>-lm-uni-v2/.
# After each train: generate_listen 0,1,2 (plus the + REF), then the next axis.
#
#   ./scripts/train_uni_v2_plus.sh 0 energy gender tempo distortion breath rhyme triphop live
#   ./scripts/train_uni_v2_plus.sh 1 rapslow grit hurt joy sexy tender yearn
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CATALOG="energy gender tempo distortion breath rhyme triphop live rapslow grit hurt joy sexy tender yearn"

prompts_for() {
  local axis="$1"
  local uni="conceptmod/textsliders/data/prompts-${axis}-uni-v1.yaml"
  case "$axis" in
    *-minus*|minus-*)
      echo "[uni-v2] FAIL ${axis}: plus queue refuses minus slugs" >&2
      return 1
      ;;
  esac
  if [ ! -f "$uni" ]; then
    echo "[uni-v2] FAIL ${axis}: missing $uni" >&2
    return 1
  fi
  case "$uni" in
    *v4.yaml|*minus-uni*)
      echo "[uni-v2] FAIL ${axis}: refused $uni" >&2
      return 1
      ;;
  esac
  echo "$uni"
}

train_one() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_plus_neu pole_mode=hidden endreg=1.0"
  echo "  prompts=$prompts"
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
  local sidecar="$1" name="$2" prompts="$3"
  "$PY" - "$sidecar" "$name" "$prompts" <<'PY'
import json, sys
path, name, prompts = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.loads(open(path).read())
target = d.get("lm_target")
pole = d.get("pole_mode")
plus_neu = d.get("plus_neu")
used = d.get("prompts_file") or d.get("prompts") or ""
if target != "faithful_plus_neu":
    raise SystemExit(f"[uni-v2] FAIL {name}: sidecar lm_target={target!r} want faithful_plus_neu")
if pole != "hidden":
    raise SystemExit(f"[uni-v2] FAIL {name}: sidecar pole_mode={pole!r}")
if plus_neu is not True:
    raise SystemExit(f"[uni-v2] FAIL {name}: plus_neu={plus_neu!r}")
if "v4.yaml" in str(used) or "v4.yaml" in prompts:
    raise SystemExit(f"[uni-v2] FAIL {name}: sidecar/prompts still v4 {used!r} {prompts!r}")
if "uni-v1.yaml" not in prompts:
    raise SystemExit(f"[uni-v2] FAIL {name}: expected uni-v1 yaml, got {prompts!r}")
print(f"[uni-v2] sidecar {name} lm_target={target} pole_mode={pole} plus_neu={plus_neu} prompts={prompts}")
PY
}

sample_plus() {
  local gpu="$1" name="$2" prompts="$3" out_dir="$4"
  local save="models/${name}"
  local weights
  weights="$(lm_weights "$save" "$name")"
  [ -n "${weights:-}" ] && [ -f "$weights" ] || {
    echo "[gpu${gpu}] no weights for ${name} in $save"
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
  local name="${axis}-lm-uni-v2"
  local prompts out sidecar
  prompts="$(prompts_for "$axis")" || return 1
  out="eval/listen/uni-v2/${name}"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "eval/listen/uni-v2/logs" "$out" "models/${name}"
  echo "[uni-v2] ${name} GPU${gpu} prompts=$prompts lm_target=faithful_plus_neu pole_mode=hidden"
  if [ -f "$sidecar" ] && [ -f "models/${name}/${name}_last.safetensors" ]; then
    if assert_sidecar "$ROOT/$sidecar" "$name" "$prompts"; then
      echo "[uni-v2] SKIP train ${name} (faithful_plus_neu weights exist)"
    else
      echo "[uni-v2] FAIL ${name}: leftover non-UNI weights; not skipping" >&2
      return 1
    fi
  else
    if ! train_one "$gpu" "$name" "$prompts"; then
      return 1
    fi
    if [ -f "$sidecar" ] && ! assert_sidecar "$ROOT/$sidecar" "$name" "$prompts"; then
      return 1
    fi
  fi
  if ! sample_plus "$gpu" "$name" "$prompts" "$out"; then
    echo "[uni-v2] sample failed ${name}"
    return 1
  fi
  echo "[uni-v2] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful_plus_neu --pole_mode hidden (UNI plus)" >&2
  echo "  prompts: conceptmod/textsliders/data/prompts-<axis>-uni-v1.yaml (never v4)" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
