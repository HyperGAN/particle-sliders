#!/usr/bin/env bash
# Uni-v3 prefix-hold UNI: same-room prompts-<axis>-uni-v1.yaml (never v4, never minus).
# --lm_target faithful_plus_neu_prefix --pole_mode hidden
# Teacher: +1 last → raw h+; +1 prefix → encode(neu) prefix (yaml lyrics);
# scale 0 → h0. No minus MSE. Last token must be <|audio_start|>.
# Names are <axis>-lm-uni-prefix. Samples at eval/listen/uni-v3/<axis>-lm-uni-prefix/.
# After each train: generate_listen 0,1,2 (plus the + REF), then the next axis.
# Exam is +1; +2 may shred. recommended_range [0, 1].
#
# Worst-first, both GPUs:
#   ./scripts/train_uni_v3_prefix.sh 0 grit joy energy gender
#   ./scripts/train_uni_v3_prefix.sh 1 distortion hurt rapslow tempo
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CATALOG="grit distortion joy energy hurt rapslow gender tempo"

prompts_for() {
  local axis="$1"
  local uni="conceptmod/textsliders/data/prompts-${axis}-uni-v1.yaml"
  case "$axis" in
    *-minus*|minus-*)
      echo "[uni-v3] FAIL ${axis}: plus queue refuses minus slugs" >&2
      return 1
      ;;
  esac
  if [ ! -f "$uni" ]; then
    echo "[uni-v3] FAIL ${axis}: missing $uni" >&2
    return 1
  fi
  case "$uni" in
    *v4.yaml|*minus-uni*)
      echo "[uni-v3] FAIL ${axis}: refused $uni" >&2
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
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_plus_neu_prefix pole_mode=hidden endreg=1.0"
  echo "  prompts=$prompts"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_plus_neu_prefix --pole_mode hidden \
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
plus_neu_prefix = d.get("plus_neu_prefix")
used = d.get("prompts_file") or d.get("prompts") or ""
if target != "faithful_plus_neu_prefix":
    raise SystemExit(f"[uni-v3] FAIL {name}: sidecar lm_target={target!r} want faithful_plus_neu_prefix")
if pole != "hidden":
    raise SystemExit(f"[uni-v3] FAIL {name}: sidecar pole_mode={pole!r}")
if plus_neu is not True:
    raise SystemExit(f"[uni-v3] FAIL {name}: plus_neu={plus_neu!r}")
if plus_neu_prefix is not True:
    raise SystemExit(f"[uni-v3] FAIL {name}: plus_neu_prefix={plus_neu_prefix!r}")
if "v4.yaml" in str(used) or "v4.yaml" in prompts:
    raise SystemExit(f"[uni-v3] FAIL {name}: sidecar/prompts still v4 {used!r} {prompts!r}")
if "uni-v1.yaml" not in prompts:
    raise SystemExit(f"[uni-v3] FAIL {name}: expected uni-v1 yaml, got {prompts!r}")
if d.get("leak_positive") or d.get("leak_negative"):
    raise SystemExit(f"[uni-v3] FAIL {name}: leak_* must be omitted")
print(f"[uni-v3] sidecar {name} lm_target={target} pole_mode={pole} plus_neu={plus_neu} plus_neu_prefix={plus_neu_prefix} prompts={prompts}")
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
  echo "[gpu${gpu}] SAMPLE ${name} 0,1,2 -> $out_dir (exam +1; +2 canary)"
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
  local name="${axis}-lm-uni-prefix"
  local prompts out sidecar
  prompts="$(prompts_for "$axis")" || return 1
  out="eval/listen/uni-v3/${name}"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "eval/listen/uni-v3/logs" "$out" "models/${name}"
  echo "[uni-v3] ${name} GPU${gpu} prompts=$prompts lm_target=faithful_plus_neu_prefix pole_mode=hidden"
  if [ -f "$sidecar" ] && [ -f "models/${name}/${name}_last.safetensors" ]; then
    if assert_sidecar "$ROOT/$sidecar" "$name" "$prompts"; then
      echo "[uni-v3] SKIP train ${name} (faithful_plus_neu_prefix weights exist)"
    else
      echo "[uni-v3] FAIL ${name}: leftover non-UNI weights; not skipping" >&2
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
    echo "[uni-v3] sample failed ${name}"
    return 1
  fi
  echo "[uni-v3] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful_plus_neu_prefix --pole_mode hidden (UNI prefix-hold)" >&2
  echo "  prompts: conceptmod/textsliders/data/prompts-<axis>-uni-v1.yaml (never v4)" >&2
  echo "  worst-first both GPUs:" >&2
  echo "    $0 0 grit joy energy gender" >&2
  echo "    $0 1 distortion hurt rapslow tempo" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
