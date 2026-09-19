#!/usr/bin/env bash
# Uni-v1: plus-only LM sliders on prompts-<axis>-uni-v1.yaml.
# --lm_target faithful_plus --pole_mode hidden
# Teacher is raw h+ (no leak_*). Negative := target; not taught.
# Names are <axis>-lm-uni-v1. Samples at eval/listen/uni-v1/<axis>-lm-uni-v1/.
# After each train: sidecar check, then generate_listen at scales 0,1,2
# (plus the + REF). Do not pass v4 files.
#
#   ./scripts/train_uni_v1_lm.sh 0 energy gender tempo distortion
#   ./scripts/train_uni_v1_lm.sh 1 breath rhyme triphop live
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CATALOG="energy gender tempo distortion breath rhyme triphop live"

prompts_for() {
  local axis="$1"
  local uni="conceptmod/textsliders/data/prompts-${axis}-uni-v1.yaml"
  if [ -f "$uni" ]; then
    echo "$uni"
  else
    return 1
  fi
}

train_lm_uni_v1() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_plus pole_mode=hidden endreg=1.0"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_plus --pole_mode hidden \
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

assert_uni_v1_log() {
  local log="$1" name="$2"
  if ! grep -q 'faithful_plus: teacher=+ caption leftover-gated' "$log"; then
    echo "[uni-v1] FAIL ${name}: expected faithful_plus teacher banner" >&2
    tail -40 "$log" || true
    return 1
  fi
  if grep -q 'leftover gate:' "$log"; then
    echo "[uni-v1] FAIL ${name}: leftover-gate ran (uni-v1 must omit leak_*)" >&2
    grep -E 'leftover gate|declared leak' "$log" || true
    return 1
  fi
  if ! grep -q 'pole_mode=hidden' "$log" && ! grep -q 'minus_mse=off' "$log"; then
    echo "[uni-v1] FAIL ${name}: expected plus-only hidden banner" >&2
    tail -40 "$log" || true
    return 1
  fi
  echo "[uni-v1] faithful_plus pole_mode=hidden no leftover-gate ok ${name}"
}

assert_uni_v1_sidecar() {
  local sidecar="$1" name="$2"
  "$PY" - "$sidecar" "$name" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
d = json.loads(open(path).read())
target = d.get("lm_target")
pole = d.get("pole_mode")
plus_only = d.get("plus_only")
leak_p = d.get("leak_positive") or ""
leak_n = d.get("leak_negative") or ""
if target != "faithful_plus":
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar lm_target={target!r}")
if pole != "hidden":
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar pole_mode={pole!r}")
if plus_only is not True:
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar plus_only={plus_only!r}")
if leak_p or leak_n:
    raise SystemExit(f"[uni-v1] FAIL {name}: sidecar leak_* {leak_p!r}/{leak_n!r}")
print(f"[uni-v1] sidecar {name} lm_target={target} pole_mode={pole} plus_only={plus_only}")
PY
}

append_metrics() {
  local name="$1" sidecar="$2"
  local metrics="eval/listen/uni-v1/metrics.tsv"
  mkdir -p "eval/listen/uni-v1"
  if [ ! -f "$metrics" ]; then
    printf 'name\tlm_target\tpole_mode\tplus_only\tsteps\tcos_pos\tcollapse\tloss\n' > "$metrics"
  fi
  "$PY" - "$sidecar" "$name" "$metrics" <<'PY'
import json, sys
path, name, out = sys.argv[1:4]
d = json.loads(open(path).read())
last = d.get("last") or {}
def f(key, default=""):
    v = last.get(key, default)
    if isinstance(v, float):
        return f"{v:.6f}"
    return "" if v is None else str(v)
row = "\t".join([
    name,
    str(d.get("lm_target", "")),
    str(d.get("pole_mode", "")),
    str(d.get("plus_only", "")),
    str(d.get("steps", "")),
    f("cos_pos"),
    f("collapse"),
    f("loss"),
])
with open(out, "a") as fh:
    fh.write(row + "\n")
print(
    f"[uni-v1] metrics {name} lm_target={d.get('lm_target')} "
    f"pole={d.get('pole_mode')} plus_only={d.get('plus_only')} "
    f"steps={d.get('steps')} c+={last.get('cos_pos')} "
    f"col={last.get('collapse')} loss={last.get('loss')}"
)
PY
}

sample_uni() {
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
  local name="${axis}-lm-uni-v1"
  local prompts out logs sidecar
  prompts="$(prompts_for "$axis")" || {
    echo "[uni-v1] no uni-v1 file for $axis" >&2
    return 1
  }
  case "$prompts" in
    *uni-v1.yaml) ;;
    *)
      echo "[uni-v1] FAIL ${axis}: refused non-uni prompts $prompts" >&2
      return 1
      ;;
  esac
  out="eval/listen/uni-v1/${name}"
  logs="eval/listen/uni-v1/logs"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "$logs" "$out" "models/${name}"
  echo "[uni-v1] flags ${name} GPU${gpu} recipe=faithful_plus pole_mode=hidden endreg=1.0"
  echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
  echo "    --name ${name} --lm_target faithful_plus --pole_mode hidden \\"
  echo "    --prompts_file ${prompts} \\"
  echo "    --save_dir $ROOT/models/${name} \\"
  echo "    --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --endreg_weight 1.0 --device 0"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[uni-v1] SKIP train ${name} (weights exist)"
  else
    echo "[uni-v1] ${name} GPU${gpu} prompts=$prompts lm_target=faithful_plus"
    if ! train_lm_uni_v1 "$gpu" "$name" "$prompts"; then
      echo "[uni-v1] train failed ${name}"
      return 1
    fi
  fi
  if [ -f "models/${name}/${name}_train.log" ]; then
    if ! assert_uni_v1_log "models/${name}/${name}_train.log" "$name"; then
      return 1
    fi
  fi
  if [ -f "$sidecar" ]; then
    if ! assert_uni_v1_sidecar "$ROOT/$sidecar" "$name"; then
      return 1
    fi
    append_metrics "$name" "$ROOT/$sidecar" || true
  fi
  if ! sample_uni "$gpu" "$name" "$prompts" "$out"; then
    echo "[uni-v1] sample failed ${name}"
    return 1
  fi
  echo "[uni-v1] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful_plus --pole_mode hidden --endreg_weight 1.0" >&2
  echo "  prompts: conceptmod/textsliders/data/prompts-<axis>-uni-v1.yaml" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
