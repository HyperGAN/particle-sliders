#!/usr/bin/env bash
# v21: leftover-gate odd + half leak-pair even leftover (PR #39 / #41)
# --lm_target faithful_even_blend --even_blend_scale 0.5 --pole_mode hidden
# Same knobs as v20 besides the teacher. Pair-odd cos will look worse than
# v9 — that is the half-even subtract, not a miss.
# Gender has no leak_*; this flag is a no-op there. Queue leaky yamls.
# Names are <axis>-lm-v21. Samples at eval/listen/v21/<axis>-lm-v21/.
# After each train: sidecar metrics + check_slider_gate.py, then
# generate_listen at scales -2,-1,0,1,2 (plus REFs).
#
#   ./scripts/train_v21_lm.sh 0 energy tempo distortion
#   ./scripts/train_v21_lm.sh 1 breath rhyme triphop
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CATALOG="energy breath tempo distortion rhyme triphop"

prompts_for() {
  local axis="$1"
  local v4="conceptmod/textsliders/data/prompts-${axis}-v4.yaml"
  if [ -f "$v4" ]; then
    echo "$v4"
  else
    return 1
  fi
}

train_lm_v21() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_even_blend even_blend_scale=0.5 pole_mode=hidden endreg=1.0"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_even_blend --even_blend_scale 0.5 --pole_mode hidden \
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

assert_v21_log() {
  local log="$1" name="$2"
  if ! grep -q 'faithful_even_blend: leftover-gate odd' "$log"; then
    echo "[v21] FAIL ${name}: expected even-blend teacher banner" >&2
    tail -40 "$log" || true
    return 1
  fi
  if ! grep -q 'subtract 0.5 of leak-pair even leftover' "$log"; then
    echo "[v21] FAIL ${name}: expected even_blend_scale=0.5 in teacher banner" >&2
    tail -40 "$log" || true
    return 1
  fi
  if ! grep -q 'pole_mode=hidden' "$log"; then
    echo "[v21] FAIL ${name}: expected pole_mode=hidden" >&2
    tail -40 "$log" || true
    return 1
  fi
  echo "[v21] even-blend 0.5 pole_mode=hidden ok ${name}"
}

assert_v21_sidecar() {
  local sidecar="$1" name="$2"
  "$PY" - "$sidecar" "$name" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
d = json.loads(open(path).read())
scale = d.get("even_blend_scale")
target = d.get("lm_target")
if target != "faithful_even_blend":
    raise SystemExit(f"[v21] FAIL {name}: sidecar lm_target={target!r}")
if scale != 0.5:
    raise SystemExit(f"[v21] FAIL {name}: sidecar even_blend_scale={scale!r} (want 0.5)")
print(f"[v21] sidecar {name} even_blend_scale={scale} lm_target={target}")
PY
}

append_metrics() {
  local name="$1" sidecar="$2" gate_status="$3"
  local metrics="eval/listen/v21/metrics.tsv"
  mkdir -p "eval/listen/v21"
  if [ ! -f "$metrics" ]; then
    printf 'name\tlm_target\teven_blend_scale\tpole_mode\tsteps\tcos_pos\tcos_neg\tcollapse\tedrift_p\tedrift_n\tloss\tgate\n' > "$metrics"
  fi
  "$PY" - "$sidecar" "$name" "$gate_status" "$metrics" <<'PY'
import json, sys
path, name, gate, out = sys.argv[1:5]
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
    str(d.get("even_blend_scale", "")),
    str(d.get("pole_mode", "")),
    str(d.get("steps", "")),
    f("cos_pos"),
    f("cos_neg"),
    f("collapse"),
    f("edrift_p"),
    f("edrift_n"),
    f("loss"),
    gate,
])
with open(out, "a") as fh:
    fh.write(row + "\n")
print(
    f"[v21] metrics {name} lm_target={d.get('lm_target')} "
    f"even_blend_scale={d.get('even_blend_scale')} "
    f"pole={d.get('pole_mode')} "
    f"steps={d.get('steps')} "
    f"c+={last.get('cos_pos'):.3f} c-={last.get('cos_neg'):.3f} "
    f"col={last.get('collapse'):.3f} e={last.get('edrift_p'):.3f}/{last.get('edrift_n'):.3f} "
    f"loss={last.get('loss'):.4f} gate={gate}"
)
PY
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-v21"
  local prompts out logs sidecar
  prompts="$(prompts_for "$axis")" || {
    echo "[v21] no v4 pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v21/${name}"
  logs="eval/listen/v21/logs"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "$logs" "$out" "models/${name}"
  echo "[v21] flags ${name} GPU${gpu} recipe=faithful_even_blend even_blend_scale=0.5 pole_mode=hidden endreg=1.0"
  echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
  echo "    --name ${name} --lm_target faithful_even_blend --even_blend_scale 0.5 --pole_mode hidden \\"
  echo "    --prompts_file ${prompts} \\"
  echo "    --save_dir $ROOT/models/${name} \\"
  echo "    --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --endreg_weight 1.0 --device 0"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v21] SKIP train ${name} (weights exist)"
  else
    echo "[v21] ${name} GPU${gpu} prompts=$prompts lm_target=faithful_even_blend even_blend_scale=0.5"
    if ! train_lm_v21 "$gpu" "$name" "$prompts"; then
      echo "[v21] train failed ${name}"
      return 1
    fi
    if ! assert_v21_log "models/${name}/${name}_train.log" "$name"; then
      return 1
    fi
  fi
  if [ -f "$sidecar" ]; then
    if ! assert_v21_sidecar "$ROOT/$sidecar" "$name"; then
      return 1
    fi
  fi
  local gate="FAIL"
  echo "[v21] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v21] GATE OK ${name}"
    gate="OK"
  else
    echo "[v21] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if [ -f "$sidecar" ]; then
    append_metrics "$name" "$ROOT/$sidecar" "$gate" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v21] sample failed ${name}"
    return 1
  fi
  echo "[v21] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful_even_blend --even_blend_scale 0.5 --pole_mode hidden --endreg_weight 1.0" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
