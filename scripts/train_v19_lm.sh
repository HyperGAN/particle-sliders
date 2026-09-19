#!/usr/bin/env bash
# v19: hidden MSE onto the raw caption poles
# --lm_target symmetric --common_beta 1 --pole_mode hidden
# (docs/lm-sheet-goodhart.md: β=1 is t± = h±; v9 ignores --common_beta).
# Names are <axis>-lm-v19. Samples at eval/listen/v19/<axis>-lm-v19/.
# After each train: sidecar metrics + check_slider_gate.py, then
# generate_listen at scales -2,-1,0,1,2 (plus REFs).
#
#   ./scripts/train_v19_lm.sh 0 gender energy tempo distortion rapslow
#   ./scripts/train_v19_lm.sh 1 live breath rhyme triphop
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CATALOG="gender energy tempo distortion rapslow live breath rhyme triphop"

prompts_for() {
  local axis="$1"
  local v4="conceptmod/textsliders/data/prompts-${axis}-v4.yaml"
  if [ -f "$v4" ]; then
    echo "$v4"
  else
    return 1
  fi
}

assert_v19_log() {
  local log="$1" name="$2"
  if grep -q 'ignoring --common_beta' "$log"; then
    echo "[v19] FAIL ${name}: trainer ignored --common_beta (wrong lm_target?)" >&2
    return 1
  fi
  if ! grep -q 'pole_mode=hidden' "$log"; then
    echo "[v19] FAIL ${name}: expected pole_mode=hidden" >&2
    tail -40 "$log" || true
    return 1
  fi
  echo "[v19] pole_mode=hidden common_beta=1 ok ${name}"
}

append_metrics() {
  local name="$1" sidecar="$2" gate_status="$3"
  local metrics="eval/listen/v19/metrics.tsv"
  mkdir -p "eval/listen/v19"
  if [ ! -f "$metrics" ]; then
    printf 'name\tlm_target\tcommon_beta\tpole_mode\tsteps\tcos_pos\tcos_neg\tcollapse\tedrift_p\tedrift_n\tloss\tgate\n' > "$metrics"
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
    str(d.get("common_beta", "")),
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
    f"[v19] metrics {name} lm_target={d.get('lm_target')} "
    f"beta={d.get('common_beta')} pole={d.get('pole_mode')} "
    f"steps={d.get('steps')} "
    f"c+={last.get('cos_pos'):.3f} c-={last.get('cos_neg'):.3f} "
    f"col={last.get('collapse'):.3f} e={last.get('edrift_p'):.3f}/{last.get('edrift_n'):.3f} "
    f"loss={last.get('loss'):.4f} gate={gate}"
)
PY
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-v19"
  local prompts out logs sidecar
  prompts="$(prompts_for "$axis")" || {
    echo "[v19] no v4 pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v19/${name}"
  logs="eval/listen/v19/logs"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "$logs" "$out" "models/${name}"
  echo "[v19] flags ${name} GPU${gpu} recipe=symmetric common_beta=1 pole_mode=hidden"
  echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
  echo "    --name ${name} --lm_target symmetric --common_beta 1 --pole_mode hidden \\"
  echo "    --prompts_file ${prompts} \\"
  echo "    --save_dir $ROOT/models/${name} \\"
  echo "    --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --device 0"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v19] SKIP train ${name} (weights exist)"
  else
    echo "[v19] ${name} GPU${gpu} prompts=$prompts lm_target=symmetric common_beta=1"
    if ! train_lm_symmetric_beta1 "$gpu" "$name" "$prompts"; then
      echo "[v19] train failed ${name}"
      return 1
    fi
    if ! assert_v19_log "models/${name}/${name}_train.log" "$name"; then
      return 1
    fi
  fi
  local gate="FAIL"
  echo "[v19] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v19] GATE OK ${name}"
    gate="OK"
  else
    echo "[v19] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if [ -f "$sidecar" ]; then
    append_metrics "$name" "$ROOT/$sidecar" "$gate" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v19] sample failed ${name}"
    return 1
  fi
  echo "[v19] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target symmetric --common_beta 1 --pole_mode hidden" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
