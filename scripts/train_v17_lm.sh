#!/usr/bin/env bash
# v15 recipe on current main: gender+live default v9 (no leak_*, hold_ê=0);
# leaky axes --lm_target pair_odd_sub_e (pair-odd − ê_⊥, hold_ê=0).
# Hidden MSE onto that teacher (--pole_mode hidden). Do not pass --hold_weight.
# Names are <axis>-lm-v17. Samples at eval/listen/v17/<axis>-lm-v17/.
#
#   ./scripts/train_v17_lm.sh 0 gender energy tempo distortion rapslow
#   ./scripts/train_v17_lm.sh 1 live breath rhyme triphop
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

CLEAN_AXES="gender live"
LEAKY_AXES="energy tempo distortion rapslow breath rhyme triphop"

is_clean() {
  case " $CLEAN_AXES " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

prompts_for() {
  local axis="$1"
  local v4="conceptmod/textsliders/data/prompts-${axis}-v4.yaml"
  if [ -f "$v4" ]; then
    echo "$v4"
  else
    return 1
  fi
}

assert_v17_log() {
  local log="$1" name="$2" clean="$3"
  if grep -q 'hold_ê=8' "$log"; then
    echo "[v17] FAIL ${name}: log has hold_ê=8 (silent λ=8 — forgot pair_odd_sub_e?)" >&2
    return 1
  fi
  if ! grep -q 'hold_ê=0' "$log"; then
    echo "[v17] FAIL ${name}: log did not say hold_ê=0" >&2
    tail -40 "$log" || true
    return 1
  fi
  if [ "$clean" = "1" ]; then
    if ! grep -q 'v9: teacher=full pair-odd, hold_ê=0' "$log"; then
      echo "[v17] FAIL ${name}: expected v9 hold_ê=0" >&2
      return 1
    fi
  else
    if ! grep -q 'pair_odd_sub_e: teacher=pair-odd − ê_⊥, hold_ê=0' "$log"; then
      echo "[v17] FAIL ${name}: expected pair_odd_sub_e hold_ê=0" >&2
      return 1
    fi
  fi
  if ! grep -q 'pole_mode=hidden' "$log"; then
    echo "[v17] FAIL ${name}: expected pole_mode=hidden (v15 MSE, not semantic_kl)" >&2
    return 1
  fi
  echo "[v17] hold_ê=0 pole_mode=hidden ok ${name}"
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-v17"
  local prompts out logs
  prompts="$(prompts_for "$axis")" || {
    echo "[v17] no v4 pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v17/${name}"
  logs="eval/listen/v17/logs"
  mkdir -p "$logs" "$out" "models/${name}"
  if is_clean "$axis"; then
    echo "[v17] flags ${name} GPU${gpu} recipe=v9 pole_mode=hidden (no leak_*, hold_ê=0)"
    echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
    echo "    --name ${name} --prompts_file ${prompts} --pole_mode hidden \\"
    echo "    --save_dir $ROOT/models/${name} \\"
    echo "    --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --device 0"
  else
    echo "[v17] flags ${name} GPU${gpu} recipe=pair_odd_sub_e pole_mode=hidden (hold_ê=0)"
    echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
    echo "    --name ${name} --lm_target pair_odd_sub_e --pole_mode hidden --prompts_file ${prompts} \\"
    echo "    --save_dir $ROOT/models/${name} \\"
    echo "    --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --device 0"
  fi
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v17] SKIP train ${name} (weights exist)"
  else
    if is_clean "$axis"; then
      echo "[v17] ${name} GPU${gpu} prompts=$prompts lm_target=v9 hold_ê=0"
      if ! train_lm_v9 "$gpu" "$name" "$prompts"; then
        echo "[v17] train failed ${name}"
        return 1
      fi
      if ! assert_v17_log "models/${name}/${name}_train.log" "$name" 1; then
        return 1
      fi
    else
      echo "[v17] ${name} GPU${gpu} prompts=$prompts lm_target=pair_odd_sub_e hold_ê=0"
      if ! train_lm_pair_odd_sub_e "$gpu" "$name" "$prompts"; then
        echo "[v17] train failed ${name}"
        return 1
      fi
      if ! assert_v17_log "models/${name}/${name}_train.log" "$name" 0; then
        return 1
      fi
    fi
  fi
  if [ -f "models/${name}/${name}_last.json" ]; then
    "$PY" - "$ROOT/models/${name}/${name}_last.json" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read())
last = d.get("last") or {}
print(
    f"[v17] metrics {d.get('name')} lm_target={d.get('lm_target')} "
    f"hold={d.get('hold_weight')} steps={d.get('steps')} "
    f"c+={last.get('cos_pos'):.3f} c-={last.get('cos_neg'):.3f} "
    f"col={last.get('collapse'):.3f} e={last.get('edrift_p'):.3f}/{last.get('edrift_n'):.3f} "
    f"loss={last.get('loss'):.4f}"
)
PY
  fi
  echo "[v17] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v17] GATE OK ${name}"
  else
    echo "[v17] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v17] sample failed ${name}"
    return 1
  fi
  echo "[v17] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  clean (v9): $CLEAN_AXES" >&2
  echo "  leaky (pair_odd_sub_e): $LEAKY_AXES" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
