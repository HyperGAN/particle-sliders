#!/usr/bin/env bash
# Retrain failed / lopsided LM halves with the v6 loss recipe and NO endreg.
# Writes new *-v6b checkpoints so a failed seed cannot wipe a shipped file.
#
# Gate: collapse < 0 and min(window cos+, cos-) >= 0.45.
# GPU: pass as $1 (default 1). Studio should not be on that card.
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

GPU="${1:-1}"
MIN_COS=0.45

lm_window_ok() {
  local json="$1"
  [ -n "${json:-}" ] && [ -f "$json" ] || return 1
  python3 - "$json" "$MIN_COS" <<'PY'
import json, sys
path, floor = sys.argv[1], float(sys.argv[2])
d = json.loads(open(path).read())
win = ((d.get("early_stop") or {}).get("metrics")) or (d.get("last") or {})
col = win.get("collapse")
cp = win.get("cos_pos")
cn = win.get("cos_neg")
if col is None or cp is None or cn is None:
    sys.exit(1)
print(f"collapse={col:.3f} cos+={cp:.3f} cos-={cn:.3f}")
ok = col < 0 and cp >= floor and cn >= floor
sys.exit(0 if ok else 2)
PY
}

train_lm_strict() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  local seeds=(7 17 27)
  mkdir -p "$save"
  for seed in "${seeds[@]}"; do
    echo "[gpu${gpu}] START ${name} seed=${seed}"
    rm -f "$save"/*.safetensors "$save"/*_last.json
    if gpu_run "$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
        --name "$name" \
        --prompts_file "$prompts" \
        --save_dir "$ROOT/$save" \
        --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed "$seed" --device 0 \
        --save_every 0 \
        --target_mode faithful \
        --pole_mode semantic_kl --pole_temperature 1 \
        --planreg_mode semantic_kl --planreg_weight 0.3 --planreg_temperature 1 \
        --collapse_weight 1 --collapse_cos 0 \
        --endreg_weight 0 --no-early_stop \
        > "$save/${name}_train.log" 2>&1; then
      if lm_window_ok "$(lm_last_json "$save" "$name")"; then
        echo "[gpu${gpu}] DONE ${name} seed=${seed}"
        return 0
      fi
      echo "[gpu${gpu}] WEAK ${name} seed=${seed} — retry"
    else
      echo "[gpu${gpu}] FAIL ${name} seed=${seed}"
      tail -30 "$save/${name}_train.log" || true
    fi
  done
  echo "[gpu${gpu}] GIVE UP ${name}"
  return 1
}

echo "[retrain-weak-lm] GPU ${GPU}, no endreg, min cos ${MIN_COS}"
fail=0
train_lm_strict "$GPU" yearn-lm-v6b conceptmod/textsliders/data/prompts-yearn-v6.yaml || fail=1
train_lm_strict "$GPU" hurt-lm-v6b conceptmod/textsliders/data/prompts-hurt-v6.yaml || fail=1
train_lm_strict "$GPU" live-lm-v6b conceptmod/textsliders/data/prompts-live-v6.yaml || fail=1
train_lm_strict "$GPU" breath-lm-v6b conceptmod/textsliders/data/prompts-breath-v6.yaml || fail=1
train_lm_strict "$GPU" tender-lm-v6b conceptmod/textsliders/data/prompts-tender-v6.yaml || fail=1
train_lm_strict "$GPU" grit-lm-v6b conceptmod/textsliders/data/prompts-grit-v6.yaml || fail=1
echo "[retrain-weak-lm] finished fail=${fail}"
exit "$fail"
