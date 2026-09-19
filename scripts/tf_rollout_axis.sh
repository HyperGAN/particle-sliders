#!/usr/bin/env bash
# Roll one TF axis through the v10 derate-doctrine pipeline.
#   usage: tf_rollout_axis.sh <axis> <gpu>
# Produces models/<axis>-tf-v10/<axis>-tf-v10_derateXX.{safetensors,json}
# and eval/listen/v10-roll/<axis>-tf-v10-derate/ (20s ladder for ears).
set -euo pipefail
AXIS="$1"; GPU="$2"
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
TRAJ_FRAC="${TF_TRAJ_FRAC:-0.5}"   # closed-loop dose; 0.5 won the ship-strength metric
cd "$ROOT"
PROMPTS="$ROOT/conceptmod/textsliders/data/prompts-${AXIS}-tf-v7.yaml"
[[ -f "$PROMPTS" ]] || { echo "no pair file for $AXIS"; exit 1; }
SAVE="$ROOT/models/${AXIS}-tf-v10"

# 1. train (traj25 recipe, in-training gate at full range)
CUDA_VISIBLE_DEVICES="$GPU" "$PY" conceptmod/textsliders/train_lora_music3.py \
  --name "${AXIS}-tf-v10" --prompts_file "$PROMPTS" \
  --save_dir "$SAVE" --cache_dir "$ROOT/cache/${AXIS}-tf-v7" \
  --rank 8 --alpha 8 --lr 2e-3 --steps 500 --duration 4 --seed 7 --device 0 \
  --targets full --xt_mode anchor --bidirectional --target_mode axis \
  --traj_frac 0.25 --loss nmse --x0_per_row 8 --cond_seeds 7 --grad_clip norm \
  --render_gate_every 25 --render_gate_steps 20 --render_gate_scales=-2,-1,0,1,2

W_RAW="$SAVE/${AXIS}-tf-v10_alpha8.0_rank8_full_renderbest.safetensors"
[[ -f "$W_RAW" ]] || W_RAW="$SAVE/${AXIS}-tf-v10_alpha8.0_rank8_full_last.safetensors"

# 2. score renderbest raw
RAW_JSON="$ROOT/eval/gate/${AXIS}-tf-v10-renderbest.json"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/tf_gate_metric.py --weights "$W_RAW" \
  --prompts_file "$PROMPTS" --cache_dir "$ROOT/cache/${AXIS}-tf-v7" > "$RAW_JSON" 2>&1

# 3. solve alpha so user +/-2 stays inside caps (hottest <= 1.5), 4% margin
read -r HOTTEST _ <<< "$($PY - "$RAW_JSON" <<'PYEOF'
import json, sys
t = open(sys.argv[1]).read()
d = json.loads(t[t.find('{'):])
e = d['entries'][0]
hot = max(max(r['ratio_vs_zero'], 1/r['ratio_vs_zero']) for r in e['rows'])
print(f"{hot:.4f}")
PYEOF
)"
DERATE=$("$PY" -c "print(min(1.0, round(1.5 / ($HOTTEST * 1.04) , 3)))")
ALPHA=$("$PY" -c "print(round(8.0 * $DERATE, 2))")
echo "$AXIS: hottest_raw=$HOTTEST -> alpha=$ALPHA (derate $DERATE)"

# 4. emit ship weights + sidecar
SHIP_STEM="${AXIS}-tf-v10_derate$(echo "$DERATE" | tr -d '.')"
cp "$W_RAW" "$SAVE/${SHIP_STEM}.safetensors"
"$PY" - <<PYEOF
import json
src = "$SAVE/${AXIS}-tf-v10_alpha8.0_rank8_full_last.json"
m = json.load(open(src))
m["alpha"] = $ALPHA
m["unit_scale"] = 1.0
json.dump(m, open("$SAVE/${SHIP_STEM}.json", "w"), indent=2)
PYEOF

# 5. ritual ladder for ears
CUDA_VISIBLE_DEVICES="$GPU" "$PY" conceptmod/textsliders/generate_listen.py \
  --weights "$SAVE/${SHIP_STEM}.safetensors" --prompts_file "$PROMPTS" \
  --name "${AXIS}-tf-v10-ship" \
  --out_dir "$ROOT/eval/listen/v10-roll/${AXIS}-tf-v10-derate" \
  --scales=-2,-1,0,1,2 --duration 20 --seed 7 --device 0

# 6. confirm the shipped config gates clean at USER scales
"$PY" scripts/tf_render_gain.py --ladder_dir "$ROOT/eval/listen/v10-roll/${AXIS}-tf-v10-derate" \
  > "$ROOT/eval/gate/${AXIS}-tf-v10-ship-gain.txt"
echo "$AXIS rollout complete"
