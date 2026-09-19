#!/usr/bin/env bash
# v22: leftover-gated real poles + dual-band pole loss
# --lm_target faithful_sub_e_if_unused --pole_mode dual_band --blind_cut 0.012
# Same leftover gate as v20 (subtract ê only when |ê̂_⊥ · â| < 0.50);
# pole loss is semantic-band KL plus hidden MSE on the blind band.
# Default --blind_cut 0 is the fixture-head exact-null setting. Live Music 3
# has no exact null (0 dims). Cut 0.05 already covers ~90% of the 4096-wide
# hidden — that is full hidden MSE, not a band. Live SVD: 0.012 → ~48 dims.
# Names are <axis>-lm-v22. Samples at eval/listen/v22/<axis>-lm-v22/.
# After each train: sidecar metrics + check_slider_gate.py, then
# generate_listen at scales -2,-1,0,1,2 (plus REFs).
# Leaky yamls only (gender/live have no leak_*; leftover-gate is a no-op).
#
#   ./scripts/train_v22_lm.sh 0 energy tempo distortion rapslow
#   ./scripts/train_v22_lm.sh 1 breath rhyme triphop
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

BLIND_CUT=0.012
# Fail closed if the live projector is empty or eats most of hidden (4096).
BLIND_DIMS_MIN=8
BLIND_DIMS_MAX=512
CATALOG="energy tempo distortion rapslow breath rhyme triphop"

prompts_for() {
  local axis="$1"
  local v4="conceptmod/textsliders/data/prompts-${axis}-v4.yaml"
  if [ -f "$v4" ]; then
    echo "$v4"
  else
    return 1
  fi
}

train_lm_v22() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_sub_e_if_unused pole_mode=dual_band blind_cut=${BLIND_CUT} endreg=1.0"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_sub_e_if_unused --pole_mode dual_band \
      --blind_cut "$BLIND_CUT" \
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

assert_v22_log() {
  local log="$1" name="$2"
  if ! grep -q 'faithful_sub_e_if_unused: subtract leftover' "$log"; then
    echo "[v22] FAIL ${name}: expected leftover-gate teacher banner" >&2
    tail -40 "$log" || true
    return 1
  fi
  if ! grep -q 'pole_mode=dual_band' "$log"; then
    echo "[v22] FAIL ${name}: expected pole_mode=dual_band" >&2
    tail -40 "$log" || true
    return 1
  fi
  if grep -q 'this is exactly pole_mode=semantic_kl' "$log"; then
    echo "[v22] FAIL ${name}: blind band empty (semantic_kl in disguise). Raise --blind_cut." >&2
    grep -E 'pole_mode=dual_band|WARNING' "$log" || true
    return 1
  fi
  local dims
  dims="$(sed -n 's/.*on the \([0-9][0-9]*\) blind dims.*/\1/p' "$log" | head -1)"
  if [ -z "${dims:-}" ]; then
    echo "[v22] FAIL ${name}: no 'N blind dims' line" >&2
    return 1
  fi
  if [ "$dims" -lt "$BLIND_DIMS_MIN" ] || [ "$dims" -gt "$BLIND_DIMS_MAX" ]; then
    echo "[v22] FAIL ${name}: blind_dims=${dims} not a band (want ${BLIND_DIMS_MIN}..${BLIND_DIMS_MAX})" >&2
    grep -E 'pole_mode=dual_band|WARNING' "$log" || true
    return 1
  fi
  echo "[v22] leftover-gate pole_mode=dual_band cut=${BLIND_CUT} blind_dims=${dims} ok ${name}"
}

assert_v22_sidecar() {
  local sidecar="$1" name="$2"
  "$PY" - "$sidecar" "$name" "$BLIND_CUT" "$BLIND_DIMS_MIN" "$BLIND_DIMS_MAX" <<'PY'
import json, sys
path, name, want_cut, dmin, dmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = json.loads(open(path).read())
target = d.get("lm_target")
pole = d.get("pole_mode")
cut = d.get("blind_cut")
dims = d.get("blind_dims")
if target != "faithful_sub_e_if_unused":
    raise SystemExit(f"[v22] FAIL {name}: sidecar lm_target={target!r}")
if pole != "dual_band":
    raise SystemExit(f"[v22] FAIL {name}: sidecar pole_mode={pole!r}")
if cut is None or abs(float(cut) - want_cut) > 1e-9:
    raise SystemExit(f"[v22] FAIL {name}: sidecar blind_cut={cut!r} (want {want_cut})")
if dims is None or not (dmin <= int(dims) <= dmax):
    raise SystemExit(f"[v22] FAIL {name}: sidecar blind_dims={dims!r} not in {dmin}..{dmax}")
print(
    f"[v22] sidecar {name} lm_target={target} pole_mode={pole} "
    f"blind_weight={d.get('blind_weight')} blind_cut={cut} "
    f"blind_dims={dims}"
)
PY
}

append_metrics() {
  local name="$1" sidecar="$2" gate_status="$3"
  local metrics="eval/listen/v22/metrics.tsv"
  mkdir -p "eval/listen/v22"
  if [ ! -f "$metrics" ]; then
    printf 'name\tlm_target\tpole_mode\tblind_weight\tblind_cut\tblind_dims\tsteps\tcos_pos\tcos_neg\tcollapse\tedrift_p\tedrift_n\tloss\tgate\n' > "$metrics"
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
    str(d.get("pole_mode", "")),
    str(d.get("blind_weight", "")),
    str(d.get("blind_cut", "")),
    str(d.get("blind_dims", "")),
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
    f"[v22] metrics {name} lm_target={d.get('lm_target')} "
    f"pole={d.get('pole_mode')} "
    f"blind_w={d.get('blind_weight')} cut={d.get('blind_cut')} dims={d.get('blind_dims')} "
    f"steps={d.get('steps')} "
    f"c+={last.get('cos_pos'):.3f} c-={last.get('cos_neg'):.3f} "
    f"col={last.get('collapse'):.3f} e={last.get('edrift_p'):.3f}/{last.get('edrift_n'):.3f} "
    f"loss={last.get('loss'):.4f} gate={gate}"
)
PY
}

run_one() {
  local gpu="$1" axis="$2"
  local name="${axis}-lm-v22"
  local prompts out logs sidecar
  prompts="$(prompts_for "$axis")" || {
    echo "[v22] no v4 pair file for $axis" >&2
    return 1
  }
  out="eval/listen/v22/${name}"
  logs="eval/listen/v22/logs"
  sidecar="models/${name}/${name}_last.json"
  mkdir -p "$logs" "$out" "models/${name}"
  echo "[v22] flags ${name} GPU${gpu} recipe=faithful_sub_e_if_unused pole_mode=dual_band blind_cut=${BLIND_CUT} endreg=1.0"
  echo "  $PY -u conceptmod/textsliders/train_lm_slider_music3.py \\"
  echo "    --name ${name} --lm_target faithful_sub_e_if_unused --pole_mode dual_band --blind_cut ${BLIND_CUT} \\"
  echo "    --prompts_file ${prompts} \\"
  echo "    --save_dir $ROOT/models/${name} \\"
  echo "    --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop --endreg_weight 1.0 --device 0"
  if [ -f "models/${name}/${name}_last.safetensors" ]; then
    echo "[v22] SKIP train ${name} (weights exist)"
  else
    echo "[v22] ${name} GPU${gpu} prompts=$prompts lm_target=faithful_sub_e_if_unused pole_mode=dual_band blind_cut=${BLIND_CUT}"
    if ! train_lm_v22 "$gpu" "$name" "$prompts"; then
      echo "[v22] train failed ${name}"
      return 1
    fi
  fi
  if ! assert_v22_log "models/${name}/${name}_train.log" "$name"; then
    return 1
  fi
  if [ -f "$sidecar" ]; then
    if ! assert_v22_sidecar "$ROOT/$sidecar" "$name"; then
      return 1
    fi
  fi
  local gate="FAIL"
  echo "[v22] gate ${name}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_slider_gate.py \
      "$ROOT/models/${name}" \
      --leakage_prompts "$ROOT/$prompts" --leakage_row 0 --device 0 \
      > "$logs/${name}-gate.log" 2>&1; then
    echo "[v22] GATE OK ${name}"
    gate="OK"
  else
    echo "[v22] GATE FAIL ${name} (see $logs/${name}-gate.log) — still sampling"
    tail -20 "$logs/${name}-gate.log" || true
  fi
  if [ -f "$sidecar" ]; then
    append_metrics "$name" "$ROOT/$sidecar" "$gate" || true
  fi
  if ! sample_lm "$gpu" "$name" "$prompts" "$out"; then
    echo "[v22] sample failed ${name}"
    return 1
  fi
  echo "[v22] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis> [axis...]" >&2
  echo "  catalog: $CATALOG" >&2
  echo "  recipe: --lm_target faithful_sub_e_if_unused --pole_mode dual_band --blind_cut ${BLIND_CUT} --endreg_weight 1.0" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for axis in "$@"; do
  run_one "$GPU" "$axis" || fail=1
done
exit "$fail"
