#!/usr/bin/env bash
# Adv uni-lyric campaign: the train_uni_lyric.sh loop plus the training-only
# ParticleGAN-style head (--adv_weight 0.01, RpGAN logistic + b_cap D).
# Same UNI lyric-hold recipe otherwise:
# --lm_target faithful_plus_neu_lyric --pole_mode hidden, rank 8, 800 steps,
# seed 7, endreg 1.0, no early stop. Checkpoints stay LoRA-only; D is
# discarded in-loop (sidecar "adv" block records the hyperparameters).
#
# New dirs (shipped catalog untouched):
#   models/adv-uni-v1/<name>/  +  eval/listen/adv-uni-v1/<name>/
# After each train: generate_listen 0,1,2 (plus the + REF), then next axis.
#
# Both GPUs (14 + 14):
#   ./scripts/train_uni_lyric_adv.sh 0 <names...>
#   ./scripts/train_uni_lyric_adv.sh 1 <names...>
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

MODEL_ROOT="models/adv-uni-v1"
LISTEN_ROOT="eval/listen/adv-uni-v1"
ADV_WEIGHT="0.01"

# Full uni-lyric catalog: run name -> prompts file (mirrors shipped sidecars).
prompts_for() {
  case "$1" in
    breath-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-breath-uni-v1.yaml" ;;
    distortion-clean-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-distortion-minus-uni-v1.yaml" ;;
    distortion-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-distortion-uni-v2.yaml" ;;
    energy-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-energy-uni-v2.yaml" ;;
    energy-quiet-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-energy-minus-uni-v1.yaml" ;;
    gender-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-gender-uni-v2.yaml" ;;
    gender-male-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-gender-minus-uni-v1.yaml" ;;
    grit-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-grit-uni-v2.yaml" ;;
    grit-smooth-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-grit-minus-uni-v1.yaml" ;;
    hurt-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-hurt-uni-v2.yaml" ;;
    hurt-numb-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-hurt-minus-uni-v1.yaml" ;;
    joy-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-joy-uni-v2.yaml" ;;
    joy-somber-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-joy-minus-uni-v1.yaml" ;;
    live-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-live-uni-v1.yaml" ;;
    live-studio-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-live-minus-uni-v1.yaml" ;;
    rapslow-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-rapslow-uni-v2.yaml" ;;
    rhyme-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-rhyme-uni-v1.yaml" ;;
    rhyme-prose-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-rhyme-minus-uni-v1.yaml" ;;
    sexy-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-sexy-uni-v1.yaml" ;;
    sexy-plain-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-sexy-minus-uni-v1.yaml" ;;
    tempo-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-tempo-uni-v2.yaml" ;;
    tempo-slow-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-tempo-minus-uni-v1.yaml" ;;
    tender-fierce-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-tender-minus-uni-v1.yaml" ;;
    tender-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-tender-uni-v1.yaml" ;;
    triphop-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-triphop-uni-v1.yaml" ;;
    triphop-pop-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-triphop-minus-uni-v1.yaml" ;;
    yearn-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-yearn-uni-v1.yaml" ;;
    yearn-settled-lm-uni-lyric) echo "conceptmod/textsliders/data/prompts-yearn-minus-uni-v1.yaml" ;;
    *) echo "[adv-uni] FAIL: unknown run name $1" >&2; return 1 ;;
  esac
}

train_one() {
  local gpu="$1" name="$2" prompts="$3"
  local save="${MODEL_ROOT}/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lyric-hold + adv ${ADV_WEIGHT}"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_plus_neu_lyric --pole_mode hidden \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \
      --no-early_stop --endreg_weight 1.0 \
      --adv_weight "$ADV_WEIGHT" \
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
  ADV_WANT="$ADV_WEIGHT" "$PY" - "$sidecar" "$name" <<'PY'
import json, os, sys
path, name = sys.argv[1], sys.argv[2]
d = json.loads(open(path).read())
want = float(os.environ["ADV_WANT"])
checks = {
    "lm_target": (d.get("lm_target"), "faithful_plus_neu_lyric"),
    "pole_mode": (d.get("pole_mode"), "hidden"),
    "plus_neu": (d.get("plus_neu"), True),
    "plus_neu_lyric": (d.get("plus_neu_lyric"), True),
}
for key, (got, want_v) in checks.items():
    if got != want_v:
        raise SystemExit(f"[adv-uni] FAIL {name}: sidecar {key}={got!r} want {want_v!r}")
if d.get("plus_neu_prefix"):
    raise SystemExit(f"[adv-uni] FAIL {name}: prefix-hold, want lyric-hold")
adv = d.get("adv") or {}
if adv.get("enabled") is not True or abs(float(adv.get("weight", -1)) - want) > 1e-12:
    raise SystemExit(f"[adv-uni] FAIL {name}: sidecar adv={adv!r} want enabled/weight={want}")
print(f"[adv-uni] sidecar {name} lyric-hold + adv {want} ok")
PY
}

sample_plus() {
  local gpu="$1" name="$2" prompts="$3" out_dir="$4"
  local save="${MODEL_ROOT}/${name}"
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
  local gpu="$1" name="$2"
  local prompts out sidecar
  prompts="$(prompts_for "$name")" || return 1
  [ -f "$ROOT/$prompts" ] || { echo "[adv-uni] FAIL ${name}: missing $prompts"; return 1; }
  out="${LISTEN_ROOT}/${name}"
  sidecar="${MODEL_ROOT}/${name}/${name}_last.json"
  mkdir -p "${LISTEN_ROOT}/logs" "$out" "${MODEL_ROOT}/${name}"
  echo "[adv-uni] ${name} GPU${gpu} prompts=$prompts"
  if [ -f "$sidecar" ] && [ -f "${MODEL_ROOT}/${name}/${name}_last.safetensors" ]; then
    if assert_sidecar "$ROOT/$sidecar" "$name"; then
      echo "[adv-uni] SKIP train ${name} (adv weights exist)"
    else
      echo "[adv-uni] FAIL ${name}: leftover non-adv weights; not skipping" >&2
      return 1
    fi
  else
    train_one "$gpu" "$name" "$prompts" || return 1
    assert_sidecar "$ROOT/$sidecar" "$name" || return 1
  fi
  if ! sample_plus "$gpu" "$name" "$prompts" "$out"; then
    echo "[adv-uni] sample failed ${name}"
    return 1
  fi
  echo "[adv-uni] done ${name} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <run-name> [run-name...]" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for name in "$@"; do
  run_one "$GPU" "$name" || fail=1
done
exit "$fail"
