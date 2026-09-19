#!/usr/bin/env bash
# UNI lyric-hold for plus AND minus poles. Same loss as uni-lyric:
#   --lm_target faithful_plus_neu_lyric --pole_mode hidden
# Teacher: +1 last → raw h+; +1 lyric tokens → encode(neu) lyrics;
# Vocal Details stay free; scale 0 → h0. No minus MSE. Empty lyrics fail closed.
#
# Tokens:
#   <axis>        plus pole. yaml: uni-v2 if present else uni-v1.
#                 name: <axis>-lm-uni-lyric
#   <axis>-minus  minus pole. yaml: prompts-<axis>-minus-uni-v1.yaml
#                 name: <axis>-<slug>-lm-uni-lyric  (slug = minus plus_label)
#
# Listen: yaml neu caption + yaml lyrics, LoRA 0,1,2. REF is + caption off.
# Exam is +1. Out: eval/listen/uni-lyric/<name>/
#
#   ./scripts/train_uni_lyric_poles.sh 0 breath rhyme triphop live \
#       grit-minus distortion-minus joy-minus energy-minus hurt-minus rapslow-minus gender-minus
#   ./scripts/train_uni_lyric_poles.sh 1 sexy tender yearn tempo-minus \
#       breath-minus rhyme-minus triphop-minus live-minus sexy-minus tender-minus yearn-minus
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck source=train_v6_lib.sh
. "$ROOT/scripts/train_v6_lib.sh"

# minus yaml plus_label → name slug. Axis prefix keeps breath-clean ≠ distortion-clean.
declare -A MINUS_SLUG=(
  [energy]=quiet
  [gender]=male
  [tempo]=slow
  [distortion]=clean
  [breath]=clean
  [rhyme]=prose
  [triphop]=pop
  [live]=studio
  [rapslow]=slow
  [grit]=smooth
  [hurt]=numb
  [joy]=somber
  [sexy]=plain
  [tender]=fierce
  [yearn]=settled
)

PLUS_AXES="breath rhyme triphop live sexy tender yearn grit distortion joy energy hurt rapslow gender tempo"

# Name collisions on the minus pole: keep the stronger of each pair.
# Clean: distortion-clean (guitar tone) beat breath-clean (airless, c+ ~0.02).
# Slow: tempo-slow (BPM) beats rapslow-slow (held sung notes, leftover of rap).
SKIP_TOKENS="breath-minus rapslow-minus"

parse_token() {
  # sets: AXIS POLE NAME
  local tok="$1"
  AXIS=""
  POLE=""
  NAME=""
  case "$tok" in
    *-minus)
      AXIS="${tok%-minus}"
      POLE=minus
      ;;
    minus:*)
      AXIS="${tok#minus:}"
      POLE=minus
      ;;
    plus:*)
      AXIS="${tok#plus:}"
      POLE=plus
      ;;
    *)
      AXIS="$tok"
      POLE=plus
      ;;
  esac
  case "$AXIS" in
    ""|*/*|*..)
      echo "[uni-lyric] FAIL bad axis token $tok" >&2
      return 1
      ;;
  esac
  if [ "$POLE" = plus ]; then
    NAME="${AXIS}-lm-uni-lyric"
  else
    local slug="${MINUS_SLUG[$AXIS]:-}"
    if [ -z "$slug" ]; then
      echo "[uni-lyric] FAIL unknown minus axis $AXIS" >&2
      return 1
    fi
    NAME="${AXIS}-${slug}-lm-uni-lyric"
  fi
}

prompts_for() {
  local axis="$1" pole="$2"
  local uni
  if [ "$pole" = minus ]; then
    uni="conceptmod/textsliders/data/prompts-${axis}-minus-uni-v1.yaml"
  else
    uni="conceptmod/textsliders/data/prompts-${axis}-uni-v2.yaml"
    if [ ! -f "$uni" ]; then
      uni="conceptmod/textsliders/data/prompts-${axis}-uni-v1.yaml"
    fi
  fi
  if [ ! -f "$uni" ]; then
    echo "[uni-lyric] FAIL ${axis} ${pole}: missing $uni" >&2
    return 1
  fi
  case "$uni" in
    *v4.yaml)
      echo "[uni-lyric] FAIL ${axis}: refused $uni" >&2
      return 1
      ;;
  esac
  if [ "$pole" = plus ]; then
    case "$uni" in
      *minus-uni*)
        echo "[uni-lyric] FAIL ${axis}: plus queue got minus yaml $uni" >&2
        return 1
        ;;
    esac
  else
    case "$uni" in
      *minus-uni*) ;;
      *)
        echo "[uni-lyric] FAIL ${axis}: minus queue expected minus-uni yaml, got $uni" >&2
        return 1
        ;;
    esac
  fi
  "$PY" - "$ROOT/$uni" "$axis" "$pole" <<'PY' || return 1
import sys, yaml
path, axis, pole = sys.argv[1], sys.argv[2], sys.argv[3]
raw = yaml.safe_load(open(path, encoding="utf-8"))
if raw.get("leak_positive") or raw.get("leak_negative"):
    raise SystemExit(f"[uni-lyric] FAIL {axis} {pole}: yaml declares leak_*")
rows = raw.get("rows") or []
if not rows:
    raise SystemExit(f"[uni-lyric] FAIL {axis} {pole}: empty rows")
for i, row in enumerate(rows):
    lyrics = str((row or {}).get("lyrics") or "").strip()
    if not lyrics:
        raise SystemExit(f"[uni-lyric] FAIL {axis} {pole} row {i}: empty lyrics")
    if not str((row or {}).get("target") or "").strip():
        raise SystemExit(f"[uni-lyric] FAIL {axis} {pole} row {i}: missing target")
    if not str((row or {}).get("neutral") or "").strip():
        raise SystemExit(f"[uni-lyric] FAIL {axis} {pole} row {i}: missing neutral")
    if not str((row or {}).get("positive") or "").strip():
        raise SystemExit(f"[uni-lyric] FAIL {axis} {pole} row {i}: missing positive")
print(
    f"[uni-lyric] yaml {axis} {pole} rows={len(rows)} plus_label={raw.get('plus_label')!r} "
    f"lyrics=ok leak_*=omitted",
    file=sys.stderr,
)
PY
  echo "$uni"
}

train_one() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=faithful_plus_neu_lyric pole_mode=hidden endreg=1.0"
  echo "  prompts=$prompts"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target faithful_plus_neu_lyric --pole_mode hidden \
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
  local sidecar="$1" name="$2" prompts="$3" pole="$4"
  "$PY" - "$sidecar" "$name" "$prompts" "$pole" <<'PY'
import json, sys
path, name, prompts, pole = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.loads(open(path).read())
target = d.get("lm_target")
pole_mode = d.get("pole_mode")
plus_neu = d.get("plus_neu")
plus_neu_lyric = d.get("plus_neu_lyric")
plus_neu_prefix = d.get("plus_neu_prefix")
used = d.get("prompts_file") or d.get("prompts") or ""
if target != "faithful_plus_neu_lyric":
    raise SystemExit(f"[uni-lyric] FAIL {name}: sidecar lm_target={target!r}")
if pole_mode != "hidden":
    raise SystemExit(f"[uni-lyric] FAIL {name}: sidecar pole_mode={pole_mode!r}")
if plus_neu is not True:
    raise SystemExit(f"[uni-lyric] FAIL {name}: plus_neu={plus_neu!r}")
if plus_neu_lyric is not True:
    raise SystemExit(f"[uni-lyric] FAIL {name}: plus_neu_lyric={plus_neu_lyric!r}")
if plus_neu_prefix:
    raise SystemExit(f"[uni-lyric] FAIL {name}: plus_neu_prefix={plus_neu_prefix!r}")
blob = f"{used} {prompts}"
if "v4.yaml" in blob:
    raise SystemExit(f"[uni-lyric] FAIL {name}: still v4 {used!r} {prompts!r}")
if pole == "minus" and "minus-uni" not in prompts:
    raise SystemExit(f"[uni-lyric] FAIL {name}: minus expected minus-uni yaml, got {prompts!r}")
if pole == "plus" and "minus-uni" in prompts:
    raise SystemExit(f"[uni-lyric] FAIL {name}: plus got minus yaml {prompts!r}")
if pole == "plus" and "uni-v1.yaml" not in prompts and "uni-v2.yaml" not in prompts:
    raise SystemExit(f"[uni-lyric] FAIL {name}: plus expected uni-v1/v2 yaml, got {prompts!r}")
if d.get("leak_positive") or d.get("leak_negative"):
    raise SystemExit(f"[uni-lyric] FAIL {name}: leak_* must be omitted")
print(
    f"[uni-lyric] sidecar {name} lm_target={target} plus_neu_lyric={plus_neu_lyric} "
    f"pole={pole} prompts={prompts}"
)
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
  local gpu="$1" tok="$2"
  local AXIS POLE NAME prompts out sidecar
  case " $SKIP_TOKENS " in
    *" $tok "*)
      echo "[uni-lyric] SKIP $tok (weaker name-collision minus)"
      return 0
      ;;
  esac
  parse_token "$tok" || return 1
  prompts="$(prompts_for "$AXIS" "$POLE")" || return 1
  out="eval/listen/uni-lyric/${NAME}"
  sidecar="models/${NAME}/${NAME}_last.json"
  mkdir -p "eval/listen/uni-lyric/logs" "$out" "models/${NAME}"
  echo "[uni-lyric] ${NAME} GPU${gpu} pole=${POLE} prompts=$prompts lm_target=faithful_plus_neu_lyric"
  if [ -f "$sidecar" ] && [ -f "models/${NAME}/${NAME}_last.safetensors" ]; then
    if assert_sidecar "$ROOT/$sidecar" "$NAME" "$prompts" "$POLE"; then
      echo "[uni-lyric] SKIP train ${NAME} (faithful_plus_neu_lyric weights exist)"
    else
      echo "[uni-lyric] FAIL ${NAME}: leftover non-lyric weights; not skipping" >&2
      return 1
    fi
  else
    if ! train_one "$gpu" "$NAME" "$prompts"; then
      return 1
    fi
    if [ -f "$sidecar" ] && ! assert_sidecar "$ROOT/$sidecar" "$NAME" "$prompts" "$POLE"; then
      return 1
    fi
  fi
  if ! sample_plus "$gpu" "$NAME" "$prompts" "$out"; then
    echo "[uni-lyric] sample failed ${NAME}"
    return 1
  fi
  echo "[uni-lyric] done ${NAME} -> $out"
}

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu> <axis|axis-minus> [token...]" >&2
  echo "  plus catalog: $PLUS_AXES" >&2
  echo "  minus token: <axis>-minus" >&2
  echo "  recipe: --lm_target faithful_plus_neu_lyric --pole_mode hidden" >&2
  exit 2
fi
GPU="$1"
shift
fail=0
for tok in "$@"; do
  run_one "$GPU" "$tok" || fail=1
done
exit "$fail"
