#!/usr/bin/env bash
# After the plus UNI queues exit, train the minus UNI LoRAs on both GPUs.
set -uo pipefail
ROOT=/ml2/music/sliders-conceptmod
cd "$ROOT"
LOG="$ROOT/eval/listen/uni-v1/logs/minus-waiter.log"
mkdir -p "$ROOT/eval/listen/uni-v1/logs"

wait_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date -Is) waiting for plus_neu pid $pid" >>"$LOG"
    sleep 30
  done
}

PLUS0="${1:-}"
PLUS1="${2:-}"
if [ -n "$PLUS0" ]; then wait_pid "$PLUS0"; fi
if [ -n "$PLUS1" ]; then wait_pid "$PLUS1"; fi

echo "$(date -Is) plus UNI queues done; starting minus UNI" >>"$LOG"
"$ROOT/scripts/train_uni_v1_minus.sh" 0 energy gender tempo distortion breath rhyme triphop live \
  >"$ROOT/eval/listen/uni-v1/logs/gpu0-minus.log" 2>&1 &
echo $! >"$ROOT/eval/listen/uni-v1/logs/gpu0-minus.pid"
"$ROOT/scripts/train_uni_v1_minus.sh" 1 rapslow grit hurt joy sexy tender yearn \
  >"$ROOT/eval/listen/uni-v1/logs/gpu1-minus.log" 2>&1 &
echo $! >"$ROOT/eval/listen/uni-v1/logs/gpu1-minus.pid"
echo "$(date -Is) minus UNI launched gpu0=$(cat "$ROOT/eval/listen/uni-v1/logs/gpu0-minus.pid") gpu1=$(cat "$ROOT/eval/listen/uni-v1/logs/gpu1-minus.pid")" >>"$LOG"
wait || true
echo "$(date -Is) minus UNI queues finished" >>"$LOG"
