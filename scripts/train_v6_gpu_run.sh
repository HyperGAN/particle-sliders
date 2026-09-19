#!/usr/bin/env bash
# Locked body for gpu_run: wait until GPU $1 has enough free memory, then exec.
set -euo pipefail
gpu="$1"
min_free="$2"
shift 2

free_mib() {
  nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print int($1)}'
}

apps() {
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader \
    | awk 'NF && $1+0 == $1 { n++ } END { print n+0 }'
}

while true; do
  free="$(free_mib)"
  n="$(apps)"
  if [ "$n" -eq 0 ] && [ "$free" -ge "$min_free" ]; then
    break
  fi
  echo "[gpu${gpu}] occupied (free=${free} MiB compute-apps=${n}); wait 30s" >&2
  sleep 30
done

echo "[gpu${gpu}] exec $*" >&2
export CUDA_VISIBLE_DEVICES="$gpu"
exec "$@"
