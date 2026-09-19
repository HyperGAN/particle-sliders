#!/usr/bin/env bash
# Kill the live GPU0 rapslow-slow lyric-hold job if the old queue still launches it.
set -u
ROOT=/ml2/music/sliders-conceptmod
WRAPPER=1258050
while kill -0 "$WRAPPER" 2>/dev/null; do
  while read -r pid; do
    [ -n "$pid" ] || continue
    comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
    case "$comm" in
      python|python3) kill "$pid" 2>/dev/null || true ;;
    esac
  done < <(pgrep -f '[p]ython .*--name rapslow-slow-lm-uni-lyric' || true)
  sleep 8
done
rm -rf "$ROOT/models/rapslow-slow-lm-uni-lyric" \
  "$ROOT/eval/listen/uni-lyric/rapslow-slow-lm-uni-lyric"
echo DONE
