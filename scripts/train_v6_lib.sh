# Shared GPU exclusivity for train_v6_*.sh. Source after ROOT is set.
#
# One trainer python per GPU. flock serializes our scripts; the inner helper
# also waits until the card is empty of other CUDA apps (the studio parks a
# full MiniMax on both devices — stacking a trainer on top OOMs).

_gpu_lock_dir="${ROOT}/cache/gpu-locks"
mkdir -p "$_gpu_lock_dir"

# MiB free we insist on before launching. MiniMax LM + LoRA + endreg is ~22 GiB;
# TF + x0 cache is similar. A parked studio copy is ~25 GiB, so this refuses to
# share a card with the desk.
_GPU_FREE_MIN_MIB=32000

gpu_run() {
  local gpu="$1"
  shift
  local lock="$_gpu_lock_dir/gpu${gpu}.lock"
  echo "[gpu${gpu}] acquire $lock"
  flock "$lock" "$ROOT/scripts/train_v6_gpu_run.sh" "$gpu" "$_GPU_FREE_MIN_MIB" "$@"
}

# VRAM-only wait. gpu_run also demands zero compute-apps, which parks forever
# when nano-work-server (or a desk that failed to take CUDA) holds a sliver.
wait_vram() {
  local gpu="$1"
  local min_mib="${2:-$_GPU_FREE_MIN_MIB}"
  while true; do
    local free
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
      | awk '{print int($1)}')"
    if [ "$free" -ge "$min_mib" ]; then
      return 0
    fi
    echo "[gpu${gpu}] wait vram free=${free} MiB (need ${min_mib})" >&2
    sleep 30
  done
}

# TF trainer writes <name>_alpha*_rank*_full_last.json, not <name>_last.json.
# Using the wrong path made collapse_ok fail-open as COLLAPSED after a good run.
lm_last_json() {
  echo "$1/${2}_last.json"
}

tf_last_json() {
  local save="$1" name="$2"
  local f="$save/${name}_last.json"
  if [ -f "$f" ]; then
    echo "$f"
    return
  fi
  ls -1t "$save"/${name}_alpha*_last.json 2>/dev/null \
    | grep -v unit_last | grep -v comfyui | head -1 || true
}

collapse_ok() {
  local json="$1" kind="$2"
  [ -n "${json:-}" ] && [ -f "$json" ] || return 1
  python3 - "$json" "$kind" <<'PY'
import json, sys
path, kind = sys.argv[1], sys.argv[2]
try:
    d = json.loads(open(path).read())
except Exception:
    sys.exit(1)
if kind == "lm":
    col = (d.get("last") or {}).get("collapse")
else:
    col = (d.get("eval") or {}).get("collapse")
if col is None:
    sys.exit(1)
print(f"collapse={col:.3f}")
sys.exit(0 if col < 0 else 2)
PY
}

already_good() {
  local save="$1" name="$2" kind="$3"
  local json
  if [ "$kind" = lm ]; then
    json="$(lm_last_json "$save" "$name")"
  else
    json="$(tf_last_json "$save" "$name")"
  fi
  collapse_ok "$json" "$kind"
}

train_lm() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  local seeds=(7 17 27)
  mkdir -p "$save"
  if already_good "$save" "$name" lm; then
    echo "[gpu${gpu}] SKIP ${name} (collapse ok)"
    return 0
  fi
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
      if collapse_ok "$(lm_last_json "$save" "$name")" lm; then
        echo "[gpu${gpu}] DONE ${name} seed=${seed}"
        return 0
      fi
      echo "[gpu${gpu}] COLLAPSED ${name} seed=${seed} — retry"
    else
      echo "[gpu${gpu}] FAIL ${name} seed=${seed}"
      tail -30 "$save/${name}_train.log" || true
    fi
  done
  echo "[gpu${gpu}] GIVE UP ${name}"
  return 1
}

# Bare --lm_target v9 after 3fa7bdb: full pair-odd + hold ê_⊥û (λ=8) when
# YAML leak_positive/leak_negative is set; hold 0 if no ê. slider_positive
# is a name/probe, not the teacher. Hold dir is ê−(ê·û)û (hardcoded slider).
# Do not pass --project_align_min, --hold_weight, or Hub --anchor_weight /
# --leakage_floor. On a leaky yaml this is silent λ=8 — leaky catalog
# axes use train_lm_pair_odd_sub_e instead.
train_lm_v9() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=v9 pole_mode=hidden (pair-odd + hold ê_⊥û)"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target v9 \
      --pole_mode hidden \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop \
      --save_every 0 --device 0 \
      > "$save/${name}_train.log" 2>&1; then
    echo "[gpu${gpu}] TRAIN OK ${name}"
    return 0
  fi
  echo "[gpu${gpu}] TRAIN FAIL ${name}"
  tail -40 "$save/${name}_train.log" || true
  return 1
}

# Pair-odd after subtracting leftover ê from the poles. Hold is 0 even
# when YAML leak_* is set. Do not pass --hold_weight. Log must say hold_ê=0.
train_lm_pair_odd_sub_e() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=pair_odd_sub_e pole_mode=hidden (pair-odd − ê, hold_ê=0)"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target pair_odd_sub_e \
      --pole_mode hidden \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop \
      --save_every 0 --device 0 \
      > "$save/${name}_train.log" 2>&1; then
    echo "[gpu${gpu}] TRAIN OK ${name}"
    return 0
  fi
  echo "[gpu${gpu}] TRAIN FAIL ${name}"
  tail -40 "$save/${name}_train.log" || true
  return 1
}

# Hidden MSE onto the raw caption poles: t± = h± (β=1). The on-sheet
# target from docs/lm-sheet-goodhart.md. v9 ignores --common_beta (κ=0),
# so this is --lm_target symmetric, not v9. Do not pass Hub leash /
# project+hold. endreg stays trainer default 1.0.
train_lm_symmetric_beta1() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=symmetric common_beta=1 pole_mode=hidden"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target symmetric --common_beta 1 --pole_mode hidden \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop \
      --save_every 0 --device 0 \
      > "$save/${name}_train.log" 2>&1; then
    echo "[gpu${gpu}] TRAIN OK ${name}"
    return 0
  fi
  echo "[gpu${gpu}] TRAIN FAIL ${name}"
  tail -40 "$save/${name}_train.log" || true
  return 1
}

# Pair-symmetric, no Hub leash, no project+hold. Gender's clean pair.
# Do not pass --slider_positive / --slider_negative / --anchor_weight /
# --leakage_floor. endreg stays trainer default 1.0.
train_lm_symmetric() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=symmetric (no hub leash, no project)"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target symmetric \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop \
      --save_every 0 --device 0 \
      > "$save/${name}_train.log" 2>&1; then
    echo "[gpu${gpu}] TRAIN OK ${name}"
    return 0
  fi
  echo "[gpu${gpu}] TRAIN FAIL ${name}"
  tail -40 "$save/${name}_train.log" || true
  return 1
}

# Shipped Hub recipe: --lm_target hub = --symmetric + anchor_weight 0.3 +
# --anchor_autocal --leakage_floor -0.9. Energy until the new 2-D loss lands.
# Do not pass --lm_target v9 / --project_align_min. YAML shorts are log-only.
train_lm_hub() {
  local gpu="$1" name="$2" prompts="$3"
  local save="models/${name}"
  mkdir -p "$save"
  wait_vram "$gpu"
  echo "[gpu${gpu}] TRAIN ${name} lm_target=hub (symmetric + κ floor) endreg=default"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
      --name "$name" \
      --prompts_file "$prompts" \
      --save_dir "$ROOT/$save" \
      --lm_target hub \
      --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 --no-early_stop \
      --save_every 0 --device 0 \
      > "$save/${name}_train.log" 2>&1; then
    echo "[gpu${gpu}] TRAIN OK ${name}"
    return 0
  fi
  echo "[gpu${gpu}] TRAIN FAIL ${name}"
  tail -40 "$save/${name}_train.log" || true
  return 1
}

lm_weights() {
  local save="$1" name="$2"
  ls -t "$save/${name}_last.safetensors" "$save/${name}_unit_last.safetensors" 2>/dev/null | head -1
}

# One-card ritual. Prefer sample_lm_split after a v11 train.
sample_lm() {
  local gpu="$1" name="$2" prompts="$3" out_dir="$4"
  local save="models/${name}"
  local weights
  weights="$(lm_weights "$save" "$name")"
  [ -n "${weights:-}" ] && [ -f "$weights" ] || {
    echo "[gpu${gpu}] no weights for ${name} in $save"
    return 1
  }
  wait_vram "$gpu"
  echo "[gpu${gpu}] SAMPLE ${name} -2,-1,0,1,2 -> $out_dir"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u conceptmod/textsliders/generate_listen.py \
    --weights "$weights" \
    --prompts_file "$prompts" \
    --name "$name" --kind lm \
    --out_dir "$out_dir" \
    --scales=-2,-1,0,1,2 --duration 20 --seed 7 --device 0
}

# Split the 5-point ladder across both cards, then a resume pass writes LISTEN.md.
sample_lm_split() {
  local name="$1" prompts="$2" out_dir="$3"
  local save="models/${name}"
  local weights
  weights="$(lm_weights "$save" "$name")"
  [ -n "${weights:-}" ] && [ -f "$weights" ] || {
    echo "[sample-split] no weights for ${name} in $save"
    return 1
  }
  mkdir -p "$out_dir"
  wait_vram 0
  wait_vram 1
  echo "[sample-split] ${name} GPU0=-2,-1,0 GPU1=+1,+2+REFs"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u conceptmod/textsliders/generate_listen.py \
    --weights "$weights" --prompts_file "$prompts" \
    --name "$name" --kind lm --out_dir "$out_dir" \
    --scales=-2,-1,0 --start_index 1 --no-refs \
    --duration 20 --seed 7 --device 0 \
    > "$out_dir/sample_gpu0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" -u conceptmod/textsliders/generate_listen.py \
    --weights "$weights" --prompts_file "$prompts" \
    --name "$name" --kind lm --out_dir "$out_dir" \
    --scales=1,2 --start_index 4 \
    --duration 20 --seed 7 --device 0 \
    > "$out_dir/sample_gpu1.log" 2>&1 &
  local p1=$!
  local e0=0 e1=0
  wait "$p0" || e0=$?
  wait "$p1" || e1=$?
  if [ "$e0" -ne 0 ]; then
    echo "[sample-split] GPU0 failed (see $out_dir/sample_gpu0.log)"
    tail -20 "$out_dir/sample_gpu0.log" || true
  fi
  if [ "$e1" -ne 0 ]; then
    echo "[sample-split] GPU1 failed (see $out_dir/sample_gpu1.log)"
    tail -20 "$out_dir/sample_gpu1.log" || true
  fi
  echo "[sample-split] resume LISTEN.md (and any missing clips) on GPU1"
  wait_vram 1
  CUDA_VISIBLE_DEVICES=1 "$PY" -u conceptmod/textsliders/generate_listen.py \
    --weights "$weights" --prompts_file "$prompts" \
    --name "$name" --kind lm --out_dir "$out_dir" \
    --scales=-2,-1,0,1,2 --duration 20 --seed 7 --device 0 \
    > "$out_dir/sample_resume.log" 2>&1
  local er=$?
  if [ "$er" -ne 0 ]; then
    echo "[sample-split] resume failed (see $out_dir/sample_resume.log)"
    tail -20 "$out_dir/sample_resume.log" || true
    return "$er"
  fi
  echo "[sample-split] OK ${name} -> $out_dir"
  return 0
}

train_tf() {
  local gpu="$1" name="$2" prompts="$3"
  shift 3
  local save="models/${name}"
  local seeds=(7 17 27)
  mkdir -p "$save" "cache/${name}"
  if already_good "$save" "$name" tf; then
    echo "[gpu${gpu}] SKIP ${name} (collapse ok)"
    return 0
  fi
  for seed in "${seeds[@]}"; do
    echo "[gpu${gpu}] START ${name} seed=${seed}"
    rm -f "$save"/*.safetensors "$save"/*_last.json "$save"/*_best.safetensors
    if gpu_run "$gpu" "$PY" -u conceptmod/textsliders/train_lora_music3.py \
        --name "$name" \
        --prompts_file "$prompts" \
        --save_dir "$ROOT/$save" \
        --cache_dir "$ROOT/cache/${name}" \
        --rank 8 --alpha 8 --lr 2e-3 --steps 500 --duration 4 --seed "$seed" --device 0 \
        --targets full --xt_mode anchor --bidirectional \
        --target_mode axis --traj_frac 0 --loss nmse --gain_penalty 0 --uncond_weight 0 \
        --x0_per_row 8 --cond_seeds "$seed" --grad_clip norm \
        "$@" \
        > "$save/${name}_train.log" 2>&1; then
      if collapse_ok "$(tf_last_json "$save" "$name")" tf; then
        echo "[gpu${gpu}] DONE ${name} seed=${seed}"
        return 0
      fi
      echo "[gpu${gpu}] COLLAPSED ${name} seed=${seed} — retry"
    else
      echo "[gpu${gpu}] FAIL ${name} seed=${seed}"
      tail -30 "$save/${name}_train.log" || true
    fi
  done
  echo "[gpu${gpu}] GIVE UP ${name}"
  return 1
}
