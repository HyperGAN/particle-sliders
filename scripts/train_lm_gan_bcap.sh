#!/usr/bin/env bash
# Repaired deterministic-caption GAN: the listening-preferred tx-smoke card.
# Optional miners reweight prompt rows;
# they are not the movable latent prior in the Gaussian reference.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MUSIC3_PYTHON:-/home/mikkel/anaconda3/envs/minimax-music3/bin/python}"
GPU="${1:-1}"
NAME="${2:-repaired-tx-smoke}"
PROMPTS="${3:-conceptmod/textsliders/data/prompts-gender-uni-v2.yaml}"
STEPS="${STEPS:-120}"
SEED="${SEED:-7}"
SAVE_EVERY="${SAVE_EVERY:-0}"
STATE_ARGS=()
if [[ "${SAVE_STATE:-0}" == "1" ]]; then STATE_ARGS+=(--save_training_state); fi
if [[ -n "${RESUME_STATE:-}" ]]; then STATE_ARGS+=(--resume_state "$RESUME_STATE"); fi
MINERS="${MINERS:-0}"
CONDITION="${CONDITION:-none}"
FM_MODE="${FM_MODE:-batch}"
FM_WEIGHT="${FM_WEIGHT:-1}"
ADV_WEIGHT="${ADV_WEIGHT:-1}"
SCHEDULE="${SCHEDULE:-constant}"
SAVE="${SAVE_DIR:-$ROOT/models/gan-bcap-repair/$NAME}"
cd "$ROOT"
mkdir -p "$SAVE"
if [[ -e "$SAVE/${NAME}_train.jsonl" || -e "$SAVE/${NAME}_last.safetensors" ]]; then
  echo "Run already exists: $SAVE. Choose a new NAME or SAVE_DIR." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HOME=/ml2/music/.cache/huggingface
export PYTHONPATH="$ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PY" -u conceptmod/textsliders/train_lm_slider_music3.py \
  --name "$NAME" --prompts_file "$PROMPTS" --save_dir "$SAVE" \
  --lm_target faithful_plus_neu_lyric --pole_mode hidden \
  --rank 8 --alpha 8 --lr 5e-4 --steps "$STEPS" --seed "$SEED" \
  --no-early_stop --endreg_weight 1 --save_every "$SAVE_EVERY" --device 0 \
  --adv_arch tx --adv_in scaled --adv_readout mean_last --adv_condition "$CONDITION" \
  --adv_weight "$ADV_WEIGHT" --fm_weight "$FM_WEIGHT" --fm_mode "$FM_MODE" --pole_weight 0 --lyrichold_weight 0 \
  --adv_reg_coeff 1 --adv_reg_kappa 1 --adv_batch 4 \
  --gan_beta1 0 --gan_lr_schedule "$SCHEDULE" --grad_account \
  --parts "$MINERS" "${STATE_ARGS[@]}"
