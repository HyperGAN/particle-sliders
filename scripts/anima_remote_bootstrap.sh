#!/usr/bin/env bash
set -euo pipefail
cd /workspace/anima
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/huggingface
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
python3 -m pip install --quiet uv
uv venv .venv-anima --python 3.12.13
uv pip sync --python .venv-anima/bin/python configs/anima/requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
apt-get update -qq
apt-get install -y -qq rsync
.venv-anima/bin/python -m pytest tests/lumen_studio -q > artifacts/anima/cpu-tests.log 2>&1
.venv-anima/bin/python -u -c 'from lumen_studio.prepare import convert; print(convert("artifacts/anima/model"))' > artifacts/anima/conversion.log 2>&1
printf 'Runtime ready\n' > artifacts/anima/ready.txt
