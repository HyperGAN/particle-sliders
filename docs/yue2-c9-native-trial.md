# YuE2 metal: matched c9_g4x trial

Status: CPU verification complete; no native GPU spend yet.
The studio had 2 active and 58 queued jobs at preflight. The user explicitly chose to preserve that queue and share GPU 1; the studio stays running.

Existing trainers are reused. The opt-in `--propose_only_c9_g4x` changes only optimizer learning rates.
Both module-level RECIPE dictionaries, the shared GAN updates, Music ARM_B, live v9 defaults and locked AdvConfig remain unchanged.
The explicit LR flag is pinned in the resume signature; crossing control/ablation resumes fails.
There are no particles, additional G losses, schedule changes, or critic changes.

| Run | G LR | D LR | Schedule | Train seed | Updates |
|---|---:|---:|---|---:|---:|
| Production control | 0.0005 | 0.00075 | constant | 7 | 600 |
| c9_g4x (propose-only) | 0.002 | 0.003 | constant | 7 | 600 |
| Existing gan_plus_neu | 0.0005 | 0.0005 | delayed cosine, hold 80, floor 0.05 | 7 | 600 |

Four balanced shuffled train rows; two held-out rows, seeds 1709/2903, scales 0/0.5/1 plus a positive-caption reference and an **unscored -1 canary**: 20 clips per run. All captions are sound-only.

The optional held-out report measures caption-delta projection, orthogonal residual ratio, relative target error, and bitwise scale-zero identity. These are **hidden-geometry diagnostics**, not calibrated audible cover/leak or lyric-preservation gates. Toy gates remain unchanged.

## Full planned commands (printed before GPU spend)

Runtime: the existing isolated YuE2 Python; no installation. Physical GPU 1 maps to `--device cuda:0`.
Each two-step preflight retains a 600-step schedule horizon and resumes into its own campaign.

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/ml2/music/.cache/huggingface
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/ml2/music/.cache/sliders-yue2-c9-native-20260916
unset TRANSFORMERS_CACHE
```

### control

```bash
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/train_lora_yue2_arm_b.py --recipe unipolar_gan --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-uni-control-600-s7-20260916 --name metal-yue2-uni-control-600-s7-20260916 --steps 600 --seed 7 --device cuda:0 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --until 2
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/scripts/train_yue2_arm_b_campaign.py --recipe unipolar_gan --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-uni-control-600-s7-20260916 --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-c9-20260916/control --name metal-yue2-uni-control-600-s7-20260916 --steps 600 --seed 7 --gpu 1 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --eval_prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml --include_canary --hidden_diagnostics
```

Expanded campaign children:

```bash
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/train_lora_yue2_arm_b.py --recipe unipolar_gan --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-uni-control-600-s7-20260916 --name metal-yue2-uni-control-600-s7-20260916 --steps 600 --seed 7 --device cuda:0 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/scripts/evaluate_yue2_arm_b.py --recipe unipolar_gan --weights /ml2/music/sliders-conceptmod/models/metal-yue2-uni-control-600-s7-20260916/metal-yue2-uni-control-600-s7-20260916_last.safetensors --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-c9-20260916/control --include_canary --hidden_diagnostics
```

### c9-g4x

```bash
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/train_lora_yue2_arm_b.py --recipe unipolar_gan --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-uni-c9-g4x-600-s7-20260916 --name metal-yue2-uni-c9-g4x-600-s7-20260916 --steps 600 --seed 7 --device cuda:0 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --propose_only_c9_g4x --until 2
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/scripts/train_yue2_arm_b_campaign.py --recipe unipolar_gan --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-uni-c9-g4x-600-s7-20260916 --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-c9-20260916/c9-g4x --name metal-yue2-uni-c9-g4x-600-s7-20260916 --steps 600 --seed 7 --gpu 1 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --eval_prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml --include_canary --hidden_diagnostics --propose_only_c9_g4x
```

Expanded campaign children:

```bash
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/train_lora_yue2_arm_b.py --recipe unipolar_gan --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-uni-c9-g4x-600-s7-20260916 --name metal-yue2-uni-c9-g4x-600-s7-20260916 --steps 600 --seed 7 --device cuda:0 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --propose_only_c9_g4x
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/scripts/evaluate_yue2_arm_b.py --recipe unipolar_gan --weights /ml2/music/sliders-conceptmod/models/metal-yue2-uni-c9-g4x-600-s7-20260916/metal-yue2-uni-c9-g4x-600-s7-20260916_last.safetensors --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-c9-20260916/c9-g4x --include_canary --hidden_diagnostics
```

### plus-neu

```bash
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/train_lora_yue2_arm_b.py --recipe gan_plus_neu --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-plus-neu-600-s7-20260916-r2 --name metal-yue2-plus-neu-600-s7-20260916-r2 --steps 600 --seed 7 --device cuda:0 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --until 2
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/scripts/train_yue2_arm_b_campaign.py --recipe gan_plus_neu --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-plus-neu-600-s7-20260916-r2 --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-c9-20260916/plus-neu --name metal-yue2-plus-neu-600-s7-20260916-r2 --steps 600 --seed 7 --gpu 1 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml --eval_prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml --include_canary --hidden_diagnostics
```

Expanded campaign children:

```bash
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/train_lora_yue2_arm_b.py --recipe gan_plus_neu --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-plus-neu-600-s7-20260916-r2 --name metal-yue2-plus-neu-600-s7-20260916-r2 --steps 600 --seed 7 --device cuda:0 --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml
/ml2/music/.cache/yue2-test-env/bin/python -u /ml2/music/.cache/sliders-yue2-c9-native-20260916/scripts/evaluate_yue2_arm_b.py --recipe gan_plus_neu --weights /ml2/music/sliders-conceptmod/models/metal-yue2-plus-neu-600-s7-20260916-r2/metal-yue2-plus-neu-600-s7-20260916-r2_last.safetensors --prompts_file /ml2/music/.cache/sliders-yue2-c9-native-20260916/conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-c9-20260916/plus-neu --include_canary --hidden_diagnostics
```

## Verification

106 selected tests passed, including independent RpGAN/cap equation parity, eager/checkpointed native gradients, exact c9 resume, forbidden-G-companion rejection, the actual c9 flag on both unipolar cells at 600 for seeds 0/1/7, and Music/default locks. No MSE call is allowed in the c9 acceptance fits.

The CPU schedule audit reran both `c0_production` and `c9_g4x` at 600/1200/3400 on seeds 0/1/7. It exited 1 as expected because the control fails requested early budgets. The ablation passes every requested budget/cell/seed. Compact evidence and unchanged-source hashes: [yue2-c9-native-toy-proof.json](yue2-c9-native-toy-proof.json).

| Arm | Budget | Divergent cover range | Close cover range | Off-caption | Neutral hold | Both cells, all seeds |
|---|---:|---:|---:|---:|---:|---|
| c0_production | 600 | 0.4823–0.4836 | 0.5855–0.5866 | 0.0833 max | 1.0000 min | FAIL |
| c0_production | 1200 | 0.6584–0.6606 | 0.9313–0.9325 | 0.0000 max | 1.0000 min | FAIL |
| c0_production | 3400 | 0.9312–0.9314 | 0.9337–0.9342 | 0.0000 max | 1.0000 min | PASS |
| c9_g4x | 600 | 0.9252–0.9274 | 0.9285–0.9309 | 0.0000 max | 1.0000 min | PASS |
| c9_g4x | 1200 | 0.9311–0.9316 | 0.9336–0.9345 | 0.0000 max | 1.0000 min | PASS |
| c9_g4x | 3400 | 0.9312–0.9318 | 0.9335–0.9338 | 0.0000 max | 1.0000 min | PASS |

The formulation leaderboard also reran with `--polarity both`; the existing `rpgan_bcap_plus_neu` row remains an in-box UNI HIT and fails the bipolar continuation exam. No UNI decision uses that bipolar failure or the -1 canary. The schedule ablation is assessed separately by the existing UniPG-C runner and the native-flag acceptance test; it is not silently added to or promoted by the formulation board.

Native GPU hours: **0**. No new weights, audio, native winner, or stability claim yet.
