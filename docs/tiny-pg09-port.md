# TinyQwen shared PG 0.9 run

Historical fe54d05 PG0.9.0 audit receipt. Current installed API0.6.0 / PG0.9.1
cohort and public legacy CLI retirement are documented in
[text-package-acceptance.md](text-package-acceptance.md). These older receipts
are preserved and do not qualify the replacement package candidate.

`tiny_llm_pg09.PG09TinyRun` now calls
`particle_sliders.pg09.ConditionalPG09` for every optimizer update, sampler,
training checkpoint and restore. It reuses `TinyQwenResponseAdapter`, the native
attention q/k/v/o `TinyParticleSlider`, its actual 128×4 bank, and the shared
`particle_sliders.reference.RoutedMLP` for training and inference. There is no
local game fallback. The CPU CLI explicitly selects a shared core checkout and
records the imported module path and hash.

This is protocol **`tiny-qwen-shared-pg090-neutral-v2`**, a new cohort. The former
`tiny-qwen-pg090-neutral-v1` local `PairedPG09` states are rejected. The shared
protocol uses raw native last-token response coordinates, frozen neutral response
plus sign/time critic conditions, balanced independent G/D row streams, and the
public `bispf_optimized` preset by default. These differ from the prior
unconditional, whitened local-loop experiment. No old experiment is relabeled.

Students receive neutral token IDs; targets are frozen positive readouts. G
optimizes the PG adversarial scalar plus the public prior regularizer, applied
once; D optimizes PG adversarial loss plus its public penalty. No MSE is optimized.
Trainable adapter leaves stay FP32 on FP32 or BF16 frozen hosts. All routed bank
rows are active and owned exactly once by the shared optimizer.

Native 32-wide fixtures use release/auto. Native 1024-wide Qwen requires the
explicit reviewed `alz-continuous-v1` / `continuous` source profile; release/auto
fails before model execution or adapter attachment. No reshape or automatic
profile switch is provided.

Checkpoints bind the resolved recipe, core/PG/native sources, architecture,
frozen responses, token IDs, model identity, optimizers, regularizer, sampling
coverage and private RNG streams. CPU-fixture vocabulary is retained, including
words introduced by inference before saving. Invalid restore leaves game state
and tokenizer unchanged. Portable FP32 exports validate protocol, conditioning,
inference sources, public recipe, model identity, tokenizer, tensor schema/dtype
and finite values before attachment. Negative, zero and positive inference scales
retain the native branch behavior; training remains unipolar +1.

```bash
CORE=/mnt/nvme/sliders/text-video-core-3e94b13-20261017
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$CORE/packages/particle-sliders-core/src:$PWD" \
  /mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest -q \
  -p no:cacheprovider -o pythonpath="$CORE/packages/particle-sliders-core/src" \
  tests/test_tiny_pg09_port.py tests/test_tiny_llm_slider.py tests/test_pg09_text_video.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/nvme/sliders/qualification/envs/pg090/bin/python scripts/qualify_tiny_pg09_cpu.py \
  --core "$CORE" --output /tmp/tiny-shared-pg09-new-run --until 2
```

The CLI has only synthetic CPU fixtures and a six-update maximum. Rerun with
`--until 4` to continue its saved state in another process. It cannot download or
load pretrained weights. The repository's historical embedded core predates
PG09; do not install its old PG dependency into the qualified environment.
The shared-package extraction's reviewed wheel/pin is a separate integration
gate; no installed environment or actively pinned source was retargeted here.

`train_lora_tiny_llm.py --recipe particle_bridge` and `qualify_captions.py` remain
the archival positive-caption Rp/VIC campaign and its recovery observer. They do
not execute PG 0.9 and must not be described as shared PG 0.9 training. Their
native routed forward now calls the shared `RoutedMLP` with exact seeded state
and output parity. Existing raw/EMA artifacts and controller defaults are intact.

Pretrained model/zero-scale validation, live export identity verification,
native runtime fit, throughput and language/generation quality remain with the
family owner and sole coordinator admission. See
[tiny-pg09-package-audit.md](tiny-pg09-package-audit.md).
