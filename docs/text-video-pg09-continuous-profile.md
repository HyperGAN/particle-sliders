# Explicit continuous-profile text adapters

Qualified against immutable core `3e94b13eb5d6cd1ceee3f018f019a68603f35696`,
module SHA256 `426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256`.
The existing single ALZ overlay is used only through PYTHONPATH at
`/tmp/alz-pg09-domain-proposal`; no package installation or patch application
occurred. Core validates its entire reviewed49-file profile, base release/wheel
and patch digest, and binds the actual declared/resolved domain and runtime.

`TinyQwenResponseAdapter` now accepts explicit `pg_source_profile` and
`penalty_input_domain` options, and calls the core's public
`validate_pg09_domain` before response computation. Its actual existing routed
bank and attention branches are unchanged. The released-auto default still
rejects native1024 before adapter/backend access beyond reading model config.

New `BonsaiReadoutResponseAdapter` wraps the existing `BonsaiParticleHead` and its
actual128x4 bank, using the unchanged frozen-backend encode/hidden interface.
An admitted caller can supply the existing checksummed `FrozenReadouts` cache
without starting a server. Released-auto rejects native5120 before backend/head
access. No GGUF autograd, in-attention LoRA, or generation quality is claimed.

Explicit construction must use the same options for adapter and shared game:

```python
adapter = TinyQwenResponseAdapter(
    backend, existing_slider, model_identity=verified_identity,
    pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
game = ConditionalPG09(adapter, prepared_rows, critic,
    conditions=frozen_conditions, particles=adapter.particles,
    **adapter.domain_options, preset='bispf_optimized', batch_size=2,
    total_steps=4, seed=15)
```

Only `ConditionalPG09` performs losses, optimizers, regularization, sampling and
training state. Fixed critic conditions retain the complete frozen neutral
context plus sign/time coordinates. The actual bank participates in generated
responses and is owned once. There is no automatic profile selection, shape
disguise, projection or MSE fallback.

CPU evidence under `evidence/pg09-text-video/`:

- `continuous-text-tests.xml`: **10passed**, both presets at actual native1024
  Tiny and5120 Bonsai response widths on synthetic/native CPU backends. G/D/bank
  parameter changes, optimizer ownership, frozen Tiny base, continuous penalty,
  GAN-only equations, safe weights-only checkpoint loading, source/profile
  binding, exact resume/export and zero-scale behavior pass.
- `release-profile-regression.xml`: **31passed** (the prior23 native text/video
  checks and8 asset preparation gates), unchanged release dependency.
- `continuous-text-fresh-process.json`: four Tiny/Bonsai × both-preset cases,
  checkpoint2 then independent-process2 continuation, exact losses/full optimizer,
  regularizer, bank, RNG, parameters and seeded evaluation. Two additional actual
  narrow Tiny64 processes reject release↔ALZ checkpoint transfer with live state
  and global Python/NumPy/Torch RNG unchanged.

The two initial domain-logging assertions were corrected to inspect the public
penalty's actual `last_stats` fields (`input_domain`/`resolved_domain`);
`diagnostics()` intentionally returns a smaller summary. No training or penalty
implementation was changed to satisfy those assertions.

All tests hide CUDA and use no pretrained assets. These are backend/host software
gates, not real-weight or quality validation. Core owner retains model/process
and GPU admission; no lease is implied by profile qualification. LTX default
connector/mask/hold protocol is being specified separately; H3 remains stopped
pending genuine native AV response geometry and verified model admission.

Reproduction:

```bash
CORE=/mnt/nvme/sliders/text-video-core-3e94b13-20261017
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/tmp/alz-pg09-domain-proposal:$CORE/packages/particle-sliders-core/src:$PWD:$PWD/tests" \
  /mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest -q \
  -p no:cacheprovider -o pythonpath="$CORE/packages/particle-sliders-core/src" \
  tests/test_pg09_wide_text.py
/mnt/nvme/sliders/qualification/envs/pg090/bin/python \
  scripts/qualify_wide_text_pg09_cpu.py --core "$CORE" \
  --overlay /tmp/alz-pg09-domain-proposal --output /tmp/text-pg09-wide-replay
```
