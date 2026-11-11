# Opt-in Cascade Stage C response cohort

This isolated package is the shared-game alternative for the family rooted at
`../textsliders/train_lora_cascade.py`. The historical file, `prompt_pair.loss`,
configuration, environment and artifact names are unchanged. Select this package
explicitly in a new environment and use a new cohort/output directory.

**Draft dependency admission:** the package pins core candidate
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` from draft PR #148, PG 0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0.
Core lifecycle corrections/admission remain with the core owner. This is a tested
candidate integration, not permission to finalize that core pin or release it.

The new cohort is an adapter-only positive-teacher GAN. Shared
`ConditionalPG09` owns adversarial mathematics, regularization, optimizers and
sampling; the common response lifecycle owns update/EMA/resume/export/cleanup.
Actual HyperGAN-built HNDL graphs execute attention LoRA down/up projections and
the shared HNDL FiLM spatial critic. The verified builder checks installed HNDL
Python sources, exact HyperGAN clean-build provenance and builder closure.
The adapter graph bytes also participate in model/export identity. There is no
local GAN objective and no particle bank or hidden reconstruction loss.

Stage C predicts image-embedding fields `[1,C,H,W]` at native continuous
`timestep_ratio` in `[0,1]`, using all token text, pooled text and image embeddings.
The spatial response remains intact through the PG penalty. Target rows use the
frozen positive condition; the student uses the neutral condition. CFG applies
the adapter only to the conditional branch. Training has scale +1; observation
accepts any finite signed strength, with zero executing the exact frozen base.
Attention projections `to_q/to_k/to_v/to_out.0` use a new rank-r, alpha/rank=1
HNDL LoRA cohort. Historical c3lier/conv LoRA artifacts are not relabeled.

The Stage C prior is distinct from Stage B's effnet-conditioned latent decoder
and Stage A's image codec. **Neither Stage B nor Stage A is trained here.** The
package advertises numerical Stage C responses only. It rejects media rendering
before native allocation; DDPMWuerstchen sampler plus Stage B/Stage A parity and
full pretrained/GPU quality are unqualified. It rejects flow/Euler trajectories,
wrong cache/base identities, wrong geometry, nonfinite conditions, and unsupported
precision. No wide-vector guard is bypassed.

## Install and use

Use Python 3.11–3.14 in a separate environment. For CPU conformance:

```bash
python -m venv .venv-cascade
.venv-cascade/bin/pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-cascade/bin/pip install ./conceptmod/cascade-shared
```

The tiny fixture instantiates a random native Diffusers `StableCascadeUNet`,
not pretrained weights. It downloads no model and selects no GPU:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference
from particle_sliders_cascade.backend import fixture_rows

config = SessionConfig('cascade_stage_c', 'new-cascade-cpu-cohort',
                       dummy=True, steps=4, ema_decay=.9)
with open_response(config) as session:
    session.train(stop_after=2)
    session.checkpoint('new-cascade/training.pt')
with open_response(config, resume='new-cascade/training.pt') as session:
    session.train()
    response = session.evaluate(strength=-.5)
    session.export_inference('new-cascade/native.pt', selection='ema')
with open_inference('new-cascade/native.pt', operation='response') as inference:
    response = inference.response(row=fixture_rows()[0], strength=.5)
```

The common CLI also executes the registered factory:
`particle-slider train --interface response --config new-config.json --out new-run`.
Config JSON is `config.as_dict()`. The shared CLI owns its own artifact envelope;
use `session.export_inference` above for the cache-independent v2 artifact.

For an already staged standalone Stage C prior directory, set `dummy=False`,
`model_dir`, explicit native `shape=[1,C,H,W]`, `cache_file`, `cache_sha256`.
Only local safetensors loading is allowed. Cache format is a weights-only-loadable
dictionary `{'model_identity': model_identity(staged_model), 'rows': [...]}`;
each row is a dictionary of `ResponseRow` fields with frozen CPU FP32 tensors.
Each conditioning dictionary must contain `embeds=[1,T,text_width]`,
`pooled=[1,1,pooled_width]`, and `image_embeds=[1,I,image_width]`. These must be
captured with the exact native tokenizer/text encoder, timestep and latent;
this package does not claim a prompt-to-cache capture pipeline. The base model's
actual state/configuration are checked against the cache before adapter hooks.
Staged runtime injection (`model=..., rows=...`) supports local integration tests.
Borrowed modes, gradient flags/objects, buffers and checkpoint attributes are
restored on close or partial construction/training failure.

Native-only v2 export omits cache paths, rows, teachers, critic and optimizers.
Its factory loads only the local base and HNDL adapter, validates the base/source
identity before attaching hooks, and accepts fresh native rows. A response is
not a generated image. Native-only inference cannot resume training.

## Tests

Run installed packages; the repository top-level pytest configuration otherwise
adds its historical core source directory to `sys.path`:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-cascade/bin/python -m pytest --noconftest -c /dev/null \
  --import-mode=importlib conceptmod/cascade-shared/tests -q
```

Tests cover installed source identity, matched HNDL outputs/VJPs/critic second
derivatives, actual shared updates, exact resume/EMA, zero/signed semantics,
native-only inference, local tiny safetensors staging/cache checks, preflight
rejections, rollback, borrowed state and partial-failure cleanup. These are tiny
CPU software fixtures, not actual pretrained model or complete Cascade
qualification. No full models, GPU training or release artifacts are produced.
