# SDXL CPU preview validation

This is a new opt-in adapter-only shared response cohort. It is not a
pretrained-model, GPU-training, sampler, image-quality or text-encoder-extraction
qualification. No model weights were downloaded and CUDA was not initialized.

Validated with Python 3.12.15, Torch 2.6.0+cpu, Diffusers 0.35.1,
Transformers 4.57.6, PG 0.9.4, HNDL 0.10.0, clean HyperGAN develop
5f368a3770cf5bdae0448995c9f76e29af8049f7 and core PR 148 revision
7a8a321d04323d49319d0b08e6093a0ddce36b27. `cohort.json` verifies the complete
installed core Python/JSON/HNDL source closure, and `verified_builder()` verifies
the actual HyperGAN/HNDL sources and framework build provenance before hooks.

Executed commands from the isolated task workspace (the checkout is `sdxl/`):

```sh
.venv/bin/python -m build --no-isolation sdxl/packages/sdxl-particle-preview --outdir evidence/dist
.venv/bin/python -m build --wheel --no-isolation evidence/sdist-rebuild/sdxl_particle_preview-0.1.0 --outdir evidence/sdist-wheel
.venv/bin/pip install --force-reinstall --no-deps evidence/sdist-wheel/sdxl_particle_preview-0.1.0-py3-none-any.whl
.venv/bin/python -m pytest sdxl/packages/sdxl-particle-preview/tests -q
.venv/bin/pip check
.venv/bin/pip install --dry-run --ignore-installed --report evidence/clean-public-cpu-resolution.json --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' ./sdxl/packages/sdxl-particle-preview
git -C sdxl diff --check
```

Results: wheel and sdist built; wheel rebuilt from extracted sdist; installed
package Python/JSON/HNDL bytes matched every source file; **16 tests passed**;
`pip check` reported **No broken requirements found**; clean public CPU resolver
succeeded, including exact public core and HyperGAN VCS revisions; diff check
passed. All historical functions (`flush`, `train`, `main`, `train_lora`) have
identical ASTs to base 3c25f785d14073e6f627c0fa9d2297a041c79913. Default
requirements and shared-core files are unchanged.

The fixtures execute a real randomly initialized small `UNet2DConditionModel`
with SDXL `text_time` additions, native math attention, and real DDIM scaling.
Both epsilon/v-prediction contracts are checked without converting their output
coordinates. Token, pooled and size/crop-ID perturbations each change native
outputs. HNDL ordinary-LoRA graph outputs and input/weight VJPs match the native
Linear reference exactly. A complete matched-weight UNet comparison with the
actual historical `LoRAModule` uses FP32 tolerances rtol=1e-5, atol=1e-6 for
outputs/VJPs; different leading-dimension execution permits small roundoff.
The real HNDL field critic matches the native FieldCritic, including input and
condition VJPs and penalty-style input-gradient second derivatives.

Both public BISPF presets execute actual shared updates and exact uninterrupted
versus split resume. Tests cover EMA state/export and identical native-only EMA
response after removing the training rows, strengths 0/0.5/1, nested scale
restoration, CLI execution, changed model/profile/shape/conditioning refusals,
historical packet refusal before factory resolution, builder tamper, retained
wide-vector guards, malformed EMA rejection, late-load rollback, borrowed
mode/gradient restoration, and cleanup on prepare/update/remove-hook failures.

There are no repository GitHub Actions workflows at this base. Local checks
above are the verification receipt; remote check status must be read on the
published draft head. Root owns mandatory Claude actual-diff review, fixes and
return review, final core admission, routing and merge. This work does not
publish a package/model release or merge the draft.
