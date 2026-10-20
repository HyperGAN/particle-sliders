# Fibo-particle-slider — experimental native EDIT backend

This integrates **`briaai/Fibo-2.0-edit-turbo`**, revision
`177f2d4a06b6587144ef5c2b8452a4f1fe2ab4b7`, through the installable
`particle_sliders.fibo_native` API. It is CPU contract qualified only. There
are no pretrained results, fitted weights, native memory-fit or image-quality
claims. No GPU work is authorized by this document.

The [official model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)
says Turbo is **EDIT only**, distilled for **4 steps, guidance 1**, and **not
intended for fine-tuning**; it recommends Fibo 2.0 Dev for that purpose. This
experimental adapter retains the specifically requested Turbo target. Dev is
not substituted. A successful CPU optimizer update does not overturn that
upstream caveat.

## Sources, access and license

The public rendered card was inspected on 2026-10-20. The anonymous raw README
endpoint returned HTTP 401. The model is automatically gated, but its form
requires organization/use details and explicit agreement to Bria's Privacy
Policy and Terms & Conditions. No form, legal agreement or credential creation
was performed. Existing authenticated access has not been established in this
task's CPU runtime. Model-card text and API metadata are available without
submitting that form.

The weights are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/);
[commercial use requires a Bria agreement](https://platform.bria.ai/licensing/pricing?model=fibo-2).
The shared repository's code license does not relicense the base model or
future model-derived adapters. Weight publication is outside this change.

| Inspected component | Exact source |
| --- | --- |
| Official model | [pinned revision](https://huggingface.co/briaai/Fibo-2.0-edit-turbo/tree/177f2d4a06b6587144ef5c2b8452a4f1fe2ab4b7) |
| Official demo and requirements | [Bria Space at `3e709136`](https://huggingface.co/spaces/briaai/Fibo-2.0-editing/tree/3e7091360e6c1fa2be7f00ee03fb94d8d42b7171) |
| Native pipeline | [diffusers `60f95274`, pipeline](https://github.com/muhammadali-afridi/diffusers/blob/60f95274bf07ed705d675eb09d99b6ae125b601e/src/diffusers/pipelines/bria_fibo2/pipeline_bria_fibo2.py) |
| Native transformer | [diffusers `60f95274`, transformer](https://github.com/muhammadali-afridi/diffusers/blob/60f95274bf07ed705d675eb09d99b6ae125b601e/src/diffusers/models/transformers/transformer_bria_fibo2.py) |
| Upstream integration | [diffusers PR 14991](https://github.com/huggingface/diffusers/pull/14991) |
| Shared core stack | [draft PR 141, `cdfb80a`](https://github.com/HyperGAN/particle-sliders/pull/141) |

Exact source SHA256s, public model-file sizes/Git blob OIDs, gate status and
defaults are packaged in `fibo_provenance.json`. The public gated API redacts
LFS SHA256s. They are recorded as unknown, never as made-up hashes. The offline
loader checks actual staged content against each upstream Git LFS pointer's
blob OID, including its size and computed SHA256. Staging must provide a
complete file checksum receipt. Ordinary Hub snapshot symlinks to immutable
blobs remain usable. No shared cache is moved, cleaned, modified or evicted.

## Native edit and training contract

The official pipeline owns VGL normalization, mask thresholding and gray-fill,
aspect-ratio resize/crop, text-render images/fonts, Qwen processing, VAE BN,
sigma shifting and scheduler updates. A native source image is always
required; the card's maximum of four ordered references is enforced. Masks
require exactly one source and the same image size. JSON prompts retain the
native `edit_instruction`; plain instructions are also accepted by upstream.
There is no implicit Bria/Gemini service call or API key creation.

`capture_pair` runs the four-step native neutral trajectory and a separate
native positive pass for its encoding. It captures the actual transformer
inputs, then computes both frozen teacher velocities at **the same neutral
x/t and ordered references**. Six Qwen3-VL bundles and their boolean padding
mask are kept; reference images condition both Qwen and the VAE. Native
Flux2 latents remain `[1,C,H,W]`; the transformer owns patching, RoPE planes
and clean-reference time zero. Raw scheduler time is divided by 1000 exactly
once. Native output is flow velocity without negation. No FLUX text-to-image
packing function or alternative sampler is introduced.

The routed projection reuses `TinyParticleProjection` and its existing
`RoutedMLP`: rank/alpha 8, one actual shared 128x4 particle parameter, FP32
adapter leaves, zero-initialized up maps. Only trunk/refiner attention Q/K/V/O
and injection cross-attention Q/K/V/O are adapted. Perceiver, VLM, VAE,
modulation, embeddings, FFNs and output head remain frozen. This is a routed
adapter, not an ordinary LoRA matrix that can be merged or loaded by PEFT.
Exports retain the particles and routed maps. Strength zero directly calls
each original projection. The supported unipolar range is 0–1; nonlinear
image response across strengths is expected.

`build_fibo_game` calls the existing `ConditionalPG09` on the genuine spatial
velocity field. It selects released **PG 0.9.1 `bispf_optimized`**, its native
DualNorm direction-blend optimizer, non-saturating loss, Born-Infeld critic
penalty, constraint-geometry backward, SBNP prior regularizer and native LR
policy. All participating particle rows are bound through the public
optimizer API. No copied game, hand-written loss, VIC overlay, fixed cap,
MSE or new optimizer is added. Trainable host execution is FP32 as required
by that shared API; the staged frozen model retains BF16. The original
gmix/legacy `winning_formulation()` implementation is unchanged.

Gaussian reference plus `(generated - positive_target) / response_scale`
is a critic coordinate, not an optimized squared error. The response scale
is caller-declared and checkpoint-bound. Critic features reuse the shared
all-coordinate conditioning sketch and collision gate, including masks and
ordered reference tensors. They contain meaningful timestep/context
coordinates, not row IDs. Guidance is always 1; the unused negative feature
dictionary aliases neutral, without a negative model pass.

## Local API after explicit native admission

The CPU package has no mandatory diffusers/Pillow dependency for import.
Pretrained execution additionally requires the **inspected** diffusers commit
above, the official processor/text encoder runtime and a previously approved
local model snapshot. The official Space requirements used Python 3.12,
`transformers>=4.57`, `accelerate`, torch 2.11.0, torchvision 0.26.0, Pillow,
and that diffusers commit. Its ZeroGPU/aokit packages are demo optimizations,
not required by this eager backend. None were installed or executed here.

```python
from particle_sliders.fibo_native import (
    staged_pipeline, FiboParticleSlider, FiboEditResponseAdapter,
    write_cache, FiboCachedRows, build_fibo_game,
)

# Run only when the resource owner admits the native phase and access exists.
# identity includes model_id, model_revision, diffusers_commit and a full
# staged_files_sha256 mapping. The loader verifies local content/source pins.
pipe = staged_pipeline(local_snapshot, identity, device=admitted_device)
slider = FiboParticleSlider(pipe.transformer)
backend = FiboEditResponseAdapter(pipe, slider, model_identity=identity)
prepared = backend.capture_pair(neutral_vgl, positive_vgl, references, seed=7)
index_sha256 = write_cache(new_cache_directory, prepared, model_identity=identity)
rows = FiboCachedRows(new_cache_directory, index_sha256=index_sha256,
                      model_identity=identity, device=admitted_device)
game = build_fibo_game(backend, rows, response_scale=declared_scale,
                       total_steps=admitted_steps, batch_size=2)
metrics = game.step()
```

The cache index binds model, protocol, condition policy, all context/row
checksums and native shape. Long prompt encodings are content-addressed and
deduplicated. Provider construction reads no response tensors; only selected
rows and their bounded context LRU move to the backend device. It does not
re-encode prompts during training. Returned rows are private snapshots.

Full training state uses the existing `game.state_dict()` / transactional
`load_state_dict()` API, including both optimizers, public regularizer,
sampling/exposure, all consumed RNG states, model/source/runtime identities
and completed steps. `game.export_state_dict()` is inference-only. Standalone
native inference uses `slider.save()` / `FiboParticleSlider.load()` with
safetensors, strict model/protocol/target identities, finite exact FP32 schema,
and complete preflight before host mutation. Frozen weights are excluded.

`backend.render_edit(..., scale=0/.5/1)` uses the official four-step edit
pipeline and its native decoder. It uses private generators and a temporary
scheduler copy, preserving the staged pipeline's scheduler object/state.
Its native FP32 inter-step latents, VAE patch/BN decode, reference/mask/VGL
path are retained. It never calls the shared generic Euler trajectory path.

## CPU evidence and remaining qualification

The Fibo tests mock the pretrained transformer/pipeline and execute actual
released PG 0.9.1. They cover native default forwarding, same-state teachers,
references/order/masks/padding/time, text-only rejection, actual particle and
adapter updates, native recipe ownership, checksum/lazy cache behavior,
exact split resume, inference export and native mock render reload parity at
0/.5/1. Packaging and shared-state regression checks are recorded separately.
This does not validate the pinned diffusers runtime with real model weights.

Recorded on Pop, 2026-10-20, with CUDA hidden, network-offline environment
flags and the previously installed CPU torch 2.14.1 runtime:

| Executed scope | Result |
| --- | --- |
| Fibo plus shared game/state/native-binding/package regression set | 190 passed, zero skipped, 12.49s |
| Final Fibo and extracted-sdist/wheel import gates | 23 passed, zero skipped; includes 20 Fibo tests |
| Python undefined/unused-name checks and `git diff --check` | Passed |

These counts overlap. The installed PG0.9.1 sources were verified by the shared
core's existing 49-file manifest. The final packaging import blocks diffusers,
transformers, PEFT and model-product imports and asserts CUDA is uninitialized.
CPU torch2.14.1 evidence does not qualify the official demo's torch2.11 GPU stack.

Before the parent schedules any GPU slot:

| Requirement | Planning value; not a measured fit |
| --- | --- |
| Model files | 25,466,702,382 bytes (23.72 GiB) of safetensors: transformer 16.25 GB, Qwen 8.88 GB, VAE 0.34 GB; plus processor/fonts/configs |
| Disk | Allow at least 35–40 GB for one approved snapshot and bounded smoke cache/exports; retain existing resident model caches |
| CPU RAM | Prefer 64 GB during model staging/encoding and teacher-cache preparation; not measured |
| Inference device | Official demo uses a 48 GB ZeroGPU slice. Propose one 48 GB device for a one-source native parity smoke; local memory fit is unmeasured |
| Training qualification | Prefer one 80 GB device for the first 1024px forward/backward/PG step, or a separately measured 48 GB checkpointed/offloaded plan; no multi-GPU/offload fit is claimed |
| Initial bounded gates | One source, 4 steps, CFG1; raw native/cached velocity parity; zero-scale exact identity; finite connected gradients and public PG update; save/reload/resume; strengths 0/.5/1; then mask and multiple references |
| Additional work | Preservation/edit-quality metrics and held-out references, stability, native BF16/FP32 adapter behavior, checkpointed backward and GPU exact-resume limits |

No slot, cloud hardware, paid capacity, sharding, weights download or model
terms acceptance is requested automatically. MOSS remains the GPU scheduling
owner. The native source currently supports five context images internally;
this backend follows the released Turbo card/Space limit of four.
One captured pair entails two native four-step passes for conditioning and
trajectory capture, then eight matched raw teacher forwards: sixteen DiT
forwards total plus two native VLM/reference preparation passes.

This change is proposed inside the existing `HyperGAN/particle-sliders`
repository, stacked on draft PR 141. Creation of the separately named public
`HyperGAN/Fibo-particle-slider` repository is pending with the parent. A local
standalone wrapper may pin this shared backend once its review commit exists.
There is no new public repository, weight publication, deployment or merge.
