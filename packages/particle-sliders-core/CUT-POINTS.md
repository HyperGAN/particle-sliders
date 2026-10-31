# Cut points: attachment, readout and lifecycle boundaries

This guide reviews the existing native bindings at
`3c25f785d14073e6f627c0fa9d2297a041c79913` (core 0.11.0 / released PG 0.9.3),
plus the additive selective PEFT helper described below. Dependency/profile modernization is separate;
use the exact compatibility receipt for the source you install. Nothing here
admits a new model, migrates a checkpoint or qualifies a pretrained sampler.

There is **no public `CutPoint` object or generic layer-index option**. In this
codebase the term can refer to three different boundaries:

| Boundary | Actual interface | Owner |
|---|---|---|
| Adapter attachment | Product target selector and attached native adapter; `AdapterParameters(named, scale_context)` exposes its real trainable leaves | Product/native binding |
| Response readout | `prepare(row)`, `generated(prepared)`, `metadata()` on a response adapter | Native binding; core validates the response contract |
| Session integration | `Backend` or `ResponseBackend` returns `Binding(game, close, ...)` | Product constructs; common lifecycle runs and closes |

Attachment and readout need not use the same tensor. An AR adapter changes
attention projections and reads the final token hidden state after the native
trunk/final normalization. An image adapter changes projections and exposes a
native spatial prediction field. Neither is a hook that splits the pretrained
model into independently interchangeable halves. ParticleGAN owns GAN math;
these boundaries do not introduce a reconstruction objective.

## Attachment: specify the module and the tensor operation

For a new backend, record the canonical native module paths, module types,
input/output feature widths and the exact operation being changed. Include
surrounding normalization/residual placement from that model's pinned source.
Selecting a layer index alone is insufficient. The existing selectors are:

| Implementation | Selection and placement |
|---|---|
| `fibo_native.target_modules` | Exact Q/K/V/O `Linear` leaves in `noise_refiner.i.attention` and `layers.i.attention`; also `layers.i.cross_attention` at configured injection layers. `to_out.0` is the O leaf. Expected configured paths must equal actual regex matches; aliases and an already attached particle wrapper are refused. |
| `yue2_native.attention_targets` | `model.layers.i.self_attn.{q_proj,k_proj,v_proj,o_proj}` across the configured native YuE2 decoder. Architecture, finite parameters and existing routed wrappers are checked. This selector does not generally detect aliased projections. |
| `native_lora.lora_target_plan` | Class-name and `train_method` selection used by `LoRANetwork` and Music3. Returns `(export_name, module)` without allocating, attaching or consuming RNG. Overlapping class selections deduplicate module identities. These are historical name-based policies, not universal semantic attention discovery. |
| Iris / Qwen product banks | Product-owned attachment; `IrisResponseAdapter` and `QwenImage21ResponseAdapter` reuse the real bank and particle parameter. Their constructors do not select target layers. |

Do not replace these native policies with a shared suffix match: Fibo's trunk,
refiner and configured injection-attention requirements differ from YuE2's AR
decoder and Music3's class-based plan. Use the same product selector for attach,
export validation and identity; a second independently maintained list can drift.

`TinyParticleProjection` replaces the selected host's `forward` and computes:

```text
base = original_linear(x)
delta = up(RoutedMLP(down(x), actual_shared_particles))
output = base + delta * (multiplier * alpha / rank)
```

The branch reads the **same input** as that Linear, then adds to its output
before the caller's later operations. It does not wrap a whole attention block,
move native norms or add another trunk residual. Q/K/V projections therefore
change values before native attention processing; O changes its projection
output. Exact norm placement inside the enclosing native model remains owned
by that model, and must be documented by its integration.

For this primitive, the last axis is `in_features`; arbitrary leading axes
survive as `[..., out_features]`. Routing operates per leading coordinate; it
does not pool tokens or interpret spatial axes. This kernel property is **not**
permission to pass arbitrary tensors through the GAN's native-domain gate.
The native input/base may use reduced precision; adapter leaves are FP32, the
branch explicitly casts to its weights, and the delta returns to input dtype
and device. A wrong final feature width fails in the Linear operation. There is
no mask, timestep or tuple-output argument on this projection API.

## Readout: preserve genuine geometry, conditioning and time

`FrozenResponseAdapter` prepares detached teacher/neutral responses and
recomputes the student through the attached adapter. `ImageResponseAdapter`
accepts frozen native encodings, not strings, and currently supports DiT/UNet
adaptation with frozen text encoders. Live text-encoder training needs a separate
binding such as Anima's live encoding path; substituting frozen embeddings
would sever that gradient path.

| Readout | Supported coordinates and important limits |
|---|---|
| Image families | One example: genuine `[1,C,H,W]` spatial fields. Krea2 and Qwen Image 2.1 require `[1,C,1,H,W]`, with singleton image-frame axis. Native Flux/Krea/Qwen packing is reversed before the response reaches PG. Cascade is a spatial image-embedding prediction, not RGB. |
| Fibo edit | Native FP32 input `[1,C,H,W]`, channels equal configured `in_channels`, patch-divisible spatial sizes. Transformer owns patch/context packing; response is raw flow velocity. Training uses matched neutral/positive reference fields at the same captured neutral `x,t`, four guidance-free steps and no local trajectory solver. |
| AR last hidden | `ARLastHiddenResponseAdapter` requires finite `[1, hidden_size]`. Music3's binding reads `model.model(...).last_hidden_state[:, -1]` on one **unpadded** row with all-ones attention mask and `use_cache=False`. YuE2 reads the last native AR state through its supplied backend, not logits/NAR outputs. No acoustic rendering follows from this readout. |

The shared `validate_pg09_domain` gate in the inspected source accepts genuine
4D/5D continuous fields and flat continuous vectors of width below 170 on its
exact release/auto pair. Wide native AR states remain refused. Do not project,
truncate, tile, relabel as tokens or reshape a vector into a fake image to pass
this gate. A tiny 32 fixture proves software behavior, not a native 2048/4096
backend. Check the installed profile's explicit gate when adopting a later pin.

Mask meanings are native, not interchangeable:

- Fibo conditioning `mask` is a boolean **valid-token padding mask** `[1,T]`
  for `[1,bundles,T,features]` embeddings. `None` means no explicit padding
  mask; an explicit mask must contain a valid token. Shape, dtype and all-false
  masks are rejected. Ordered `context_0..context_N` are 1..4 reference lattices.
  A pixel edit mask goes to the native pipeline and requires exactly one
  same-sized source image; it is not an adapter token gate or a GAN loss mask.
- Qwen 2.1 `with_encoding` forwards `prompt_embeds_mask` as
  `encoder_hidden_states_mask` and constructs native `img_mask` from the
  all-false `image_pad_mask` plus target image slots. Any true image-prefix bit
  is refused by the text-only training protocol. `predict` returns the native
  target-image suffix; per-row forwards preserve prefix-dependent RoPE layout.
  These helpers do not supply comprehensive shape/dtype validation for all
  inputs; the product and native pipeline must validate their encoding layout.
- The AR response rows use token tuples, with no padding mask API. A nonzero
  `history_tokens` must leave a prompt prefix and match the exact suffix in
  both contexts. This is continuation-history identity, not diffusion time.
  Music3's separate `_gather_last_hidden` helper uses `mask.sum()-1`, which
  assumes contiguous right padding; it is not a general left-padding selector.

There is no shared token-region mask or automatic timestep router. Image binding
conventions declare native time explicitly: Qwen/Flux/Krea/Fibo divide raw time
by 1000, ZImage uses `(1000-t)/1000` and negates output, Supra uses increasing
flow time, and SD/SDXL retain scheduler-specific input scaling/prediction type.
See `ImageResponseAdapter.conventions` and the backend's `metadata()` for the
exact family. Do not introduce another conversion in a session callback.
Mixed-precision generic trajectories, discrete/non-flow trajectories and Iris
rendering have explicit qualification refusals. Native production samplers
belong in product render callbacks, not a generic core solver.

## Trainable, scale and cleanup ownership

`AdapterParameters` registers the actual nonempty, unique, trainable FP32 leaves
without cloning them. A routed bank must be exposed once under `particles`;
adapter-only GANs use `particles=None`. Frozen base and fixed adapters must stay
outside optimizer ownership, while autograd can still pass through them to an
earlier adapter or trajectory latent. `eval()` does not disable autograd; a
`no_grad()`/`detach()` on the student path does.

Teacher capture runs frozen/no-grad at scale 0. The generated response remains
differentiable. Core rejects detached or disconnected adapter parameters and
checks finite gradients. It does **not** require every gradient to be nonzero
on every update. In particular, zero-up initialization preserves the base but
initially yields zero gradients upstream of the up matrix. Test nonzero routed
gradients with a warmed-up up matrix as well as the zero-initialized case;
do not change initialization or GAN mathematics to satisfy a probe.

Scale has one native owner: `AdapterParameters.scaled(s)` delegates to the
product context and records the effective scale for checkpoint replay. The
projection applies the public strength once to its residual, retaining its
existing alpha/rank factor. Local ordinary LoRA may also have documented
sequence gains. `lora_scale_context` scales all supported local/PEFT layers
and **every entry of each PEFT scaling dict in its supplied roots**; it does not
identify fixed versus trainable adapter names. A product with fixed companion
adapters must use an explicitly selective context and test that their weights
and scales remain unchanged. The additive
[`peft_adapter_scale_context`](PEFT-SCALING.md) supplies strict exact-name
selection for supported PEFT Linear leaves; an existing product-owned selective
wrapper remains valid when independently verified. No product is migrated here.

| Backend/binding | Scale owner in inspected source | Selective-helper adoption |
|---|---|---|
| Fibo | `FiboParticleSlider.scaled` | Owns routed projections; PEFT helper does not apply |
| YuE2 | `ParticleSlider.scaled` | Owns routed projections; PEFT helper does not apply |
| Music3 | `ParticleNetwork.scaled` | Owns routed projections; PEFT helper does not apply |
| Iris / Qwen product banks | Product `adapter.active(particles, strength=...)` | Preserve actual bank ownership; PEFT helper does not apply |
| Anima / Supra compatibility bindings | `_adapter_parameters` delegates to historical `_lora_scale_context` | Full-root PEFT mixed hosts require explicit selected names or a qualified product context; no automatic change |
| Image PEFT / Krea2 compatibility binding | `image_lora_parameters` delegates to historical `_lora_scale_context` | Same fixed-companion hazard on mixed roots; adoption belongs to the product factory |
| H3/Pika Flex task | Product-owned custom wrapper, outside this reviewed diff | Inspect that wrapper and coordinate with its owner; no forced replacement or qualification claim |

At scale 0 the routed Tiny/Music3/YuE2 branches explicitly bypass their delta.
For the same deterministic host inputs this preserves the base output exactly.
Do not generalize bit equality to every ordinary LoRA implementation, stochastic
backend, reduced-precision kernel or full native sampler. Compare with matched
RNG/native state and declare numerical tolerances where needed. A projection's
residual is linear in its multiplier at fixed input; a complete multi-layer
model or sampled output is generally nonlinear in slider strength. Unipolar
range limits and signed support are backend capabilities, not a global choice.
PEFT numeric scale zero does not bypass A/B execution; nonfinite deltas can
still produce NaN. The new helper preserves fixed companions but does not change
that native PEFT numerical behavior.

Scale must survive backward/recomputation. Common lifecycle `advance` owns the
outer context through the update; selected checkpoint blocks additionally
capture each forward's scale. Use `checkpoint_with_adapter_scale` handles or the
model's supported public checkpoint hook, never an uncaptured legacy wrapper.
See [CHECKPOINT-OWNERSHIP.md](CHECKPOINT-OWNERSHIP.md) for cleanup and conflicts.

The **factory is the lifetime owner**. It must unwind partial attachment, capture
borrowed model state before mutation, retain hook/checkpoint handles and return
an idempotent `Binding.close`. Temporary Fibo capture hooks remove themselves in
`finally`. `native_cleanup` restores modes, requires-grad flags and checkpoint
settings; it does **not** remove projection-forward replacements. Likewise,
`TinyParticleProjection` has no universal detach handle. Restore those forwards
through the product owner, with ownership checks before overwriting another
wrapper. Close checkpoint wrappers before broader adapter/model restoration.
Session `close` marks itself closed before calling the callback; an unresolved
cleanup conflict needs a retained native owner/ledger, not a second session close.

## Public extension pattern and identity

Backend authors should compose the existing contracts rather than teaching the
common lifecycle another model's module names:

```python
from particle_sliders.lifecycle import Binding
from particle_sliders.pg09_backends import AdapterParameters

# Inside the product's complete prepare(options) factory:
# 1. Validate capabilities/options before loading. Verify local base/fixed-adapter
#    revisions and content; retain restoration state before attaching anything.
# 2. Resolve one canonical native target plan; attach the actual product slider.
network = AdapterParameters(slider.named_parameters(), slider.scaled)
# 3. The native response adapter exposes this same network and metadata(),
#    prepare(row), generated(prepared); no detached student tensors.
# 4. Construct ConditionalPG09 with genuine response_domain/shape and the real
#    bank (or particles=None). Return only after all admission checks succeed.
binding = Binding(game, close=close_native, observation=observe_native)
```

This is an ownership sketch, not a standalone factory. The complete runnable
registration examples are in [IMAGE-LIFECYCLE.md](IMAGE-LIFECYCLE.md),
[RESPONSE-SESSIONS.md](RESPONSE-SESSIONS.md) and `examples/response_fixture`.
`Backend.validate`/`preflight` and capability description allocate no weights.
`ResponseBackend` additionally checks actual response shape/domain/profile/arm
against its declaration. Unsupported operations fail before preparation where
the descriptors can determine refusal. Backend registration alone is not
qualification; do not bypass native preflight, operation gates or domain refusal.

Record a stable cut-point description in product metadata/receipts: native path
and role, input/output axes/dtypes, placement, mask/time/CFG semantics, scale
owner/range and target/readout convention. No new required checkpoint field is
introduced here. Existing training signatures bind backend metadata, native
architectures, adapter names/schema, conditioning bytes, recipe and implementation
source identities; full session packets add factory/lifecycle/config/capability
identity. Native-only v2 has a separate product-supplied model/protocol identity
and no training cache requirement. Mismatches are rejected before state mutation.

These checks bind **what the product actually supplies**. Source/architecture
hashes do not authenticate frozen weight values. A filesystem path and layer
count are insufficient evidence of a model or fixed-adapter revision. Fibo's
staging receipt verifies exact revisions and staged content; older Music3 native
loader identity is path/architecture-oriented. New integrations need verified
content/revision receipts, not a claim that the core invents them. Changing
source-bound runtime code creates a new cohort even if tensor schemas match;
use the old pinned loader for old checkpoints. Existing runtime modules and
packet formats remain unchanged; the new opt-in helper records its own source
identity and selection, which a product must include in its metadata on adoption.

Regular-LoRA distillation is a separate approximation: `fit_routed_down` fits
the teacher's routed bottleneck by a homogeneous linear down map while retaining
the up matrix. It is not an exact conversion of a nonlinear routed adapter.
The product must retain native teacher/model/fixed-adapter identities, target
paths, calibration data/procedure and ridge settings, source/dependency pins,
raw/EMA selection and held-out native-versus-distilled error/quality results.
Label the artifact as distilled and record its approximation limits. The shared
ridge solver neither writes that complete receipt nor qualifies a voice/model.

## Review findings and conformance evidence

The native geometry/selection code lives in compatibility/native binding modules;
the session lifecycle contains no model-specific projection names or solver.
This is the boundary to preserve. Some legacy discovery uses `_roots`,
`_adapted_nn_modules` and bounded attribute traversal (`backend`, `runtime`,
`model`, `transformer`, `slider`, `adapter`) for provenance. Arbitrarily hiding a
new native owner behind another attribute is not a supported identity seam.
Prefer the explicit `AdapterParameters`/factory interfaces; extending provenance
traversal needs its own source/identity review.

Projection-attachment rollback/detach is intentionally factory-owned; missing
rollback/removal in a particular factory would be a defect, not evidence that a
global registry is required. Remaining native-owner work includes verified model
identities in old bindings and product adoption/qualification of selective PEFT
scaling where needed. The shared helper closes the explicit-selection API gap;
it does not establish every product's fixed-companion guarantees. A new generic
hook registry would not resolve those ownership/provenance obligations. Snapshotting conditioning
also overlaps between `ImageResponseAdapter.prepare` and its base, but removing
that duplication changes source identity without fixing a demonstrated cut-point
bug, so it is left intact.

`tests/test_cut_points.py` exercises module-selection purity/deduplication,
projection axis preservation, fixed-input fractional scaling and zero bypass,
wrong feature widths, distinct injection/readout tensors, detached teacher versus
live student gradients, unchanged frozen host values and continuation-history
refusal. Existing checkpoint ownership, Fibo native, image lifecycle, response
lifecycle, dependency direction and strict state tests cover adjacent contracts.
The new tests also exercise Fibo padding-mask semantics/refusals and temporary
capture-hook cleanup after repeated native failure while preserving foreign hooks.

Executed on 2026-10-31 against the source above: **143 passed in 7.81s**, including
16 new cut-point cases, with Python 3.14.4, Torch 2.15.0+cpu, official
ParticleGAN 0.9.3, pytest 9.2.0, NumPy 2.5.4, safetensors 0.8.0 and Pillow 13.0.0.
The installed PG Python source-map digest matched
`966027b710f3abb48ece4cfe0261f12fd119fc8e29eb43aa2f8957e406ecdae1`.
From the repository root in the isolated test environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q packages/particle-sliders-core/tests/{test_cut_points,test_checkpoint_ownership,test_conditional_pg09,test_fibo_native,test_image_lifecycle,test_response_lifecycle,test_dependency_direction,test_compact_inference}.py
```

No native libraries or model weights were installed. CPU fixtures do not
qualify pretrained layouts, fixed product adapters, GPU kernels or acoustic/media
outputs, and passing tests do not remove the follow-ups above.

That receipt predates the optional real-PEFT follow-up. See
[PEFT-SCALING.md](PEFT-SCALING.md) and its separate test receipt for the additive
helper; historical runtime/helper behavior is not silently changed.
