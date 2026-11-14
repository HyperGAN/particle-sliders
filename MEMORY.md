# Memory requirements

These are **calculated planning budgets**, in GiB (2³⁰ bytes), dated 2026-11-14.
They are not measured peaks or hard allocator limits. This repository supplies
the engine; each model's root `MEMORY.md` supplies its complete requirement.
The [combined repository index](docs/memory-requirements.md) links every filled
product profile and its largest documented GPU/host planning budget.

| Operation / bounded profile | GPUs | Calculated VRAM per GPU | Calculated host RAM | Limits |
| --- | --- | --- | --- | --- |
| Command planning / controller only | 0 | 0 GiB | 1 GiB | At most 1 MiB of request/manifest JSON; no numerical backend imported |
| Numerical engine **add-on example** | Product-defined | **2 GiB additional** | **4 GiB additional** | ≤1,000,000 FP32 trainables including critic/bank, ≤32 MiB response batch, ≤128 MiB CPU response cache, ≤2 protected losses; base/activation/KV memory excluded |
| Complete training or sampling | Product-defined | Product `MEMORY.md` | Product `MEMORY.md` | Model, precision, input length/geometry and residency must be selected |

There is no finite model-independent maximum: backend weights, sequence length,
native response size and caller-supplied caches are not globally bounded here.
Do not add the example allowance again if a product's ledger already includes it.

## Calculation

For a controller, allow 512 MiB for Python/dependencies, 64 MiB for four expanded
copies of 1 MiB JSON at a conservative 16× expansion, and 1 MiB streaming hash
storage. This is 577 MiB; rounding to **1 GiB** leaves scheduling headroom.
These are planning assumptions, not enforced input limits.

For the numerical example, let `T = 1,000,000` trainable values. FP32 weights and
gradients alone use `8T = 8,000,000` bytes. The default zero-momentum DualNorm
optimizer does **not** imply two Adam moment arrays. DirectionBlend additionally
uses original parameters, protected gradients, displacements and FP64 projected
normals. Budget **96T bytes in total** for trainables and these transient
directions at two protected losses, rather than counting persistent state only.
Add ten 32 MiB response/noise/penalty buffers, 512 MiB solver scratch and 1 GiB
CUDA/runtime allowance: `96,000,000 / 2^30 + 320/1024 + 0.5 + 1 = 1.902 GiB`,
rounded to **2 GiB**. Larger matrices, more protected losses or another optimizer
need a new ledger.

The host example allows 1.5 GiB framework/interpreter storage, 1 GiB checkpoint
serialization headroom, 128 MiB cached responses, 64 MiB metadata and `12T` bytes
for additional checkpoint tensor copies: **2.699 GiB**, rounded to **4 GiB**.
This excludes native model loading and all additional product caches.

Sources: [the paired native game](packages/particle-sliders-core/src/particle_sliders/pg09.py),
[optimizer policy](packages/particle-sliders-core/src/particle_sliders/optimizer_policy.py),
[lifecycle and serialization](packages/particle-sliders-core/src/particle_sliders/lifecycle.py),
and [exact PG0.9.5 source manifest](packages/particle-sliders-core/src/particle_sliders/pg095_release.json).
The manifest identifies the recipe and `optim/{dualnorm,constraint_geometry,direction_blend}.py`
implementations used for state accounting. No weight-EMA copy is budgeted;
fresh sessions use `ema_decay=None` and raw weights.

## Required product format

Every product keeps **`MEMORY.md` at its repository root**, linked near the top
of its README. Fill the first table with numeric GPU/host planning budgets for
its supported profiles. Include input limits, precision, GPU count and placement,
attention/checkpointing, offload/residency, component bytes, adapter/critic/state,
activations/KV, CPU loading/cache and explicit workspace headroom. Use the maximum
of non-overlapping stages and the sum of concurrently resident components.

Use parameter/config/index arithmetic and cite exact model revisions and source
paths. Quantized storage includes scale metadata and unquantized islands;
checkpoint bytes, scheduler caps and card capacity are not runtime requirements.
Distinguish calculated budgets from any separately retained measurements. Recompute
the document when model, precision, shapes, backend or residency changes.
