# Modern shared optimizer and evaluation contract

This source target is core0.11.0 on exact official `particlegan==0.9.3`, wheel
`fd7ee424d4c67e1dc3cf0b9dd345222049a67e8c293c30c8a3c42d54805affec`, source
profile `release-0.9.3`. No0.5 overlay, PG0.9.1 alias, unofficial continuous-domain
proposal, source-path injection or model download is admitted by this target.
Earlier091/093 cohorts and artifacts remain immutable. This change is a new
source/numerical-policy cohort, not an old-checkpoint migration or package release.

## Gram ownership and actual dispatch

`ConditionalPG09(..., optimizer_dense_profile='auto')` selects upstream
`gram_fp64_svd_diagnostic_v2` for full DualNorm with zero momentum, smoothing.001,
and native/batched SVD. The common policy changes only the upstream Recipe's
`optimizer_dense_profile` field; all factorization/guard/fallback/optimizer math
lives in official ParticleGAN. `auto` retains unsupported optimizer laws, with
an explicit reason; an explicit unsupported Gram request rejects. Supported
selectors are `auto`, `default`, `gram_fp64_svd_diagnostic_v2` and the stricter
`gram_fp64_guarded_v1` (which deliberately has no rank fallback).

Dense FP32 CPU/CUDA two-dimensional DualNorm leaves participate. Bank rownorm,
1D/vector updates, convolution/quintic/per-offset paths and Adam/formulation arms
retain upstream implementations. No formulation, matrix shape, rank, smoothing,
prior or global Torch setting is silently changed to force eligibility.

```python
# A complete registered product factory owns game construction. New products
# pass this selector to ConditionalPG09; consumers supply only local options.
with open_session(config) as run:
    run.train(stop_after=2)
    print(run.game.optimizer_diagnostics)
    run.checkpoint('training.pt')
```

Diagnostics expose actual upstream calls/Gram passes/rank ambiguities/SVD
fallbacks, eligible dense-leaf counts and original group roles/algorithms.
A profile flag is not proof of a Gram call: a vector-only adapter can have zero
eligible matrices. Detached diagnostic counters/timers are not checkpoint state
and upstream resets them after a successful load; rejected loads preserve them.

The fallback catches ONLY the upstream exact Gram rank-ambiguity condition. It
uses the same device's FP64 SVD (CUDA gesvd), original FP32 strict rank cutoff
and absolute.001 smoothing. Ambiguous SVD rank decisions are diagnostic, not
rank certification. Other guard/technical failures propagate. An optimizer may
have written earlier leaves before failure; the shared game becomes unusable
for training/export/checkpoint until the entire matching pre-step state is
restored. No partial retry with another solver or changed loss is introduced.
Autograd through this optimizer solver is unsupported; model forward/backward
and their parameter gradients remain native and are tested separately.

## Budgets, read-only splits and native observations

The shared game accepts a declared positive `total_steps`; session continuation
uses that original absolute budget. A product's previous fixed64 contract must
be explicitly revised for a new256-update cohort, never by editing an old packet.

A product factory keeps its native dev provider alive and transfers it with the
same cleanup/observation owner:

```python
from particle_sliders.lifecycle import Binding, EvaluationRows

# Within the product's complete factory, after local native preparation:
binding = Binding(game, close_all_owned_resources, observation=observe_native,
    evaluation_rows={'dev': EvaluationRows(dev_rows, dev_rows.identity())})
```

```python
# Consumer: train rows remain the default; no private provider mutation needed.
with open_session(config) as run:
    dev_field = run.evaluate(index=0, split='dev', strength=1., seed=7)
```

`train` always uses the original training provider. A named split must be
registered explicitly with stable data/protocol identity, bound in the session
manifest; changing that identity rejects resume before mutation. Unknown splits
and invalid index types/negative bounds reject before native access. Lazy length,
fetch and response forward run within the same seeded observation transaction,
including failure restoration. The product observation hook owns native caches,
buffers/samplers and other state outside the adapter. `Binding.close` owns every
transferred provider; the common session closes it once. CLI supports
`particle-slider evaluate --split dev ...`; output receipts identify split/row/
seed/strength. This exposes native fields, not a media renderer.

Native products with their own guarded field-fetch path can instead supply
`Binding(..., response=observations.response, observation=observe_native)`.
The callback signature is `response(index=0, *, split='train', strength=1., seed=0)`.
The core enters the same observation transaction before invoking it. The callback
must validate supported splits and bounds before fetch, preserve native admission
and return guards, and return a detached response snapshot. Negative/noninteger
indices and empty/nonstring splits reject centrally before invocation. Other
unsupported splits reject inside the native transaction before provider access.
Do not combine `response` and `evaluation_rows`. Callback source hash and qualified
name enter the manifest; native DEV data/protocol identity must be explicit in
factory config. No private training-provider substitution is required. This is
an observation callback, not a second training lifecycle.

Rendering remains `Binding.render` with allocation-free request preflight and
capability gates. MOSS owns its native four-step sampling/Off reference; the
shared core does not invent a sampler or own a competing run renderer. HyperGAN
owns reporting. Raw288-column historical MOSS and modernXY290-column critics
are different architectures; use fresh matched initialization or a separately
reviewed migration, not exact-resume claims.

## MOSS and Fibo adoption

Both owners must consume the exact new wheel/source map/profile receipt and
produce fresh installed product qualification. Old strict locks are not relaxed.
MOSS's112-field official093 Recipe is preserved except the selected dense profile;
its native builder/budget/render/dev-provider closure stays product-owned.

Fibo portable HNDL definitions are included from the separately authored
network-only candidate. Use `particle-sliders-core[fibo-hndl]` in a separate
Python>=3.11,<3.15 environment with exact HNDL0.9.0 and HyperGAN source
`bff6de3d5162f151180d64513b662eb79fbebb2a`. `hypergan-hndl` exposes the same
shared dependencies for other native network builders. The old optional
readout `[hndl]` remains a distinct HNDL0.8.2 architecture/environment; do not
co-install the incompatible extras or silently substitute readout networks.
Fibo's new product target explicitly selects `hypergan-hndl-v1`; legacy network
profiles remain distinguishable for old artifacts and are not relabeled.
The shared Fibo factory uses this modern PG source/default Gram policy, preserving
native EDIT geometry/references and four-step CFG1 semantics. The native model
remains a pretrained product dependency. Existing56-test091 evidence is historical;
only the new artifact's receipt states modern qualification actually executed.

No GPU benchmark or pretrained training is part of this change. CPU numerical
fixtures do not establish CUDA speed, native fit, convergence, edit quality or
waveform generation. No performance improvement is claimed without measurements.
