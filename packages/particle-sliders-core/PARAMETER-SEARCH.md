# Shared raw parameter search

`particle-slider-search` runs bounded, resumable hyperparameter searches through
model-owned backend bindings. The shared runner owns proposals, matched step
budgets, retained response coordinates, exact checkpoint replay, and ranking.
A product binding owns model allocation, native responses, admission and full
trial state, delegating optimizer mathematics to its declared engine.
For model setup, see the [product repositories](../../docs/products.md).
Importing the search contract does not import Torch or any product runtime.

The current raw-only config schema is `particle-sliders-raw-parameter-search-v2`; historical manifests/checkpoints are refused,
not migrated or relabelled. Historical validation receipts remain under
`legacy/parameter-search/20261104/`; they do not qualify this implementation.

## Ranking and confirmation

The default score is RAW teacher/student FP64 operand-first relative SSE, used
as a proxy for relative effect:

```
example_error = sum((float(S) - float(T)) ** 2) / sum((float(T) - float(N)) ** 2)
score = mean(example_error for each example)
```

Each sum pools every declared coordinate and position within that example.
Examples receive equal weight. Operands must reach CPU as original coordinate
values (for example, `tensor.detach().cpu().reshape(-1).tolist()`), before any
subtraction or reduction; do not provide low-precision precomputed differences,
projections, scalar summaries, or generated text comparisons. The target is the
actual backend training target, including a signed formulation if applicable.
Neutral and teacher identities are frozen across candidates and seeds.

This proxy is an acceptable default for search. Measuring the intended effect
in matched finished outputs is an optional improvement to the search objective,
not a prerequisite for using this API or its existing results. It does not
require repeating completed training, replacing historical scores or adding a
fresh confirmation pass. The current runner retains its one fixed proxy metric;
an output-effect search would declare its scorer, matched cases and objective
as a separate explicit protocol rather than add a second ranking gate.

Each FP64 coordinate term enters an exact streaming integer superaccumulator,
rounded once per example statistic. Chunk boundaries cannot affect the score.
There is no denominator epsilon: `min_edit_energy` defaults to zero and must stay zero. Zero/nonfinite
pooled teacher energy or nonfinite primary SSE/ratio makes the example
unscoreable and refuses that run. A zero-edit position is valid when its whole
example has positive pooled energy.

Lower finite scores always rank first, without tolerance bands. Simplicity and
candidate IDs break only exact score ties. Architecture and regularizer-disabled
cohorts have separate initialization/rungs; their finalists may be compared only
inside the same frozen target/grid/budget/seed protocol.

The finalist in each cohort is measured at the final common step budget with
independent training seeds and then a heldout complete grid. Heldout data never
feeds exploration/refinement. `confirmed_raw_metric` means this measurement and
exact replay evidence exist; it is not a quality or generalization claim. It
does not require an error below one, positive signed gain, every scene improving,
an interior optimum, native-model evidence, trained OFF, or media approval.
Those measurements/limits are diagnostics only. Optional invalid media
attestations are recorded as diagnostic errors and cannot veto the score. Later
external media drift is reported by the offline auditor as a nonblocking
diagnostic discrepancy; all retained RAW evidence remains strictly checked.
Overflow in preservation energy or signed-gain diagnostics becomes `null` and
cannot veto an otherwise finite primary score. Optional algebraic controls can
be computed with `metric.algebraic_controls`; they are not native OFF evidence
and are not automatically required by search.

## CPU example

After installing this branch's core wheel:

```sh
particle-slider-search plan --config examples/parameter_search/vector.json
particle-slider-search run --config examples/parameter_search/vector.json --out /tmp/raw-vector-search
particle-slider-search audit --out /tmp/raw-vector-search
```

The vector and coupled image/audio backends are tiny deterministic Python
quadratic-game fixtures. Their actual generator, critic, and particle parameters
train and respond to independent rates, regularizer strength, population, and
temperature. They are not upstream ParticleGAN optimizer mathematics, pretrained
models, media validation, or native quality evidence. Their fixed response
coordinates qualify the generic orchestration and formula only.

All configs explicitly declare `selection="raw"` and `ema_decay=null` in backend
options. Any weight-EMA request is refused before resolving a backend. No weight
shadows are allocated by these fixtures. Shared lifecycle bridges also refuse
nonraw selection/configuration. Upstream ParticleGAN scalar optimizer statistics
are unchanged.

Resource bounds are physical update calls (including exact replay/recovery),
trial count, artifact bytes, and checkpoint frequency. Budgets are fixed actual
completed steps against one frozen schedule horizon. Elapsed time is diagnostic
only, with no training/campaign deadline. No accelerator is selected by default:
execution requires an explicit device and `--allow-native` for non-CPU backends,
and product admission still applies. `plan` calls only allocation-free `describe`.

The runner exclusively locks its output directory. It writes immutable evidence
receipts before nomination, retains failed/orphaned outputs, and refuses changed
source/config/frozen identity or modified retained bytes on resume. Repeating an
identical completed command reuses evidence without updates. This source-bound
resume intentionally refuses even an implementation change; start a new run.
The offline auditor verifies all raw metric artifacts, full checkpoints, raw
exports, replay receipts, proposal/rung order, and independent confirmation.
Checksums establish local artifact consistency, not an adversarial signature.

## Product backend contract

Register one `particle_sliders.search_backends` entry point exposing `describe`
and `open`. `describe(options)` must be allocation-free, validate raw-only
configuration, and return `Capabilities` with a complete frozen model/teacher/
precision/data/coordinate/schedule/source/device/determinism identity; disjoint
train/dev/heldout grids; dev/heldout teacher-neutral digests; and actual supported
parameters. Unsupported, disabled, ambiguous, and no-op knobs must be refused.
The independent `generator_lr`, `critic_lr`, and `particle_lr` roles correspond
to optimizers that really participate; an absent optimizer is not advertised.
A regularizer that can be disabled declares an objective cohort, and any changed
parameter architecture declares an architecture cohort.

`open` returns the `Trial` protocol. It must own its runtime and close resources,
report actual effective values and real optimizer rates/parameter counts (each
role's base rate must equal its advertised absolute candidate rate),
complete exactly one update per `step`, and save/restore full raw weights,
optimizers, streams, RNG, native state, horizon, and completed counters. Observation
and raw export must preserve all training/native/RNG/mode state. The runner checks
state digests and ambient RNG and performs an exact one-update restore/replay
for every candidate/seed. Product bindings must enforce raw-only behavior
internally as well; a descriptor alone cannot prove a third-party implementation.

`LifecycleTrial` delegates to the existing shared lifecycle rather than inventing
a training loop. A product supplies an observer that binds frozen identity,
actual effective parameters/groups, initial state, native state, and full raw
coordinate chunks using the session observation transaction. This bridge has a
real small CPU lifecycle regression against released ParticleGAN; native product
backends remain individually unqualified until their own installed tests pass.
