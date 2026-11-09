# Catastrophic-output eligibility

The shared guard stops a candidate when a **qualified assessment of its actual
finished output** detects catastrophic collapse. Weak or absent slider effect is
not catastrophic failure. Rank eligible candidates separately by the one
case-weighted intended-effect ON-minus-OFF mean; retain negatives, zeros and
all tied maxima. Do not combine that score with failure probability, beauty,
preservation, teacher SSE, GAN loss or response magnitude.

The implementation is reusable infrastructure, **inactive unless a product
explicitly supplies a qualified observer and genuine finished-output probe**.
It is not an all-product deployment or a universal renderer. A qualified image
observer does not establish audio coverage. Existing model environments,
FULL/RAW packets, source identities, old scores and completed searches stay
unchanged. No EMA or optimizer update is added.

## Public API

`particle_sliders.candidate_failure` is stdlib-only. Its public pieces are:

- `FileRecord(path, bytes, sha256)` binds an actual artifact or metadata source.
- `QualifiedDetector(detector_contract, actual_instrument_source,
  actual_qualification=None, judgment_kind="probability")` requires a genuine,
  matching qualification record before an assessment or guarded model opening.
- `ObservationCache(root, detector, observe)` retains one first judgment per
  unique output bytes/native contract and exact observer identity. Its product
  adapter receives only `(output_source, media, native_contract)`, not candidate,
  LR, alpha, loss, effect score or expected labels. Artifact paths are for local
  loading; adapters must not put identifying paths into observer prompts.
- `FinishedOutput(output_source, media, native_contract)` identifies a product's
  completed image, text, waveform or discrete output. Numerical velocity fields
  and hidden vectors are not finished outputs. An explicitly legal empty output
  is still a genuine outcome to assess under its native contract; an empty probe
  iterable means missing evidence, not a healthy model.
- `CandidateGuard(state_path, candidate_id=..., scientific_protocol=...,
  media=..., cache=...)` persists eligibility for the exact candidate/protocol.
  Share its state across that candidate's seeds/rungs. Keep the checkpoint and
  inference case/strength in outer provenance, outside the observer prompt.
- `open_guarded_session(open_session, guard, checkpoint=..., probe=...,
  every=..., resume_checkpoint=None)` checks before the product opens a model,
  then delegates to the original session's public `train`/`step` API. For a
  saved-state resume, pass the exact `resume_checkpoint` `FileRecord`; a missing
  assessment cannot become an implicit pass. Direct
  `GuardedSession.restore(checkpoint_record, load_packet=product_loader)`
  verifies that exact assessed file before/after the product loader and passes
  only that loaded packet to the original session. Unbound packet dictionaries
  and other unassessed snapshots are refused.

The product supplies two callbacks: `checkpoint(session)` **commits a real full
training checkpoint** and returns its `FileRecord`; `probe(session, checkpoint)`
yields its genuine `FinishedOutput` records. Reuse saved artifacts whenever they
already exist. The checkpoint callback must reuse its actual recorded digest,
not hash/read a large checkpoint twice merely to construct metadata. The probe
owns native geometry, conditioning, rendering/decoding, allowed native outcomes
and read-only training/RNG observation. No common arbitrary-prompt generator is
invented. A response-only backend without a finished-output provider remains
unmeasured.

The wrapper invokes these callbacks synchronously at the declared checkpoint
interval and at the requested train endpoint. It commits FULL state **before**
rendering or asking an observer. A failed/unknown assessment raises before the
next update. A failure can only be detected in an output actually assessed;
the declared probe cadence is not a claim of observation at every update.
`session.checkpoint`, archival exports, native cleanup and diagnostic reuse are
not deleted or rewritten when eligibility fails.

Before promotion, call `guard.require_selection(exact_checkpoint_record)`.
A pass at another snapshot is not a pass for this checkpoint. An actual fail is
persistent: reopening the journal cannot resume, update or select that arm.
Changing inference scope or observer identity requires an explicit new binding,
not clearing a failed state or overwriting its first observations.

## Typed observer and qualification records

An actual qualification JSON uses schema
`Universal-catastrophic-detector-actual-qualification-v1`, exact `accepted: true`
and `qualified: true`, the exact `detector_contract` `FileRecord`, exact
`actual_instrument_source` object, `judgment_kind` and a `covered_media` list.
Root/product qualification evidence may include additional immutable source,
fixture and first-observation records. Merely naming an observer or passing one
image fixture does not produce this qualification. A missing, rejected,
unreadable, mismatched or uncovered qualification is unmeasured.

The observer returns `status` (`pass`, `fail` or `unmeasured`), `reason`,
`output_source`, `detector_contract`, `actual_instrument_source`, and one honest
typed judgment:

| Kind | Judgment | Interpretation |
| --- | --- | --- |
| `probability` | finite `failure_probability` in [0,1] | fail at >=0.95; pass at <=0.05; intermediate unmeasured |
| `categorical` | `categorical_failure: true/false/null` and `failure_probability: null` | true fails; false passes; null is unmeasured |

Integer0/1 or a categorical label is never presented as confidence or subjected
to the probability thresholds. The categorical observer's actual profile/source
identity and matching qualification must explicitly bind that interpretation.
Native legality prechecks remain separately identifiable; do not claim a model
heard a zero-frame waveform or assessed malformed media it never received.

The core verifies finished-output bytes/SHA before observing or reusing a cached
first, and again after an actual observer call. Admission/configuration or
missing-byte failures before an observer call produce retained **no-call**
diagnostics, allowing the first real assessment once genuinely available.
A committed ambiguous call, actual first error, failed verdict or uncertain
first is retained and never automatically repeated. Candidate journals are
atomically replaced with file and directory fsync; candidate/cache locks avoid
concurrent replacement of a failed state.

| State | Updates/resume | Selection |
| --- | --- | --- |
| new, qualified, before first declared boundary | permitted within the declared probe cadence | unmeasured |
| pass | permitted until the next required boundary | only the exact assessed checkpoint |
| failed | refused; retain FULL/RAW/old scores | refused |
| unmeasured | blocked pending explicit resolution; not a model-failure claim | refused |

## Search integration and existing behavior

`Run(..., failure_guard=product_guard_factory)` explicitly enables the same
persistent guard for the existing search owner. The factory maps its trial row
to the candidate-wide `CandidateGuard`. `Trial.finished_outputs(split,
selection)` is the product hook; `LifecycleTrial` delegates to
`observer.finished_outputs(session, split, selection)`. At each committed
nonzero checkpoint, the guard assesses that provider once. It never substitutes
`responses()` or SSE for finished media. Failed/unmeasured candidates are
excluded individually, and other candidates/cohorts continue. A failed arm's
saved numerical observations remain; an empty eligible set has no nomination.

The legacy `particle-slider-search` command still ranks its fixed historical
SSE diagnostic, with its original tie/confirmation behavior. This guard does
not turn it into a finished-effect selector or silently add a product probe.
The common CLI's existing profile/session workers are not claimed to be wired
across all products by this API addition. Products/controllers must explicitly
bind the qualified guard and their native finished-output provider. Default
unguarded legacy behavior is not an automatic healthy verdict. Current LR
recommendations still require the sole finished-effect workflow.

## Validation

`tests/test_candidate_failure.py` exercises the actual controller/search paths
with fake counter-only sessions and tiny temporary artifacts. It checks
checkpoint-before-probe, stop/no-extra-update, persistent reopen/resume refusal,
rank exclusion without dropping other candidates or changing a metric, native
empty outcomes, unknown/error behavior, first-call durability, qualification
becoming available after a no-call diagnostic, output-byte mismatches, honest
categorical judgments and rejection of vector-only probes. It imports neither
Torch nor a numerical framework and performs no native model call or training.
