# Historical distillation adoption

The YuE2 distillation and refinement scripts are historical cohort wrappers and
migration inputs. They are not a model-independent shared-core training entry
point or part of the default shared-core 0.14 wheel.

Their historical regression, refinement, selection and diagnostic calculations
remain unchanged. In particular, the old FP32 subtraction, epsilon denominators
and averaged region/strength MSE are **not** the current RAW FP64 operand-first,
per-example pooled relative-SSE ranking protocol. An integrity receipt proves
which historical inputs produced an artifact; it does not give that artifact
current metric qualification or authorize its use to rank a new campaign.
Future product and metric migration is separate work.

New first-stage, refinement and audio attempts bind their complete effective
cohort: script/helper/core source content, the native runtime source digest,
model/VAE shard and configuration identities, tokenizer content, model parameter
layout, Python/numerical package versions, execution/generation settings,
teacher weights, release catalog, prompt/calibration rows, held-out token files,
and stage settings. Refinement additionally binds the verified first-stage
receipt. Audio additionally binds the verified refinement receipt and target
checkpoint. Output manifests hash every required checkpoint, sidecar, rollout,
recording, semantic-token file and per-record JSON.

The native pipeline supplies its verified full weight/config and runtime
identities. These scripts do not identify a model by path, mtime, a friendly
model name, or an unverified partial marker.

A completion or rollout is adopted only when its identity and output hashes
match. Old markers and caches without identity are refused and remain on disk.
Mismatches, missing outputs, changed outputs and uncommitted exports or partial
audio are explicit refusals; they are not deleted, silently promoted, resumed
as full optimizer checkpoints, or rerolled. Use a new output directory for a
new attempt. Matching, sealed teacher rollouts may be reused after an
interrupted calibration before an export was written. Published JSON receipts
are write-once, and per-stage directory locks exclude simultaneous writers.
Error reports get unique filenames so a refusal preserves earlier errors too.

These adoption checks have standard-library-only regression tests. They do not
load models, execute a native runtime, or claim a new native/RAW qualification.
