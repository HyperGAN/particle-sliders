# Clean main integration candidate

Base: 20fdf98d25455582e7e7191eef8ef51cf73173ae (locally fetched origin/main).
Historical review milestone fd89e1c remains unchanged on its preserved review
branch. PR142 must retain its reviewed target; do not retarget it to main.
This separate candidate is a file-level transplant, not a cherry-pick of PR141
or a merge of its unrelated ninety-commit ancestry.

`clean-core-file-closure.json` records the selected paths. Main's package/license,
teacher/reference provenance and retired alias remain. Legacy compatibility
changes required by released PG0.9.1 are explicit in the diff. The shared
ConditionalPG09 engine and image/native modules come from the reviewed
snapshots; lifecycle 0.8 adds separately reviewed native-only inference and
extension hooks. No GPU experiment, MOSS campaign, audio/text launcher, model
weights or qualification receipt is transplanted. No derivative imports remain.

Review this diff against the exact base, then select one exact shared commit for
all derivative proposals. Derivatives adopt their own native inference factory
and hook/capability declarations before running installed-wheel CPU conformance.
Do not silently relabel old checkpoints: source updates mean new cohorts.
All original reference branches/checkpoints/receipts stay on their original pins.

Music3/YuE2 modality/profile support and Sana's archive-faithful online native
provider require subsequent bounded work. HyperGAN external run reporting has
an independent upstream implementation; use the observer seam rather than
adding a competing renderer. See compact-inference-plan.md for the deferred
scope, and NATIVE-INFERENCE.md for the precise supported 0.8 contract.
