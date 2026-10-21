# Clean main integration candidate

Public main base:17cd484ef00601389256a30f6d52fc79991bcb3d.
The clean reviewed lifecycle history starts at20fdf98d25455582e7e7191eef8ef51cf73173ae
and ends atf6807a91546e43e198645a3f8925d2dc7c9e03b4 (officialPG0.9.1).
The merge preserves those immutable ancestors and the main README update.
Core0.8.3 adds only a narrow legacy compatibility layer and integration docs.
Historical review milestone fd89e1c remains unchanged on its preserved review
branch. PR142 must retain its reviewed target; do not retarget it to main.
This separate candidate is a file-level transplant, not a cherry-pick of PR141
or a merge of its unrelated ninety-commit ancestry.

`clean-core-file-closure.json` records the selected paths. Main's package/license,
teacher/reference provenance and retired alias remain. Legacy compatibility
changes required by released PG0.9.1 are explicit in the diff. The shared
ConditionalPG09 engine and image/native modules come from the reviewed
snapshots; lifecycle 0.8 adds separately reviewed native-only inference and
extension hooks. No GPU experiment, MOSS campaign, model weights or raw qualification payload
is transplanted. The minimal five-module numerical audio closure and standalone
response CLI retain their exact inventoried historical sources. No derivative imports remain.

Review this diff against the exact base, then select one exact shared commit for
all derivative proposals. Derivatives adopt their own native inference factory
and hook/capability declarations before running installed-wheel CPU conformance.
Do not silently relabel old checkpoints: source updates mean new cohorts.
All original reference branches/checkpoints/receipts stay on their original pins.

Numerical Music3/YuE2 response registration, CLI v2 export, provider observation
and reversible checkpoint ownership are included. Official native wide response
refusal remains; Sana's archive-faithful provider migration is separate work. HyperGAN external run reporting has
an independent upstream implementation; use the observer seam rather than
adding a competing renderer. See compact-inference-plan.md for the deferred
scope, and NATIVE-INFERENCE.md for the precise supported 0.8 contract.
