# FinalForm and Supra FinalBoss: audited GAN rerun proposals

These are **proposals awaiting resource-owner admission**, not completed reruns.
Only new isolated files were added. Active canonical MOSS/shared-core sources,
original caches, exports, images, queues and services were not changed. No live
model weights were opened or loaded; existing CPU teacher tensors and prior
weight-verification manifests were read. Other held queues are not classified
as quality failures here.

New bridge: `conceptmod/textsliders/pg09_remaining_campaigns.py`. Shared core
`ConditionalPG09` owns every GAN objective, optimizer, RNG stream, balanced
sampler and step. Both proposals explicitly select public `bispf_optimized`,
native continuous fields and adapter-GAN-only (`particles=None`). There is no
copied training loop, MSE optimizer/resume/fallback, or disconnected particle
bank. Resource admission and the process/checkpoint shell remain owner work.

| Case | Actual training data | Frozen edit RMS | Proposed full cap | Balanced coverage |
| --- | --- | --- | --- | --- |
| MOSS FinalForm | 24 captions ×16 seeds ×2 paths ×10 times ×2 signs =15360 rows | 0.2522578997899252 | 3840 updates ×8 | two complete D and G cycles |
| Supra FinalBoss | 240 edited states +30 preservation states; native [1,4,32,32] | 0.32844100447280516 | 28800 updates ×4 | 192 complete D and G cycles over600 logical rows |

The audit and source byte hashes are in `evidence/pg09-remaining-campaigns`.
`proposals/*.proposal.json` bind those facts and remain `DRAFT_NOT_ADMITTED`.
Regenerate source paths/hashes after any reviewed integration and check actual
Python import origins before loading components. No proposal is a dispatch file.

## FinalForm

All384 training bundles and two development bundles are retained. The scale was
computed over training deltas only, with every target SHA checked through the
existing lazy provider. Original embeddings are BF16; latents, times, base and
deltas are FP32. Native timestep is1000×sigma. Original teacher contrast is
positive minus neutral, while inference CFG uses an explicit empty prompt.

`build_final_form_campaign` binds an already-admitted transformer and existing
`MossCampaignRows`; it reuses `build_campaign_run` without editing it. It rejects
heldout data, shortened3840/b8 settings and incomplete15360-row data. Rank4,
alpha4, seed1000, the4101 context/time/sign features and conditional-only CFG
are reused. Native Torch2.9.1 and the reviewed source-verified PG0.9 overlay are
required: the prior BadIntent Torch2.11 parity failure must not be repeated or
waived. A three-update probe uses the full3840-step recipe and needs both
in-memory and disk-backed fresh-process exact resume before sustained training.

The preserved MSE run stopped at **254**, with matched renders at **200**. Its
recipe planned1600; it never completed1600. Both200 and254 adapter hashes are
recorded. Compare GAN3840, MSE200 and MSE254 under the same new conditional-only
CFG/empty-negative/no-skip protocol, labeling each accurately. Retain archived
images. Use both original development prompts, seed29001,512px,50 steps and all
five existing variants: off/direct-positive/direct-positive-CFG1/minus/plus.
This yields30 common-protocol images across three arms.

The proposed generator budget is30720 signed forwards, versus4064 at MSE254
and25600 for the old planned1600 cohort. It provides complete balanced coverage,
not a small fixture run. Actual memory/time remain subject to the admitted probe.

## Supra FinalBoss

Archived source/current native runtime, trainer and vendored DiT match byte for
byte. Model revision10dec6e4..., native DiT source640851f4..., Flan-T5 revision
7bcac572..., VAE31f26fde..., rank16 and alpha16 are pinned by the audit. The old
runtime used FP32 weights with CUDA BF16 autocast under Torch2.11.0. It preserved
checkpoint-stored unconditional context. No inferred replacement is acceptable.

Training cache: six subjects ×2 seeds7063/7064 ×2 frozen trajectories ×10
positions (t0,.1,...,.9) =240 edited states. Three preservation prompts ×10 times
=30 states, with target exactly equal to frozen base. Both teacher responses use
CFG3 and the same cached native latent/time. The training-only RMS above measures
`target-neutral`; preservation states are excluded from scale fitting.

Separate validation cache contains240 edited velocities,60 preservation
velocities and12 endpoints, seeds39001/39002. It is excluded from training and
scale. Bridge heldout and fruit are excluded from selection, matching the archive.
Validation path labels were not stored; their optional condition feature is
explicitly unknown(-1), rather than inventing a path. Training path labels are
required. The new provider uses pooled masked neutral T5 features normalized by
pooled RMS plus native time/scale/strength/guidance/path/preservation:774 features
with the real768-wide encoder, no row ID.

**Prerequisite:19 required contexts are not archived.** The owner must admit a
bounded preparation phase on the actual native runtime. Call
`export_live_supra_contexts(runtime, config, path, runtime_identity=...)` with
verified model/T5 revisions. This live wrapper checks the exact19-context config
(canonical SHA d32639ad...) and exact model/T5 revisions before delegation. The
generic exporter also rejects more than19 unique contexts before any encode or
output. It calls native encode at most19 times (18 T5 calls
when the stored unconditional is present), performs zero updates, and writes a
new safetensors file plus actual runtime source hash, Torch version, revision
identity and every embedding/mask tensor hash. There are no placeholder contexts
or downloads. Use `load_supra_contexts` with the reviewed file hash. The campaign
builder requires every context to be present and byte-identical in the admitted
native runtime; it cannot silently re-encode a missing prompt.

`SupraCampaignRows` retains the13MiB cache on CPU and transfers selected fields
only. Two copies per edited state and four per preservation state give600 logical
rows, an exact4:1 edit/preservation frequency on this actual cache. This retains
the old one-in-five preservation exposure. The old optimized MSE weight0.1 is
**removed**, an intentional change required by the GAN-only objective; it is not
carried into another loss or represented as equivalent GAN mathematics. All
preservation rows remain native frozen-base targets in the conditional game.

`build_supra_campaign` binds the admitted native runtime/ordinary LoRA to the
shared core and existing native-field critic. It requires fresh zero-up native
LoRA; reviewed full GAN state is restored only after building. An already loaded
MSE adapter is rejected. No optimizer or loss is copied.
Full28800/b4 gives115200 generator examples:92160 edited and23040 preservation,
matching the prior documented cap's exposure allocation and192 balanced cycles.
The six-hour wall cap remains hard; canary timing must establish whether this
fits, with no automatic extension or budget reduction. Save every400 updates and
retain a1600 comparison milestone. Probe stops after3 on this full28800 recipe.

The old product applied the adapter to both CFG branches. The new adapter uses
conditional-only CFG for training/inference; **every old and new comparison arm
must use that same correction**. Frozen teacher batch2 versus split branch
arithmetic must pass the unchanged measured teacher-parity gate on admitted
hardware before updates. No tolerance relaxation is proposed. Native Euler uses
positive dt=1/50 and t0→1; initial noise must use native CUDA `runtime.noise`, not
CPU-generated noise. Rendering rejects latent nonfinites before VAE and float
VAE nonfinites before image conversion.

Compare GAN28800, actual MSE1600 benchmark and selected MSE27200 converged export.
The selected run stopped28400; its comparison-grid label "Original / step400"
actually referenced1600. Preserve that provenance correction. Match verification
knight/cave/bridge/fruit, seeds42/1234,256px,50 steps, CFG3, scales0/.5/1 and
frozen-positive teacher. This yields96 images over three arms. Fruit teacher is
its unchanged prompt, retained as a control. Existing reviews report poor bridge
transfer despite good trained effects; numerical convergence is not visual success.

## Handoff blockers and example

```python
# Called only by the resource owner's admitted worker; no components loaded here.
contexts = load_supra_contexts(frozen_context_path, reviewed_context_sha, config)
rows = SupraCampaignRows(teacher_cache, cache_sha256=reviewed_teacher_sha,
                         config=config, contexts=contexts, split='train')
run = build_supra_campaign(admitted_runtime, rows,
                          model_identity=verified_frozen_model_identity,
                          total_steps=28800, batch_size=4, response_scale=0.32844100447280516)
# Owner runs teacher_parity, disk-backed probe and generic shared-core step/checkpoint shell.
```

Pending actions are resource admission (both cards are occupied by BI/Afterimage),
Supra's bounded19-context preparation/export, actual full-model teacher parity,
memory/timing canaries, reviewed worker/checkpoint/render shells, regenerated
actual-source pins, fresh-process GPU resume/export/frozen-base gates, full budgets
and matched heldout review. None was attempted by this CPU preparation session.
The FinalForm worker must bind200/254 comparison paths rather than inheriting the
BadIntent/Afterimage MSE1600 restriction. Canonical core and active workers stay
unchanged throughout this proposal preparation.
