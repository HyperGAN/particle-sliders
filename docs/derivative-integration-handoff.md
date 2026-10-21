# Parent-owned integration milestone

Finish/review the shared interface first. The parent then creates one fresh
Codex task for each verified derivative in `derivative-inventory.json`; this
interface task does not spawn those tasks or claim that generic support has
migrated them. Resolve Sana's canonical product repository separately. Preserve
existing source/result cohorts, unpublished work, and all current merge holds.

## Starting evidence

Use the exact shared interface commit and build/source receipts in the local
milestone handoff. Its branch is stacked on the reviewed f797/cdf context, not
an approved main integration. `image-lifecycle-integration.md` specifies the
separate package-only main candidate. Do not merge PR141's unrelated stack or
retarget PR142 to main. A later clean candidate needs a new common pin/locks.

The four local image integrations prepared before the fresh-task workflow was
clarified are **reference drafts**, not published/merged migrations. Their local
heads and source archives are in the milestone receipt. Fresh tasks can review
and reuse them, then report remaining rough spots back to the interface owner.
Original public review work must remain available:

| Product | Existing review payload to preserve |
|---|---|
| Iris | PR1 `7aef066491c10fb98277d59cb9c0469b5bf9be8b` |
| Krea2 | PR4 `afb329014caffac825b9c29b9ae85ec23e0d7f72` |
| Qwen Image | PR1 `31e797f290c58bda74bb2e7e69ad2aeef2b4244b` |
| Fibo | PR1 `10ee426c0c35c9f11e365b8af4b80c679f233bde` |

For Anima, YuE2, MOSS, Sana, Ming, Music3, Supra and further products, the source
inventory identifies existing bindings and gaps. It is not an actual product
entrypoint audit. Each fresh task should inspect its current PRs and checkpoints
before selecting a base; a default-branch inventory observation is insufficient
to discard unmerged work or rebind a running cohort.

## Integration task instructions

1. Inspect the product's real loading/train/resume/evaluate/export/infer paths,
   native dependencies and source/cohort receipts. Record the actual generator
   arm and parameter ownership. No dummy-only stamp imports qualify adoption.
2. Depend on particle-sliders, which depends on ParticleGAN/HyperGAN. Product
   factories/callbacks point into core; core must not import the derivative.
   Keep native model dependencies in product environments. No copied GAN loop,
   penalty, prior or optimizer law. Use the dependency-direction guard.
3. Bind one complete owned factory to the public lifecycle. Preserve native
   conditioning, geometry, time, discrete/flow policy, real bank or particles=None,
   strength/reference/precision gates and cleanup. Reject unsupported rendering
   before allocation. Do not map AR hidden-state evaluation to waveform output.
4. Run bounded CPU-only installed-wheel conformance outside source trees:
   exact split resume including metrics/EMA/RNG/coverage; invalid restore before
   mutation; adapter-only export/load; exact zero and supported signed/fractional
   strengths; observation state/RNG preservation; setup/update failure cleanup;
   capability and dependency diagnostics. No model downloads or GPU scheduling.
5. Read `hypergan-interface-seams.md`. HyperGAN owns run rendering; the session
   exposes on_step/manifest/artifact seams. Native inference rendering is a
   separate callback. Definitions should use the agreed HyperGAN/HANDL builder;
   inventory custom definitions and return gaps to that separate task. Preserve
   native pretrained backbones behind explicit reviewed boundaries.
6. Return exact local heads, wheels/source hashes, commands/receipts, implemented
   versus unmigrated entrypoints, and concrete interface rough spots. Do not
   publish, merge, schedule GPU work or update active training pins here. The
   parent coordinates review, updates to this interface and repeat compatibility
   checks until each derivative fits.

## Known boundaries to carry forward

- New source pins are new cohorts. No silent old checkpoint rebinding or result
  receipt relabeling. Legacy schemas remain available through original loaders.
- The initial image inference loader reconstructs its prepared game/cache to
  enforce strict source/model/data identity. A compact inference-only loader is
  a future interface iteration, not hidden caller assembly.
- One legacy MOSS core-to-product import remains explicitly inventoried and
  frozen by the guard. Migration should inject its slider factory in a new
  source cohort. Ongoing MOSS GPU experiments are untouched.
- HyperGAN run-renderer integration and replacement of current raw-torch custom
  definitions are not completed by this session API. No competing framework or
  unreviewed pretrained-backbone rewrite is included.
- CPU fixtures establish code contracts; pretrained sampling/decode quality,
  memory, convergence and GPU parity remain separately unqualified.

## 0.8 follow-up contract (new cohort)

The preserved fd89e1c/0.7 milestone is not rewritten. Read
`packages/particle-sliders-core/NATIVE-INFERENCE.md` before adopting 0.8.
Add a separate `particle_sliders.inference_backends` native-only factory and
`Binding.inference` specification, then use `export_inference`/`open_inference`.
Do not rebuild a teacher cache/game during v2 inference. Test in fresh Python
with training files absent, training constructors forbidden, and raw/EMA labels
verified. V1 artifacts retain their original explicit loader; they are not
converted automatically.

Declare real `Capabilities.operations`, `unavailable_reasons`, precisions and
render formats. Empty operations/precisions honestly describe Ming's missing
native implementation. No generic solver-parity reason is imposed on unrelated
backends; Iris must supply its own solver-parity reason. Latent-only formats
now work as the default. `Backend.preflight` handles native requests before
allocation; `Binding.before_update` handles pure native admission. Keep Anima's
gates until its callback is integrated/tested. `Binding.observation` restores
online/native provider state around observation, and optional `native_cleanup`
and `new_directory` helpers make factory ownership explicit. These helpers do
not prove that any unmodified derivative has adopted them.

The session-level inference gate is earlier diagnostics, not a repaired PG
training bypass: PG already blocked updates from inference exports. Nested CUDA
preparation devices are discovered without routing CPU indices through CUDA.
Each derivative still owns truthful native device support and source-bound tests.

Sana is confirmed as a shared-repository prototype (task-14 audit
779333e9e8dd1af48bd9d892949cd0e7a8677fb7), not a new standalone repository.
Music3/YuE2 response modality/profile and the needed audio source closure are a
subsequent bounded revision. Retain historical pins for absent audio modules.
HyperGAN external RunReporter bridging and matched-weight HNDL migration are
separate tasks; use existing on_step/manifest/detached observation references.

For main use the separate clean image candidate. Its selected closure excludes
historical MOSS/audio/campaign files; core AST dependency guard has zero product
edges there. Do not update all derivative pins to a historical stack by default.
Choose/review the common exact head first, then rerun each isolated product's
installed-wheel conformance and retain its original receipts unchanged.
