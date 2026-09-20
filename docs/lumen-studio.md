# NTC Image Studio

NTC Image Studio is a standalone text-to-image workspace in `lumen_studio/`. Anima Turbo
v1.1 is its first enabled backend. Other model backends are a follow-up; the
queue, history, comparisons, reviews, and HTTP application are model-independent.
The model catalog is `lumen_studio/models.py`, and Anima's implementation is
`lumen_studio/backends/anima.py`.

This is a separate application from Music Studio. It does not modify Music's
processes, ports, environments, trainers, adapters, or defaults.

The interface shares NTC Music Studio's warm ink/brass colors, serif wordmark,
and compact faders. New renders automatically use each atmosphere's latest
current EMA snapshot (highest update, with registration time breaking ties).
Archived and incompatible-model snapshots are excluded. There is no checkpoint
dropdown; saved-image rerenders still use their original immutable hashes.
**Max** sets one fader to 1.00 without resetting the other proportions or Energy.
The generation form has one resolution selector with shape icons. Anima offers
768px square (default), 768×1024 portrait, 1024×768 landscape, and 512px/1024px
squares. It generates one image at a time; the Seed field is blank (random) by
default and accepts a fixed value, including zero.

Image history and training samples group matched comparisons by request, case,
seed, rendering settings and checkpoint identity. Off/On badges and ordered
energy sweeps are visible in the listing. Definition previews use Neutral/Target
labels so they cannot be mistaken for trained-adapter results. Group headings
show atmosphere, sample type, definition and training update where applicable.
These groups form a compact thumbnail grid. Small status badges sit below the
artwork, with energy shown beside them. Opening a thumbnail enlarges the
selected image; the viewer's **Compare** button reveals the matched reference.
Blind review hides labels and shuffles individual images. Clicking the modal
backdrop or pressing Escape closes it; clicks within the dialog stay open.

## Install and start

Use the dedicated Python 3.12.13 environment and pinned lock; do not install the
repository's legacy Music requirements into it.

```bash
uv venv .venv-anima --python 3.12.13
uv pip sync --python .venv-anima/bin/python configs/anima/requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu126
.venv-anima/bin/python -m lumen_studio prepare-model
.venv-anima/bin/python -m lumen_studio compile
.venv-anima/bin/python -m lumen_studio serve --port 8876
# A separate terminal; choose the assigned physical GPU:
.venv-anima/bin/python -m lumen_studio worker --gpu 0
```

Open <http://127.0.0.1:8876>. The server binds to localhost. Public deployment and
img2img are follow-ups. A RunPod installation can be reached through an SSH
local-port tunnel without making the application public.

Open `/#training` for live progress. Preparation shows committed paired
trajectories for each variation's training and development sets. After the
benchmark, pilot runs show their own 200-update target, followed by the
1,600-update campaign if review passes. The page updates automatically and
shows raw/best-so-far residuals plus development comparisons from update 20
onward. Preparation counts are not training updates. The corresponding read-only
endpoints are `/api/training/progress`, `/api/training/runs`, and
`/api/training/images`.

Run cards also show the latest 120 D/G loss readings, current particle VIC and
gradient-cap values, and average update time over the latest 20 updates. The
remaining-time estimate covers updates only; rendering and evaluation add time.
Partial log writes and updates beyond the published run step are excluded.

Expand **Latest probe details** for residual p95, cosine, gain, orthogonal error,
teacher SWD and game SWD. The breakdown selector shows full-strength residual
by denoising position, definition, character, framing or prompt wording. Its open
state and selected breakdown survive live refreshes. Archived pilot cards are
labeled separately and continue to show their original evidence.

The **Has the slider settled?** panel reports fixed-input checkpoint movement
for live and EMA weights, and bounded improvement tests against a frozen
opponent. It updates as measurements finish. Read the [measurement contract](anima-convergence.md)
for the fixed fixtures, objective definitions, provenance and interpretation.
Raw reports are available at `/api/training/convergence`.

The main mixer currently offers Candlelit and Moonlit. Theatrical is retired: its
checkpoints and images remain reproducible, while its runs and definitions appear
in expandable history sections. New Theatrical training requests are rejected.
Replacement atmosphere previews are explicitly labeled in image history and do
not create a selectable slider before qualification. Dusk (amber/violet twilight)
is the selected replacement, labeled **In development** in the mixer. Its draft
definitions live in `configs/anima/candidates/`; the initial two preview drafts
need preservation improvements before target caching or training.

The Training page also shows checkpoint quality reviews: completed atmosphere
wins and major regressions, pending review coverage, and the current development
choice. A failed selection checkpoint remains visible. With eight reviewed cases,
the 90% preservation requirement permits no major regression. Browser checks on
desktop and mobile verify that these failures are displayed without page overflow.

Once all six target caches are ready, `scripts/anima_campaign_pilots.py --gpu 0`
runs the six benchmark configurations and automatically queues all three pilots
with the fastest configuration that passes parity. Stop the separate worker
before starting it; this campaign process owns the coordinator afterward. The
Training view displays the active benchmark phase and counters. The script does
not approve pilot quality or start the full 1,600-update runs.

The driver verifies real-model resume before admitting a pilot. GPU execution
uses deterministic algorithms and a fixed cuBLAS workspace. An identical seeded
four-update replay must reproduce every adapter state tensor and gradient before
benchmark timing can qualify a configuration. Resume also checks critic and
optimizer state, sampler streams, EMA, and all ten exported Studio predictions.
Adam's non-capturable step counters remain on CPU when restoring a checkpoint.

The model defaults to 768px, ten steps, and guidance one. Eight and twelve steps
are supported. There is no ineffective negative-prompt field for this backend.
The seed is saved with each image. API batches increment the starting seed; comparison
rows use the same prompt, seed, model, checkpoint hashes, and rendering settings.

## Model and runtime identity

`configs/anima/model.lock.json` pins the official CircleStone Turbo v1.1
checkpoint revision `f973fc41ec7545364ac9776c2440285f43ff2a30`, source tensor hashes,
tokenizers, and the upstream Diffusers converter commit
`d5baa4fb548294f47dbca49890abd4b291204c60`.

The source checkpoint's `model.diffusion_model.` prefix is changed to `net.`
before the unmodified official converter runs. All converted files are hashed.
Diffusers embeds installation paths in its modular index; identity normalizes
only those location fields, while load-time verification still checks the actual
file bytes. Runtime source and the full dependency lock have separate hashes.
Native exports bind to that portable model/runtime identity.

The pinned Cosmos transformer repeats its padding mask across the batch. The
runtime supplies a singleton mask; supplying one mask per image accidentally
squares that batch dimension. The correction is recorded in
`artifacts/anima/runtime-fixes/padding-batch-v1/`. Existing frozen targets retain
their original hashes and identities. Reuse requires an explicit compatibility
certificate binding the old/new runtime identities and all six cache
fingerprints, with exact single-image teacher equality at all ten timesteps on
both paths. The trainer records that certificate's hash. This does not approve
larger microbatches: output, gradient, and update parity still gate each benchmark
configuration independently.

The renderer, target builder, and probes share encoding, velocity prediction,
Euler schedule, and VAE decoding. Velocity losses accumulate in FP32. Euler
receives BF16 model output, matching the official pipeline's output-dtype
rounding at every step; keeping Euler in FP32 instead fails trajectory parity.
The image latent always retains its singleton time axis.

The official model and derivative license is documented at
<https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md>.

## Definition contract and datasets

The source catalogs are `configs/anima/definitions.json` and `characters.json`.
There are six training definitions and two held-out lighting paraphrases per
variation: Candlelit, Moonlit, and Theatrical. Character identity never appears
in a definition. All characters are original adults.

The compiler builds positive and neutral prompts from one immutable shared
record. Quality tags, character, outfit, pose, framing, scene, and medium are
identical. Neutral lighting defaults to unspecified, rather than an opposite
lighting concept. Explicit neutral clauses are stored in the expanded manifest
and reproduced in corresponding prompts.

There are 12 training characters, four development characters, and eight final
test characters. Each variation has 24 training rows, with four characters per
definition and two definitions per character. Each definition spans both
presentations and all three skin-tone categories. Sixteen seeds per row yield
384 paired trajectories per variation. Development includes bare prompts.
Final-test characters and seeds are excluded from the development API.

Before caching, inspect `artifacts/anima/manifests/train.json`, render the paired
references, and record visual qualification tied to its exact manifest hash:

```bash
.venv-anima/bin/python -m lumen_studio references
# Run worker, inspect the paired images, and save the reviewed qualification.
.venv-anima/bin/python -m lumen_studio cache candlelit --split train --gpu 0
.venv-anima/bin/python -m lumen_studio cache candlelit --split dev --gpu 0
```

Qualification must contain `manifest_sha256` and `passed: true`, plus the review
evidence. A changed manifest invalidates it. A source-catalog validation pass
alone is not visual qualification.

Draft catalogs can be compiled and rendered without replacing the active
definitions. Before promoting a revision, archive the current catalog with
`lumen_studio.dataset.archive_catalog(root)`. The archive retains the exact
character/definition catalogs and expanded training/development manifests.
Existing exports and targets keep their original hashes. Selection and galleries
accept an archived global manifest hash only when every row for that variation
is identical to the current rows. Revised variations require fresh targets,
normalization, pilot evidence and reviews; their old runs remain visible as
history. Pilot reviews are bound to the active run's target-cache fingerprint.
Archived-run checkpoints cannot qualify a new run, including a fresh-seed retry
that reuses the exact same target cache and definitions.

`scripts/anima_reuse_targets.py` can carry unchanged complete paired rows into a
reviewed catalog revision. Its default is a read-only audit; `--publish` requires
matching reference qualification. It verifies source hashes and all 20 trajectory
positions, compares the complete expanded row including prompts and seeds, and
checks exact tensors after serialization. Changed rows are left for the official
target preparer. The preparer commits the new complete index, and the revised run
fits new normalization; old normalization and pilot approval cannot be reused.

## Training and normalized mixing

Frozen neutral and positive trajectories start with identical noise. Both
teachers are evaluated at each shared state at all ten positions on both paths.
Velocity fields are flattened in full, without spatial averaging. Compatible
neutral trajectories are reused by model, prompt, size, seed, and scheduler.

Per-timestep paired-edit coordinate standard deviations and median-RMS gains
are fitted on training data only, using two sequential FP32 streaming passes.
The game consumes `(v_adapted(neutral) - v_frozen(positive)) / scale[t]`, exactly
`(Delta_S - Delta_T) / scale[t]` under the shared neutral reference.

The native adapter uses rank-eight attention branches, one 128×4 particle cloud,
router width 16, branch width 48, and zero-initialized up projections. The pinned
reference gmix critic has eight tokens, width 48, one layer, four heads, and a
score bound of eight. The objective is paired relativistic error plus particle
VIC. The exact lazy gradient cap runs every fourth update. Independent D/G row
and noise streams are saved in full training state.

Adam uses generator/critic/particle rates `2e-5 / 9e-4 / 6e-3`, betas `(0,.999)`,
no weight decay, effective batch eight, EMA `.995`, and no reconstruction or
perceptual auxiliary. The noise horizon stays 1,600 during the 200-update pilot,
with edit-relative initialization and absolute hold one.

```bash
.venv-anima/bin/python -m lumen_studio benchmark candlelit --gpu 0
.venv-anima/bin/python -m lumen_studio train candlelit --until 200 --microbatch 1
.venv-anima/bin/python -m lumen_studio worker --gpu 0
```

Each variation is a separate run. Full state is saved every 100 updates and at
GPU handoff. Live weights, EMA, critic, optimizers, sampler streams, RNG state,
normalization provenance, model identity, and manifest identity are resumable.
Studio accepts immutable native EMA exports. Continuing through update 1,600
requires a passed pilot qualification; it is never inferred from finite loss.

Studio Energy ranges from 0 to 10, with its original default of 0.5. For
nonnegative mix proportions `m`, strengths are `energy * m / sum(m)`. A solo
slider at Energy 10 receives coefficient 10; three equal faders receive 10/3
each. Off/On uses the selected energy. Energy sweep extends to the selected
energy when it exceeds 1, while existing sweeps within 0–1 retain their original
range. Training and the fixed development/final audits retain their pinned
strengths. The Studio extension preserves the exact arithmetic at Energy ≤1
and does not alter the model/runtime identity or training recipe. Zero
energy or all-zero proportions bypasses adapter hooks exactly. Independent
branches receive the same layer input and are summed in canonical order. The
coefficient budget is normalized; perceptual monotonicity is an empirical audit,
not a mathematical guarantee.

## Persistence and GPU coordination

The local production instance uses GPU 1 and the verified A6000 kernel policy:

```bash
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv-anima/bin/python \
  scripts/anima_cuda_host.py --gpu 1 --sm-count 82 --kernel-policy \
  --module lumen_studio -- --root artifacts/anima/remote worker --gpu 1
```

This policy reproduced 70 RTX 3090 updates exactly and then replayed the saved
production checkpoint, both frozen teachers and a 768px EMA render identically.
Its native libraries affect only this process. The model, game and training
engine hashes remain unchanged. A fresh host must repeat qualification before
using this hardware-specific policy. Large artifacts and compiled libraries stay
outside git. The RunPod rental was released after checksum-verified transfer;
the Studio address remains `http://100.90.104.57:8876/` on the private network.

SQLite/WAL stores jobs, events, checkpoint catalog, images, review notes, training
runs, and scheduler controls. SSE resumes from event IDs. Interrupted jobs return
to the queue when the coordinator recovers; cancellation and retry are durable.
Generation metadata is stored in SQLite, PNG metadata, and JSON sidecars.

One OS lock per physical GPU prevents concurrent Studio coordinators. Training
only yields after a complete update, saves state, and releases GPU allocations.
Interactive rendering gets at most four images before another 20 training
updates. Manual training pause permits continuous rendering. No code evicts or
kills another application's GPU process.

Automatic galleries run at pilot updates 20, 100, and 200, then at full-run
updates 400, 800, 1200, and 1600. This training-first schedule replaces the
intermediate 100-update galleries at the user's request. It reduces the standard
campaign from 324 to 164 images per variation while retaining all pilot and
checkpoint-selection comparisons. Quantitative probes, saved training state,
and immutable EMA checkpoints still run every 100 updates. Manual generation,
the training formulation, and final-test/mixture audit coverage are unchanged.

API routes include `/api/generate`, `/api/comparisons`, `/api/jobs`, queue
reordering, `/api/history`, `/api/catalog`, `/api/definitions`,
`/api/training/runs`, training pause, image downloads/rerender/reviews, and
`/api/events`. Changing an immutable checkpoint file fails the affected job.

## Verification and measurements

```bash
OMP_NUM_THREADS=2 .venv-anima/bin/python -m pytest tests/lumen_studio -q
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_gpu_smoke.py
```

CPU checks cover paired fields, balanced definitions, split isolation, invalid
mixes, spatial edits, wrong neutral fields, timestep ordering, frozen weights,
particle gradients, reference loss/VIC/cap derivatives, scale-zero identity,
ordering-independent mixtures, native reload, microbatch parity, exact tiny
resume, persistence, cancel/retry, restart, failures, and GPU scheduling.

`scripts/anima_verify_reference.py --reference-root <pinned-source-checkout>`
verifies source SHA256 values before comparing the actual gmix loss and parameter
derivatives with the captured ParticleGAN source. It covers active/inactive cap
steps with nonzero penalties, generator input gradients, and particle VIC.
The results are stored in `artifacts/anima/correctness/`.

The real GPU smoke compares the pinned official pipeline at all ten timesteps,
checks 8/10/12 steps and 768px rendering, and exercises the full-sized particle
game including an active second-derivative cap. Its four updates are a smoke
test, not a pilot or quality result.

Development fixtures freeze normalization, 256 projections, and evaluation
noise. Probes report full-strength raw residual RMS/p95, edit cosine/gain,
orthogonal residual, teacher SWD, and game SWD at noise .5/1/2, with definition,
character, framing, and timestep breakdowns. The chart's raw series can regress;
only the explicitly labeled running minimum is non-increasing.

Benchmark candidates use identical streams, four parity updates, 20 warmup
updates, and 50 timed updates. Output, gradient, and optimizer-update parity
must pass before a speedup is accepted. Reports include preparation/storage,
update throughput, VRAM, and save/release costs. Compilation remains gated on
eager correctness.

For visual review, `scripts/anima_review_grid.py candlelit --pilot` downloads
matched development comparisons with randomized A/B order. Record judgments
before opening the separate `mapping.json`. The grid and its A/B assignment are
immutable on reruns; changed coverage needs a fresh output directory. Complete a
copy of `ratings.template.json`, explicitly rating every preservation and leakage
check, then submit it with `scripts/anima_submit_review.py <ratings.json>`. This
maps the recorded A/B choices to the API without revealing image identities.
After the immutable final audit has
rendered all 512 images, `scripts/anima_audit_grid.py` creates 128 blinded
four-image sweeps and an unrated review template. Its mapping remains stable on
reruns. Submit the completed template with
`scripts/anima_submit_audit_review.py <ratings.json>`; `--validate-only` checks
it without writing reviews. Submission requires all 512 explicit ratings and
matched seeds, prompts, settings, checkpoint hashes and audit identity. Atmosphere
wins/ties/losses are derived from the concealed strength ratings relative to Off;
non-monotone ratings remain unchanged for the energy-ordering report. These
scripts do not invent ratings or automatically qualify a run.
`scripts/anima_step_compatibility.py candlelit --step 200` queues matched 8/12-step
768px comparisons for two development characters, including a bare prompt.
Inspect them with `anima_review_grid.py candlelit --step 200 --purpose step_compatibility`.

Quality selection, the three 200-update pilots, full 1,600-update campaigns, and
untouched-character/mixture audits must be backed by saved run evidence. Never
interpret CPU tests, four-update GPU smoke, or an untrained Studio gallery as
completion of those stages.

Measured execution comparisons, migration parity, and overheads are recorded in
[the benchmark report](anima-benchmark.md). The live Training view separates the
original configuration benchmark from measurements on the current GPU.
