# Anima / NTC Image Studio session resume

## Latest UX changes, 2026-09-20

User renamed the product **NTC Image Studio**, with visual inspiration from
`/ml2/music/app/static/`. That project was read only. Internal package paths stay
`lumen_studio` to preserve commands and identities. Visible branding, API title,
CLI description and PNG download names use NTC.

Slider checkpoint dropdowns are removed. `latestCheckpoints` automatically picks
the highest-step current compatible EMA for each offered atmosphere; archived
weights are excluded, and catalog refreshes retain mix proportions. Rerender
still uses the image's saved checkpoint hashes. Off/On, sweeps and neutral/target
references are grouped and labeled directly in both image listings. Blind mode
hides labels/grouping. Backdrop clicks close the native dialog, while clicks or
drags beginning inside it do not.

Follow-up UX corrections restore compact thumbnail groups and enlarge the
selected image by default. **Compare** in the viewer reveals its matched image.
**Max** replaces Solo and sets only that fader to 1.00; other faders and Energy
are retained. The form removes Batch and numeric Width/Height, replacing them
with model-owned resolution presets and square/portrait/landscape icons. Seed
defaults to blank (random) and remains settable, including seed zero. The hero
heading is smaller and the explanatory subheading is removed. Status badges and
energy labels sit below the thumbnails so they never cover the artwork.

Verification: all 69 Studio tests pass; desktop/mobile browser checks pass for
actual latest hashes in submitted requests, archived/incompatible exclusions,
catalog refresh, missing checkpoints, comparison grouping, blind mode, modal
backdrop/inside/drag/Escape behavior, thumbnail-to-viewer enlargement, Max keeping
other controls, all five presets, optional seed behavior and overflow. Browser fixtures never submit
generation to the real queue. Evidence: `artifacts/anima/ntc-image-ux/`.
Training formulation, weights, target definitions and Music processes unchanged.
API workers after the resolution update: 526468 (localhost), 526693 (Tailscale).
GPU worker remains 352156; runtime hash remains unchanged. Re-read live status
before any process operation.

Updated 2026-09-20; Theatrical retired from the product; user selected Dusk as its replacement. Two Dusk definition drafts were previewed and need revision before training. This session is back on Anima/NTC Image Studio only.

## Project and user intent

- Implementation repo: `/ml2/sliders/sliders-conceptmod`.
- Dedicated environment: `.venv-anima`; preserve the pinned dependencies and
  verified A6000 CUDA launch policy. Do not replace it with a generic Python launch.
- Studio: <http://100.90.104.57:8876/>; progress: `/#training`.
- Main references: `docs/lumen-studio.md`, `docs/anima-benchmark.md`,
  `configs/anima/`, and `lumen_studio/`.
- Standalone text-to-image Studio is named NTC Image Studio (formerly Lumen). Anima Turbo v1.1 is the first
  backend; additional backends, img2img and public deployment are follow-ups.
- Continue the originally authorized slider campaign, verification, development
  selection, and final held-out character/mixture audits. Preserve the ParticleGAN
  formulation. Do not treat a finished training run as a qualified slider.
- Preserve Music Studio and other GPU users. Use local GPUs; do not top up RunPod
  or incur charges beyond previously authorized existing credit. The former
  RunPod instance was already removed.

## Live state at this check

- All six target caches complete: 1,224 paired trajectories total.
- `candlelit-1600-9f82568d`: completed 1,600/1,600. Last raw development residual
  1.487609; best-so-far 1.233479. Check development image reviews before selection.
- `moonlit-1600-b36faf35`: completed 1,600/1,600. Last raw development residual
  1.584974; best-so-far 1.238437. Neither completed run is automatically qualified.
- The user requested replacing Theatrical because it looked broken in the UX.
  It is now retired from the mixer, and its runs/definitions are grouped under
  expandable history. Theatrical is not qualified for the full campaign. Original and v16/v17 pilots
  remain archived. v17 resolved a measured definition-direction conflict but its
  200-update image review still failed: three major clothing regressions across
  six development comparisons, no atmosphere wins. Do not silently promote it.
- The numerical diagnostic completed successfully and restored the regular
  GPU 1 worker under the verified A6000 CUDA policy. Inspect `/api/status` before
  launching GPU work; do not start a duplicate worker.
- Read-only APIs: `/api/training/progress`, `/api/training/runs`,
  `/api/training/images`. Summarize responses rather than dumping metric arrays.

## Accepted user changes and constraints

- Energy range is now **0–10**, explicitly requested by the user. Nonnegative
  mix faders still share the energy budget; zero bypass remains exact. High
  energy can change clothes/settings; successful API rendering is not quality
  qualification. Original strength-0-to-1 audits remain relevant.
- The user requested less time sampling and more time training. Automatic
  previews were reduced to updates 20, 100, 200, then 400/800/1200/1600.
  Checkpoints and probes remain every 100 updates. Do not restore the heavier
  preview schedule without a reason.
- Seed 7 is the run's deterministic initialization/sampling seed; it is not a
  replacement for the dataset's explicit image seeds.
- Preserve paired-prompt shared fields, train-only normalization, frozen runtime
  identity, split isolation, bare-prompt tests, and raw versus best-so-far metrics.
- Pinning/parity work has been completed; consult the benchmark evidence before
  changing batching, kernels, precision, checkpointing or optimizer execution.

## Next work

The numerical convergence assessment requested in this turn is complete. For
continued campaign work, review Candlelit/Moonlit development checkpoints and
qualify a replacement for retired Theatrical within the unchanged formulation. The new tests
do not establish convergence or demonstrate reliable short-search slider gains.
Use validation/development evidence for selection. Open untouched final-test
characters only for the selected checkpoints' final audit. Keep regressions,
energy-ordering failures and mixture failures visible.

## Jev review and global skill, 2026-09-20

User requested Jev comparisons and asked to make the TypeSafe skill available
across projects. Copied SKILL.md and LICENSE, verified byte-identical, from
`/ml2/music/.agents/skills/typesafe-ai` into
`/home/mikkel/.codex/skills/typesafe-ai`.

Jev currently takes text only. A first screen used recorded Codex observations
of 32 randomized A/B development pairs, then Jev scored appeal, atmosphere and
unrelated change severity. It is not independent visual validation. Details and
guidance: `docs/anima-jev-review.md`; reusable tool:
`scripts/anima_jev_review.py`; raw requests/responses and originals:
`artifacts/anima/review/jev-20260920/`. No extra GPU renders, final-test access,
training changes, formal review submissions or checkpoint promotions occurred.

Appeal wins: Candlelit 400 and 1600 both 7/8; Moonlit 400 6/8 with two ties,
Moonlit 800 7/8. Later checkpoints have higher described unrelated-change scores.
Prioritize comparing energy curves and inspecting clothing/crop/medium drift.
Jev's next-concern choice changed under an A/B label swap; treat it as advisory.

Subsequent user feedback rejected the usefulness of this text-mediated image
judging workflow. Exclude the Jev scores from checkpoint selection and future
training guidance; inspect matched images directly. The global TypeSafe skill
remains installed for suitable text tasks. Existing Jev artifacts are historical
only, and there are no scheduled Jev evaluation jobs.

## Numerical convergence request, 2026-09-20

The user can no longer distinguish improvement visually and wants the numerical
assessment completed. Implemented fixed-input prediction movement for both live
and EMA checkpoints (1200–1600), plus 16-update D-only and G-only responses on
disposable copies of checkpoints 1400 and 1600, each with two independent sampler
seeds. Production formulation, saved weights and optimizer states remain intact.

Implementation: `lumen_studio/convergence.py`, `scripts/anima_convergence.py`.
Contract: `docs/anima-convergence.md`. Dashboard: `/#training`, endpoint
`/api/training/convergence`. Artifacts:
`artifacts/anima/remote/diagnostics/convergence/`. The artifact supervisor is
`artifacts/anima/convergence-supervisor.py`; its `supervisor.json` stores the
original worker command, diagnostic PID, exit status and restored worker PID.

18 relevant CPU/API tests passed. D/G response phases reproduce production
parameters and optimizer states exactly on active/inactive cap steps. Browser
checks pass on desktop and mobile, including fully populated report fixtures.
Candlelit finished all ten prediction passes and eight response trials. Last
100-update prediction movement: live 54.0%, EMA 37.7% of prior edit. Final D-only
trials improved; G-only results were mixed (-0.3493 and +0.1304 objective
improvement, the latter interval crosses zero). Original source hashes are
unchanged and all existing probe metrics reproduce within 1.2e-7. Convergence
is not established. Moonlit also completed all ten prediction passes and eight
response trials. Its last-100-update movement is live 64.6%, EMA 37.6%. Both final
D-only trials improved; both G-only improvement intervals cross zero. Its probe
metrics also reproduce within 1.2e-7, and original sources are unchanged.
Both reports are complete and served on the dashboard. Supervisor exit code 0
restored the regular worker (recorded PID 352156; re-read live status). Detailed
results and limitations are in `docs/anima-convergence.md`. No final-test data, checkpoint promotion, new gallery, or Jev review
is part of this assessment.

## Theatrical replacement, 2026-09-20

Product availability is separate from immutable model/training contracts in
`lumen_studio/atmospheres.py`. `/api/catalog` exposes `mixer_variations` and
`retired_variations`. Theatrical checkpoints remain in the complete catalog for
history/rerender, but no longer appear in the mixer; new Theatrical training is
rejected. Its failed runs and definitions remain under expandable history.
Candlelit/Moonlit weights, target caches, definitions and runtime identity are
unchanged. Desktop/mobile mixer, Solo, retired history, and definition checks pass.

The user selected **Dusk — amber/violet twilight** after an optional direction
question. Dusk is the replacement direction; Neon previews are historical drafts
and must not guide new training. Draft specs live in
`configs/anima/candidates/`. The isolated draft compiler
`lumen_studio/candidates.py` reuses the production shared-field layouts and prompt
assembler, with 6 training/2 evaluation definitions and isolated train/dev rows.
It leaves active manifests, registered variations and model identity untouched.
Queue previews with `scripts/anima_atmosphere_candidate.py <spec> --queue`.
Four training-character pairs were rendered per Neon draft; six per Dusk draft.
Drafts and
queued job IDs are persisted under `artifacts/anima/remote/definition-candidates/`.

v1 had strong color but major hair/clothing/scene changes in all four pairs.
v2 matched lighting geometry in each neutral/positive pair and improved three
cases, but the seated character still changed hairstyle/crop. v3 used simpler
color-only wording: one case ignored the lighting target, and two changed hair.
Neon is not qualified. Dusk v1 records explicit matched neutral-lighting
clauses and six amber/violet training definitions plus two unseen paraphrases.
Both Dusk batches cover all six training definitions on six training characters,
leaving final-test characters untouched. V1 has the intended warm/cool mood but
changes a braid, pose, clothing and background elements in several pairs. V2
removes broad twilight-atmosphere wording; its first case preserves the character
well but has little violet, and other cases still change clothing, hair or pose.
Neither draft is qualified. Direct inspection notes and job IDs are saved with
each immutable candidate. No targets have been cached and no replacement
training has started. Dusk is the user-selected direction, not a renamed
Theatrical checkpoint. The mixer labels Dusk **In development**; candidate image
captions identify Dusk and their draft/definition IDs. Sixteen relevant Studio
API and candidate compiler/coverage tests pass. Desktop/mobile checks pass.

Normal GPU 1 worker PID 352156 remained active throughout these renders; no new
training or diagnostic worker was started. API PIDs after the catalog update:
471876 (localhost) and 472000 (100.90.104.57); always recheck live processes. The
runtime hash is still `ca564ceaf401f96198258d773ef0c50eb586fe5b9937122b95ee4e3226cea076`.
Further work needs a Dusk definition revision and qualified paired references,
then explicit compatibility work to register a new slider name without
invalidating existing checkpoints. Do not add Dusk to the hashed legacy
`contracts.VARIATIONS` or reuse Theatrical's identity as a shortcut.
