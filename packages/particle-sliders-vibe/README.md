# Particle Sliders Vibe

Persistent best-of-N training-data curation with model-specific workflows.
Keep successful prompts, generate replacements for weaker ones, and retain
every finished output and judgment as data accumulates.

The base collector installs without Torch or a particular core version.
It can coexist with a product's pinned Particle Sliders training runtime.
The separate `particle_sliders_vibe` namespace leaves that core's source closure
unchanged. Install `[nanoodle]` for the workflow adapters.

Use `particle-slider-vibe --vibe ... --workflow product.module:factory --out ...`,
or `particle-slider collect` in a core version with collection dispatch.
Full protocol, persistence and scoring details are in
[VIBE-COLLECTION.md](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/VIBE-COLLECTION.md).
Anima supplies the first product renderer and frozen-target manifest handoff. The [shared `--vibe` instructions](https://github.com/HyperGAN/particle-sliders/blob/main/docs/vibe.md) take a qualified collection through training, alpha selection and release.

The collector stores its provider request namespace in `request-scope.json`. Keep
this file with the collection when resuming or moving it. Each fresh collection
has its own request IDs, including when two collections use the same settings.
The `request_id` in each call receipt identifies the provider request. When an
older collection resumes, recorded calls keep their original IDs and completed
rounds keep their original identity.

If the proposer returns an invalid batch, the collector saves a failed receipt
and stops. Running the same command again requests a replacement batch under a
fresh ID. Completed renders and scores are reused. A pending provider call still
requires reconciliation before `--retry-pending` retries its original ID.


Image products that implement a native `VibeHost` can call `image_cli.entry` to train a
character slider from its phrase. Install the `image` extra alongside the host's CUDA and
ParticleGAN dependencies. The workflow keeps per-scene image judgments and structural
checks, retries unqualified scenes, freezes matched fields on both trajectories, and
searches a constant strength after the 448-update RAW export. Open `teachers.html` in the
run directory to see the starting image, all three candidates and the selection for
each scene and round. The measured run prints the ranking and confirmation scores,
working case counts and structure verdicts. Product READMEs give their model
locations, sampler and replay commands.

## Train and sample a text vibe

Install the `text` extra in a CUDA environment. For Glimmer, install both
`text` and `glimmer`. Set `CUDA_VISIBLE_DEVICES` to one GPU UUID and provide
`NANOGPT_API_KEY` for Jev to score the teacher and evaluation replies.
Generation from a saved slider needs only the local runtime and snapshot.
The text extra pins PyTorch 2.13.0. Install its CUDA build for your driver
first when using a platform-specific wheel index.

```sh
particle-text-vibe --model tiny-aya --snapshot "$AYA_SNAPSHOT" \
  --vibe "playful explanations with an unexpected rhyme" --out ./rhyming-aya
particle-text-vibe --model tiny-aya --snapshot "$AYA_SNAPSHOT" \
  --adapter ./rhyming-aya/slider.safetensors --prompt "Explain why leaves change color."
```

Use `--model glimmer` for Muse Glimmer 30B. The supported snapshot revisions
are `af89d219b53ed9b13b8a4645f9c8028973510324` for Tiny Aya Global and
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68` for Muse Glimmer. Supply the
local Hugging Face snapshot directory named for that revision. Tiny Aya uses
BF16 and Glimmer uses NF4 with BF16 compute. Glimmer generates in the ATEM
user channel. Both use greedy decoding.

The trainer compares three prompt treatments per context and retries missing
contexts with a new task. It requires 16 qualifying contexts. For each kept
pair it freezes both final-layer hidden responses at matched continuation
tokens. Both text products use Anima's rank-eight routed attention branches,
128 × 4 particle bank, and the same `ConditionalParticleGAN` stock EMPF
update from ParticleGAN 0.12.0. New DOWN matrices and the particle bank receive
scoped initialization. The backbone stays frozen and all UP matrices start
at zero. The critic operates on final-token hidden vectors. The default
recipe takes 448 updates on the 1,024-step optimizer schedule, batch two,
with critic and particle rates of 0.009 and 0.03. The generator rate is
0.00075 for Tiny Aya and 0.000375 for Glimmer. The original Glimmer rate of
0.00075 produced degenerate repetition at update 448 on the rhyme run. Model
weight EMA is disabled. Each quarter of the update budget saves RAW weights and
checks a generated reply for catastrophic failure before the next update.

Use `--frozen-source PRIOR_RUN` with the same phrase and base revision to train a
fresh candidate on identical teaching data. Its manifest binds the teacher
manifest and targets by SHA-256. Set `--g-lr` to test a generator rate while
keeping the critic and particle rates fixed.

The run directory contains the teacher verdicts, frozen targets, RAW weights,
strength leaderboard and matched held-out replies. Calibration compares
constant, early and late token strengths on a separate panel. The numeric
strength and schedule travel in the exported adapter. Qualification includes
an intact held-out panel and exact token replay after reloading the export.
`result.json` records the measured effect, including weak or negative gains.
