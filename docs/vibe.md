# From a vibe to a released slider

Use this template when writing the `--vibe` instructions for a `*-particle-sliders` product. The deliverable is a saved RAW adapter with a measured strength and matched examples showing what it does. Collection chooses what to teach, ParticleGAN learns the native response, and alpha selection chooses how strongly to apply the saved adapter. The product owns its renderer, teacher fields and finished-output rubric. [Collection interface](../packages/particle-sliders-core/VIBE-COLLECTION.md) · [Product entry points](products.md)

## 1. Collect matched teachers

Describe the intended change in the user's language. Say what should carry it and what should remain recognizable. For a character edit, change the character in a fixed scene. For a whole-frame style, let the whole frame change. For a voice treatment, vary speakers and utterances while holding the words and speaker fixed within each pair.

Generate an ordinary prompt and several treatments for each context. Render them with the frozen model at the same seed and native settings. Ask a modality-capable Jev judge how clearly each finished output conveys the concept. Retain pairs with a low enough neutral rating, a high enough positive rating and a visible or audible gain. Compare successful treatments across contexts for repeated designs when variety is part of the intended effect. Keep the prompts, seeds, native outputs, rubric, raw judgments and selection decisions. Aim for at least 12 distinct qualified contexts for a comparison like Anima's. Report the actual count if fewer qualify.

**Anima example.** For Final Boss, the neutral must score at most 0.35 and the positive at least 0.75. Three collections supplied 14 qualified pairs. Two repeated a design already retained, leaving 12. In one pair, a man holding a coffee mug in a greenhouse scored 0.01. Giving that same man an iron crown, a black armored coat and broad shoulders scored 0.927. [Anima collection instructions](https://github.com/HyperGAN/anima-particle-sliders/blob/main/VIBE.md) · [Frozen teacher binding](https://github.com/HyperGAN/anima-particle-sliders/blob/main/research/final-boss-vibe12/anima-final-boss-vibe12-20261124/freeze.py)

## 2. Freeze the targets and train comparable adapters

Freeze the selected prompt pairs, seeds, base-model identity, sampler and teacher responses. Share this source cache across training recipes. A recipe that changes a target should record the transformation, its parameters and the derived cache fingerprint. For diffusion, compare the teacher's neutral and positive predictions at the **same latent and timestep** on each captured path. For another architecture, document its native response and the matched condition in which it was captured. [Shared response sessions](../packages/particle-sliders-core/RESPONSE-SESSIONS.md)

**Anima example.** Twelve scenes × two trajectories × ten denoising positions gave 240 fields. The late-cues recipe used the usual Boss-minus-neutral edit at positions 1–7. At positions 8–10 it blended 65% of that edit with 35% of the stronger cue’s edit, evaluated at the same state. A soft foreground mask weighted the subject at 1.0 and the background at 0.45. Each field was rescaled to the original edit norm. Stock ParticleGAN trained the adapter for 192 updates at batch size 2 with the base model frozen and `ema_decay=None`. [Target implementation](https://github.com/HyperGAN/anima-particle-sliders/blob/main/research/final-boss-vibe12/anima-final-boss-vibe12-20261124/adapters/spatial_stock.py) · [Training recipe and fields](https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/final-boss-raw192/README.md)

Compare recipes at the same update budget, with matched initialization and native evaluation cases. Use raw weights throughout and save the RAW artifact's hash. Start a new product from its own trained rates and supported budget. Keep the stock ParticleGAN update when experimenting with teacher construction. [Training lifecycle](../packages/particle-sliders-core/TRAINING-COMMAND.md) · [Learning-rate selection](../packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md)

## 3. Find the best measured alpha for each saved RAW

Choose the finished-output score before the search and use it for both tuning and the leaderboard. Specify what counts as effect, what an actual defect looks or sounds like, and whether repeating the same solution across contexts is undesirable. Judge OFF and ON on matched prompts and seeds. For a concept like Final Boss, a clean output with zero concept gain earns zero. Score deliberate changes according to the concept. Record missing judgments as unmeasured.

For an image concept, start with these three Jev questions and calibrate their rating anchors on examples from the product. Ask the first on OFF and ON individually, the second on the finished ON, and the third on pairs of ON outputs from different contexts.

| Measurement | Reusable question |
| --- | --- |
| Concept strength | How clearly does the [intended subject or region] convey [concept]? |
| Accidental damage | What visible rendering accident would need repair for this to read as an intentional finished work? |
| Repetition, when relevant | How much does the same underlying [design or treatment] repeat across these two [subjects or contexts]? |

Use the **same saved RAW** for every probe and keep the matched renders, judgments and resolved gains. For native multi-step qualifications that compare schedules, evaluate constant, early and late strength on the sampler's own steps. Where the product supports it, start at α 1 and probe 0.75 and 1.25 on a small search panel. Compare finalists and relevant earlier settings on a common development panel. Use scalar strength on a single application. Name the product's search entry point and objective. The shared `alpha-tune <slider-definition>` command uses the profiles and cases in the slider definition. [Native schedule guide](strength-schedules.md) · [Shared alpha-tune implementation](../packages/particle-sliders-core/src/particle_sliders/alpha_tune.py)

Anima's unattended Final Boss, 1990s Anime and Visually Appealing Character studies freeze each recipe's rates and RAW first, then search one constant alpha applied to all ten denoising steps. For 1990s Anime and Visually Appealing Character, select alpha by the mean matched gain in the intended effect. [Anima study protocol](https://github.com/HyperGAN/anima-particle-sliders/blob/main/VIBE.md#rate-and-checkpoint-selection)

**Anima Final Boss.** Choose alpha and rank the board by the **working Boss score**. The goal is one intact final boss in the original scene. Once the character reads as a final boss, greater intensity earns no extra credit. Record signed Boss gain as a diagnostic and leave likeness to the original person out of the score.

1. Score each character as `effect × quality × (1 − scene recolored)`. Effect rises from zero at the higher of the OFF Boss reading and 0.25 to full credit at 0.75. An OFF already at or above 0.75 has zero available gain. For quality, take the minimum of the tile-based drawing integrity relative to OFF and the two whole-image readings for a single figure and a head connected to the body. The paired recolor reading discounts an edit in proportion to how much it washes the surroundings in one color. A broken drawing, split figure or missing head should earn zero. Keep the qualified structural verdict alongside the score. [Working Boss definition](https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/post-training/README.md#what-on-should-mean) · [Per-image formula](https://github.com/HyperGAN/anima-particle-sliders/blob/main/lumen_studio/study/working_score.py)
2. For each pair of ON images, average Jev's design-repetition readings in both image orders. For character scores `a` and `b` and mean repetition `r`, the pair earns `(a + b − r × min(a, b)) / 2`. Average all pairs in the panel. [Pair formula](https://github.com/HyperGAN/anima-particle-sliders/blob/main/lumen_studio/study/design_score.py)
3. On the working Boss board, each frozen RAW has its own alpha search on four characters. Select the highest measured working Boss score, breaking exact ties toward the lower alpha. Freeze that setting before scoring the eight held-out characters. Rank by their 28-pair mean, reported on a 0–100 scale. [Working Boss search](https://github.com/HyperGAN/anima-particle-sliders/blob/main/research/post-training/working_search.py) · [Leaderboard with samples](https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/post-training/README.md#every-trained-recipe)

## 4. Publish the winner and show its samples

Rank one best measured setting per training recipe on the same cases, at the same update count. Show matched OFF and ON samples in each leaderboard entry and label the released default on its exact row. Publishing the winner should select its RAW, training-target fingerprint, alpha, schedule, and resolved per-step gains together. Link the source, judge evidence and pinned artifacts. Load the default through the normal replay path and verify a reference output against the scored sample. Identify which scores are development measurements and which, if any, used fresh confirmation cases.

**Historical Anima release.** The timed late-cues RAW192 release used the earlier **distinct Boss score**. Across 27 search probes and five full-panel comparisons, early α 1.15625 scored **28.84/100** on eight development characters. Early α 1.0 scored 26.39 and early α 1.25 scored 28.27. “Late cues” describes teacher construction during training. “Early strength” describes adapter application at inference. Its ten gains were approximately 1.7344 for seven steps, 1.1177 for one and 0.5781 for two. Keep those replays and failed trials with their original metric. [Historical search path](https://github.com/HyperGAN/anima-particle-sliders/blob/6b32f5715a86ec530171dcdc55d3cca23cb7ca24/docs/final-boss-raw192/vibe12/alpha-tuning/late-cues/search.json) · [Historical release manifest](https://github.com/HyperGAN/anima-particle-sliders/blob/6b32f5715a86ec530171dcdc55d3cca23cb7ca24/docs/final-boss-raw192/default.json)

## What to put in a product's VIBE.md

| Section | Fill in for this product |
| --- | --- |
| Collect | Actual command, concept and scope, matched generation settings, judge questions, qualification and diversity rules, retained context count, manifest path. |
| Train | Native response and stage, cache and derivation, training entry point, fixed comparison budget, and saved RAW identity. |
| Tune | Search command or worker, supported alpha range and profiles, exact score, search and development case lists, measured path, winning alpha and resolved gains. |
| Use the default | Leaderboard row with matched samples, release manifest, artifact revision and hash, replay command and verified reference. |
