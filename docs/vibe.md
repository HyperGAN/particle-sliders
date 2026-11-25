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

1. Use the **same saved RAW** for every probe. On a multi-step sampler, compare constant, early and late strength on the native grid. For audio or text generators that support scheduling, use the native generation or refinement positions where the adapter actually runs. Use scalar strength on a single application. [Native schedule guide](strength-schedules.md) Where supported, start each profile at α 1.0 and probe 0.75 and 1.25. Stay within the product's supported range. Anima multiplied the step by 1.5 when a neighboring probe won and by 0.5 when the center won. Keep the strongest measured candidate even if a larger alpha looks worse.
2. Use a small search panel to explore, then compare finalists on a separate, common development panel. Reuse matching renders and judgments. Anima used four characters during search, up to nine probes per profile, then eight other characters for final selection. Include earlier measured settings for the same RAW and shared-strength controls. Keep a record of the probes, the full-panel comparisons and their scores.
3. For Final Boss, Jev scores each character as `max(0, Boss_ON − Boss_OFF) × Boss_ON³ × (1 − visible_damage)`, with all ratings in [0, 1]. It also rates design repetition for every pair of characters in both image orders. For case scores `a` and `b` and repetition `r`, the pair earns `(a + b − r × min(a, b)) / 2`. Average the 28 pairs for eight characters and multiply by 100. That rewards distinct convincing bosses and discounts an adapter that repeatedly puts everyone in the same costume. Use this rubric for the Final Boss task. For a different vibe, define the corresponding concept and decide which variation it should preserve. [Per-image formula](https://github.com/HyperGAN/anima-particle-sliders/blob/main/release_tools/final_boss_visible_damage.py) · [Pair formula](https://github.com/HyperGAN/anima-particle-sliders/blob/main/release_tools/final_boss_design_scores.py)

The shared `alpha-tune <slider-definition>` command loads a saved RAW and uses the profiles, cases and Jev workflow supported by that implementation. Anima's research worker implements the adaptive 1.0/0.25 search and cross-character objective described here. Name the actual search entry point and objective in each product's instructions. [Shared alpha-tune implementation](../packages/particle-sliders-core/src/particle_sliders/alpha_tune.py) · [Anima search and replay instructions](https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/final-boss-raw192/vibe12/alpha-tuning/README.md)

## 4. Publish the winner and show its samples

Rank one best measured setting per training recipe on the same cases, at the same update count. Show matched OFF and ON samples in each leaderboard entry and label the released default on its exact row. Publishing the winner should select its RAW, training-target fingerprint, alpha, schedule, and resolved per-step gains together. Link the source, judge evidence and pinned artifacts. Load the default through the normal replay path and verify a reference output against the scored sample. Identify which scores are development measurements and which, if any, used fresh confirmation cases.

**Anima example.** Across 27 search probes and five full-panel comparisons, late cues at early α 1.15625 scored **28.84/100** on eight development characters. Early α 1.0 scored 26.39 and early α 1.25 scored 28.27. “Late cues” describes teacher construction during training. “Early strength” describes adapter application at inference. Its ten gains were approximately 1.7344 for seven steps, 1.1177 for one and 0.5781 for two. A different product or concept needs its own measured winner. [Scored search path](https://github.com/HyperGAN/anima-particle-sliders/blob/6b32f5715a86ec530171dcdc55d3cca23cb7ca24/docs/final-boss-raw192/vibe12/alpha-tuning/late-cues/search.json) · [Released manifest for this run](https://github.com/HyperGAN/anima-particle-sliders/blob/6b32f5715a86ec530171dcdc55d3cca23cb7ca24/docs/final-boss-raw192/default.json) · [Current leaderboard with samples](https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/final-boss-raw192/vibe12/distinct-designs.md)

## What to put in a product's VIBE.md

| Section | Fill in for this product |
| --- | --- |
| Collect | Actual command, concept and scope, matched generation settings, judge questions, qualification and diversity rules, retained context count, manifest path. |
| Train | Native response and stage, cache and derivation, training entry point, fixed comparison budget, and saved RAW identity. |
| Tune | Search command or worker, supported alpha range and profiles, exact score, search and development case lists, measured path, winning alpha and resolved gains. |
| Use the default | Leaderboard row with matched samples, release manifest, artifact revision and hash, replay command and verified reference. |
