# Jev development review, 2026-09-20

Status: retired from checkpoint-selection guidance after user feedback. The
descriptions already contain evaluative language, so Jev's scores are not an
independent check of visual quality. Keep this experiment as an audit record;
do not use its win counts or recommendations to select or qualify checkpoints.
Use direct matched-image reviews for those decisions.

TypeSafe's `typesafe-ai` skill is installed globally at
`/home/mikkel/.codex/skills/typesafe-ai/SKILL.md`, with its LICENSE. Both files
match the source under `/ml2/music/.agents/skills/typesafe-ai` byte for byte.

Jev 1.13 accepts text only. This workflow sends **Codex's recorded visual
observations**, not image pixels, to pinned model `jev-1.13.0`. Results are
judgments of those descriptions, not independent visual validation. Sources:
[model contract](https://docs.typesafe.ai/models),
[HTTP API](https://docs.typesafe.ai/api),
[composite scoring](https://docs.typesafe.ai/patterns/composite-scoring).

## First screen

Used existing randomized A/B development grids: Candlelit updates 400 and 1600,
Moonlit updates 400 and 800, eight cases each. All compare Off with Energy 1 at
matched prompt, seed, model, size and steps. No additional GPU rendering was
required, no final-test characters were opened, and training was unchanged.

Observations were recorded before opening the A/B identity maps. The observer
knew each grid's variation and checkpoint step; the Jev requests withhold steps,
checkpoint hashes, On/Off identities, residuals and file paths. Jev receives
separate questions for overall visual appeal, intended atmosphere, unrelated
change severity and the next concern to inspect. Choice questions include ties
and insufficient evidence. The descriptions remain an observer-dependent source
of bias even with randomized A/B order.

| Snapshot | Appeal wins / ties / losses | Atmosphere wins / ties / losses | Mean unrelated-change score, 0–3 |
| --- | --- | --- | --- |
| Candlelit 400 | 7 / 0 / 1 | 7 / 0 / 1 | 1.51 |
| Candlelit 1600 | 7 / 0 / 1 | 8 / 0 / 0 | 1.98 |
| Moonlit 400 | 6 / 2 / 0 | 7 / 1 / 0 | 1.76 |
| Moonlit 800 | 7 / 0 / 1 | 7 / 1 / 0 | 2.10 |

Counts are preference choices based on descriptions, not calibrated win
probabilities. The change score uses ordinal levels: lighting only; small detail
changes; noticeable outfit/face/framing/layout differences; substantial
identity/outfit/composition/medium change. Its mean is descriptive, not a formal
quality or preservation gate. These eight-case screens are correlated and do not
establish generalization or qualify any checkpoint.

## Guidance from the evidence

- The existing images show visible atmosphere gains, so residual alone should
  not choose the checkpoint. The same 7/8 appeal preference at Candlelit 400 and
  1600 accompanies increased unrelated-change severity at the later checkpoint.
  Compare their energy-response curves before choosing; later is not uniformly
  better. No new energy sweep has been run as part of this screen.
- Keep rewarding dramatic shadows and color. The purple-tunic case gives a
  useful example: cool dark architecture and a clear half-face shadow improve
  atmosphere while retaining the main subject and pose.
- Examine clothing and crop drift separately: Moonlit 800 changes a green
  turtleneck/suspenders close portrait to a cream turtleneck waist-up portrait.
  Jev still prefers its described appeal, showing why an appeal score cannot
  substitute for preservation evidence.
- Inspect medium drift in the bare-prompt red-haired character case: later
  outputs develop a photoreal face and textured sweater against illustrated
  scenery. Jev's appeal choice disfavors these examples despite stronger warmth
  in Candlelit. Verify at full size before recording formal reviews.
- The `focus` question is not reliable enough to direct experiments
  automatically: its concern changed from medium to atmosphere when A/B labels
  were reversed on the same descriptions. Use the recorded observations and
  separate judgments to choose a concrete investigation.

No Jev answers were submitted as human reviews, no qualification gates were
relaxed, and no checkpoint was promoted. The original ParticleGAN formulation,
training prompts, noise schedule and current run continue unchanged.

## Reproduce and inspect

Artifacts live outside git at `artifacts/anima/review/jev-20260920/`:

- `observations.json`: recorded A/B visual observations and limitations.
- Per-checkpoint directories: randomized grids, immutable mapping and originals.
- `jev/*.request.json`: exact outbound JSON, with credentials excluded.
- `jev/*.response.json`: raw answers, request hashes, usage and request latency.
- `jev/*.provenance.json`: observation, mapping and original image hashes.
- `jev/summary.json`: all case-level answers, mapped locally to Off/On.
- `jev/calibration.*.json`: two A/B reversal checks and an explicitly synthetic
  identical-image description control, kept separate from development counts.

```bash
.venv-anima/bin/python scripts/anima_jev_review.py \
  artifacts/anima/review/jev-20260920/observations.json --dry-run
.venv-anima/bin/python scripts/anima_jev_review.py \
  artifacts/anima/review/jev-20260920/observations.json
```

`TYPESAFE_API_KEY` stays in the environment. Requests and responses are cached
immutably; changing evidence or questions requires a fresh output directory.
The tool validates pairing and response contracts and never writes Studio
reviews or training state.

All four production requests returned the pinned model and complete, valid
answers. Two preference choices followed their images under an A/B text swap;
the identical-description control produced ties and zero drift. Some confidence
values and the next-concern choice changed under swapping, so these are limited
sanity checks, not judge calibration for this domain. Total use including that
control: 29,189 input tokens and 6,410 output tokens across five calls.

## Concrete example

The exact Moonlit 800 request includes case 4:

> Both show the same dark-haired man in a belted purple tunic, hands on hips,
> frontal anime composition. A has bright neutral courtyard and relatively even
> face lighting. B has dark blue-gray architecture, strong half-face shadow,
> cooler floor and brighter isolated pillar edge. B changes face and hand detail
> and backdrop construction, but character, main outfit and pose remain
> recognizable. B has distinctly stronger cool dramatic lighting.

Its appeal question asks which is a more compelling, visually coherent
illustration, using only those observations and the stated goal. Separate
questions ask about atmosphere and unrelated changes. Jev chose B for appeal
and atmosphere and gave unrelated changes 1.65/3. Its reported confidence of
1.0 in B is conditional on the supplied text, not proof of aesthetic quality or
observer accuracy.
