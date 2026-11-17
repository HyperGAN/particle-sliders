# Adaptive training-data collection

`particle-slider collect --vibe` is a separate path from training configuration
files. It discovers useful teacher contexts by judging finished outputs. The
base model stays frozen. Collection does not update adapters or select learning
rates/checkpoints.

```sh
particle-slider collect --vibe 'fantasy rpg final boss, epic battle scene' \
  --workflow my_product.vibe:factory --out runs/boss-data \
  --pool-size 8 --replace 4 --best-of 3 --rounds 4
```

The product factory returns an object with a JSON `identity` and three methods:

```python
def propose(vibe, count, history, *, request_id):
    # Return exactly count new {"neutral": "...", "positive": "..."} pairs.
    # history contains every round's prompts, finished-output scores and fitness.
    ...

def render(prompt, seed, directory, *, request_id):
    # Run the frozen model; return JSON, e.g. {"path": "/absolute/image.png"}.
    ...

def score(vibe, output, *, request_id):
    # Judge the complete output: {"score": 0.89, "raw": {...}}.
    # score is finite and normalized to [0,1]. Retain the original judgment.
    ...
```

The optional `close()` releases resources. Include model, sampler, workflow,
rubric and scorer identities in `identity`; changing them requires a new
collection. Models, native environments, GPU admission and target preparation
belong to the product. No model code or Torch is imported by the collector.

Each round renders N matched positive/neutral outputs per prompt pair. Every
candidate receives the same seeds. Fitness is the equal-case mean positive score
minus neutral score. The next round retains `pool_size - replace` top prompt
pairs and generates replacements informed by the entire measured history.
Stable ordering handles ties without adding a second fitness metric. The
best-of-N context for each pair is retained for product-owned qualification and
target preparation; every other output, judgment, negative gain and tie remains
in the round history. No confidence, beauty, preservation or teacher SSE score
is used as a tiebreak. This curation score does not establish generalization.

Increase `--rounds` on the same command to collect more data over time. It is a
total round count, not an instruction to repeat completed work. `collection.json`
contains the prompt pool, all cases and selected contexts. `calls/*.json` commits
each proposal, generation and score independently; `outputs/` retains media with
SHA256 identities. One file lock prevents concurrent writers.

A pending call means its outcome is uncertain. The collector does not silently
replay it after interruption. Reconcile its stable request ID with the provider;
use `--retry-pending` only after deciding that retry is appropriate. Stable IDs
are passed to product callbacks, but Nanoodle currently provides no guarantee of
server-side idempotency for those IDs. Completed calls are reused. A failed
proposal, generation or judge never becomes a zero score or a synthetic target.

## Nanoodle and Jev

Install the optional `particle-sliders-vibe[nanoodle]` add-on in the collector's
environment. Until its first release, install from
`./packages/particle-sliders-vibe[nanoodle]` in this source checkout. It has no
Torch/core dependency and leaves existing native core source closures intact.
`particle_sliders_vibe.nanoodle.NanoodleWorkflows` runs saved Nanoodle graphs:

- Prompt workflow: `Context` text input, `Pairs` JSON-list output.
- Score workflow: `Criterion` text plus `Image` media input; a `judge` Decide
  node in score mode with an explicit model, question and 2–10 ordered levels.

The default `judge_provider='nanogpt'` uses an explicitly configured NanoGPT
decision-model route and `NANOGPT_API_KEY`. Its live decision-model catalog lists
the image-capable Jev ID `typesafe/jev-1.14`; `typesafe/jev-latest` and `liquid/d1`
currently advertise text-only input. Choose the exact image-capable model
explicitly. A TypeSafe vendor prefix in the NanoGPT model ID does not cause a
direct TypeSafe request. The optional `judge_provider='typesafe'` transport is
available separately, but is not needed for this NanoGPT path.
The adapter does not substitute a different judge. It
retains the full decision distribution and normalizes Nanoodle's one-based
score as `(score - 1) / (levels - 1)`. Missing/invalid distributions fail before
selection. NanoGPT can round its score and probabilities independently to two
decimals. Validation allows only the resulting bounded rounding discrepancy;
the reported score remains the measurement, and the original distribution plus
validation diagnostics are retained. `NANOGPT_API_KEY` stays in the environment and is never included in
the collection. Generation and judging can use independent workflows. Definitive
provider 4xx rejections are recorded as failed calls and can be retried after
their cause is resolved; uncertain timeouts remain pending and require explicit
reconciliation. Provider rejection never becomes a fabricated score.

The first product implementation is `lumen_studio.vibe` in
[Anima Particle Sliders](https://github.com/HyperGAN/anima-particle-sliders).
It adds a CUDA-only renderer, model-specific prompt graph and qualified manifest
that the existing frozen trajectory cache can consume. See its `VIBE.md` for
native setup and its preserved training source pins.

The workflow follows the successful-context curation described in
[the Tiny Aya article](https://huggingface.co/blog/ntc-ai/particle-sliders-learn-the-behavior-keep-the-quest).
Adaptive prompt replacement is an extension of that fixed-candidate experiment.
