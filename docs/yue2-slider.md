# YuE2 concept sliders

An opt-in composition backend for `m-a-p/YuE2-3B`, using the
[official YuE2 runtime](https://github.com/multimodal-art-projection/YuE).
Training and inference live entirely in `sliders-conceptmod`.

## Installation

Use a separate environment: the official runtime pins torch and transformers
versions that differ from the Music 3 environment. Never install this project's
old `requirements.txt`. The implementation targets upstream commit
`ef1936f2ee39fe8de486a0f47a481c95f8d4da87`.

```bash
uv venv --python 3.12 .venv-yue2
uv pip install --python .venv-yue2/bin/python \
  'yue2-infer @ git+https://github.com/multimodal-art-projection/YuE.git@ef1936f2ee39fe8de486a0f47a481c95f8d4da87' \
  PyYAML pytest
```

All model loading is local-only by default; `--allow_hub` opts into downloading.
`--model_id` / `--vae_id` also accept local directories. `--revision`,
`--vae_revision` and `--cache_dir` pin the model inputs and cache location.
Training only loads the composition model and tokenizer; the VAE is needed for
rendering. Upstream code is Apache 2.0; model weights have their separate
[CC BY-NC 4.0 license](https://huggingface.co/m-a-p/YuE2-3B).

## Train: Music Arm B

New metal training uses the explicit Arm B entry point. This is a different
recipe from the historical UNI16 female experiments below.

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --save_dir models/metal-yue2-arm-b --steps 600
```

The source recipe is [Music Arm B](music-arm-b-gates.md):
`faithful_guard_e`, last-token MLP, RpGAN, exact vendored `b_cap` with
coefficient and threshold 1, pole weight 1, FM 0, parts 0, VICReg 0.
Both +1 and -1 are trained. The metal prompts declare metal/clean band as
the concept axis and singer gender as the unrelated axis. The shared guard
removes only the admissible odd leftover; it preserves the caption midpoint
and falls back to the original poles when subtraction would destroy the axis.

The generator objective, averaged over rows, is exactly:

```text
MSE(h+, target+) + MSE(h-, target-)
+ 0.5 * (RpGAN_G(+) + RpGAN_G(-))
+ 0.5 * (end_margin_MSE(+) + end_margin_MSE(-))
```

The toy cover term maps to the single pole term above; it is not added again.
The ending term is the existing locked Music transfer ending term, evaluated
on YuE2's native music-end versus semantic-code log odds. There is no feature
matching, lyric hold, plan loss, learned zero anchor, second critic, row mining,
EMA, or parameter-step cap. Scale zero is the exact base model.

The critic is Music's two-layer 256-wide MLP on the music-start hidden delta.
Both scores and the gradient penalty use fixed teacher-RMS coordinates. The
vendored `GradRegularizer` receives the MLP core and the already scaled
vectors, preserving the cap's units without dividing twice. It uses exact
autograd, both real and fake samples including zeros, every update; no finite
differences, lazy interval, interpolation penalty, or annealed threshold.

Rank/alpha 8, four distinct rows per update, generator AdamW 0.0005
(weight decay 1e-6), discriminator Adam 0.00075, betas (0, 0.999), constant
learning rates, and generator gradient-value clipping at 1 come from the
locked Music transfer. Only AR q/k/v/o adapters train; base AR/NAR/VAE weights
stay frozen. Each draw gets a fresh base continuation and a new seed, and rows
follow balanced shuffled passes from the first update. No step-601 switch.

`--until 2` limits a preflight to two updates without changing the recipe.
Rerun the same command without `--until` to continue from the complete state
(adapter, critic, optimizers, sampler, cached targets, and CPU/CUDA RNG).
Source/config/model changes reject resume. Every 100 updates and at a requested
endpoint, exports and complete states are retained; SIGTERM finishes the current
update and saves. A run lock prevents concurrent trainers sharing a directory.

To train and automatically render all held-out cases afterward:

```bash
.venv-yue2/bin/python scripts/train_yue2_arm_b_campaign.py \
  --save_dir models/metal-yue2-arm-b --steps 600 --gpu 1 \
  --output_dir eval/listen/yue2-metal-arm-b
```

The comparison page has two held-out prompts, two fixed seeds, scales
-1/0/0.5/1, and both caption references (24 clips). It retains original native
artifacts and publishes a local download only after rendering succeeds.
Progress is in `status.json`; training and evaluation logs stay in the run
folder. This does not rank intermediate checkpoints or publish to the Hub.
The listening page refreshes its loss and alignment charts every second, with
raw values, optional moving averages, and hover inspection. Chart data is
exported to `training-metrics.json` from the completed training-log records.

See [the formulation audit](yue2-arm-b-verification.md) and
`tests/test_yue2_arm_b.py` for numeric update parity and native model checks.
The CPU tests verify implementation, not audible quality. Metal quality must be
assessed from held-out matched renders at -1, 0, 0.5 and 1.

## Historical UNI16 training

The earlier UNI16 training workflow uses fresh sampled histories and shuffled prompt
rows **from update 1**, with one unchanged objective, batch size and optimizer
setup through the requested endpoint:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2_fresh.py \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-female.yaml \
  --save_dir models/female-yue2-fresh --steps 3400
```

It uses one row per update, balanced shuffled passes, fresh 250-token base
histories, and the same RpGAN + feature matching + end-margin objective described
below. Rank/alpha, rates and clipping stay fixed; there is no step-601 change or
parameter-update cap. The complete game state is saved every 20 updates. Running
the same command resumes it exactly; `--until` can stop at a milestone without
changing the declared `--steps` horizon. Native CUDA graphs are used only for
adapter-off base-history sampling. Adapter training and rendering retain eager
execution. Graph/eager histories need not be token-identical, so the sampling
backend is pinned for the run.

The local fresh 3400 campaign also automates milestone renders, integrity checks, CPU scoring and the existing
quality-based ranking rule. It preserves the original 600-step trial as a
separate listening reference.

### Historical fixed-history 600-update trainer

From the project root, on physical GPU 1 while the studio uses GPU 0:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2.py \
  --config_file conceptmod/textsliders/data/config-yue2.yaml \
  --prompts_file conceptmod/textsliders/data/prompts-yue2.yaml \
  --save_dir models/breath-yue2-ar --allow_hub
```

The YAML uses `rows` with `positive`, `neutral`, and shared `lyrics`. Optional
`attributes` repeat a pair with the same sound description on both sides.
Use instruments, vocal register, breath, room and timing; never named references.
The included breath prompts and lyric sheets are original sound-only examples.

The default `--recipe uni16` ports the winning voice-slider warm-up from
the historical UNI16 formulation. It trains **AR attention only**:
`model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}` (112 projections on the
released 28-layer model). NAR attention, both MLP paths, embeddings, normalization,
output heads and the VAE remain frozen. The existing `lm_adv.SpanTransformerD`,
RpGAN pairing losses, calibrated `b_cap`, and `lm_gan.feature_mean_surrogate`
are reused directly. The critic sees the target and student changes at matching
lyric tokens plus the final music-start token, divided by fixed teacher RMS.
Captions, ABC delimiters and padding are excluded from those spans.

The generator objective is adversarial comparison + exact batch-mean critic
feature matching + base end-margin MSE, with all three coefficients 1. There is
**no direct hidden-state MSE and no lyric hold**. End margin is the native
music-end logit minus logsumexp over YuE2's 32768 semantic-token logits on a
fixed base-model history. History sampling uses YuE2's native off-mode CFG.
Scale 0 bypasses the adapter exactly and therefore needs no learned anchor.

The winning warm-up settings are rank/alpha 8, zero LoRA-up initialization,
four rows per update, 600 updates, 250-token fixed histories, generator LR
0.0005 and critic LR 0.00075, beta1 0, constant rates, and elementwise gradient
clipping at 1. The critic has width 128, two layers, four heads and mean/last
readout. Its calibrated gradient cap has coefficient and threshold 1.
The update-norm cap and fresh-history phase of the later Music 3 continuation
are not part of this 600-update warm-up. This port rejects larger UNI16 budgets
rather than silently applying fixed-history warm-up beyond that phase.

The original provisional loss remains available as `--recipe hidden` for
explicit ablations. It is not the winning formulation.

`--train_tokens` is the ending-supervision history budget, not an audio-duration
control. `--max_seq_len` rejects overly long input without silently cutting lyrics.
Training uses native unpadded causal AR forwards and activation checkpointing.
The native NAR velocity method is inference-only and is not used as a train loss.

Outputs are native `.safetensors`, JSON sidecars and a JSONL loss log. The tensor
header retains rank, alpha, exact target names, base model identity, prompt pairs,
seed and recipe. These weights are not interchangeable with Music 3 or PEFT LoRAs.
UNI16 also saves a full `_state.pt` with the critic, both optimizers, exact cached
teachers/histories and RNG state. `--resume_state PATH --steps 600` resumes the
same run, preserving completed updates; recipe/model/prompt mismatches are rejected.

### Female voice, 600 updates

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2.py \
  --config_file conceptmod/textsliders/data/config-yue2-female-uni16.yaml \
  --save_dir models/female-yue2-uni16-600 --allow_hub
```

The four female training pairs and four separate evaluation sheets come from
the winning campaign. Its Music 3 caption layout is flattened into YuE2 style
text; lyrics, arrangements, tempos and the neutral→female distinction are
retained. Files: `prompts-yue2-female.yaml` and `prompts-yue2-female-eval.yaml`.

The September 16 trial completed all 600 updates on GPU 1 in 15 minutes of
training, with an exact export/full-state match and ten held-out 30-second
previews. The female-vocal diagnostic improved in 3/4 matched pairs; quality
diagnostics were mixed. It remains an experimental listening candidate, with
positive user listening feedback, but no established lyric-preservation verdict.
That feedback applies to the female UNI16 trial, not the new metal Arm B recipe.

## Render a scale comparison

Save a section-tagged original lyric sheet to `lyrics.txt`, then:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/infer_yue2.py \
  --weights models/breath-yue2-ar/breath-yue2-ar_last.safetensors \
  --style 'English, slow acoustic pop, close clear lead vocal, fingerpicked steel strings, soft bass, brushed snare, small dry room.' \
  --lyrics_file lyrics.txt --scales=0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-breath --allow_hub
```

Every scale uses the same neutral style, lyrics and seed. Scale 0 bypasses the
adapter exactly. The adapter is active during AR planning/semantic generation
and disabled for NAR synthesis and VAE decoding. The eager torch backend keeps
the native Python LoRA forwards active. Quantization, vLLM, CUDA-graph AR execution
and AR offload are excluded from this initial integration.

Each directory retains the official audio, tokens, latents, plan, truncation
flags and model identity, plus the adapter SHA-256 and scale. Existing comparison
directories are never overwritten. `--max_tokens` caps semantic generation;
short caps can truncate music and are reported in the artifacts.

The trained mode is `--cot off`. `--cot full` / `--cot melody` and `--abc_file`
are available for experiments using the native planning protocol, but the recipe
does not establish transfer to those modes. Negative scales are trained for Arm B; they are extrapolation for UNI16.
The recipe is experimental: loss reduction and successful rendering do not prove
audible concept control or lyric preservation. Gate trained releases on matched
off/on listening comparisons and lyric checks.

## Verification

```bash
.venv-yue2/bin/python -m pytest tests/test_yue2_slider.py tests/test_yue2_uni.py \
  tests/test_yue2_fresh.py tests/test_yue2_arm_b.py tests/test_music_arm_b.py -q
.venv-yue2/bin/python conceptmod/textsliders/train_lora_yue2.py \
  --dummy --steps 2 --train_tokens 4 --save_dir /tmp/yue2-smoke
```

`--dummy` uses a tiny random instance of the **official architecture** with a byte
tokenizer and synthetic continuations. It needs the runtime, but no GPU, model
download or VAE. Its exports are explicitly marked and rejected by the render
loader; dummy results are never evidence of audio quality.

Local verification on September 16, 2026: **36 tests passed**, including the YuE2
tests and shared critic/feature-matching tests, with the pinned
YuE2 source and transformers 4.57.6 (Python 3.11.15, torch 2.11.0+cu128).
Before the UNI16 port, on physical GPU 1 (RTX A6000), the provisional hidden-MSE
recipe completed three rank-2
updates with an eight-token training continuation; loss went from 0.007738 to
0.007698. Reloading that export rendered scales 0 and 1 with a 128-token cap:
both produced finite, non-silent 48 kHz stereo audio of about 5.1 seconds, with
verified artifact manifests and correctly reported semantic truncation. The
two recordings were identical in this tiny probe. This establishes train/load/
render operation, **not audible control**; a trained slider release still needs
the listening and lyric-preservation gates above. Local smoke artifacts are in
`/ml2/music/.cache/yue2-smoke/` and are not shipped as trained sliders.
