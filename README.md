# particle-sliders

**Sliders for generative models: turn one concept up or down with a number.**
A slider is a small adapter trained on a frozen model. At strength 0 the model
is unchanged; at strength 1 it applies one learned edit, such as a female voice,
a metal arrangement or a lighting style. The caption, lyrics and seed stay
fixed. Music comes first here (YuE2 and MiniMax Music 3), followed by image
and video backends.

This repository also ships
[`particle-sliders-core`](packages/particle-sliders-core), the installable
training core that every HyperGAN slider product pins.

[Listen](#listen) · [Pick your path](#pick-your-path) · [How it works](#how-it-works)
· [Use the core](#use-the-shared-core) · [Train](#train-a-slider) · [Tests](#tests)
· [Repository map](#repository-map)

## Listen

| Project | Hugging Face links |
|---|---|
| **YuE2** · routed particles | [Project / weights](https://huggingface.co/ntc-ai/yue2-concept-sliders) · [Live demo](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders) · [Listening gallery](https://huggingface.co/ntc-ai/yue2-concept-sliders#press-play--original-particles) |
| **Music 3** · LM LoRA | [Project / weights](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders) · [Live demo](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders) · [Listening gallery](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders#press-play) |

Each Off/On pair below uses the same neutral caption, lyrics and seed. Only
the slider strength changes. These are short, token-capped YuE2 excerpts. The
full gallery also has half strength and positive-caption references.

| Control | Matched MP3 samples |
|---|---|
| Female voice | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/female/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/female/row0-seed1709-on.mp3) |
| Metal | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/metal/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/metal/row0-seed1709-on.mp3) |
| House | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/house/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/house/row0-seed1709-on.mp3) |

Both projects publish **16 voice and genre controls**, with four matched
Off/On comparisons per control. YuE2's original particle weights and its
ordinary distilled LoRAs are different artifacts. The demo and the samples
above use the **original particles**. The
[distillation guide](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/DISTILLATION.md)
covers standard LoRA loading and the approximation it introduces.

## Pick your path

| I want to… | Start here |
|---|---|
| Hear or use a published slider | [Listen](#listen): Hub weights, live demos, galleries |
| Build a slider product on the shared core | [Use the shared core](#use-the-shared-core) |
| Train a new slider on a supported model | [Train a slider](#train-a-slider) |
| Understand the math and the evidence | [docs/math.md](docs/math.md) |
| Find a backend guide, prompt file or experiment write-up | [docs/README.md](docs/README.md) |

## How it works

Each adapted layer becomes $f_s(x)=f_0(x)+s\frac{\alpha}{r}\Delta(x)$. Here
$f_0$ is the frozen layer, $\Delta$ is the learned correction and $s$ is the
slider strength. The preferred formulation, first used for YuE2, builds
$\Delta$ from a **routed particle cloud**. Each input picks a soft mixture of
learned vectors, and that mixture conditions a small nonlinear branch. Because
routing stays active at inference, these weights cannot be merged into an
ordinary LoRA.

Training is a **paired-error adversarial game**. A frozen teacher reads a
positive caption, such as "female vocal", and the adapted model reads the
neutral caption. A critic sees shared noise, alone or plus the
student–teacher error, and the adapter learns to make the two
indistinguishable. There is no output MSE and no feature-matching term. Only a
critic gradient cap and a particle variance/covariance regularizer are added.
The game draws on [ParticleGAN](https://github.com/255BITS/ParticleGAN).

[docs/math.md](docs/math.md) has the full equations, the published YuE2
release settings, the archived training curves and the older LoRA, diffusion
and language-model objectives used by the other backends.

## Use the shared core

Products such as
[anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders),
[krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders) and
[supra-concept-sliders](https://github.com/HyperGAN/supra-concept-sliders)
pin `particle-sliders-core` and train `winning_formulation()`. They do not
re-implement routed particles, the critic, `locked_shared` or
`GradRegularizer`. Install your PyTorch build first, then pin a full commit:

```bash
python -m pip install \
  "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<commit>#subdirectory=packages/particle-sliders-core"
```

```python
from particle_sliders import winning_formulation, FormulationGame

stamp = winning_formulation()
stamp.require(stamp.as_dict())     # raises if a product drifts from the stamp
bridge = stamp.bridge()            # gmix routed-particle adapter
critic = stamp.critic(training_targets, neutrals=training_neutrals)
```

The architecture is gmix. The formulation parameters are provisional:
`particle-gmix-1600-v2` holds until
[ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) crowns a full
live leaderboard winner. Products pick up a new overlay by bumping the pin,
not by copying knobs.

- [Package README](packages/particle-sliders-core/README.md): install, API and `FormulationGame` train step
- [Winning formulation](docs/winning-formulation.md): the product stamp and what products may override
- [Shared-core architecture](docs/shared-core.md): what lives in the core and what stays in each product

`concept-slider-core` / `concept_slider_core` is a deprecated alias from the
Anima extraction. Formulation toys live in
[HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod).

## Train a slider

Backends are separate. A checkpoint belongs to its base model, adapter host
and training recipe, and adapter formats are not interchangeable.

| Backend / guide | Adapter host and status |
|---|---|
| **[YuE2 particles](docs/yue2-slider.md)** | AR attention; NAR and VAE frozen. Preferred formulation; published 16-control release. |
| **[Music 3 LM](MUSIC3.md)** | Qwen3 attention for voice and composition. Published 16-control release. |
| **[Music 3 flow](conceptmod/textsliders/train_lora_music3.py)** | Acoustic flow transformer. Earlier mix and production controls. |
| **[Music 3 particles](conceptmod/textsliders/train_lora_music3_particle.py)** | Nonlinear branches on Music 3. Experimental transfer of the particle game. |
| [Krea 2](docs/krea-slider.md) | Text encoder and/or DiT. Opt-in images. |
| [Anima](docs/anima-slider.md) | Conditioner or DiT. Opt-in images. |
| [Supra2-IMG](docs/supra-slider.md) | Cross-attn or DiT on `SupraLabs/Supra2-IMG`. Opt-in images. |
| [Z-Image Turbo](docs/zimage-slider.md) | DiT attention. Opt-in images. |
| [Sana 0.6B](docs/sana-slider.md) | Cross-attention or LoRA. Small image experiments. |
| [LTX-2.5](docs/ltx25-slider.md) | Text encoder and video connectors. Opt-in video. |
| [MiniMax-H3](docs/minimax-h3-slider.md) | Omni-Transformer. Opt-in video and audio. |
| [Tiny LLM](docs/tiny-llm-slider.md) | Qwen3-0.6B attention. Particle-bridge test target. |
| [Bonsai GGUF](docs/bonsai-gguf-slider.md) | Frozen readouts. Opt-in particle experiment. |
| [Legacy images](conceptmod/textsliders/) | SD 1.x/2.x, SDXL, SD3, Flux and Stable Cascade trainers. |

Each backend needs its own environment. There is no universal installer.
Base-model weights and their runtimes are separate prerequisites. The commands
below pin `CUDA_VISIBLE_DEVICES=1`, the training card on the shared music
workstation, which appears as logical `cuda:0`. Change it for your machine.

### YuE2

YuE2 runs on its native runtime in its own environment, pinned to this
upstream revision:

```bash
uv venv --python 3.12 .venv-yue2
uv pip install --python .venv-yue2/bin/python \
  'yue2-infer @ git+https://github.com/multimodal-art-projection/YuE.git@ef1936f2ee39fe8de486a0f47a481c95f8d4da87' \
  PyYAML pytest
```

For ready-made controls, start with the
[original particle weights](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/weights/particle-1200-v1)
and the [live demo](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders).
To train a **routed-particle slider with the current experimental defaults**:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python \
  conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --recipe particle_bridge \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --save_dir models/metal-yue2-particles --steps 1200
```

These defaults are a later experiment, not a replay of the September 17
release. The trainer expects cached model weights or a local directory passed
as `--model_id`. Training needs the composition model and tokenizer; rendering
also needs the VAE. To compare an exported adapter, put section-tagged lyrics
in `lyrics.txt`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/infer_yue2.py \
  --weights /path/to/adapter.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales=0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

The renderer keeps inputs matched and records model identity, weight hash,
scale and truncation flags. `--allow_hub` permits downloads; `--model_id` and
`--vae_id` take local model directories. Use a fresh output directory for each
run. The [YuE2 guide](docs/yue2-slider.md) covers exact resume, recipe
variants, native adapter loading, generation modes and runtime limits.

### MiniMax Music 3

Use a Music 3 environment whose Diffusers install provides the MiniMax Music 3
pipeline and model classes (the local studio env is `minimax-music3`). For
**published adapters**, follow the
[native loading example](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/main/usage.md)
or the [ComfyUI guide](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/main/comfyui/README.md).
Keep each native `.safetensors` next to its JSON sidecar. Start with one adapter
at strengths 0 and 1, holding caption, lyrics and seed fixed. In ComfyUI the
released LM adapters apply to the text encoder/CLIP side.

To train an **acoustic flow slider** on the included energy prompts:

```bash
conda activate minimax-music3
CUDA_VISIBLE_DEVICES=1 python conceptmod/textsliders/train_lora_music3.py \
  --name energy-example \
  --model_dir /path/to/MiniMax-Music3 \
  --prompts_file conceptmod/textsliders/data/prompts-energy-tf-v7.yaml \
  --save_dir models/energy-example \
  --rank 8 --alpha 8 --steps 500 --seed 7 --device 0
```

This uses the acoustic trainer's defaults (`full` targets, anchored latents,
NMSE). It does **not** train the published voice/genre LM recipe. For LM
training and the release campaign, start with [MUSIC3.md](MUSIC3.md), the
[warm-up campaign](analysis/uni16_20260906/README.md) and the
[fresh-continuation campaign](analysis/uni16_fresh3400_20260912/README.md).
Campaign scripts keep local model/cache paths and recovery-state requirements.
They are research records, not one-command installers. Matched acoustic sweeps
use the [recipe-comparison pipeline](slider_pipeline/README.md).

The **September 16, 2026 Music 3 release** contains 16 unipolar rank-8 LM
adapters: female, male, lo-fi, pop, hip-hop, R&B, indie rock, pop punk, metal,
country, acoustic folk, house, disco funk, K-pop, reggaeton and afrobeats. It
includes native weights, ComfyUI conversions and 64 matched Off/On
comparisons. The [release card](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)
records the per-control checkpoint choices (1,000 to 3,400 updates) and
limitations.

### Images and video

Follow the backend's guide from the table. Each guide gives its adapter host,
teacher space, environment and sampling settings. Several backends accept
`--dummy` for a CPU integration check with no Hub access and no GPU. The
upstream Concept Sliders notebooks, eval scripts and paired-image trainer are
kept unmaintained in [legacy/](legacy/).

### Judging a slider

Hold the **neutral caption, lyrics, seed and generation settings fixed** and
change only adapter strength. Include an adapter-off positive-caption
reference. A passing CPU test, a lower loss or a completed render does not by
itself show useful musical control. Describe concepts through **sound** and
never put real artist, band, songwriter, producer or album names in prompts,
lyrics, titles, notes or sidecars. The full evaluation policy is in
[docs/math.md](docs/math.md#evaluation), with gate definitions in
[SCORING.md](SCORING.md) and [LM-SCORING.md](LM-SCORING.md).

## Tests

The CPU suite covers target geometry, the shared adversarial core, particle
mechanics and backend wiring with small or dummy models:

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch                  # or the CUDA build for your machine
pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest -q
```

Modules that need the native YuE2 runtime or the parent music workspace's
`app` package skip when those are absent. For a fast check of the core
contracts:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest -q \
  packages/particle-sliders-core/tests \
  tests/test_2d_slider_geometry.py tests/test_lm_gan.py tests/test_yue2_particle_bridge.py
```

GPU training, listening and campaign reproduction need the corresponding
models and recovery artifacts. Each backend guide lists its own tests and
dummy training commands.

## Repository map

| Path | What it is |
|---|---|
| [packages/particle-sliders-core](packages/particle-sliders-core) | **Stable.** Installable shared core and `winning_formulation()` |
| [conceptmod/textsliders/](conceptmod/textsliders/) | Trainers, backends, adapters, objectives and inference for every model |
| [conceptmod/textsliders/gan_v2/](conceptmod/textsliders/gan_v2/) | Music 3 span critics, game updates and recovery states |
| [conceptmod/textsliders/reward_game/](conceptmod/textsliders/reward_game/) | Reward-model research with its own [acceptance checks](conceptmod/textsliders/reward_game/README.md) |
| [conceptmod/textsliders/data/](conceptmod/textsliders/data/) | Prompt and config YAMLs; see the [prompt catalog](docs/prompts.md) |
| [docs/](docs/README.md) | Backend guides, math, experiment write-ups and release-card sources, all indexed |
| [MUSIC3.md](MUSIC3.md) · [SCORING.md](SCORING.md) · [LM-SCORING.md](LM-SCORING.md) | Music 3 operator guide, acoustic gates, LM composition checks |
| [analysis/](analysis/) | Dated campaigns, audits and 2-D geometry fixtures ([slider2d](analysis/slider2d/)); some need local artifacts |
| [slider_pipeline/](slider_pipeline/README.md) | Matched acoustic recipe comparisons and render gates |
| [slider_selection/](slider_selection/README.md) | Listening, features and checkpoint-selection experiments |
| [scripts/](scripts/) | Render, probe, evaluate, dashboard and publish entry points |
| [tests/](tests/) | CPU contracts and backend integration |
| [eval/listen/](eval/listen/README.md) | Listening notes; audio stays local |
| `models/`, `cache/` | Local run outputs (git-ignored, apart from a few JSON sidecars) |
| [scratchpad/](scratchpad/) | Working notes and data read by the blind-spot and A/B scripts |
| [legacy/](legacy/) | Upstream Concept Sliders notebooks, eval scripts, paired-image trainer and their pinned `requirements.txt` |

Published weights and recordings are distributed on the Hugging Face Hub.

## Lineage and license

This is a substantially divergent fork of
[Concept Sliders](https://github.com/rohitgandikota/sliders), previously named
`sliders-conceptmod`, with its own music objectives, adapter architectures and
listening tools. The original [paper](https://arxiv.org/abs/2311.12092) and
implementation introduced LoRA-based concept control for diffusion models.
Cite that work when building on it:

```bibtex
@article{gandikota2023sliders,
  title={Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Tingrui Zhou and Antonio Torralba and David Bau},
  journal={arXiv preprint arXiv:2311.12092},
  year={2023}
}
```

The repository retains the upstream [MIT license](LICENSE). Base models,
runtimes and vendored code keep their own licenses; this license does not
relicense model weights.
