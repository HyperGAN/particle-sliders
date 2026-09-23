# particle-sliders

**Sliders for generative models: turn one concept up or down with a number.**
A slider is a small adapter trained on a frozen model. At strength 0 the model
is unchanged; at strength 1 it applies one learned edit, such as a female voice,
a metal arrangement or a lighting style. Music comes first here: YuE2 and
MiniMax Music 3 generate songs from a style caption and lyrics. Image and video
backends follow.

This repository also ships
[`particle-sliders-core`](packages/particle-sliders-core), the installable
training core that HyperGAN slider products are meant to pin.

| I want to… | Start here |
|---|---|
| Hear or use a published slider | [Listen](#listen) |
| Build a slider product on the shared core | [Use the shared core](#use-the-shared-core) |
| Train a new slider on a supported model | [Train a slider](#train-a-slider) |
| Run the tests | [Tests](#tests) |
| Understand the math and the evidence | [docs/math.md](docs/math.md) |
| Find a backend guide, prompt file or experiment write-up | [docs/README.md](docs/README.md) |

## Listen

| Project | Hugging Face links |
|---|---|
| **YuE2** · routed particles | [Project / weights](https://huggingface.co/ntc-ai/yue2-concept-sliders) · [Live demo](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders) · [Listening gallery](https://huggingface.co/ntc-ai/yue2-concept-sliders#press-play--original-particles) |
| **Music 3** · LM LoRA | [Project / weights](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders) · [Live demo](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders) · [Listening gallery](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders#press-play) |
| **Anima** · images | [Candlelit / Moonlit samples](https://huggingface.co/ntc-ai/anima-concept-sliders) |

Each Off/On pair below uses the same neutral caption, lyrics and seed. Only
the slider strength changes. These are short, token-capped YuE2 excerpts from
the September 17 `particle-1200-v1` export. The full gallery also has half
strength and positive-caption references.

| Control | Matched MP3 samples |
|---|---|
| Female voice | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/female/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/female/row0-seed1709-on.mp3) |
| Metal | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/metal/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/metal/row0-seed1709-on.mp3) |
| House | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/house/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/house/row0-seed1709-on.mp3) |

Both music projects publish the same **16 voice and genre controls**, with
four matched Off/On comparisons each: female, male, lo-fi, pop, hip-hop, R&B,
indie rock, pop punk, metal, country, acoustic folk, house, disco funk, K-pop,
reggaeton and afrobeats.

### Use a published slider

- **YuE2.** Try the live demo, or download the
  [native particle weights](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/weights/particle-1200-v1)
  and render them with `infer_yue2.py` (YuE2 environment and a GPU; see
  [YuE2](#yue2)). The demo and samples use these native particles. Tools that
  only load standard LoRAs need the distilled LoRAs, which are an approximation;
  see the [distillation guide](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/DISTILLATION.md).
  The same Hub project also records the later `particle-gmix-1600-v2` run that
  the shared core is pinned to.
- **Music 3.** Follow the
  [native loading example](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/main/usage.md)
  or the [ComfyUI guide](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/main/comfyui/README.md).
  Keep each native `.safetensors` next to its JSON sidecar. In ComfyUI the
  released LM adapters apply to the text encoder/CLIP side. The September 16
  release has 16 unipolar rank-8 LM adapters, ComfyUI conversions and 64
  matched Off/On comparisons. Its
  [release card](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)
  records the per-control checkpoint choices (1,000 to 3,400 updates) and
  limitations.

Start with one adapter at strengths 0 and 1.

## How it works

Each adapted layer becomes $f_s(x)=f_0(x)+s\,\Delta(x)$. Here
$f_0$ is the frozen layer, $\Delta$ is the learned correction and $s$ is the
slider strength. The preferred formulation, first used for YuE2, builds
$\Delta$ from a **routed particle cloud**. Each input picks a soft mixture of
learned vectors, and that mixture conditions a small nonlinear branch. Because
routing stays active at inference, these weights cannot be merged into an
ordinary LoRA.

The preferred formulation trains through a **paired-error adversarial game**.
A frozen teacher reads a positive caption, such as "female vocal", and the
adapted model reads the neutral caption. A critic sees shared noise, alone or
plus the student–teacher error, and the adapter learns to make the two
indistinguishable. There is no output MSE, feature matching, lyric hold or
ending loss. Only a critic gradient cap and a particle variance/covariance
regularizer are added. The game draws on
[ParticleGAN](https://github.com/255BITS/ParticleGAN). The released Music 3 LM
sliders use an earlier span-GAN that adds feature matching and ending
supervision.

[docs/math.md](docs/math.md) has the full equations, the published YuE2
release settings, the archived training curves and the older LoRA, diffusion
and language-model objectives used by the other backends.

## Use the shared core

Products such as
[anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders),
[krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders) and
[supra-concept-sliders](https://github.com/HyperGAN/supra-concept-sliders) are
meant to pin `particle-sliders-core` and train `winning_formulation()`, not
re-implement the routed particles, critic or losses. Each product's migration
status is in
[docs/winning-formulation.md](docs/winning-formulation.md#product-follow-ups).
Install your PyTorch build first, then pin a full commit of `main`. The git
revision, not the package version, identifies the stamp.

```bash
python -m pip install \
  "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<commit>#subdirectory=packages/particle-sliders-core"
```

```python
from particle_sliders import winning_formulation, FormulationGame

stamp = winning_formulation()
config = {**stamp.as_dict(), "g_lr": 2e-5}  # only stamp.model_surface_keys may differ
stamp.require(config)                       # raises if a formulation key drifts

# targets / neutrals: [N, rank] paired positive and neutral features from your model
game = FormulationGame(stamp, config, critic_targets=targets, critic_neutrals=neutrals)
for step in range(1, steps + 1):            # steps count from 1
    stats = game.step(step, features)       # features: [batch, stamp.spec["adapter_rank"]]
```

The **architecture** (gmix: routed particles plus a global-mix critic) is
fixed. Its **tuning parameters** (`particle-gmix-1600-v2`) are provisional
until [ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) picks a
winner. Products get new values by bumping the pin, never by editing them
locally. The runnable reference is
[test_formulation_game.py](packages/particle-sliders-core/tests/test_formulation_game.py).

- [Package README](packages/particle-sliders-core/README.md): install, API and `FormulationGame` train step
- [Winning formulation](docs/winning-formulation.md): the product stamp and what products may override
- [Shared-core architecture](docs/shared-core.md): what lives in the core and what stays in each product

`concept-slider-core` / `concept_slider_core` is a deprecated alias from the
Anima extraction. Formulation toys live in
[HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). The research
trainers stay in this tree until each product repo moves its own train/infer
surface.

## Train a slider

```bash
git clone https://github.com/HyperGAN/particle-sliders.git
cd particle-sliders
```

Backends are separate. A checkpoint belongs to its base model, adapter host
and training recipe, and adapter formats are not interchangeable.

| Backend / guide | Adapter host and status |
|---|---|
| **[YuE2 particles](docs/yue2-slider.md)** | AR attention; NAR and VAE frozen. Preferred formulation; published 16-control release. |
| **[Music 3 LM](MUSIC3.md)** | Qwen3 attention for voice and composition. Published 16-control release. |
| **[Music 3 flow](MUSIC3.md)** | Acoustic flow transformer (`train_lora_music3.py`). Earlier mix and production controls. |
| **[Music 3 particles](conceptmod/textsliders/train_lora_music3_particle.py)** (script) | Nonlinear branches on Music 3. Experimental transfer of the particle game. |
| [Krea 2](docs/krea-slider.md) | Text encoder and/or DiT. Opt-in images. |
| [Anima](docs/anima-slider.md) | Conditioner or DiT. Opt-in images. |
| [Supra2-IMG](docs/supra-slider.md) | Cross-attn or DiT on `SupraLabs/Supra2-IMG`. Opt-in images. |
| [Z-Image Turbo](docs/zimage-slider.md) | DiT attention. Opt-in images. |
| [Sana 0.6B](docs/sana-slider.md) | Cross-attention or LoRA. Small image experiments. |
| [LTX-2.5](docs/ltx25-slider.md) | Text encoder and video connectors. Opt-in video. |
| [MiniMax-H3](docs/minimax-h3-slider.md) | Omni-Transformer. Opt-in video and audio. |
| [Tiny LLM](docs/tiny-llm-slider.md) | Qwen3-0.6B attention. Particle-bridge test target. |
| [Bonsai GGUF](docs/bonsai-gguf-slider.md) | Frozen readouts. Opt-in particle experiment. |
| [Diffusion images](conceptmod/textsliders/) | SD 1.x/2.x, SDXL, SD3, Flux and Stable Cascade trainers (this fork's, not `legacy/`). |

Each backend needs its own environment. There is no universal installer.
Base-model weights and their runtimes are separate prerequisites. On the
shared music workstation, train and render on **physical GPU 1**; the studio
uses GPU 0. The commands below set `CUDA_VISIBLE_DEVICES=1`, which exposes that
card as logical `cuda:0`, so pass `--device 0` or `--device cuda:0` as the
script expects. On another machine, change it.

### YuE2

YuE2 runs on its native runtime in its own environment, pinned to this
upstream revision:

```bash
uv venv --python 3.12 .venv-yue2
uv pip install --python .venv-yue2/bin/python \
  'yue2-infer @ git+https://github.com/multimodal-art-projection/YuE.git@ef1936f2ee39fe8de486a0f47a481c95f8d4da87' \
  PyYAML pytest
```

Training reads base models from the local Hugging Face cache and never
downloads. Fetch `m-a-p/YuE2-3B`, plus `m-a-p/YuE2-Vae` for rendering (both
[CC BY-NC 4.0](https://huggingface.co/m-a-p/YuE2-3B)), for example with
`hf download`. Or pass local directories with `--model_id` / `--vae_id`. To
train a **routed-particle slider with the current experimental defaults**:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python \
  conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --recipe particle_bridge --name metal-yue2-particles \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --save_dir models/metal-yue2-particles --steps 1200
```

This writes `models/metal-yue2-particles/metal-yue2-particles_last.safetensors`
(the EMA export), plus `_live_last` and per-step copies. These defaults
(`--critic mlp`, paired-edit normalization, continuation histories) are a later
experiment. They replay neither the September 17 release (`particle-1200-v1`)
nor the `particle-gmix-1600-v2` core formulation, which needs `--critic gmix`
and the settings in the
[drift table](docs/yue2-slider.md#train-routed-particles). For another concept,
see the [prompt catalog](docs/prompts.md#yue2--particle-smokes). The 16
published controls' prompt sheets are in
[analysis/yue2_uni16_1200_20260917/prompts/](analysis/yue2_uni16_1200_20260917/prompts/).

To compare scales, put original section-tagged lyrics (`[verse]`, `[chorus]`,
…) in `lyrics.txt`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/infer_yue2.py \
  --weights models/metal-yue2-particles/metal-yue2-particles_last.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales=0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

The renderer also accepts the published particle weights directly. Audio lands
in `scale_0`, `scale_0.5` and `scale_1` under the output directory, with model
identity, weight hash, scale and truncation flags recorded. Use a fresh output
directory for each run. The style must describe sound: phrases such as "in the
style of" or "sounds like" are rejected. `--allow_hub` permits downloads, and
`--model_id` / `--vae_id` take local model directories. The
[YuE2 guide](docs/yue2-slider.md) covers exact resume, recipe variants, native
adapter loading, generation modes and runtime limits.

### MiniMax Music 3

Music 3 training needs a Diffusers install with the MiniMax Music 3 pipeline
and local model weights. On the workstation that is the pinned build in the
`minimax-music3` conda env; other Diffusers releases may differ.
`train_lora_music3.py` defaults to the workstation's model directory, so pass
`--model_dir` elsewhere. Published adapters are covered under
[Use a published slider](#use-a-published-slider).

To train an **acoustic flow slider** on the included energy prompts:

```bash
conda activate minimax-music3   # workstation env
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
Campaign scripts keep local model/cache paths, manifests and recovery-state
requirements. They are research records, not portable one-command installers.
Matched acoustic sweeps use the
[recipe-comparison pipeline](slider_pipeline/README.md).

### Images and video

Follow the backend's guide from the table. Each guide gives its adapter host,
teacher space, environment and sampling settings. Several backends accept
`--dummy` for a CPU integration check with no Hub access and no GPU. The
upstream Concept Sliders notebooks, eval scripts and paired-image trainer are
kept unmaintained in [legacy/](legacy/).

### Judging a slider

Hold the **neutral caption, lyrics, seed and generation settings fixed** and
change only adapter strength. Include an adapter-off positive-caption
reference: render the positive caption at `--scales=0` into its own output
directory. A passing CPU test, a lower loss or a completed render does not by
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
pip install torch --index-url https://download.pytorch.org/whl/cpu  # skip if you have a CUDA build
pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest -q
```

Tests that need the music workstation (the parent workspace's `app` package,
its pinned Music 3 Diffusers build or local campaign files) or the native YuE2
runtime skip elsewhere and run on the workstation. The full CPU run takes about
ten minutes. For a check of the core contracts in under a minute:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest -q \
  packages/particle-sliders-core/tests tests/test_particle_sliders_formulation.py \
  tests/test_2d_slider_geometry.py tests/test_lm_gan.py tests/test_yue2_particle_bridge.py
```

To run the YuE2 runtime tests, use the file list in the YuE2 guide's
[Verification](docs/yue2-slider.md#verification) section with
`.venv-yue2/bin/python -m pytest`. The bare suite there also collects core
tests that need `particlegan`, which that environment does not install.

GPU training, listening and campaign reproduction need the corresponding
models and recovery artifacts. Most backend guides list their own tests and a
`--dummy` training command.

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
| [eval/listen/](eval/listen/README.md) | Local render outputs and listening notes; audio is git-ignored |
| `models/`, `cache/` | Local run outputs (git-ignored, apart from a few JSON sidecars) |
| [scratchpad/](scratchpad/) | Working notes (blind spots, Goodhart gates, A/B listening protocol); `scripts/blindspot_analyze.py` reads its local scan CSVs here |
| [legacy/](legacy/) | Upstream Concept Sliders notebooks, eval scripts, paired-image trainer and their pinned `requirements.txt` |

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
