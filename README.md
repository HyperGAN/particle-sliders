# sliders-conceptmod

**Train continuous controls for music, image and video generators.** A slider
learns a small adapter from contrasting descriptions, then changes the model's
behavior with a numeric strength while you keep the generation prompt fixed.

This is a substantially extended research fork of
[Concept Sliders](https://github.com/rohitgandikota/sliders). It includes
**MiniMax Music 3 and YuE2**, language-model and flow-transformer adapters,
adversarial training, routed-particle experiments, and tools for matched
listening comparisons. The original image-slider code remains part of the
repository; the methods added here have their own objectives and validation.

**Music 3:** [Listen and download](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)
· [Interactive demo](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)
· [Training notes](MUSIC3.md)

**YuE2:** [Weights and release notes](https://huggingface.co/ntc-ai/yue2-concept-sliders)
· [Interactive demo](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders)
· [Setup, training and rendering](docs/yue2-slider.md)

[Models](#models-and-status) · [Get started](#getting-started)
· [Math](#how-it-works) · [Music 3](#minimax-music-3) · [YuE2](#yue2)
· [Evaluation](#evaluation) · [Repository map](#repository-map)

## Models and status

These are separate backends. A checkpoint belongs to its base model, adapter
host and training recipe; adapter formats are not interchangeable.

| Backend / guide | Adapter host and status |
|---|---|
| **[Music 3 LM](MUSIC3.md)** | Qwen3 attention for voice and composition. Published 16-control release. |
| **[Music 3 flow](conceptmod/textsliders/train_lora_music3.py)** | Acoustic flow transformer. Earlier mix and production controls. |
| **[YuE2](docs/yue2-slider.md)** | AR attention; NAR and VAE frozen. Experimental composition controls. |
| **[Routed particles](#routed-particle-adapters)** | Nonlinear branches for Music 3 and YuE2. Experimental GAN. |
| [Krea 2](docs/krea-slider.md) | Text encoder and/or DiT. Opt-in images. |
| [Anima](docs/anima-slider.md) | Conditioner or DiT. Opt-in images. |
| [Z-Image Turbo](docs/zimage-slider.md) | DiT attention. Opt-in images. |
| [Sana 0.6B](docs/sana-slider.md) | Cross-attention or LoRA. Small image experiments. |
| [LTX-2.5](docs/ltx25-slider.md) | Text encoder and video connectors. Opt-in video. |
| [MiniMax-H3](docs/minimax-h3-slider.md) | Omni-Transformer. Opt-in video and audio. |
| [Tiny LLM](docs/tiny-llm-slider.md) | Qwen3-0.6B attention. Particle-bridge test target. |
| [Bonsai GGUF](docs/bonsai-gguf-slider.md) | Frozen readouts. Opt-in particle experiment. |
| [Legacy images](conceptmod/textsliders/) | SD 1.x/2.x, SDXL, SD3, Flux and Stable Cascade trainers. |

The **September 16, 2026 Music 3 release** contains 16 unipolar rank-8 LM
adapters: female, male, lo-fi, pop, hip-hop, R&B, indie rock, pop punk,
metal, country, acoustic folk, house, disco funk, K-pop, reggaeton and
afrobeats. It includes native weights, ComfyUI conversions and 64 matched
Off/On comparisons. Selected checkpoints range from 1,000 to 3,400 updates;
the [release card](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)
records the per-control choices and limitations.

YuE2 and particle experiments have separate validation records. A successful
CPU test, lower training loss or completed render does not establish useful
musical control. Reward-model experiments also live here and have their own
[methods and acceptance checks](conceptmod/textsliders/reward_game/README.md).

## How it works

### A strength-controlled adapter

For a linear layer with frozen weight $W_0$, ordinary LoRA learns factors
$A\in\mathbb R^{r\times d_{in}}$ and $B\in\mathbb R^{d_{out}\times r}$:

```math
W(s)=W_0+s\frac{\alpha}{r}BA.
```

$r$ is the rank, $\alpha$ is the adapter normalization and $s$ is the slider
strength. The up projection starts at zero. At $s=0$ the adapter contributes
nothing; at $s=1$ it applies the learned unit edit. The complete generator can
respond nonlinearly even though this weight update is linear. See
[`lora.py`](conceptmod/textsliders/lora.py).

**Unipolar** recipes teach neutral → positive at $+1$. Negative strengths are
untrained extrapolation, not a learned opposite. **Bipolar** recipes explicitly
teach both signs. Published female and male controls, for example, are separate
unipolar adapters.

At inference the slider operates on the **neutral caption**. Changing to the
positive caption with the adapter off is a teacher reference, a different
treatment from applying the slider.

### Diffusion and flow targets

Let $f_0(x_t,t,c)$ be the frozen model's prediction under caption $c$, with
neutral, positive and negative captions $c_0,c_+,c_-$. A bipolar axis target is

```math
\begin{aligned}
a_t &= g\big[f_0(x_t,t,c_+)-f_0(x_t,t,c_-)\big],\\
\widehat f_s &= f_0(x_t,t,c_0)+s\,a_t.
\end{aligned}
```

The student sees $c_0$ and learns to match $\widehat f_s$. The original image
formulation acts on noise predictions. Music 3's acoustic trainer acts on
flow velocities and, by default, normalizes its fitting loss:

```math
\mathcal L_{\mathrm{NMSE}}=
\frac{\mathrm{MSE}\big(f_\theta(x_t,t,c_0;s),\widehat f_s\big)}
{\max\big(\mathrm{mean}[(s a_t)^2],10^{-8}\big)}.
```

Its training inputs are anchored to generated clean latents,
$x_t=(1-t)\epsilon+t x_0$, using Music 3's noise-to-clean time convention.
The alternative `pole` target learns each caption's displacement from neutral
instead of forcing symmetric movement along $c_+-c_-$. Executable definitions:
[`slider_targets.py`](conceptmod/textsliders/slider_targets.py).

### Language-model targets and the common component

Music generation also depends on the autoregressive model that plans the
composition. Let $h_0,h_+,h_-$ be its frozen prompt states. Decompose the pair as

```math
\begin{aligned}
a &= \tfrac12(h_+-h_-),\\
b &= \tfrac12(h_++h_-)-h_0,\\
h_\pm &= h_0+b\pm a.
\end{aligned}
```

The older `v9` target $h_0\pm a$ discards $b$, the information shared by both
pole captions beyond the neutral caption. A perfectly fitted symmetric axis
can therefore miss the states occupied by either real caption. Faithful-pole
and unipolar targets retain that information. Lyric-token holds, role-specific
targets and declared leakage directions address different preservation problems.

See the [target geometry](docs/lm-sheet-goodhart.md),
[positive/neutral formulation](docs/lm-plus-neu-exam.md) and
[lyric preservation study](docs/lm-lyric-hold.md). The generic LM trainer still
defaults to `--lm_target v9 --pole_mode hidden`; **those defaults do not
reproduce the published adversarial release**.

### The released Music 3 span-GAN objective

The released recipe compares corresponding lyric-token states plus the
audio-start state. Frozen neutral and positive sequences provide $H_0,H_+$;
the adapted model on the neutral caption provides $H_\theta$. With fixed
teacher-RMS calibration $\sigma$:

```math
x_+=(H_+-H_0)/\sigma,\qquad x_\theta=(H_\theta-H_0)/\sigma.
```

A transformer critic $D$ learns a relativistic paired comparison. Write
$\mathrm{sp}(z)=\log(1+e^z)$:

```math
\mathcal L_D=\mathbb E[\mathrm{sp}(D(x_\theta)-D(x_+))]
+\mathcal R_{\mathrm{cap}},
```

```math
\begin{aligned}
\mathcal L_G &= \mathbb E[\mathrm{sp}(D(x_+)-D(x_\theta))]\\
&\quad +\mathcal L_{\mathrm{FM}}+\mathcal L_{\mathrm{end}}.
\end{aligned}
```

$\phi$ denotes critic features. Feature matching compares **batch means**.

```math
\begin{aligned}
\mu_\theta &= \mathbb E[\phi(x_\theta)],\\
\mu_+ &= \mathbb E[\phi(x_+)],\\
\mathcal L_{\mathrm{FM}} &= \mathrm{MSE}(\mu_\theta,\mu_+).
\end{aligned}
```

The one-sided input-gradient cap uses $g_x=\lVert\nabla_xD(x)\rVert_2$ in
calibrated critic coordinates:

```math
\mathcal R_{\mathrm{cap}}=\frac{\lambda}{2}
\sum_{x\in\{x_+,x_\theta\}}
\mathbb E\!\left[\max(g_x-\kappa,0)^2\right].
```

The release uses $\lambda=\kappa=1$.

Ending supervision matches the base model's end-versus-continuation margin
on the same base-generated token history:

```math
\begin{aligned}
m &= \ell_{\mathrm{audio\_end}}-\log\sum_{j\in\mathcal S}\exp(\ell_j),\\
\mathcal L_{\mathrm{end}} &= \mathrm{MSE}(m_\theta,m_0),
\end{aligned}
```

where $\mathcal S$ is the semantic-token band. All three generator terms have
coefficient 1. Explicit lyric hold and direct hidden-state MSE are disabled
in this release. The first 600 updates use four rows per batch and fixed base
histories. Continuation uses one row per update in balanced shuffled passes,
with a fresh base history for ending supervision and a per-parameter update
norm cap of 2. Prompt-state teachers remain fixed. See the
[release formulation](docs/hub-formulation-fresh-selected.md) and
[`gan_v2/`](conceptmod/textsliders/gan_v2/) for implementation details.

YuE2 has its own recipes. `gan_plus_neu` uses the paired logistic game at
scales 0 and +1 without the feature-matching or ending losses above. Historical
`uni16` recipes include those auxiliary terms. The [YuE2 guide](docs/yue2-slider.md)
and checkpoint metadata identify which objective a run actually used.

### Routed-particle adapters

The particle experiment replaces a linear low-rank branch with a routed
nonlinear branch. Each projection has its own router and MLP; a slider shares
one cloud $P\in\mathbb R^{128\times4}$ across its projections:

```math
\begin{aligned}
u &= Ax,\quad q=\mathrm{router}(u),\\
z &= \mathrm{softmax}(qP^\top/\sqrt4)P,\\
\Delta(x) &= B\,\mathrm{MLP}([u,z]).
\end{aligned}
```

The layer returns its frozen output plus $s(\alpha/r)\Delta(x)$. Routing runs
at both training and inference. These checkpoints contain routers, MLPs and
particles, so they **cannot be merged as an ordinary $BA$ LoRA**.

For normalized paired error $e=T(h_\theta)-T(h_+)$, the critic compares
$x_r=n$ with $x_f=n+e$, using the same Gaussian noise $n$ within a pair:

```math
\begin{aligned}
\mathcal L_D &= \mathbb E[\mathrm{sp}(D(x_f)-D(x_r))]\\
&\quad +\mathcal R_{\mathrm{cap}},\\
\mathcal L_G &= \mathbb E[\mathrm{sp}(D(x_r)-D(x_f))]\\
&\quad +\mathcal L_{\mathrm{VIC}}(P).
\end{aligned}
```

The particle regularizer encourages per-coordinate sample standard deviation
of at least 1 and penalizes off-diagonal sample covariance. It operates on a
sample of the cloud, not on predicted outputs. The game uses separate D/G
minibatches, lazy gradient-cap evaluation and EMA exports. It has no output
reconstruction MSE.

The [reference audit](docs/yue2-particle-bridge.md) records the original
absolute-target normalization and 8,000-step noise schedule. Native experiments
also support paired-edit normalization, run-budget noise annealing/holds and
alternative critics. Read each run's recipe and metadata for those settings;
the reference proof does not validate every later native variant. Shared math:
[`particle_bridge_gan.py`](conceptmod/textsliders/particle_bridge_gan.py).

## Getting started

```bash
git clone https://github.com/mikkel/sliders-conceptmod.git
cd sliders-conceptmod
```

Choose an environment for the backend you intend to use. **Do not install the
root `requirements.txt` into a Music 3, YuE2 or modern image/video environment.**
It is the inherited diffusion-era dependency list, not a universal installer.
Base-model weights and their compatible runtimes are separate prerequisites.

On the shared music workstation, train and render on **physical GPU 1** while
the studio uses GPU 0. `CUDA_VISIBLE_DEVICES=1` exposes that card as logical
`cuda:0`; use `--device 0` or `--device cuda:0` according to the script.
The examples below follow this assignment.

### MiniMax Music 3

Use a working Music 3 environment with the MiniMax Music 3 pipeline and model
classes available in its compatible Diffusers installation. The local studio
environment is `minimax-music3`. The acoustic trainer loads from a local model
directory; override `--model_dir` when your weights are elsewhere.

For **published adapters**, use the
[native loading example](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/main/usage.md)
or the [ComfyUI guide](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/main/comfyui/README.md).
Keep each native `.safetensors` with its JSON sidecar. Start with one adapter
at strengths 0 and 1, holding caption, lyrics and seed fixed. The released LM
adapters apply to the text encoder/CLIP side in ComfyUI.

To train an **acoustic flow slider** using the included energy prompts:

```bash
conda activate minimax-music3
CUDA_VISIBLE_DEVICES=1 python conceptmod/textsliders/train_lora_music3.py \
  --name energy-example \
  --model_dir /path/to/MiniMax-Music3 \
  --prompts_file conceptmod/textsliders/data/prompts-energy-tf-v7.yaml \
  --save_dir models/energy-example \
  --rank 8 --alpha 8 --steps 500 --seed 7 --device 0
```

This uses the acoustic trainer's `full` targets, anchored latents and NMSE
defaults. It does not train the published voice/genre LM recipe. For LM
training and the release campaign, start with [MUSIC3.md](MUSIC3.md),
the [warm-up campaign](analysis/uni16_20260906/README.md) and
the [fresh-continuation campaign](analysis/uni16_fresh3400_20260912/README.md).
Campaign scripts retain local model/cache paths, manifests and recovery-state
requirements; they are research records, not a portable one-command installer.

Use the [recipe-comparison pipeline](slider_pipeline/README.md) for matched
acoustic training/rendering sweeps. Music 3's nonlinear particle experiment has
a separate entry point,
[`train_lora_music3_particle.py`](conceptmod/textsliders/train_lora_music3_particle.py).

### YuE2

YuE2 uses its native runtime in a separate environment. The integration targets
this pinned upstream revision:

```bash
uv venv --python 3.12 .venv-yue2
uv pip install --python .venv-yue2/bin/python \
  'yue2-infer @ git+https://github.com/multimodal-art-projection/YuE.git@ef1936f2ee39fe8de486a0f47a481c95f8d4da87' \
  PyYAML pytest
```

An explicit **experimental positive/neutral GAN** run:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python \
  conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --recipe gan_plus_neu \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --save_dir models/metal-yue2-example --steps 600 \
  --propose_only_lr_scale 0.2
```

The 0.2 rate multiplier is the opt-in
[native stability trial](docs/yue2-gan-stability.md), not a universal quality
recommendation. This trainer expects cached model weights, or a local directory
passed as `--model_id`. Training needs the composition model and tokenizer;
rendering also needs the VAE. The renderer below accepts `--allow_hub` to permit
downloads, plus `--model_id` and `--vae_id` for local model directories.

To compare an exported adapter, put original section-tagged lyrics in
`lyrics.txt`, then substitute the exported weight path:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yue2/bin/python conceptmod/textsliders/infer_yue2.py \
  --weights /path/to/adapter.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales=0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

The renderer preserves matched inputs and records model identity, weight hash,
scale and truncation flags. Use a fresh output directory for another run.
The [full guide](docs/yue2-slider.md) covers exact resume, recipe variants,
native adapter loading, generation modes and runtime limits.

### Images and video

Follow the backend's guide in the model table. Each guide specifies its adapter
host, teacher space, environment and sampling settings. Several backends expose
`--dummy` for CPU integration checks. Legacy image inference notebooks remain
at the repository root; paired-image training lives in
[`trainscripts/imagesliders/`](trainscripts/imagesliders/).

## Evaluation

A useful comparison keeps the **neutral caption, lyrics, seed and generation
settings fixed**, and changes only adapter strength. Include an adapter-off
positive-caption reference to show what the base model can do with that prompt.
Retain failures, natural endings and truncation flags in the comparison.

Assess concept movement alongside audio quality, lyric preservation, unintended
changes and behavior across held-out prompts/seeds. Hidden-state cosine or a
GAN loss alone cannot answer those questions. Acoustic ladder gates and LM
composition checks measure different behavior; see [SCORING.md](SCORING.md),
[LM-SCORING.md](LM-SCORING.md) and the
[listening/selection tools](slider_selection/README.md).

The published Music 3 checkpoint policy compares enjoyment and production
quality separately, keeps candidates within 0.2 of each best clean mean, and
prefers the later qualifying checkpoint. Style similarity and lyric scores
are diagnostics, not terms in that selection rule. Four short matched clips
per candidate support a shortlist, not a claim of full-song or stacked-adapter
reliability. The release card retains the full selection record.

Describe training concepts through **sound**: instruments, playing, vocal
register, breath, mic distance, room and timing. Never put real artist, band,
songwriter, producer or album names in prompts, lyrics, titles, listening notes
or checkpoint sidecars. Checkpoints trained on named references must be retired
and retrained from sound-only prompts.

## Development and verification

From the repository root, in a compatible environment with PyTorch, PyYAML,
Safetensors and pytest, this CPU suite exercises target geometry, the shared
adversarial core and particle mechanics:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest -q \
  tests/test_2d_slider_geometry.py \
  tests/test_lm_gan.py \
  tests/test_yue2_particle_bridge.py
```

Native YuE2 tests additionally require its runtime; optional-runtime cases can
skip when it is absent. See each backend guide for its tests and dummy training
commands. The full historical suite also exercises studio integrations and
campaign artifacts: it needs additional dependencies such as SciPy, the parent
music workspace's `app` package for those integrations, and local campaign
fixtures. Native GPU training, listening and campaign reproduction require the
corresponding models and recovery artifacts.

## Repository map

| Source | Purpose |
|---|---|
| [Backends and trainers](conceptmod/textsliders/) | Model loading, adapters, objectives and inference |
| [GAN engine](conceptmod/textsliders/gan_v2/) | Span critics, game updates and recovery states |
| [Reward research](conceptmod/textsliders/reward_game/) | Reward-guided adapters and acceptance checks |
| [Geometry fixtures](analysis/slider2d/) | Small distribution and objective studies |
| [Campaigns](analysis/) | Source, audits and notes; some need local artifacts |
| [Comparison pipeline](slider_pipeline/) | Matched acoustic recipes and render gates |
| [Listening tools](slider_selection/) | Listening, features and selection experiments |
| [Scripts](scripts/) | Evaluation, dashboards and packaging |
| [Documentation](docs/) | Backend guides, math and release-card sources |
| [Tests](tests/) | CPU contracts and backend integration |

Local run outputs go in `models/`, `cache/` and `eval/listen/`. Published weights
and recordings are distributed on the Hub.

## Lineage and license

The original [Concept Sliders paper](https://arxiv.org/abs/2311.12092) and
[implementation](https://github.com/rohitgandikota/sliders) introduced LoRA-based
concept control for diffusion models. Cite that work when building on it:

```bibtex
@article{gandikota2023sliders,
  title={Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Tingrui Zhou and Antonio Torralba and David Bau},
  journal={arXiv preprint arXiv:2311.12092},
  year={2023}
}
```

This fork's music, adversarial, particle and newer image/video extensions are
documented in the linked source and experiment records. The repository retains
the upstream [MIT license](LICENSE). Base models, runtimes and vendored code
retain their respective licenses; this license does not relicense model weights.
