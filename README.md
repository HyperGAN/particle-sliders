# Particle Sliders

Particle Sliders trains small adapters that give a frozen generative model a
controllable edit: make an image shinier, change a musical style, or steer a
language model's response. The adapter's strength is the slider.

This repository provides **`particle-sliders-core`**, the shared training engine,
sessions, adapter controls, checkpoints, evaluation and execution tools.
**Model loading, prompts, training recipes and generation belong to the
[product repositories](docs/products.md).** Start there to train or use a slider.

| Model family | Usage and model setup |
|---|---|
| Images | [Anima](https://github.com/HyperGAN/anima-particle-sliders), [Supra](https://github.com/HyperGAN/supra-particle-sliders), [Krea2](https://github.com/HyperGAN/krea2-particle-sliders), [Ming](https://github.com/HyperGAN/ming-particle-sliders), [Iris](https://github.com/HyperGAN/iris-particle-sliders) |
| More image models | [Qwen Image](https://github.com/HyperGAN/qwen-image-particle-sliders), [Z-Image](https://github.com/HyperGAN/zimage-particle-sliders), [P-H3](https://github.com/HyperGAN/p-h3-image-particle-sliders), [MOSS](https://github.com/HyperGAN/moss-image-sliders) |
| Music and audio | [Music3](https://github.com/HyperGAN/music3-particle-sliders), [YuE2](https://github.com/HyperGAN/yue2-particle-sliders) |
| Language models | [Glimmer](https://github.com/HyperGAN/glimmer-particle-sliders), [Qwen3.5](https://github.com/HyperGAN/qwen35-particle-sliders), [Glide](https://github.com/HyperGAN/glide-particle-sliders) |

Each product documents its own supported operations and dependencies. These
links do not imply that every product supports the same core version, native
training, or rendering.

## How training works

A product constructs matched teacher/student examples. For an image edit,
for example, the frozen teacher sees “a shiny red cup” while the student sees
“a red cup.” Both receive the same latent and timestep. The student uses the
frozen base plus a trainable adapter; the teacher stays frozen. Audio and text
products choose their own matching inputs and native response coordinates.
The teacher and student can share one frozen base, with the adapter disabled for
the teacher pass.

The goal is to make the student's neutral-prompt response acquire the teacher's
edit. Training compares their numerical responses inside the model, rather than
requiring a rendered image, waveform, or generated text at every update.
At inference, the product applies the learned adapter at a chosen strength.
Strength scales the adapter branch; a complete generated output need not vary
linearly, and the product defines supported strengths and OFF behavior.

## Math: a paired residual game

For matched example $i$, let $T_i$ be the frozen teacher target and
$S_\theta(i)$ the student's response. With positive response scale $a$:

$$
e_i = \frac{S_\theta(i)-T_i}{a}, \qquad
n_i \sim \mathcal{N}(0,\sigma^2 I).
$$

The conditional critic sees a real reference $n_i$ and a fake reference
$n_i+e_i$, with the same example condition $c_i$. When the student matches its
teacher, $e_i=0$ and the pair is identical. ParticleGAN supplies the adversarial
loss, critic penalty, optimizer rules and, when a real routed particle bank
participates, its regularizer. The response error constructs the game;
**training does not minimize a separately added MSE loss**.

One update trains the critic, then trains the adapter through the frozen critic.
The frozen native base and teacher are outside the optimizers. See
[the math guide](docs/math.md) for the default Sinkhorn-relativistic loss,
conditioning, optional particle bank, and the distinction between the training
objective and the search metric.

## Shared core

Current core **0.14.2** requires Python 3.11+, PyTorch 2.6–2.x and exact
**`particlegan==0.9.5`**. Products pin reviewed source commits and retain their
own compatible native environments.

```sh
# From this repository; product setup instructions live in the links above.
python -m pip install './packages/particle-sliders-core[serialization]'
```

| Integrating a product | Guide |
|---|---|
| Decide what belongs in core and what the product owns | [Distribution boundary](packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md) |
| Bind teacher/student responses and use the shared game | [Response sessions](packages/particle-sliders-core/RESPONSE-SESSIONS.md) · [Engine contract](packages/particle-sliders-core/CONDITIONAL-PG09-CONTRACT.md) |
| Train, evaluate, resume and export an image adapter | [Image sessions](packages/particle-sliders-core/IMAGE-LIFECYCLE.md) |
| Load an adapter without rebuilding training resources | [Native inference](packages/particle-sliders-core/NATIVE-INFERENCE.md) |
| Plan and dispatch training in the product's environment | [Training command](packages/particle-sliders-core/TRAINING-COMMAND.md) |
| Select rates from intended effect | [Evaluation and saved-output workflow](packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md) · [Historical SSE runner](packages/particle-sliders-core/PARAMETER-SEARCH.md) |
| Stop catastrophically failed candidates | [Qualified finished-output guard API](packages/particle-sliders-core/CATASTROPHIC-OUTPUT-GUARD.md) |
| Build and test the core | [Development](docs/development.md) |

Fresh sessions explicitly set `ema_decay=None`. Training, checkpoints, selection,
exports and inference use raw current weights. ParticleGAN's scalar optimizer
statistics are unchanged; they are not model-weight averaging.

Select rates by one number: the equal-case mean intended-effect intensity in
finished ON outputs minus matched OFF. Retain negative gains, zeros and all
tied maxima. Teacher/student relative error is diagnostic only, never a selector
or tiebreak. Reuse completed searches, saved RAW checkpoints, outputs and first
measurements; no extra confirmation is required.

The existing `particle-slider-search` runner implements its historical fixed
SSE protocol, not this finished-output selection policy. It has no output-effect
scorer callback, and `--media-review` does not change its ranking. Use the
product's supported inference path and a frozen external output instrument as
described in [the evaluation guide](packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md).
Historical runner results and declared protocols remain intact.

## Project and history

Particle Sliders has developed into a shared, model-independent project. It
builds on [Concept Sliders](https://github.com/rohitgandikota/sliders) and the
[sliders-conceptmod lineage](https://github.com/ntc-ai/sliders-conceptmod).
The original attribution and [MIT license](LICENSE) remain intact.

Historical embedded model code, experiments and old checkpoint loaders are not
part of the current core wheel. Use their original source pins to reproduce
old runs; see [history and retained evidence](docs/history.md).
The [documentation index](docs/README.md) lists the current guides.
