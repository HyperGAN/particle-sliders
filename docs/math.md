# Slider math, training evidence and evaluation

The formulations behind this repository's sliders, moved here from the root
[README](../README.md) so that page can stay a short entry point. YuE2's
routed-particle game is the preferred formulation; the Music 3 and inherited
image/flow objectives explain the other backends and how this fork evolved.

[YuE2 formulation](#yue2-the-lead-formulation) · [Training evidence](#training-evidence)
· [Other formulations](#other-training-formulations) · [Evaluation](#evaluation)

## YuE2: the lead formulation

The learned particles, paired adversarial loss, gradient cap and particle
variance/covariance regularizer draw on
[ParticleGAN](https://github.com/255BITS/ParticleGAN), pinned to
[revision `441fdf42`](https://github.com/255BITS/ParticleGAN/tree/441fdf42dd2c0905af312a303add422f700c0ac2)
for this release. The routed transformer adapter is this fork's YuE2 integration.

A frozen YuE2 teacher reads a positive caption. The adapted model reads the
neutral caption and learns to reproduce the teacher's prompt-state behavior.
Only branches on the AR model's 112 q/k/v/o projections train; the base AR,
NAR and VAE weights stay frozen. The published release trains on four caption
pairs and reads the final prompt state at the start of music generation.

### Terms and notation

| Term | Meaning here |
|---|---|
| **Teacher / student** | The frozen base model on the positive caption / the adapted model on the neutral caption. A hidden state is the model's internal feature vector. |
| **AR / NAR / VAE** | Autoregressive token generation / non-autoregressive decoding / variational autoencoder for audio. These base-model components remain frozen. |
| **LoRA / rank** | Low-rank adaptation / the dimension of its small internal bottleneck, here 8. |
| **Particle / router / MLP** | A learned vector / the network that chooses a mixture of those vectors / a multilayer perceptron, or small feed-forward neural network. |
| **GAN / RpGAN** | Generative adversarial network / its relativistic paired form: a critic learns to distinguish paired real and fake inputs while the adapter learns to defeat it. |
| **MSE / feature matching** | Mean squared error / a loss that matches critic features. Neither is part of YuE2's particle matching objective. |
| **VIC** | Variance and covariance regularization: keep particles spread out without correlated coordinates. |
| **EMA** | Exponential moving average of trained parameters; the smoothed weights used for exports. |

### A routed-particle adapter

Each projection has its own rank-8 down/up projections, router and nonlinear
MLP. All projections in one slider share a learned cloud
$P\in\mathbb R^{128\times4}$. In row-vector notation:

```math
\begin{aligned}
u &= xA^\top,\\
q &= \mathrm{router}(u),\\
z &= \mathrm{softmax}(qP^\top/\sqrt4)P,\\
\Delta(x) &= \mathrm{MLP}([u,z])B^\top,\\
f_s(x) &= f_0(x)+s\frac{\alpha}{r}\Delta(x).
\end{aligned}
```

Here $x$ is the layer input, $A$ compresses it to feature $u$ of rank $r$,
and $B$ projects the correction back to the layer's output size. The router
produces query $q$; softmax turns its particle scores into nonnegative weights
that sum to 1, giving the mixture $z$. Square brackets $[u,z]$ concatenate
the two vectors; $\top$ means transpose. $f_0$ is the frozen layer, $\Delta$
its learned correction, and $f_s$ the result at slider strength $s$.
$\alpha/r$ sets the adapter's overall scale.

The input chooses a soft mixture of particles, which conditions the learned
correction. The up projection starts at zero; $r=\alpha=8$. Strength $s=0$
bypasses the adapter exactly, and $s=1$ applies the trained edit. Intermediate
strengths interpolate its contribution; negative strengths are untrained.
Routing remains active at inference, so these weights cannot be merged into
an ordinary $BA$ LoRA. A distilled LoRA is a separately trained approximation.

### The paired-error game

Let $h_\theta$ be the adapted neutral-caption state and $h_+$ the frozen
positive-caption state. A fixed normalization $T$ puts their error into critic
coordinates. Real and fake examples share the same Gaussian noise:

```math
\begin{aligned}
e &= T(h_\theta)-T(h_+),\\
n &\sim\mathcal N(0,\sigma_t^2 I),\\
x_r &= n,\\
x_f &= n+e.
\end{aligned}
```

At exact teacher matching, $e=0$ and the critic sees identical paired inputs.
Using $\mathrm{sp}(z)=\log(1+e^z)$, the relativistic paired game is

```math
\begin{aligned}
\mathcal L_D &= \mathbb E[\mathrm{sp}(D(x_f)-D(x_r))]\\
&\quad +\mathcal R_{\mathrm{cap}},\\
\mathcal L_G &= \mathbb E[\mathrm{sp}(D(x_r)-D(x_f))]\\
&\quad +\mathcal L_{\mathrm{VIC}}(P).
\end{aligned}
```

$G$ denotes the trainable adapter and particle cloud; $D$ is the scalar-valued
critic (discriminator). They minimize their respective losses
$\mathcal L_G$ and $\mathcal L_D$. $\mathbb E$ denotes an average over sampled
pairs, and $\mathrm{sp}$ is the smooth positive function **softplus**.
$x_r,x_f$ mean real/fake critic inputs, $n$ is shared Gaussian noise,
$\sigma_t$ its standard deviation at update $t$, and $I$ the identity matrix.

**The matching signal is purely adversarial.** There is no additional output
reconstruction MSE, feature matching, lyric hold or ending loss. The two
regularizers act on the critic's input gradients and the particle cloud:

```math
\begin{aligned}
g_x &= \lVert\nabla_xD(x)\rVert_2,\\
\mathcal R_{\mathrm{cap}} &= \tfrac12\sum_{x\in\{x_r,x_f\}}\!\mathbb E[
\max(g_x-1,0)^2].
\end{aligned}
```

The gradient $\nabla_xD(x)$ measures how much the critic score changes with its
input; $\lVert\cdot\rVert_2$ is Euclidean length. The cap penalizes only gradient
lengths above 1, keeping the critic's sensitivity bounded by a soft penalty.

For the sample covariance $C$ of 64 particles drawn without replacement, with
particle dimension $d=4$:

```math
\begin{aligned}
\mathcal L_{\mathrm{VIC}}(P)
&= \frac1d\sum_j\max(0,1-\sqrt{C_{jj}+10^{-4}})\\
&\quad +\frac1d\sum_{j\ne k}C_{jk}^2.
\end{aligned}
```

$C_{jj}$ is the sample variance of coordinate $j$; $C_{jk}$ is the covariance
between different coordinates $j$ and $k$. The small $10^{-4}$ stabilizes the
square root. The covariance uses the sample denominator $64-1$.

This encourages each particle coordinate to retain variance while discouraging
correlation between coordinates. Both regularizers have coefficient 1. In the
published release, each update uses separate D/G minibatches of 64 noise-row pairs; the gradient cap
runs every fourth update with a factor of 4. Exports use EMA with decay 0.995.

### Published release and current experiments

The **September 17, 2026 YuE2 release** exports the final EMA at 1,200 updates
for each of its 16 controls. Its normalization uses the mean and coordinatewise
sample standard deviation of frozen positive states, and its noise schedule
retains the original 8,000-update horizon:

```math
\begin{aligned}
T(h) &= \frac{h-\mu_+}{\max(\mathrm{std}(h_+),10^{-4})},\\
\sigma_t &= 0.03^{\min(t/8000,1)}.
\end{aligned}
```

Here $\mu_+$ and $\mathrm{std}(h_+)$ are fixed training-set statistics; the
maximum is coordinatewise and prevents division by a nearly zero deviation.

The release includes 64 matched Off/On comparisons; checkpoints were exported
at the fixed training budget, without an audio-quality selection pass. See the
[full release math](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/MATH.md),
[training audit](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/FORMULATION.md)
and [stability evidence](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/GAN_STABILITY.md).

**Current training defaults are a later experiment.** They use paired-edit
normalization, generated continuation histories and a run-budget noise schedule;
alternative critics and noise holds are also available. The core paired-error
game remains, but a fresh run is not an exact replay of the published release.
Implementation: [YuE2 adapter and trainer](../conceptmod/textsliders/yue2_particle_bridge.py),
[shared game](../conceptmod/textsliders/particle_bridge_gan.py),
[reference audit](yue2-particle-bridge.md).

## Training evidence

The **hip-hop run from the September 18 gmix catalog** completed 1,600 updates
with strong teacher alignment at strength +1. These graphs archive the actual
[dashboard](http://100.90.104.57:8888/yue2-gmix-catalog-20260918/#job-hiphop)
data in this repository, so viewing them does not require access to that host.
This later experiment uses paired-edit normalization, 128 continuation seeds
per caption template and a **gmix critic**: a learned global projection of the
error vector into 8 tokens, followed by one attention layer of width 48.
It uses 8 noise-row pairs per training minibatch.

![Hip-hop training: generator and discriminator losses, particle regularization, and edit-direction cosine over all 1600 updates.](assets/yue2-gmix-hiphop-20260918/training.svg)

**Reading the training curves.** Generator total is the adversarial loss plus
particle VIC; discriminator total includes its gradient cap. **Cosine** measures
alignment between the adapter's hidden-state edit and the positive teacher's
edit: 1 means the same direction, 0 means perpendicular, and −1 means opposite.
The final training cosine is **0.995**. Every update is plotted without smoothing.

![Held-out EMA probes at strength one: residual RMS and p95, teacher sliced Wasserstein distance, and fixed-noise game distance from updates 100 to 1600.](assets/yue2-gmix-hiphop-20260918/heldout.svg)

**Reading the held-out curves.** These probes use continuation seeds excluded
from training and the EMA weights, sampled every 100 updates at strength +1.
**Residual RMS** is the root mean square of the normalized student–teacher
error; **p95** is the 95th percentile of per-example RMS errors.
**SWD** means sliced Wasserstein distance: average distribution mismatch along
256 fixed random projections. Teacher SWD compares student and teacher edits;
game SWD compares noise with noise-plus-error at a fixed noise standard
deviation of 1. Lower is better for all four plotted probe metrics.
**Gain** measures how much of the teacher edit the student produces along its
direction; at strength +1, the target is 1.

| Held-out metric at +1 | Update 100 → 1600 |
|---|---|
| Residual RMS | 0.997 → **0.144** |
| Residual p95 | 1.583 → **0.216** |
| Teacher SWD | 0.663 → **0.050** |
| Game SWD, noise std = 1 | 0.360 → **0.048** |
| Mean gain, target 1 | 0.183 → **0.989** |

This is evidence of successful teacher matching for one completed run at +1.
Intermediate calibration is still imperfect: strength 0.5 has gain 0.778 at the
last probe. The dashboard reports `not_plateaued`, and its independent classifier
evaluation is pending. Audio quality needs listening; the public samples in the
[root README](../README.md#listen) belong to the separate 1,200-update release.

[Archived metrics, exact settings and reproducible plots](assets/yue2-gmix-hiphop-20260918/README.md)
include all recorded strengths, with no probe points removed.

## Other training formulations

Music 3's released LM sliders and the inherited image/flow methods use the
objectives below. These explain the other backends and the evolution of this
fork; YuE2's routed-particle game is defined above.

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
[`lora.py`](../conceptmod/textsliders/lora.py).

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
[`slider_targets.py`](../conceptmod/textsliders/slider_targets.py).

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

See the [target geometry](lm-sheet-goodhart.md),
[positive/neutral formulation](lm-plus-neu-exam.md) and
[lyric preservation study](lm-lyric-hold.md). The generic LM trainer still
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
[release formulation](hub-formulation-fresh-selected.md) and
[`gan_v2/`](../conceptmod/textsliders/gan_v2/) for implementation details.

## Evaluation

A useful comparison keeps the **neutral caption, lyrics, seed and generation
settings fixed**, and changes only adapter strength. Include an adapter-off
positive-caption reference to show what the base model can do with that prompt.
Retain failures, natural endings and truncation flags in the comparison.

Assess concept movement alongside audio quality, lyric preservation, unintended
changes and behavior across held-out prompts/seeds. Hidden-state cosine or a
GAN loss alone cannot answer those questions. Acoustic ladder gates and LM
composition checks measure different behavior; see [SCORING.md](../SCORING.md),
[LM-SCORING.md](../LM-SCORING.md) and the
[listening/selection tools](../slider_selection/README.md).

YuE2's published particle checkpoints use the final 1,200-update EMA, with no
audio-quality selection pass. Its listening gallery is the place to judge
the control and its side effects; preference for the formulation is not a
controlled quality ranking against Music 3 or every other recipe.

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
