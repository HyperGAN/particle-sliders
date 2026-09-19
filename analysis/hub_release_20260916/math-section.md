## How the sliders learn

The **new September 16 voice and genre adapters** use LoRA to change the language model that plans the music.

### One strength control

For an adapted weight matrix, the slider applies:

$$
W(s) = W_0 + s\,\frac{\alpha}{r}BA.
$$

Here \\(W_0\\) is the frozen base weight, \\(A\\) and \\(B\\) are the learned low-rank factors, and \\(s\\) is the strength. The new exports use \\(r=8\\) and \\(\alpha=8\\), so \\(s=0\\) switches the adapter off and \\(s=1\\) applies its exported change. Intermediate strengths scale the weight change smoothly; the resulting music need not change linearly. Negative strength is not a trained opposite sound.

### Voice and genre: learn the change between descriptions

Each of the sixteen new controls learns from **four caption pairs**. One caption describes a starting arrangement and the other adds the requested voice or genre. Lyrics, tempo and section tags stay fixed within each pair.

**1. Build a target change.** The frozen language model encodes the starting and target captions. We compare corresponding lyric-token states and the audio-start state. Let \\(H_0\\) be the starting-caption sequence, \\(H_+\\) the target-caption sequence, and \\(H_\theta\\) the adapted model's sequence on the starting caption:

$$
x_+ = \frac{H_+ - H_0}{\sigma}, \qquad
x_\theta = \frac{H_\theta - H_0}{\sigma}.
$$

The fixed scale \\(\sigma\\) is calibrated from the teacher changes using root mean square (RMS). It puts the target change \\(x_+\\) and the adapter's change \\(x_\theta\\) in the same units. Padding is excluded.

**2. Train a critic to recognize that change.** A small two-layer transformer \\(D\\) learns to score target changes above adapted changes. It combines the mean of valid token features with the final audio-start feature. Its relativistic pairing objective is:

$$
\mathcal L_D =
\mathbb E\!\left[\mathrm{softplus}\big(D(x_\theta)-D(x_+)\big)\right]
+ \mathcal R_{\mathrm{cap}}.
$$

Here \\(\mathbb E\\) denotes a batch average, and \\(\mathrm{softplus}(u)=\log(1+e^u)\\) penalizes the wrong ordering smoothly. The critic is used during training only.

**3. Train the adapter for the target change, its features and stopping behavior.** The adapter minimizes three terms, each with coefficient 1:

$$
\begin{aligned}
\mathcal L_G ={}&
\mathbb E\!\left[\mathrm{softplus}\big(D(x_+)-D(x_\theta)\big)\right] \\
&+ \mathrm{MSE}\!\left(\mathbb E[\phi(x_\theta)],\mathbb E[\phi(x_+)]\right)
+ \mathcal L_{\mathrm{end}}.
\end{aligned}
$$

| Term | What it teaches |
|---|---|
| Adversarial comparison | Make the adapted change resemble the target-caption change. |
| Feature matching | Match the batch means of the critic's learned features \\(\phi\\). MSE is mean squared error over feature coordinates. |
| Ending supervision | Preserve the base model's balance between continuing and ending on a supplied token history. |

Warm-up uses four rows per batch. Fresh-continuation training uses one row per update, so feature matching then compares that update's student and teacher features. The adapter has no separate explicit lyric-hold term.

**4. Preserve the stopping margin on fresh histories.** Let \\(\ell\\) denote token logits and \\(\mathcal S\\) the semantic audio-token vocabulary. On the same supplied history, compare the audio-end logit with the total semantic-continuation mass:

$$
\begin{aligned}
m &= \ell_{\mathrm{audio\_end}} - \log\sum_{j\in\mathcal S}e^{\ell_j}, \\
\mathcal L_{\mathrm{end}} &= \mathrm{MSE}(m_\theta,m_0).
\end{aligned}
$$

The first 600 updates use fixed base histories. **From update 601 onward, each update samples a fresh frozen-base continuation with the adapter disabled**, cycling through all four training rows in balanced shuffled passes. Seeds do not repeat; duplicate continuation tensors are rejected without substituting a new seed. These histories diversify ending supervision. The prompt-state style targets remain fixed per row.

The ending term preserves a stopping decision on the supplied histories. It does not establish that every newly generated full song will end naturally.

**5. Keep the critic's gradients bounded.** In the calibrated input coordinates, the cap penalizes gradient norms only above 1:

$$
\mathcal R_{\mathrm{cap}} =
\frac{1}{2}\sum_{z\in\{x_+,x_\theta\}}
\mathbb E\!\left[
\max\!\left(0,\lVert\nabla_zD(z)\rVert_2-1\right)^2
\right].
$$

The cap coefficient and threshold are both 1. Small gradients are allowed, and the initially zero adapter change remains included.

| Training setting | New September 16 release |
|---|---|
| Adapter location | Language-model attention; 36 layers × four projections |
| Starting point | A fresh zero-output rank-8, alpha-8 adapter and critic per control |
| Warm-up | 600 updates; all four prompt pairs per batch |
| Further training | One row per update; balanced shuffled passes; one fresh base continuation per update |
| Learning rates | Adapter 0.0005; critic 0.00075; constant |
| Continuation step bound | Adapter parameter-update L2 norm at most 2 |
| Training budgets | Eight runs to 3,400; eight runs to 2,000 |
| Published checkpoints | Selected per control at 1,000, 2,000, 3,000 or 3,400 |
| Published samples | 64 Off/On pairs: two arrangements × two seeds for each of all sixteen controls |

The critic, optimizers, sampler and RNG state carry across milestones. Every selected native export was checked tensor-for-tensor against its saved full state. The full [training record](evidence/uni16-fresh-selected-v2/training.json) gives the recipe and budgets.

### Checkpoint selection: quality bounds, then later steps

Selection happens after training. For one slider, let \\(\mathcal C\\) contain checkpoints that pass the technical screen, with four-clip mean enjoyment \\(E_c\\), mean production quality \\(P_c\\), and step \\(t_c\\). The quality anchors and eligible set are:

$$
\begin{aligned}
E^* &= \max_{c\in\mathcal C} E_c, \qquad
P^* = \max_{c\in\mathcal C} P_c, \\
\mathcal Q &= \left\{c\in\mathcal C :
E^*-E_c\leq0.2\;\text{ and }\;P^*-P_c\leq0.2\right\}.
\end{aligned}
$$

Among qualifying checkpoints from the new run, choose:

$$
c_{\mathrm{release}} =
\mathrm{arg\,max}_{c\in\mathcal Q_{\mathrm{new}}} t_c.
$$

If no new-run checkpoint qualifies, the legacy release can be considered under the same quality bounds; an empty eligible set requires listening to resolve the tradeoff. All sixteen current selections are from the new runs. Description similarity and cached lyric checks have zero influence on this selection. The 0.2-point tolerance is a preference rule, not statistical equivalence or a proven audible boundary. [Selection policy and sensitivity](evidence/uni16-fresh-selected-v2/README.md).
