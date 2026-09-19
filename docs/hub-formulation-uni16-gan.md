# How the sliders learn

Both releases are small LoRA adapters: they add a learned change to a frozen base model. The **voice and genre sliders** change the language model that plans the music. The **reward slider** changes the acoustic transformer that turns that plan into audio. Their training objectives are different.

## One strength control, two places to attach it

For each adapted weight matrix, the slider applies:

$$
W(s) = W_0 + s\,\frac{\alpha}{q}BA.
$$

Here \\(W_0\\) is the original weight, \\(A\\) and \\(B\\) are the learned low-rank factors, and \\(s\\) is the strength. These exports use rank \\(q=8\\) and scaling parameter \\(\alpha=8\\), so \\(s=0\\) switches the adapter off and \\(s=1\\) applies its exported change. Intermediate positive strengths scale the weights smoothly; the resulting music need not change linearly. Negative strengths were not trained as opposite sounds. The training critic and reward scorer are unnecessary during generation.

## Voice and genre: learn the change between two descriptions

Each of the 16 controls learns from four caption pairs. One caption describes the starting arrangement; the other adds the requested voice or genre. The lyrics, tempo and section tags stay fixed within each pair. During generation, use the starting caption and let the adapter supply the change.

**1. Measure what the added description changes.** The frozen language model encodes both captions. We compare their hidden states at matching lyric tokens and the audio-start token. \\(H_0\\) is the starting-caption sequence, \\(H_+\\) is the target-caption sequence, and \\(H_\theta\\) is the adapted model's sequence on the starting caption:

$$
x_+ = \frac{H_+ - H_0}{\sigma}, \qquad
x_\theta = \frac{H_\theta - H_0}{\sigma}.
$$

The fixed scale \\(\sigma\\) is calibrated from the teacher changes using root mean square (RMS). It puts both changes in the same units. \\(x_+\\) is the example to learn from; \\(x_\theta\\) is the adapter's attempt. Padding is excluded.

**2. Train a critic to recognize the target change.** A small two-layer transformer, \\(D\\), scores the changes. The critic learns to score the target above the adapter's attempt; the adapter learns to reverse that ordering. This is the adversarial part of the method, called a *relativistic pairing GAN*.

$$
\mathcal L_D = \mathbb E\!\left[\mathrm{softplus}\big(D(x_\theta)-D(x_+)\big)\right]
+ \mathcal R_{\rm cap}.
$$

Here \\(\mathbb E\\) means the average over the training batch, and \\(\mathrm{softplus}(u)=\log(1+e^u)\\) is a smooth penalty that grows when the ordering is wrong. The critic combines the mean of the valid tokens with the final audio-start token.

**3. Teach the adapter the sound, its broader features, and when to stop.** The adapter minimizes three terms:

$$
\begin{aligned}
\mathcal L_G ={}& \mathbb E\!\left[\mathrm{softplus}\big(D(x_+)-D(x_\theta)\big)\right] \\
&+ \mathrm{MSE}\!\left(\mathbb E[\phi(x_\theta)],\mathbb E[\phi(x_+)]\right)
+ \mathcal L_{\rm end}.
\end{aligned}
$$

| Term | What it teaches |
|---|---|
| Adversarial comparison | Make the adapted change resemble a change caused by the target caption. |
| Feature matching | Match the batch averages of the critic's learned features, \\(\phi\\). MSE means mean squared error across feature coordinates. |
| Ending penalty | Keep the base model's balance between continuing the music and ending it. |

The feature term compares **batch averages**, not every song to one shared target. All three terms have weight \\(1\\) in this release.

For the ending penalty, the model is fed the same token history from a frozen base-model composition. If \\(\ell\\) denotes token logits and \\(\mathcal S\\) the semantic audio-token vocabulary, the stop margin and penalty are:

$$
m = \ell_{\rm audio\_end} - \log\!\sum_{j\in\mathcal S}e^{\ell_j},
\qquad \mathcal L_{\rm end}=\mathrm{MSE}(m_\theta,m_0).
$$

This discourages a changed stopping decision on that fixed history; it does not guarantee that a newly generated song will finish naturally.

**4. Keep the critic from becoming too steep.** The cap adds a penalty only when the critic's input-gradient norm exceeds \\(1\\):

$$
\mathcal R_{\rm cap}=\frac12\sum_{z\in\{x_+,x_\theta\}}
\mathbb E\!\left[\max\!\left(0,\|\nabla_zD(z)\|_2-1\right)^2\right].
$$

The gradient uses the calibrated coordinates above. Small slopes are allowed, and the initially zero adapter change is included. The cap coefficient and threshold are both \\(1\\).

| Training setting | This release |
|---|---|
| Adapter location | Language-model attention projections (Qwen3Attention) |
| Starting point | A fresh adapter with zero output for each control |
| Training | 600 updates, then 60 more carrying that control's critic, adapter and optimizer state |
| Learning rates | Adapter: \\(0.0005\\); critic: \\(0.00075\\); constant rates |
| Batch | Four caption pairs per batch |
| Training seed | One training run per control, using RNG seed `7`; no repeated training seeds |
| Continuation step limit | Each adapter parameter update has L2 norm at most \\(2\\) |
| Explicit lyric hold | Disabled; the lyric-span change comes from the target caption |
| Listening coverage | 60 comparisons: two new arrangements × generation seeds `7` and `23` for each of 15 controls; Reggaeton audio pending |

Four different caption/lyric rows are reserved for evaluation; the published grid uses two of them. The other two remain reserved. Exact settings are in [the training record](training.json).

The earlier uni-lyric release explicitly held lyric-token states close to the starting caption. **UNI16 removes that hold.** It can change phrasing and composition as well as the requested sound, so the rendered words still need listening checks. A fixed seed makes the comparison reproducible; it does not lock the composition. The release includes all 16 update-660 exports and does not claim that every control improves quality or outperforms its update-600 version.

## Reward: learn from an audio enjoyment score

The separate **refined block step 2** adapter is trained to raise a frozen model's *Content Enjoyment* (CE) score. It starts from an earlier acoustic adapter and refines attention and feed-forward factors together. CE is an automated score on a 0–10 scale; a gain of \\(0.1\\) means one tenth of a score point.

For each training case \\(i\\), \\(y_{\theta,i}\\) is the adapted audio and \\(y_{\mathrm{off},i}\\) is the fixed Off comparison. The scorer \\(R\\) gives the gain:

$$
\Delta R_i = R(y_{\theta,i})-R(y_{\mathrm{off},i}).
$$

The adapter minimizes:

$$
\mathcal L_{\rm reward} = \frac18\sum_{i=1}^{8}
0.1\,\mathrm{softplus}\!\left(\frac{0.1-\Delta R_i}{0.1}\right)
+ 10\,\mathcal L_{\rm latent}.
$$

The first term pushes each case toward a \\(+0.1\\) score gain, with more pressure on weak or regressing cases. The margin is a training target, not a promised improvement. The second term keeps the intermediate audio representation close to the parent adapter's capture. Its normalized reconstruction error is:

$$
\mathcal L_{\rm latent} = \mathbb E_{i,c}\!\left[
\frac{\mathrm{MSE}(z_{\theta,i,c},z_{\mathrm{parent},i,c})}
{\max\!\left(\mathrm{mean}(z_{\mathrm{parent},i,c}^{\,2}),10^{-8}\right)}
\right].
$$

Here \\(z\\) is the latent audio representation, and the average covers training cases \\(i\\) and chunks \\(c\\). Dividing by the parent's mean squared value makes the error relative to that capture's scale.

| Training setting | This release |
|---|---|
| Adapter location | Acoustic transformer: 144 attention and 72 feed-forward projections |
| Starting point | A previously trained, composed acoustic adapter |
| Refinement | Two full-batch updates over eight cases: four voice-balanced families × two seeds |
| Optimizer | AdamW; learning rate \\(0.0005\\); betas \\(0.9/0.999\\); no weight decay; gradient norm clipped at \\(1\\) |
| Gradient path | Frozen scorer → frozen vocoder → final two of thirty flow steps per chunk |
| Fixed during refinement | Earlier parent states, conditioning and overlap between chunks |
| Evaluation | Ordinary full generation and native LoRA merging; direct strength \\(1\\) |

The gradient covers the tail of the rendering process, not a complete generation. The two updates are a refinement of trained parents, not the adapter's entire training history. The scorer measures two non-overlapping ten-second windows from the first twenty seconds, using an RMS-normalized measurement copy. Published recordings retain their original levels.

| Fresh evaluation | Result and meaning |
|---|---|
| CE gains versus Off | 15 of 16 comparisons improved; mean \\(+0.110204\\) points. |
| Uncertainty | 95% bootstrap interval \\([+0.080603,+0.140382]\\), resampling whole prompt families. |
| One lower score | The remaining case fell by \\(0.005768\\) points. |
| Preservation versus Off | One dynamics check failed: crest factor changed by \\(-1.4495\\) dB, beyond the allowed \\(1.3803\\) dB. Crest factor measures peak level relative to RMS level. |
| Preservation versus the original reward adapter | Twelve comparisons flagged changes in lyrics, style, voice or dynamics. |
| Still untested | Independent replication, combination with studio sliders at fixed energy, and natural full-song completion. |

**The score improved in this batch, but preservation failed.** This remains an experimental listening release; the scores do not establish a general improvement in human preference.

[Reward training recipe](../reward-refined-block/training.json) · [Fresh scorecard](../reward-refined-block/scorecard.json) · [Preservation report](../reward-refined-block/preservation.json)
