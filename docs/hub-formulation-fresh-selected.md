# Fresh-continuation slider training

Each control is a rank-8, alpha-8 adapter on the frozen Music 3 language model's attention projections. At inference, strength s applies W(s) = W0 + s (alpha / rank) BA. The adapter has no trained negative pole.

A starting caption and target caption share lyrics, tempo and section tags. The frozen model provides prompt-state sequences H0 and H+, sampled at corresponding lyric tokens and the audio-start token. The adapted model supplies Hθ on the starting caption. The target and student changes are scaled by a fixed RMS calibration σ:

$$
x_+ = (H_+ - H_0)/\sigma, \qquad x_\theta = (H_\theta - H_0)/\sigma.
$$

A two-layer transformer critic D learns the target change with a relativistic pairing objective and an input-gradient cap:

$$
\mathcal L_D = \mathbb E[\operatorname{softplus}(D(x_\theta)-D(x_+))] + \mathcal R_{\mathrm{cap}}.
$$

The adapter minimizes adversarial comparison, feature matching and ending supervision, each with coefficient 1:

$$
\mathcal L_G = \mathbb E[\operatorname{softplus}(D(x_+)-D(x_\theta))]
+ \operatorname{MSE}(\mathbb E[\phi(x_\theta)],\mathbb E[\phi(x_+)]) + \mathcal L_{\mathrm{end}}.
$$

The feature term compares batch means of critic features. Warm-up uses all four rows per batch; continuation uses a single row per update, so this comparison is then between that update's student and teacher features. Prompt-state targets stay fixed per training row.

Ending supervision compares the audio-end versus semantic-continuation margin on the same frozen-base token history:

$$
m = \ell_{\mathrm{audio\_end}} - \log\sum_{j\in\mathcal S}e^{\ell_j},
\qquad \mathcal L_{\mathrm{end}}=\operatorname{MSE}(m_\theta,m_0).
$$

During the initial 600-update warm-up, each row has a fixed base history. From update 601 onward, every update samples a fresh base-model continuation with the adapter disabled. All four training rows participate in balanced shuffled passes. Seeds do not cycle and duplicate continuation tensors are rejected without substituting another seed. Fresh continuations diversify ending supervision; they do not replace the fixed prompt-state style teachers.

The critic's gradient cap penalizes only norms above 1, with coefficient 1. Generator and critic rates are 0.0005 and 0.00075, constant throughout. The continuation limits each adapter parameter update to L2 norm 2. The critic, both optimizers, sampler and RNG states carry across milestones. Explicit lyric hold is disabled.

Eight runs completed 3400 updates and eight completed 2000 under the approved cost budget. Selection happens after training using the separate quality rule in [selection-policy.json](selection-policy.json). These selected steps are not claims of convergence. Each selected export is verified against every corresponding LoRA tensor in its pinned full state.

[Full recipe and budgets](training.json) · [Checkpoint selection](README.md) · [Native loading](../../usage.md)
