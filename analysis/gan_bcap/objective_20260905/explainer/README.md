# What the custom MMD loss measures

[Interactive curves and distance controls](index.html) · [Raw per-attempt CSV](loss-steps.csv)
· [Values and source hashes](data.json)

![Recorded music curves](loss-curves.png)

The monotonically non-increasing number is the fixed, full-batch conditional
MMD training loss: **1.16076550 → 0.87670718**, a **24.47% reduction**.
It is evaluated after every accepted or rejected Adam proposal. The data,
teacher targets, calibration and kernel stay fixed. The separate relative
hidden error in panel C is mean(prompt error norm / teacher-delta norm):
**.922256 → .759827** on training rows. Exactly zero slider displacement gives
relative error 1; exactly reproducing the teacher delta gives 0.

About 79.5% of the observed loss reduction happens in the first 30 attempts.
The final 30 only lower the loss .402%. There are 99 accepted steps and 21
rejections; 89 accepted steps were shortened. The x-axis counts attempts, with
zero at the original checkpoint 600. Label 720 therefore includes 120 MMD
attempts, of which only 99 changed the weights.

## Why the loss never increases

Adam first proposes a parameter update. The trainer tries that update, then
1/2, 1/4, 1/8, 1/16 and 1/32 of it, accepting the first one that lowers the
same measured full-batch loss. If every candidate fails, it restores both the
weights and the optimizer moments. Accepted attempts decrease the objective;
rejections create flat steps. The logs contain zero uphill transitions.

This monotonicity is enforced by the optimizer acceptance rule. It is not a
general convergence theorem for MMD, and it does not apply to the heldout
prompts or generated-audio quality. A sequence of rejected proposals can
remain flat with substantial error.

## What one comparison does

Each condition pairs the slider and positive-caption teacher at the same
lyric/history position and guidance branch. Their 4096-dimensional hidden
vectors yield a normalized distance:

```text
r = RMS(h_slider - h_teacher) / s
s = 0.2667504186620667
```

Here s is calibrated once from the positive-minus-neutral teacher deltas on
the training set. It is not recomputed from the current student.

Five fixed distance scales act like rulers for fine through coarse errors:

```text
b = [.03, .1, .3, 1, 3]
loss per pair = mean_b 2 * (1 - exp(-r² / (2 b²)))
training loss = average over positions and branches, then prompt rows
```

Exact agreement gives 0. Extremely large mismatch approaches 2. Small
mismatch behaves locally like squared error. Far beyond all five scales,
the loss saturates and its gradient also becomes small. The final .8767
is a unitless discrepancy, not a percentage wrong or a music-quality score.
Because the batch contains different distances, average loss cannot be
inverted into a unique average RMS error.

![Analytic distance response](kernel-intuition.png)

## Relation to the Gaussian experiments

The general MMD objective compares two entire distributions. Similarity
between real and generated samples lowers the objective; excessive similarity
within the generated samples raises it. That balances attraction to the
target cloud with a force against crowding all fake samples together. The
real-real term is constant for generator optimization.
[MMD definition and estimators](https://jmlr.org/papers/v13/gretton12a.html).

In this music experiment, each fixed condition produces one deterministic
teacher vector and one student vector. The two self-similarities equal 1,
so the same MMD expression becomes `2 * (1 - similarity)`. This is paired
teacher regression. Different lyrics or time positions must not be pooled
and repelled as though they were samples from one interchangeable condition.

| Property | Original Gaussian cap control | Gaussian MMD alternative | Custom music MMD |
| --- | --- | --- | --- |
| Compared outputs | Random 2-D clouds | Random 2-D clouds | Paired 4096-D hidden states |
| Trained discriminator | Fourier MLP with cap | None | None |
| Loss | Relativistic logistic plus cap on D; latent VICReg | Fixed multiscale MMD | Same kernel, conditional singleton MMD |
| Movable latent particles | 20,000 | 20,000 | None |
| Data per update | 256 fresh samples per draw | 1024 fresh samples per draw | Four fixed prompt rows, both branches, 250 cached history frames |
| Geometry | Raw 2-D Euclidean distance | Raw 2-D Euclidean distance | Full hidden RMS / frozen teacher-delta RMS |
| Optimizer control | Adam, LR decay, G/prior EMA | Adam, LR decay, G/prior EMA | Fresh Adam moments, nominal LR .0005, decrease check, no EMA |
| Budget | 7000 updates | 28000 updates | 120 proposals starting at checkpoint 600 |

Thus it is a substantial change from the original adversarial Gaussian
game. It shares the mathematical MMD kernel with the Gaussian MMD alternative,
but its conditioning, geometry, optimizer and observation coverage differ.
Using the same bandwidth numbers does not make the numerical units identical.

![Sparse Gaussian loss records](gaussian-losses.png)

The Gaussian MMD panel contains only four logged minibatch estimates. An
unbiased finite-batch MMD estimate can cross below zero from sampling noise;
that does not mean a negative population distance. It is not comparable
numerically to the deterministic, nonnegative paired music score. Likewise,
the cap control's logistic losses are different quantities.

The music curve measures imitation on cached histories. During generation,
the model chooses new histories that the loss did not evaluate. The fixed
training-set decrease therefore gives no monotonic guarantee for the song.

Reproduce these figures without loading a model:

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m analysis.gan_bcap.objective_20260905.loss_explainer
```

Each figure is available as PNG, SVG and PDF. Analytic response curves are
labeled as illustrations; recorded loss curves use unsmoothed log values.
