# Conditional feature-matching CPU probe

Row-paired feature matching at weight 1 was the best of these three arms.
It recovered the correct row targets across all three seeds, while marginal
batch-mean matching left substantial direction errors and excess magnitude.

The experiment uses the production `RowConditionalD` wrapping
`SpanTransformerD(in_mode="scaled", readout="mean_last")`: four rows,
64-dimensional inputs, eight tokens, width 32, two layers, four heads.
Each teacher is a shared random vector plus a row-specific random vector;
the last token adds another row-specific offset. The fake rows are directly
learnable tensors initialized to zero. Repeated span-token targets make
exact matching attainable despite the critic's set invariance.

Each arm runs 600 CPU steps with identical data and initialization for a
given seed. The objective is `0.1 * RpGAN_G + FM_weight * FM`; D uses RpGAN
plus `b_cap(coeff=1, kappa=1)` in fixed calibrated coordinates. Both Adam
optimizers use `betas=(0, 0.999)`, G LR 0.01, D LR 0.00075. These are constant
learning rates with no EMA, hold loss, or hidden-state MSE. The larger G LR
reflects the direct fake-coordinate parameterization, not a proposed LoRA LR.

| Seed | FM objective | FM weight | Whole nRMSE | Last-token cosine | Worst row cosine | Mean magnitude ratio | Max row magnitude ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | Batch mean | 1 | 0.4219 | 0.8323 | 0.6503 | 1.1301 | 1.3875 |
| 23 | Batch mean | 1 | 0.4150 | 0.8651 | 0.7073 | 1.0870 | 1.1313 |
| 101 | Batch mean | 1 | 0.5290 | 0.8198 | 0.7793 | 1.2182 | 1.3520 |
| 7 | Paired row | 1 | 0.0864 | 0.9935 | 0.9927 | 0.9897 | 1.0121 |
| 23 | Paired row | 1 | 0.1230 | 0.9827 | 0.9739 | 1.0114 | 1.0411 |
| 101 | Paired row | 1 | 0.0465 | 0.9978 | 0.9973 | 0.9978 | 1.0047 |
| 7 | Paired row | 10 | 0.0862 | 0.9912 | 0.9834 | 0.9965 | 1.0480 |
| 23 | Paired row | 10 | 0.1964 | 0.9718 | 0.9604 | 1.0208 | 1.1498 |
| 101 | Paired row | 10 | 0.0640 | 0.9959 | 0.9928 | 1.0069 | 1.0171 |

All fake gradients and parameters remained finite. At the paired-weight-1
endpoints, weighted adversarial input-gradient norms were 0.021–0.030 and
FM input-gradient norms were 0.0042–0.0135. These are small but nonzero;
the critic was approaching indistinguishability, with D loss 0.556–0.692
and G adversarial loss 0.703–0.878. The batch-mean arms still had G
adversarial loss above 3.2. Weight 10 increased FM gradient norms to
0.118–0.340 and gave less accurate endpoints than weight 1 on two seeds.

## Why row-paired matching is sound

Let `f_i` and `r_i` be the fake and real features for the same prompt row,
and `e_i = f_i - r_i`. With consistent averaging over feature dimensions,

```text
paired FM = mean_i ||e_i||²
          = ||mean_i e_i||² + mean_i ||e_i - mean_j e_j||²
          = batch-mean FM + variance of row errors.
```

Both terms vanish when each fake matches its own teacher. The added term
constrains conditional row correspondence. This differs from the previous
incorrect per-row-to-global-teacher-mean loss, which adds variance of fake
features themselves and shrinks legitimate teacher diversity.

For one deterministic teacher per condition, paired matching is conditional
first-moment matching. It is a learned-feature supervision term: it uses
the matching teacher features, rather than only the adversarial ranking.

## Limits and reproduction

Direct fake rows can move independently; a shared LoRA is more constrained.
This fixture does not test lyric preservation, rendering, unseen captions,
or arbitrary positional detail. Feature matching also has null directions
whenever D's features discard information. Its scale depends on the learned
feature representation, so weight 1 here is an initial live experiment,
not a calibrated universal strength.

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  sliders-conceptmod/analysis/gan_bcap/paired_fm_probe.py
```

Full settings, per-seed histories every 50 steps, gradients, and exact-match
controls are in `paired_fm_results.json`. Raw stdout is
`paired_fm_probe.log`; the executable fixture is `paired_fm_probe.py`.
