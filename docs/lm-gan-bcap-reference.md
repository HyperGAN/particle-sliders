# b_cap reference and LM transfer audit

Audited 2026-09-04 against local `/tmp/opencode/ParticleGAN`, commit
`b5ee35f9b24cf35a6b1346ba2f0856c4877aa336` (also the upstream HEAD checked
that day). The source is the code at that commit; cached repository landing
pages can show the older experiment without a gradient penalty.

## Exact reference game

For paired real/fake critic scores, the discriminator minimizes
`mean(softplus(fake_score - real_score))`. The generator minimizes
`mean(softplus(real_score - fake_score))`. The discriminator additionally pays

```text
(coeff / 2) * (mean(relu(norm(grad D(real)) - kappa)^2)
             + mean(relu(norm(grad D(fake)) - kappa)^2))
```

Gradients are L2 norms per sample, computed with `create_graph=True` in the
critic's input coordinates. Both sides include zero-valued samples. The
winning logistic configuration uses `coeff=1`, `kappa=1`, every step.
Source: [gradient regularizer](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/lib/grad_regularizers.py).

The 100-Gaussian configuration has 20,000 four-dimensional latent particles,
batch 256, Fourier-2 MLP discriminator, and 7,000 steps. Adam uses
`betas=(0, 0.999)`: G LR `6e-4`, D LR `9e-4`, prior LR `6e-3`. All learning
rates stay full for 60% of training, then follow cosine decay to a 5% floor.
Evaluation uses EMA `0.995` for **both** generator and prior.
Source: [winning configuration](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/configs/b_cap_c1p0_lr2p0_s1.yaml),
[training loop](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/experiments/train_arm.py).

Reference particles enter `G(z)`. The generator and prior optimize the same
G adversarial loss in one backward pass; VICReg regularizes sampled unique
latent vectors. The reported five-seed result is 100 modes, high-quality
fraction 0.986, and core width ratio 0.866. These are upstream toy results,
not evidence of music quality. The cap can become inactive near equilibrium;
it does not impose a nonzero slope or permanent damping.
Source: [findings and limitations](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/FINDINGS.md).

## Concrete failures in the previous LM transfer

- **Singular coordinates.** The unit/log-norm stem divides by each fake
  delta's norm. A CPU probe with 64-dimensional inputs, seed 1, and MLP
  width 16 measured input gradient norm `6.77e7` at zero, while the old cap
  excluded that fake and returned zero. At fake norm `1e-5`, its cap was
  `1.98e9`. The width-16, one-layer span critic measured `7.16e7` at zero;
  a sequence with one zero token and otherwise norm-20 tokens paid
  `9.87e14`, because the exclusion inspected entire rows rather than tokens.
  Fixed teacher calibration preserves magnitude with a finite Jacobian.
  For `u=x/s`, a cap in calibrated coordinates must measure
  `norm(grad_u D)=s*norm(grad_x D)`.
- **Different critic between phases.** The old span discriminator included
  batch standard deviation when D trained on multiple rows. G used single
  rows, where that channel was constant zero. That also invalidated the
  assumption that summing logits yields independent per-sample gradients.
- **Feature matching collapsed correct solutions.** Averaging per-row
  distances to the teacher batch mean adds a fake-feature variance penalty.
  With both fake and real features exactly `[-1, +1]`, that loss is 0.5
  and moves both rows toward zero; true batch-mean feature matching is zero.
- **Historical logs do not establish dead gradients.** The initial penalty
  in `gender-uni-tx1` and `gender-uni-tx2` was 578.23 against adversarial
  D loss 0.438. Their final direction cosines after 400 steps were 0.148
  and 0.186. Their `gadv_norm=0` fields were diagnostic defaults because
  gradient accounting was disabled. Logs remain under
  `/tmp/opencode/lm-adv/<run>/<run>_train.jsonl`.

## Prompt-row mining is a separate optional mechanism

The LM `ParticleBatch` never supplies latent inputs to G. It weights fixed
prompt rows using detached discriminator losses, seeking weak rows. This
is adaptive row mining; the movable-prior result does not transfer to it.

VICReg on these logits did not prevent query collapse: adding 50 to every
particle's first row-logit left VICReg at approximately 0.10371 while the
mean row weight became `[1, 0, 0, 0]`. Centering each column erases that bias.

The repaired miner reserves 10% uniform mass by default, guaranteeing every
row a weight of at least `0.1 / num_rows`. Its `balance_loss(idx, temp)` is
`KL(uniform || mean_softmax(logits[idx] / temp))`, evaluated with logsumexp
before the uniform mixture. It penalizes omitted rows, permits individual
queries to specialize, and retains restoring gradients at saturated
logits. The existing VICReg helper remains for latent-vector use and now
handles a single particle without NaNs. These checks are exercised in
`tests/test_lm_particles.py`; live LM comparisons with and without mining
are recorded in [the repair report](lm-gan-bcap.md).
