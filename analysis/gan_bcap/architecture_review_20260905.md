# Music GAN architecture review against the 100-Gaussian b_cap experiment

The repaired music GAN implements the reference adversarial loss and calibrated
cap correctly in the checks performed. The larger risks are the additional
feature-matching gradient, the restricted observations available to the critic,
and metrics that cannot establish output diversity or condition fidelity.
These are concrete differences from the successful Gaussian experiment, not
evidence that b_cap itself is broken or that rank 8 is necessarily insufficient.

Scope: the listening-preferred unconditioned, batch-FM, rank-8 smoke recipe;
the original 600/750/failed-900 states; and the later step-limited continuation.
The original failed 900 must not be confused with the later bounded 900, which
the user also found appealing. No training source or saved weight was changed.

## Verified findings, in priority order

### 1. High: b_cap does not protect the added feature-matching gradient

[lm_adv.py:351](../../conceptmod/textsliders/lm_adv.py#L351) scores
`head(RMSNorm(pooled))`, while
[features():364](../../conceptmod/textsliders/lm_adv.py#L364) returns the raw
`pooled` vector. The generator matches the means of those raw features.
The cap penalizes the input gradient of the **scalar score**; it does not
bound the feature Jacobian, the complete generator gradient, or the optimizer
step. It is also a soft, sample-local penalty, not a hard global Lipschitz bound.

The new CPU audit reproduces the scale mismatch on the saved 600 critic:
doubling both pooled feature sets multiplies FM by **4**, while the maximum
score change is only **0.00353**. This is a counterfactual at the feature
interface, not an exact reparameterization of the entire transformer.

The earlier crossed-checkpoint probe makes this more than an abstract concern:
with the same 750 critic, changing the adapter from 750 to failed 900 increases
raw FM from **0.417 to 62.745**. At the latter input, the FM input-gradient norm
is **2.717**, versus **0.174** for adversarial loss, while the fake-side cap
penalty is zero. These are critic-input gradients, not full LoRA gradients.
See [the earlier probe](failure_cause_findings_20260904.md).

This is a verified stabilization gap and a plausible amplifier of the failure.
Post-failure snapshots do not identify what initiated the overshoot.

Controlled intervention: compare the same continuation with FM on normalized
features, and separately with an explicit FM gradient budget. Normalizing FM
removes this raw-scale freedom; it does **not** make the scalar-score cap bound
every feature direction. Keep the existing parameter step limit during those
comparisons. Attribute adversarial, FM and end-regularizer gradients separately.

### 2. High: the training game has no direct view of generated-song diversity

The Gaussian generator produces samples of the distribution the discriminator
judges. Our discriminator judges four deterministic sequences of **prompt
hidden-state differences**, consisting of aligned lyric positions plus the
audio-start position. It does not consume generated songs, generated audio-code
trajectories, or multiple generation seeds. Batch 4 repeats all four training
rows every update; more updates do not add examples.

The generator is regularized on a cached, teacher-forced audio history, but
the active regularizer is only stop-versus-continue margin MSE:
[trainer:1034](../../conceptmod/textsliders/train_lm_slider_music3.py#L1034),
[trainer:2020](../../conceptmod/textsliders/train_lm_slider_music3.py#L2020).
`lyrichold_weight=0`, `planreg_weight=0`, and `pole_weight=0` in this recipe.
The critic itself sees the prompt prefix, not the appended audio history.

The audit constructs semantic logits with **zero end-margin error** but
**11.99 nats** of semantic-plus-EOS KL and a different most likely audio code.
The reason is exact: permuting the semantic logits preserves `logsumexp` and
therefore the end margin. This is a logit-space counterexample, not a claim
that every such permutation is reachable through this LoRA and fixed head.

Consequently, a low GAN loss and stable duration can coexist with changed
phonemes, voice, arrangement, or sampling diversity. Heldout policy KL is
broader than the end margin, but still covers fixed semantic-code histories,
excluding the residual-code decoder and full free-running audio distribution.
It measures agreement with a reference, not musical preference.

Controlled intervention: broaden training prompt/lyric support and assess
free-running outputs on reserved prompt/seed fixtures. Keep condition fidelity,
lyric preservation, duration, artifacts and seed diversity as separate outcomes.
Do not interpret a stationary four-row hidden-state game as audio convergence.

### 3. High: the critic has an explicit linear blind subspace

Before any transformer nonlinearity, the scaled input is projected from 4096
dimensions to 128:
[lm_adv.py:302](../../conceptmod/textsliders/lm_adv.py#L302),
[lm_adv.py:324](../../conceptmod/textsliders/lm_adv.py#L324).
The saved 600 projection has rank 128, leaving **3968 exactly invisible input
directions per token for that fixed critic**. Score and FM share this projection.

The audit adds a perturbation in this nullspace with RMS twice the teacher
delta RMS. The largest score change is **1.91e-6** and feature change **4.65e-6**,
consistent with float32 rounding. On the cached 600 spans, **94.57% of the
student/teacher squared error** lies in this current projection's nullspace.

That percentage is hidden-state error energy, **not audible error or percent
of musical information lost**. The learned projection can rotate at the next
D step, and the LoRA may not reach arbitrary constructed perturbations.
Nevertheless, the current G step cannot receive an adversarial or FM correction
through these directions. Low FM therefore cannot certify complete alignment.

Candidate ablation: additional independent projections or a wider critic, with
the same data and a capacity control. Merely increasing critic capacity on four
examples can make memorization worse. A separate fixed representation can
provide another view, but optimizing raw hidden MSE is not justified by the
listening preferences observed so far.

### 4. Medium–high: unconditional matching does not identify the right row mapping

`adv_condition=none` and `fm_mode=batch` remove explicit per-prompt
correspondence. Batch-mean FM is unchanged by permuting the teacher rows.

The audit constructs a wrongly permuted batch and a flat unconditional critic
with discriminator loss `log(2)`, zero D parameter gradient, zero generator
gradient, and zero FM, despite nonzero per-row error. Separately, collapsing
all feature samples to the teacher mean also gives zero FM gradient.

These claims have important boundaries: the **pairwise Rp loss is not invariant
to every row permutation for an arbitrary critic**; the first example is a
stationary witness for the adversarial-plus-FM subgame. End regularization
adds rowwise constraints to the full objective. A capable critic can detect
centroid collapse; mean matching alone cannot.

Row-conditioned critics and paired FM already exist in this repository.
The previous combined ablation sounded worse despite better hidden metrics,
so switching all those knobs together is not an established fix. Test prompt
conditioning or a small paired component independently, with new-prompt audio
as the deciding evidence. A learned training-row ID alone does not specify
how to generalize to unseen prompt conditions.

### 5. Medium: the critic intentionally discards explicit lyric-position order

The set transformer has no positional embeddings. With `mean_last`, the final
valid token is distinguished, but the other contextual hidden vectors can be
permuted without changing score or FM. Reversing those valid vectors on the
saved critic changes the score by at most **3.81e-6**.

This does **not** mean that rearranging lyric text leaves the model unchanged:
the upstream LM hidden vectors already contain contextual order. It means the
critic cannot independently judge the ordered trajectory of those vectors.

If tested, use lyric-relative positions, identical for the student and teacher,
or aligned local windows. Absolute offsets in the full caption would expose
different caption lengths as a spurious discriminator cue.

### 6. High for stopping decisions: current logs are not mode-collapse measurements

At [trainer:2166](../../conceptmod/textsliders/train_lm_slider_music3.py#L2166),
`collapse` is set to literal zero for this UNI recipe. Even in the bipolar
recipe, the field measures cosine between two slider directions, not diversity
across generated songs. It cannot answer whether output modes have collapsed.

At [trainer:2550](../../conceptmod/textsliders/train_lm_slider_music3.py#L2550),
`--grad_account` measures only the first row's weighted pole and adversarial
gradients. The pole weight is zero here. FM and end regularization, the two
other active terms, are absent from the attribution. `grad_norm` is the total
gradient **before elementwise clipping**, not the actual parameter displacement.
The separate step-limit telemetry does measure the latter when enabled.

The audio heuristic's eligibility condition is essentially non-silence:
[autonomous_audio.py:142](autonomous_audio.py#L142). Its mean score can improve
despite a minority of bad clips. Positive feminine-minus-masculine **gain** can
also leave the absolute description margin negative. Neither the heuristic
nor its confidence interval supplies an output-diversity or concept-success
certificate. The new prospective convergence study correctly marks the quality
gate as unvalidated rather than treating the score as one.

For the original failed 900, semantic-token entropy rose to **7.99 nats** from
about **2.47**, alongside large hidden drift and a pre-clip gradient spike.
That is evidence of a diffuse, changed policy on the tested histories, not
evidence of low-entropy mode collapse. Across-song collapse remains unmeasured.

Instrumentation changes for the next version:

- Represent unavailable collapse diagnostics as unavailable, not numeric zero.
- Measure weighted full-batch gradients for adversarial, FM and end losses,
  their pairwise cosine, clipping effects, actual step, and effective LoRA
  delta-weight change. Keep diagnostic backwards from changing the update.
- Report per-prompt failure rates and worst tails alongside mean preference.
- Measure within-prompt diversity over multiple seeds in frozen audio
  representations, relative to both neutral and positive-caption controls.
  Include near-duplicate counts, robust central distance, and tail mass.
- Keep target-voice consistency separate from diversity of arrangement,
  melody and rhythm: intentionally narrowing voice character is not automatically
  unwanted mode collapse. Embedding clusters are proxies, not known music modes.

Four seeds per prompt support an initial screen, not a strong diversity
certificate. Reserve more seeds for suspicious cases. Raw global variance can
be inflated by noisy outliers; the upstream Gaussian study itself corrected
such a misleading variance result.

## What changed from the successful Gaussian recipe

| Component | Pinned 100-Gaussian reproduction | Current music smoke |
| --- | --- | --- |
| Samples judged | Generated 2D points, known 100-mode target | Four lyric-span/audio-start hidden-delta sequences |
| Generator freedom | Full small MLP and 20,000 movable latent particles | Frozen LM with rank-8 LoRA on 144 attention projections |
| Input diversity | Random latent/real batches of 256 | The same four deterministic prompt rows per update |
| Critic | Fourier-input MLP | 4096→128 projection, two set-transformer blocks, normalized scalar readout |
| G objectives | Adversarial loss plus VICReg on the latent prior | Adversarial + raw batch-FM + end-margin preservation |
| Optimizers | Adam, beta1=0, G 0.0006 / D 0.0009 | AdamW G 0.0005 with decay 1e-6; Adam D 0.00075; beta1=0 |
| Late schedule | Full LR through 60%, cosine to 5% | Constant LR |
| Evaluation weights | EMA 0.995 of G and prior | Live LoRA weights |
| Update limiting | Reference optimizer rule | Elementwise G gradient clipping; actual LoRA step cap in later branches |
| Mode metrics | Known centers, occupancy, robust core widths and tails | No corresponding generated-song diversity metric |

The reference here is the locally reproduced 7000-update experiment, not every
later upstream sweep. See [our reproduction](gaussian_findings.md) and the
[pinned upstream example](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/examples/100gaussians.py).
Optional music prompt-row miners are reweighting tools, not the Gaussian
experiment's movable latent prior. Enabling them does not recreate that experiment.

The upstream [findings](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/FINDINGS.md)
attribute their cap stability to the training trajectory: the cap was active
earlier and went slack as LR annealing flattened the critic. They do not show
that a cap supplies continual damping in an arbitrary game. Our constant-LR,
additional-FM game therefore lacks two relevant ingredients of the successful
evaluation procedure: annealing and averaging. This is an ablation priority,
not proof that either omission caused the music failure.

## LoRA capacity and two implementation traps for proposed fixes

Each adapted attention matrix has `delta_W = (alpha/rank) B A`, of rank at most
8. Feedforward weights, the LM output head and downstream audio modules are
frozen. This constrains the intervention, but composing 144 adapted projections
through a nonlinear network is much more expressive than a single rank-8
output transformation. The review found **no proof that rank 8 is the active
capacity bottleneck**. Test rank only after controlling the moving objective
and training support, using audio outcomes rather than hidden MSE alone.

The actual parameter-step cap is correctly implemented, but depends on LoRA
factorization. Rescaling `B` by 10 and `A` by 1/10 preserves the effective model;
an equivalent function change can have a parameter-step norm 10 times different.
Thus cap 2 is an empirical safeguard in this parameterization, not an invariant
bound on audio change or policy KL.

Porting the toy's EMA also needs care: averaging `A` and `B` separately is not
the same as averaging `BA`. The audit uses two identical effective adapters
with opposite factor signs; averaging factors erases the adapter entirely,
whereas averaging effective weights preserves it. This is an algebraic
counterexample, not evidence of sign flips between adjacent real checkpoints.
An effective-weight EMA may need higher rank or measured recompression error.
Factor EMA can still be tested empirically if it is identified as such.

The current trainer also deliberately rejects nonconstant-LR full-state
continuation at [trainer:1983](../../conceptmod/textsliders/train_lm_slider_music3.py#L1983).
A decay experiment needs an explicit, resumable schedule branch with the
schedule origin and budget recorded; changing `STEPS` must not silently move
the annealing window of a supposedly resumed experiment.

The scale-zero anchor has zero LoRA gradient because multiplying the adapter
by zero already restores the frozen base function. The audit verifies this
on the real LoRA wrapper. That anchor cannot restrain the scale-one model.
It is redundant computation, not evidence of a broken plus-one gradient:
the preceding plus-one graph still backpropagates after the multiplier is
changed, absent gradient-checkpoint recomputation. No such recomputation is
enabled in this trainer.

## Controlled next experiments

1. Preserve the current 600 and bounded-continuation baselines. Complete the
   reserved-prompt audio comparison already running; interpret it as a test of
   the present metric and checkpoint behavior, not of any new architecture.
2. Add the missing loss-gradient and effective-update telemetry in a versioned
   research branch. Replay a fixed short continuation to ensure observation
   does not change updates. Reuse existing listened failures for gate checks.
3. From the same full 600 state and data, compare constant LR against a declared
   decay schedule, holding FM and the step cap fixed. Separately compare raw
   FM with controlled FM. Do not combine these interventions in the first test.
4. Test averaging as a separate evaluation choice with its LoRA algebra
   documented. Judge live and averaged candidates on the same heldout seeds.
5. If condition failures persist without optimizer instability, broaden prompt
   support, then test conditioning/paired structure and additional critic
   projections separately. Increase LoRA rank only with evidence of a remaining
   representation limitation.

Use repeated training seeds for promising changes before transferring a recipe
to the whole catalog. A failed quality gate means change the recipe; a plateau
of a validated audio outcome can justify retaining the best checkpoint. Neither
requires claiming the GAN has reached a certified game equilibrium.

## Verification and artifacts

- [Executable CPU audit](architecture_audit.py) and
  [recorded results](architecture_audit_20260905.json).
- Pinned upstream revision: `b5ee35f9b24cf35a6b1346ba2f0856c4877aa336`;
  the compared tracked kernels are unchanged.
- Rp D/G loss values and logit gradients match exactly in the parity probe.
- Calibrated b_cap value differs by **9.54e-7**, and parameter gradients by at
  most **4.77e-7**, in the representative scale-13 check. Zero fake inputs are
  included. This is float32 agreement, not a proof over every possible input.
- [**93 existing tests passed**](architecture_tests_20260905.txt): critic masks and double backward, batch/single
  agreement, conditional critic, exact sequential FM gradient, real tiny-model
  trainer updates and continuation, miners, and parameter-step limiting. One
  existing test emits a tensor-to-float warning; there were no failures.
- All source fingerprints match the saved 600 training state. The source state,
  span cache and seven core files remain hash-identical after the audit.
- The saved-span measurements use earlier prompt-only bf16 captures; they are
  not a numerical replay of teacher-forced training geometry.

Reproduce on CPU from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  analysis/gan_bcap/architecture_audit.py \
  --output /tmp/music-gan-architecture-audit.json
```

GPU 1 remains assigned to the prospective convergence audio study. The slider
catalog remains paused as requested; this review does not resume it.
