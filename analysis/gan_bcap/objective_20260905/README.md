# Gaussian → slider objective audit

The transfer is not the same game with a larger generator. The optimizer,
critic representation, conditioning, metric units and observations all matter.
Copying the Gaussian cap does not constrain the losses or inference branches
added by the slider trainer. This audit implements and measures counterexamples,
screens alternative objectives, and trains a concrete replacement candidate.
It does **not** establish one validated objective that solves both domains.
The completed screen contains 23 Gaussian runs, 18 saved-span canaries and
three real LoRA trials. Fixed MMD is the most reproducible alternative on
the Gaussian benchmark, but its music endpoint has inconsistent transcript
and production-quality diagnostics. **No replacement objective is accepted
from this audit.**

[Complete experimental results](results.md) · [machine-readable results](summary.json)
· [counterexamples](counterexamples.json) · [comparison plot](comparison.png)
· [isolated schedule comparison](optimizer-comparison.png)
· [matched audio](../../../eval/listen/gan-objective-20260905/index.html)
· [audio diagnostics](audio-results.md)
· [interactive loss explainer](explainer/index.html)
· [repeatable MMD leaderboard game](../mmd_game_20260905/README.md)

## What changed in the transfer

The reference is the local ParticleGAN checkout at
`b5ee35f9b24cf35a6b1346ba2f0856c4877aa336`. The production comparison is the
bounded baseline / FM-cap `gan_v2` path, including its imported original-600
optimizer state. Existing trainers, launchers, weights and experiment histories
are preserved. New executable research lives in this folder. All new GPU work
uses physical GPU 1.

| Component | Gaussian reference | Current bounded slider / FM-cap branches | Consequence |
| --- | --- | --- | --- |
| G optimizer | Adam, beta1=0, beta2=.999, LR .0006 | AdamW, same betas, LR .0005, decay 1e-6 | Small decay is a difference; it is not a demonstrated initiating cause |
| D optimizer | Adam, LR .0009 | Adam, LR .00075 | Both retain the 1.5 D/G ratio; equal nominal ratios do not equalize functional step sizes |
| Prior | 20,000 movable 4-D inputs to G, LR .006, VICReg | Fixed caption/lyric inputs; optional row miners select weights | Row mining is not latent transport and cannot reproduce its effect |
| Schedule | Full LR through 60%, cosine to 5% | Constant in the two bounded branches | The toy endpoint benefited from a particular trajectory and annealing |
| Averaging | EMA .995 on G and prior | Off for these two branches | Compare live and averaged samples separately |
| D | Fourier-2 MLP on 2-D points | Span transformer, 4096→128 projection, pooled normalized scalar score | A frozen critic has a large exact input nullspace; scalar sensitivity is not full-state observability |
| Condition | One unconditional distribution, fresh real/fake samples | Four deterministic prompt rows in the retained baseline; no critic context in those branches | A marginal match can assign the wrong teacher to a prompt |
| G objective | Relativistic logistic + latent VICReg | Relativistic logistic + raw batch FM + end margin | FM is a separate learned metric not controlled by the scalar-score cap |
| Observation | Generated output coordinates | Lyric-prefix / audio-start hidden deltas, narrow continuation safeguards | Residual depth paths, flow conditioning and the unconditional CFG branch can change outside the objective |
| Update safeguards | No G clipping in reference loop | Value clipping, parameter-step bound; optional FM gradient limiting | These are different update rules, not evidence of one underlying scalar objective |

`spectral: true` in the reference YAML requests an update-spectrum **diagnostic**.
It does not apply spectral normalization to the Gaussian discriminator.
The upstream spectrum probe linearizes a GD update and omits Adam's optimizer
state; it must not be presented as the actual Adam transition spectrum.

The reference's own [findings](https://github.com/255BITS/ParticleGAN/blob/b5ee35f9b24cf35a6b1346ba2f0856c4877aa336/FINDINGS.md)
distinguish trajectory regularization from persistent endpoint damping. The cap
becomes mostly inactive near its successful endpoint. Optimism was not a general
improvement in that study. These are toy-specific empirical findings, not a
guarantee for the music model.

## Executable failures of the proposed stability arguments

Run `python -m analysis.gan_bcap.objective_20260905.audit` in the existing conda
environment. The output is [counterexamples.json](counterexamples.json).

**The cap need not restore equilibrium.** Set the real distribution to a point
at zero, the generator point to theta, and the critic to `D(x)=phi*x`.
The losses are `D: softplus(phi*theta)+relu(abs(phi)-1)^2` and
`G: softplus(-phi*theta)`. Near the origin the cap is identically zero. The
negative-gradient field linearizes to

```text
theta_dot =  phi/2
phi_dot   = -theta/2.
```

At step size .2, simultaneous GD has spectral radius 1.00499; D-then-G GD
has radius 1.00000. Neither contracts. At an exact point match with phi=.5,
the cap is zero but the generator gradient is -.25. A zero-centered term
adds standing curvature; extragradient changes the discretization. Their
local effects are measured separately in the audit. This is a counterexample
to a general cap guarantee, not a reproduction of the actual neural Adam game.
The relevant local-convergence assumptions and manifold counterexamples are
discussed in [Which Training Methods for GANs do actually Converge?](https://proceedings.mlr.press/v80/mescheder18a.html).

**Clipping a gradient does not clip Adam's step.** Fresh scalar Adam updates
at LR .0005 are approximately .0005 for gradients 1, .01 and .0001. The
second moment largely cancels this scaling. An imported second moment need
not cancel it: a new loss can initially move much less simply because its
scale is smaller than the loss that populated the old moments. This is why
the two live trials record optimizer initialization explicitly.

**A clipped component is generally not a new scalar loss.** For
`f(x,y)=x²+2y²`, clipping its gradient to unit norm produces unequal cross
derivatives; the measured difference at (1,1) is -.17889. Such a rule may be
a useful update intervention, but it cannot generally be described as
minimizing a fixed replacement scalar objective. Likewise, a bound on each
parameter step is not a bound on total drift, and a bound on a LoRA factor
step is not automatically a bound on the product BA or the model output.

**The score and FM can miss a large hidden error.** The production 4096→128
projection has a nullspace of dimension at least 3968 per token. The audit
constructs a displacement in that nullspace with hidden RMS 10. The critic's
maximum score change is 4.3e-6 and raw feature MSE is 1.15e-11. This is an
observability limit of a frozen critic, not proof that this precise direction
is reachable by the shared LoRA. Normalizing FM does not restore discarded
coordinates. A critic that changes its projection over time can learn to see
new errors, but one small instantaneous loss does not certify their absence.

**Cap units are not dimension independent.** In the saved original-600 probe,
the spans have 102,400–114,688 valid coordinates. Teacher deltas divided by the
fixed component RMS have L2 norms 316–337. The cap uses the dual of summed L2
displacement. If the intended displacement metric is instead RMS,

```text
distance(du) = ||du||_2 / sqrt(N)
dual_norm(grad_u D) = sqrt(N) * ||grad_u D||_2.
```

Thus the saved critic's current-coordinate norms .815–1.457 correspond to
RMS-cost dual norms 271–493. The implementation's chain rule is correct;
these are **different units**, not an arithmetic bug. The Gaussian coefficient
and threshold do not select equivalent geometry automatically. Simply
multiplying the old penalty by a large dimension factor is not a validated fix.

## An inference path missing from the current losses

The installed `encoders.py` creates two token sequences: the conditional
prompt and its unconditional CFG counterpart, obtained by replacing interior
tokens with the audio-CFG token. Both pass through the same LoRA-enabled LM.
Guided logits combine them as `uncond + guidance*(cond - uncond)`. The same
two-branch guidance applies to the residual depth-code heads. The emitted flow
conditioning includes the LM and depth-decoder hidden states.

The existing GAN objectives and fixed-history semantic KL only train the
conditional branch. Even perfect agreement there allows the guided policy to
change through the other branch. A regression fixture holds conditional logits
exactly equal while an unconditional change makes guided KL exceed 4 nats.
This establishes an unobserved path, **not** the initiating cause of historical
gibberish. Teacher histories are also finite, so matching them cannot certify
the behavior on a different self-generated history.

## The most reproducible single-objective candidate

The single fixed objective tested across the most training seeds is
**conditional multiscale MMD**, with no trained discriminator:

```text
min_theta E_c MMD²_k(Q_theta(.|c), P_teacher(.|c))

k(x,y) = mean_b exp(-||x-y||² / (2 b²))
b = [.03, .1, .3, 1, 3]

MMD² = E k(x,x') + E k(y,y') - 2 E k(x,y).
```

For Gaussian samples, x and y are the raw two-dimensional coordinates, and
the unbiased estimate omits self pairs. The within-fake term is essential:
it supplies distributional repulsion. These runs retain the source's movable
latent particles, Adam schedule and G/prior EMA, but have **no VICReg, FM or
cap**. Across training seeds 7, 23 and 101, batch 1024 and 28,000 updates,
EMA HQ is .9336–.9356, every run covers 100 modes, and core width is
.989–.994 times truth. This uses four times the batch and four times the
updates of the .9876-HQ cap control, so it is not an equal-compute win.
The schedule horizon remains 7000: the extension uses the 5% LR floor.

An additional [mode-occupancy audit](mode-occupancy.json) distinguishes
covering every mode from assigning the correct probability to each mode.
The cap control's EMA total-variation error from uniform mode weights is
.1189, versus .0276–.0337 for the three extended MMD runs. All saved samples,
including tails, enter this calculation. Thus the cap wins on HQ fraction
while MMD better matches core widths and mode probabilities in these runs;
neither metric alone describes the entire Gaussian distribution. This is
another reason to avoid selecting an objective from just one scalar proxy.

For the LM, c identifies prompt, lyrics, CFG branch, shared history and
position. Coordinates are hidden deltas divided by the frozen teacher RMS
and sqrt(hidden width). A deterministic student and teacher at the same
condition are singleton distributions, so exactly the same objective becomes

```text
L = E_c mean_b 2 * (1 - exp(-RMS((h_student-h_teacher)/scale)² / (2 b²))).
```

This is a proper conditional discrepancy with zero gradient at an exact
match. All hidden coordinates contribute; there is no compressed critic
projection and no moving objective. Fixed Gaussian kernels are characteristic;
see [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html).
That property identifies the population distributions; it does not ensure
fast optimization, sufficient finite samples, or good coverage of histories.
In particular, the RBF objective saturates at large errors and its gradient
can become arbitrarily small. It supplies no coercive bound on hidden drift.

The research LoRA trial uses both CFG branches and fresh Adam moments. Every
accepted step decreases this same full-batch scalar loss, using the acceptance
rule below. Long stretches of rejected proposals near the end show why
monotone loss is weaker than convergence. Training and audio results remain
separate acceptance questions; this candidate is not a new default.

## The learned-energy alternative and its failure boundary

The other implemented candidate is **conditional energy distance in a
spectrally constrained, injective learned metric**:

```text
min_theta sup_phi in K  E_c [
    2 E_(x~Q_theta,y~P_teacher) rho(T_phi(c,x), T_phi(c,y))
    - E_(x,x'~Q_theta)         rho(T_phi(c,x), T_phi(c,x'))
    - E_(y,y'~P_teacher)       rho(T_phi(c,y), T_phi(c,y'))
]

rho(a,b) = sqrt(||a-b||² + epsilon²) - epsilon
T_phi(c,u) = [u/sqrt(H), D_phi(c,u)]
u = hidden_delta / fixed_teacher_RMS
```

The expectation is conditional: `c` identifies the prompt, lyrics, CFG branch,
shared history and position. Different prompts must not be pooled as though
they were interchangeable draws of one distribution. For the deterministic
LM hidden teacher/student at a fixed condition, both within-distribution terms
are zero, leaving one smoothed paired distance. For the Gaussian generator,
the within-fake term supplies output-space repulsion. Dropping that term for
stochastic outputs would change the optimum and permit mean collapse.

The raw-coordinate part of T preserves every hidden direction. The learned
scalar feature is an ordinary conditional MLP whose linear operators are
projected to spectral norm at most 1. Its input uses the same explicit RMS
geometry as the raw part. D maximizes exactly the distance that G minimizes;
there is no additional logit-ranking, FM or end-margin target. Both teacher
branches use the positive-caption teacher on the same cached history. The
unconditional target uses the actual positive-prompt CFG sequence geometry,
including its different prefix length.

At a conditional match the scalar loss and **both** player gradients are zero
for every feasible metric. The first-order mixed G/D derivative also vanishes,
eliminating the Dirac game's local rotational term in this fixture. The
raw-coordinate term supplies curvature in all hidden directions and the smooth
distance bounds input gradients. These are statements about this objective,
not bounds on the LM Jacobian or a global neural-network convergence proof.

These candidates relate to characteristic-kernel / energy-score training, including
[MMD GAN](https://arxiv.org/abs/1705.08584) and
[the energy-distance approach to speech modeling](https://arxiv.org/abs/2505.13181).
The research implementation is not an exact reproduction of either paper.
For a fixed metric, an independent-sample U statistic estimates the population
distance without the V statistic's self-pair bias. A learned metric introduces
additional estimation bias: [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)
explicitly distinguishes those issues. Finite-batch estimates can be negative;
clamping them to zero would change the training gradient.

The real LoRA trials add an **optimizer acceptance rule**, not another loss:
evaluate Adam's proposed update against the same full training batch and frozen
current metric; halve it until the actual scalar objective decreases. A complete
rejection restores parameters and optimizer moments. This controls the proposal
that gradient clipping misses. It does not promise that an incompletely optimized
minimax envelope decreases after the discriminator changes, or that heldout audio
quality is monotone. Both are measured separately.

For fixed MMD, there is no D update, so empirical loss descent also holds
across training steps. The tested search only tries six fractions, down to
1/32 of the Adam proposal. Rejection restores moments as well as weights;
it can therefore repeat the same unsuccessful proposal in exact arithmetic.
The observed finite-precision run has long rejection streaks with occasional
small improvements. No convergence claim follows from rollback. A subsequent
optimizer should explicitly adapt its search radius or use a descent fallback
instead of interpreting repeated rejected attempts as successful updates.

## What the experiments do and do not establish

The matched Gaussian cap control reaches 100 modes, HQ .9876 and core width
.868, close to the earlier reference reproduction. R1/R2 also works in that
configuration. Freezing the latent prior reduces the cap control to 43 modes
and HQ .0777; the fixed-prior control also removes VICReg, as in upstream.
This is direct evidence that retaining the cap while removing the source's
latent mechanism does not reproduce its outcome.

Later controls isolate two additional transfer differences at seed 7 and
7000 updates. Removing decay alone changes the cap's live HQ .9828→.2433;
EMA HQ falls .9876→.7790 and conceals much of the live degradation. Reducing
batch 256→32 while retaining decay changes EMA HQ .9876→.2292, with 94 modes
and core width 6.714 times truth. Fixed MMD also relies on the schedule:
at batch 1024, removing decay changes EMA HQ .8224→.3576 and live HQ
.8288→.1133. These are failures of distribution fidelity, not reports of NaNs.

Simple fixed energy distance, fixed multiscale RBF MMD and learned-metric
variants all have failures or material fidelity deficits in the initial
Gaussian screens. Lower hidden error on the saved-span canary cannot overrule
those results. The complete table includes the larger batches, changed energy
powers, long continuation and failed configurations rather than only the best
endpoint. No winner is selected from training-loss values.

Two 120-update energy LoRA trials and their full states are saved under
`models/conditional-energy-research-20260905` and
`models/conditional-energy-cfg-research-20260905-attempt2`. The former imports
old G moments and supervises conditional states. The latter starts fresh G
moments and covers both CFG branches. They test a concrete implementation and
its update rule; their joint differences do not isolate CFG as the cause of a
quality gain. The source-600 weights remain unchanged. Audio compares both
endpoints with that source at fixed seeds 7 and 23.

The conditional-only energy trial reduces mean training relative error
.6633→.4329 and diagnostic heldout error .7988→.7333. The both-CFG energy
trial reduces them .9223→.4275 and .9515→.6433; 107 of 120 accepted proposals
need shortening. Neither establishes a consistent free-running improvement:
ASR phrase match for the source is .625/.870 at seeds 7/23, versus
.625/.385 for conditional energy and .923/.594 for both-CFG energy. These
are singing-transcription diagnostics, not human listening verdicts.

The fixed-MMD trial is retained separately under
`models/conditional-mmd-cfg-research-20260905`, with both branches, the same
source weights and fresh moments. Its exact run outcome and rejection count
are recorded in the results table and full-state audit. Audio uses the same
two seeds and byte-identical off/reference controls, with reuse provenance
in `eval/listen/gan-objective-mmd-20260905/reuse.json`.
It accepts 99 of 120 proposals, shortens 89 and rejects 21. Its full training
objective is monotone; mean train error changes .9223→.7598 and diagnostic
heldout error .9515→.8799. This is a plateau with substantial remaining
teacher error, not convergence. All three exported endpoints exactly match
their saved full states; weights, optimizer tensors and source snapshots pass
[the integrity audit](live-validation.json).
The completed fixed-MMD audio has ASR phrase match .325/.966 at seeds 7/23,
versus .625/.870 for source 600. At seed 23 its frozen production-quality
estimate falls 8.326→6.948 despite the better transcript. Its voice-descriptor
margin is lower than source 600 at both seeds. These metrics do not replace
listening, but they do not support accepting the endpoint as a general
improvement. All 40 stored WAV files (12 distinct clips) are finite, their
checkpoint hashes and reused controls agree, and the first generations are
retained in the listening gallery.

## Why the current observations cannot certify the whole task

Every live trial uses only 250 frames of cached neutral-caption histories.
The emitted song follows newly sampled histories, and the 20-second renders
extend beyond those training frames. A function can match finitely many
observed histories exactly and differ elsewhere. For example, adding
`a * product_i ||h-h_i||² * v` is invisible at all observed histories h_i
and unconstrained outside them. This is a general identifiability
counterexample, not a claim that this exact polynomial is representable by
the rank-8 adapter.

Thus even a perfect conditional objective on these cached states cannot
certify the free-running distribution. The concrete next extension is to
apply the **same** conditional discrepancy to positive-teacher histories
and student-generated histories, with both CFG branches evaluated at each
shared history. This changes observation coverage rather than adding another
penalty. It has not been implemented or validated in this audit. No objective
alone can compensate for missing observations, insufficient model capacity
or arbitrarily large optimizer steps.

There is a further distinction when defining that extension. Freezing a
declared history measure, including previously collected student histories,
gives a fixed conditional objective. Continually sampling histories from the
current student and stopping gradients through that sampling gives a data
aggregation surrogate: it omits the derivative of the student-dependent
history distribution. It must not be advertised as the exact gradient of a
fixed full-trajectory objective. A full-trajectory distribution objective
requires accounting for that dependence, or a separately justified estimator.

The objective is coherent across conditional and unconditional distributions,
but the tested approximation has not yet earned a claim that it preserves the
Gaussian winner's quality and solves music generation across configurations.
The unresolved work is joint confirmation of distribution fidelity, multiple
training seeds, prompt/history coverage and free-running listening quality.
The current research weights do not replace the existing slider recipe.

The focused regression log is [tests.txt](tests.txt). Checks cover exact-match
stationarity, missing conditional correspondence, variance collapse, fixed
calibration, input-gradient behavior, spectral bounds, the CFG counterexample,
and update rejection/rollback. Reproduction commands and complete arguments are
in the per-run `invocation.json` / `manifest.json` files. Model dependencies are
the existing `minimax-music3` environment; no package installation was performed.
The optional audio scorer uses its already installed dependency directory
`/ml2/music/.cache/slider-quality/python` on `PYTHONPATH`. Setup failures and
preflight retries are recorded in [run-incidents.json](run-incidents.json).
The [source inventory](source-inventory.json) maps recorded Gaussian source
hashes to available exact file versions. Two early Gaussian script versions
were hashed but not archived; they are explicitly marked unavailable instead
of being represented as the current source. Every live trial captured its
complete listed source files before model loading.
