# YuE2 routed-particle bridge (explicit experiment)

`--recipe particle_bridge` transfers the user-requested `anneal-routed`
configuration from `/ml2/model-glue/configs/particle-toy-20260917.json` through
the existing YuE2 trainer and campaign. It is propose-only; existing recipes,
Music bipolar `ARM_B`, live `--lm_target v9`, and locked `AdvConfig()` defaults
are unchanged. It has a separate adapter format and cannot resume an old game.

## Reference and exact objective

Reference commit: `df70ccb2ca8f532bdcc07a343fd12bec77362523`.
Configuration SHA256:
`1ef39a623505691b8710cd37cb768452cd79f6666d109614af297e9d270d88bb`.
Pinned ParticleGAN commit: `441fdf42dd2c0905af312a303add422f700c0ac2`.
This uses `anneal-routed`, **not** the separate `anneal-routed-lr` arm.

Let `T(h) = (h - mean_train_positive) / sample_std_train_positive`, with
per-coordinate sample standard deviation floored at `1e-4`. Statistics are
fixed from training positive targets only. The native error is
`e = T(student_hidden) - T(positive_hidden)`.

```text
sigma(t) = 0.03 ** min(t / 8000, 1)
n ~ Normal(0, sigma(t)^2 I)
real = n; fake = n + e                    # same noise within each pair
D = mean softplus(D(fake) - D(real)) + b_cap
G = mean softplus(D(real) - D(fake)) + particle_VIC
```

D and G draw independent batches of 64 source/noise pairs. D updates first;
G recomputes with the updated, frozen D. `b_cap` uses exact autograd on both
inputs, L2 threshold 1, coefficient 1, every fourth update with ×4 weighting.
No output MSE, feature matching, ending, hold, anchor, or reconstruction loss.

| Setting | Value |
|---|---|
| G / D / particle LR | 0.0006 / 0.0009 / 0.006, constant |
| Optimizers | Adam, betas (0, 0.999), no weight decay |
| Critic | 3 hidden layers × 48, LeakyReLU 0.2 |
| Cloud | 128 learned particles × 4, standard-normal initialization |
| Route | softmax(q(x) Pᵀ / sqrt(4)) P on every example |
| Router / routed MLP | 3 hidden layers × 16 / 3 hidden layers × 48 |
| Particle VIC | std hinge + off-diagonal covariance, coefficient 1 |
| VIC sample | 64 particles without replacement per G update |
| EMA | 0.995, including routers and cloud; live state also saved |
| Clipping / audio sampling | none |

VIC uses sample variance/covariance, target standard deviation 1, epsilon
`1e-4`, covariance penalty divided by dimension. It does not compare output
predictions with targets. The paired-error critic fixes a limitation of a
marginal hidden-state critic: exchanging two rows' targets preserves the
marginal distribution but produces nonzero paired errors here. This is a
mechanism difference, not proof of native stability.

## Native architecture and deliberate transfer differences

YuE2 remains frozen. Each AR q/k/v/o low-rank branch becomes:

```text
features = down(x)                         # rank 8
q = router(features)
z = softmax(q Pᵀ / sqrt(4)) P              # one shared cloud per slider
delta = up(MLP(concat(features, z)))
output = original_projection(x) + scale * delta
```

Rank/alpha are 8/8. Each projection has its own router and MLP; all projections
share the same cloud, registered and optimized exactly once. `up` starts at
zero, preserving the base at initialization. GAN cloud gradients start once
that output branch moves; `particle_gan_grad_norm` measures them **before**
adding VIC. Scale zero bypasses all branches exactly. Routing is identical
during training and ordinary generation; there is no inference optimization.

This factorized insertion into a frozen transformer is an adaptation of the
reference's single 2D MLP, not an identical generator parameterization. Native
training has four prompt rows instead of 4,096 source examples. Replacement
sampling retains all 64 independent noise contributions, but deterministic
duplicate prompt rows need only one model forward per D/G phase. No held-out
row enters normalization. Native bfloat16 base activations, high-dimensional
targets, and sparse training rows are further transfer differences.

The separate `conceptmod-yue2-routed-particle-ar-v1` format stores the cloud,
routers, low-rank projections, and architecture metadata. The normal YuE2
loader dispatches this format for rendering. `_last` / `_stepN` exports are
EMA; `_live_last` / `_live_stepN` preserve raw weights. `state.pt` resumes live
G/D, both optimizers, EMA, all sampler streams, and CPU/CUDA RNG. Sources,
prompts, model, and recipe mismatches reject resume.

## CPU verification

The implementation was compared directly with the hash-checked upstream
modules loaded by model-glue. Float64 routed forward, VIC value/gradient,
Rp D/G value/gradient, and active/inactive lazy-cap value/parameter gradients
agree bit for bit. Independent full-batch update tests also cover row reuse,
and native tiny-model tests cover checkpointed backward, frozen base weights,
cloud GAN gradients, deployment round-trip, and exact split-run resume.

The actual native game runs through `analysis/slider2d/yue2_particle_exam.py`
with a frozen PairField backend and a routed low-rank student. It starts at
the unchanged base, which fails both cells. Conditional row outputs use the
original continuation scorer; **no thresholds changed**. Scale 0 is exact
by construction, like the native adapter. Results for seeds 0/1/7:

| Budget | EMA divergent cover | EMA close cover | Both cells, all seeds |
|---:|---:|---:|---|
| 600 | 0.622–0.681 | 0.699–0.738 | FAIL |
| 3400 | 0.966–0.968 | 0.962–0.978 | HIT, live and EMA |
| 8000 | 0.975–0.978 | 0.984–0.989 | HIT, live and EMA |

At both passing budgets, off-caption is 0 and neutral hold is 1 for all
six cell/seed combinations, including live weights. At 8,000, live cover is
0.966–0.977. The same weights fail the bipolar exam in all six cases; -1 is
an unscored UNI canary. At 600 one close EMA result happens to pass BI while
failing UNI. The stock `--polarity both` leaderboard was also rerun; its
published UNI/BI Markdown files remained byte-identical.

Full results and pinned source hashes: [toy audit](yue2-particle-bridge-toy.json).
The toy records finite transient G peaks of 22–30; passing endpoints do not
mean every update was smooth. Target coordinates with zero sample variance
use the reference's `1e-4` floor, which makes some directions stiff. Neither
EMA nor toy success is proof of native audio stability. There is no matched
fixed-cloud ablation here, so no claim that movable particles outperform one.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m analysis.slider2d.yue2_particle_exam --seeds 0 1 7 \
  --steps 600 3400 8000 --out /tmp/yue2-particle-toy.json
PYTHONPATH=. python analysis/slider2d/run_formulation_leaderboard.py \
  --polarity both --out /tmp/yue2-particle-board
```

## Native metal validation

Native validation uses seed 7, the existing four sound-only metal training
rows, GPU 1 shared with the studio, and a 600-update smoke budget. It retains
the full 8,000-update noise schedule; at 600, sigma is still 0.768748. A short
smoke is therefore not a completed reference convergence experiment.

```bash
CUDA_VISIBLE_DEVICES=1 python conceptmod/textsliders/train_lora_yue2_arm_b.py \
  --recipe particle_bridge --device cuda:0 --seed 7 --steps 600 --until 20 \
  --name metal-yue2-particle-bridge-s7-20260917 \
  --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-particle-bridge-s7-20260917 \
  --prompts_file conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml
# Resume through the existing campaign after native preflight:
python scripts/train_yue2_arm_b_campaign.py --recipe particle_bridge \
  --gpu 1 --seed 7 --steps 600 \
  --name metal-yue2-particle-bridge-s7-20260917 \
  --save_dir /ml2/music/sliders-conceptmod/models/metal-yue2-particle-bridge-s7-20260917 \
  --output_dir /ml2/music/sliders-conceptmod/eval/listen/yue2-metal-particle-bridge-20260917 \
  --include_canary --hidden_diagnostics
```

The listening grid uses two held-out prompts, seeds 1709/2903, scales
0/0.5/1 plus positive-caption reference and -1 canary. Hidden caption
projection, orthogonal residual, and exact-zero checks are unscored Music
diagnostics, not substitutes for calibrated audio cover/leak/lyric gates.
Live charts distinguish GAN, particle VIC, total loss, particle GAN gradients,
and noise.

### Native preflight result (source `2067705`)

The 20-update native preflight completed. GAN gradients reach the cloud from
update 2 onward. All weights are finite; live and EMA exports exactly match
their respective saved states, and source hashes match the frozen checkout.
Training normalization has no floored coordinates: standard deviations span
0.00454–0.312 across 2,048 dimensions (median 0.0525).

Final G GAN loss is 3.7654, particle VIC 0.05819, total 3.8236, and training
direction cosine 0.9508. Peak GAN loss is 4.2146 at update 11, when cosine
dips to 0.8323. The cap remains zero in this short window. **Native stability
is not established**; this is an observed oscillation, not a smooth-training
claim. Twenty updates are also too early for EMA convergence.

The exported EMA adapter produced 10 finite native audio clips: both held-out
prompts at 0/0.5/1, positive-caption reference, and -1 canary, seed 1709. These
32-token clips (~1.28 seconds) verify deployment, not musical quality. Scale
zero is bitwise base on both held-out prompt states. EMA caption projection
at +1 is 0.0525–0.0553, consistent with its early averaging lag, and does not
constitute a cover pass.

Training updates consumed 0.0283 GPU-hours of wall time on one shared GPU,
excluding model loading and render time. The 600-update campaign resumes the
same checkpoint and recipe, then automatically renders the 20-clip full grid.
Its results are pending. The studio remains running with its queue preserved.

Evidence: [native preflight audit](yue2-particle-bridge-native.json),
[live campaign](http://100.90.104.57:8888/yue2-metal-particle-bridge-20260917/),
[exact argv/environment/rates](http://100.90.104.57:8888/yue2-metal-particle-bridge-20260917/commands.json).
The existing listening-page bookmark links to this campaign while retaining
the earlier rate comparisons.

Validation: 124 selected regression checks passed (YuE2 loading, both existing
games, native rate flags, canary evaluation, the shared GAN, canonical UNI/BI
scorers, and `AdvConfig` lock). Three additional malformed particle-checkpoint
cases passed in the final 12-test particle suite; each rejects before attaching
an adapter. Served chart data includes every new metric, and the generated
dashboard JavaScript passes `node --check`.
