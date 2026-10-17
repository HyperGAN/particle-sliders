# BISPF paired-error integration

Music 3 routed-particle training defaults to BISPF. `--formulation legacy`
retains the earlier RpGAN / b_cap / VIC / EMA game. The reusable
`particle_sliders.BISPFGame` implements the same new update for products that
supply their own normalized student callback. Other native backend trainers,
`FormulationGame`, `EndpointGame` and `winning_formulation()` retain their
historical objectives. No published model has been retrained or deployed.

## Exact upstream and variant

- Repository: [255BITS/ParticleGAN](https://github.com/255BITS/ParticleGAN).
- Verified develop HEAD at implementation time:
  [`b5d2372f2739d7579ab66d6dba6e220d2eb776d9`](https://github.com/255BITS/ParticleGAN/commit/b5d2372f2739d7579ab66d6dba6e220d2eb776d9), package 0.8.0.
- Both the pip dependency and checkpoint identity pin this full SHA.
  [bispf_provenance.json](../packages/particle-sliders-core/src/particle_sliders/bispf_provenance.json)
  records inspected implementation file hashes. `locked_provenance.json`
  records the updated legacy-cap dependency while retaining the original
  teacher extraction identity.
- Inspected the supplied [family report at the pin](https://github.com/255BITS/ParticleGAN/blob/b5d2372f2739d7579ab66d6dba6e220d2eb776d9/reports/forge/families/bispf.md),
  `particlegan/bispf.py`, `recipes.py`, `training.py`, `k3p.py`,
  `moment_tether.py`, `gan_loss.py`, `docs/api.md`, the v2 Forge configuration,
  and the relevant `tests/test_bispf.py` contracts.

This integration selects **`get_recipe("bispf")` with Sinkhorn relativistic
loss and formulation-owned Adam**, using its public component factories:
`make_optimizers`, `make_loss`, `make_critic_penalty` and
`make_prior_regularizer`. The paired critic optimizer is `BISPFCriticAdam`;
its penalty is `BISPFCriticPenalty`, the loss is `SinkhornRelativisticLoss`,
and the prior regularizer is `SBNPPriorRegularizer`. The G optimizer is the
recipe's `K3PGeneratorAdam` with latent damping and direct-particle response
disabled. No critic EMA copy is allocated.

This is a supported BISPF variant, **not a reproduction of the family report's
selected DualNorm/non-saturating benchmark configuration**. That configuration
uses LR .012, D/prior multipliers 1.5/2.5, direction blending and different
loss/optimizer choices. The historical report also lists penalty values that
differ from current source defaults. We use source defaults at the pin:
`lambda_floor=.01`, `lambda_cap=1`, `ebi_beta=30`, `gamma_gauge=.05`,
`chord_blend=.02`, Sinkhorn epsilon .15 / 5 iterations, cooling threshold .020,
minimum thermal scale 1e-5 and thermal decay .95. These are not copied into a
local penalty implementation. The standalone `BISPFEngine` is not used: it
has a different objective, including an optional OT barycenter pull, and
separate thermal state.

The conditional paired task owns routing and observations. The unconditional
`GANTrainer` explicitly refuses paired conditional row policies; using its
component factories avoids pretending this task is an unconditional GAN.

## Objective, coordinates and ownership

For a sampled prompt row, let `p` be the student's neutral-caption hidden
state and `t` the frozen positive-caption teacher. Both pass through the same
fixed paired-edit normalizer. The critic sees `[B, H]` coordinates:

```
real = shared Gaussian noise                         # [B, H]
fake = shared Gaussian noise + normalize(p) - normalize(t)
```

BISPF now feeds the **neutral caption to the student**, matching inference.
The previous Music 3 code fed the positive caption to both student and teacher;
that behavior remains only in `--formulation legacy`. Teacher preparation and
the exact base behavior at slider scale zero remain intact.

Both losses are minimized. D receives `loss.d_loss(D(real), D(fake.detach()))`.
After D steps, G receives `loss.g_loss(D(fake), D(real))` plus the SBNP term.
The score tensors go directly from the hooked critic into the loss; reshaping,
copying or detaching them at that boundary would lose attached input
coordinates and cause upstream to fall back to scalar-logit coupling.
The runner collects the complete batch before the Sinkhorn loss. Computing a
separate loss per repeated prompt would change the transport problem.

The native routed adapter retains one shared **128 x 4** particle Parameter.
A prior view aliases that exact Parameter, and the recipe includes it in
exactly one optimizer group. It captures initial mean/covariance before the
first update. SBNP acts on the full routed bank, with `prior_reg=.05` applied
once; legacy VIC is off. The full-bank softmax routing stays active at both
training and inference. We do not replace it with the separate
`StratifiedBlueNoisePrior` sampler or claim stratified row-sampling coverage.
SBNP's capacity guard and dimension-dependent behavior come from upstream.

The CLI keeps the prior task rates: G .0006, D .0009 and particles .006;
all remain constant, with betas (0, .999). The paired-error reference Gaussian
has fixed sigma 1 in normalized coordinates. This is a task reference law,
not the standalone engine's scheduled/additive noise. No annealing, VIC or
serving EMA is applied in BISPF mode. Upstream residual-driven thermal
scaling is retained. Raw critic scores are required (`score_bound=0`).

The default BISPF batch is 8 (legacy: 64); `--batch_size` accepts 2..512.
Full-batch coupling retains all unique-row G graphs until backward, unlike
the old row-streaming loss. Native GPU memory requirements need measurement.
At H >= 170 upstream treats 2D inputs as a discrete/simplex asymptotic case,
and does not use the low-dimensional chord path. This affects native wide
hidden states even though they are continuous. The integration follows that
upstream heuristic rather than overriding its implementation.

## Shared API and checkpoint contract

```python
from particle_sliders import BISPFGame, GlobalMixErrorCritic

# network owns routed trainable parameters, including network.particles.
critic = GlobalMixErrorCritic(targets, neutrals=neutrals, score_bound=0.)
game = BISPFGame(network, critic, targets, seed=0, batch_size=8,
                 total_steps=1600)

def predict(row, phase):
    # Product callback retains autograd for phase="g".
    return critic.normalize(student_neutral_hidden(row, train=(phase == "g")))

metrics = game.step(1, predict)
state = game.state_dict()  # torch.save(state, path)
# Fresh, compatible network/critic/game construction, then:
game.load_state_dict(state)
metrics = game.step(game.completed_steps + 1, predict)
```

The callback must return finite float32 `[1, H]` on the game device. Steps
are contiguous, 1-based and bounded by `total_steps`. D then G receive fresh
row/noise batches from owned CPU generators. A separate model RNG stream is
forked/restored for stochastic forwards and checkpoint recomputation. The
adapter scale context covers G forward **and backward**.

Checkpoints include network/critic weights, the aliased particle prior with
its original moment buffers, both optimizer states (including BISPF controller
counters and residual EMAs), row/noise streams, CPU/CUDA model RNG state,
completed update count, normalized-target digest, resolved recipe and upstream
SHA. The exact thermal scale is saved separately because upstream's record
loader reconstructs it approximately; this preserves the next critic hook.
Restore rejects changed recipe, budget, target coordinates, sigma or upstream
identity. Keep `--steps` fixed when pausing with `--until`.

The Music 3 CLI writes atomic recovery files and the original adapter format.
Both `*_last` and `*_live_last` exports contain live BISPF weights and label
them `weights_kind=live`. Scale 0 reproduces the frozen base. Existing run
signatures intentionally reject source/objective changes: start a fresh BISPF
save directory. Older runs should be continued with their original revision
and environment; optimizer states are not converted across formulations.

ParticleGAN 0.8 removed several 0.6 exports. Historical fixed L2/autograd
b_cap callers use a thin local constructor adapter over the upstream kernel.
Unsupported retired norm/method/anneal settings raise; the removed
`finite_difference_norm` re-export is not supplied. The old
`particlegan_locked_shared()` helper retains cap/loss builders, with a
`legacy-fixed-bcap` label in place of upstream's retired stamp object.

## Validation and limits

See [the execution receipt](bispf-validation.md) for exact commands and results.
The tests cover full-batch coordinate coupling and loss gradients, particle
ownership, constant rates, nonreentrant checkpoint replay, exact stochastic
resume, the real Music 3 CLI with a weights-free 32-wide fixture, native export
loading and scale-zero identity. The dummy projection is not Qwen attention
and says nothing about generated audio quality.

No native Music 3 weights, GPU training, long training, rendering, published
model changes, job restarts, boot, driver or security changes were performed.
The new MOSS project and its workspace were not edited. Remaining work is
native model/GPU memory validation and listening-quality evaluation; the
upstream report's qualification does not transfer to this new conditional task.
