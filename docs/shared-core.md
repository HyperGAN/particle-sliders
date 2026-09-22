# Shared particle-sliders core

[HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) owns
the installable [`particle-sliders-core`](../packages/particle-sliders-core)
package and the research trainers around it. Every product repository pins
that package and imports [`winning_formulation()`](winning-formulation.md).

```mermaid
flowchart TB
  core[particle-sliders: particle-sliders-core and winning formulation]
  anima[anima-particle-sliders]
  krea[krea2-particle-sliders]
  supra[supra-concept-sliders]
  toys[conceptmod: formulation toys and the DSL board]
  core -->|pinned git subdirectory| anima
  core -->|pinned git subdirectory| krea
  core -->|pinned git subdirectory| supra
  toys -.->|experiments, not a second game| core
```

| Area | This core | Product repository |
|---|---|---|
| Winning formulation | `particle_sliders.winning_formulation()` | Call `require()`; bump the pin when the stamp id changes |
| Particle adapter | Soft routing and bottleneck MLP | Projection names, hooks, strength controls |
| Learning | Paired losses, global-mix critic, VIC, noise, `GradRegularizer` | Frozen targets, optimizer step sizes on `model_surface_keys`, update loop |
| Older endpoint stamp | `locked_shared_recipe()` | Import it from here if a run still needs it |
| Ordinary LoRA fitting | Dual ridge solve | Activation capture, calibration budget, export names |
| Evidence | Algorithm tests and the stamp pin against `V2_SPEC` | Weight hashes, prompts, samples, license, runtime lock |

Formulation toys stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod).
This package is the train and infer algorithms plus the winning stamp. It does
not vendor the ParticleGAN repository. GAN primitives that products need are
imported from `particle_sliders` (`GradRegularizer` is the published
`particlegan` 0.2.0 excerpt already pinned in-tree because this research
checkout's Music stack cannot take the `particlegan` torch requirement).

The name `concept-slider-core` / `concept_slider_core` is the legacy Anima
extraction path. New installs use `particle-sliders-core` from this GitHub
repository, not `mikkel/sliders-conceptmod`.

Krea2's current product README forbids a particle-sliders checkout and its
docs contract rejects the string `PARTICLE_SLIDERS_ROOT`. That self-contained
rule is the wrong long-term pattern. The dependency is a pinned install of
this package. A path variable is a worse install than the git subdirectory,
and omitting the dependency means the product re-implements the game.

Research trainers in this repository (YuE2, Music 3, the in-tree Supra and
image backends) are still here. This change does not delete them. Product
releases do not treat those files as a private fork of the stamp.

## Routing and the ordinary LoRA fit

With particle cloud P and a projected input u, routing computes
`z = softmax(R(u) @ P.T / sqrt(particle_dim)) @ P`. The branch predicts
`up(F(concat(u, z)))`. Particle training compares paired noisy real and fake
errors through relativistic logistic losses and regularizes the particle cloud
and the critic. The product adapter defines what those errors mean on that
backbone.

To fit an ordinary LoRA, keep the teacher's up matrix U and solve for the down
matrix A from calibration inputs X and routed bottleneck outputs H:

```text
lambda = 0.01 * max(mean(diag(X @ X.T)), 1e-8)
A = solve(X @ X.T + lambda * I, H).T @ X
delta(x) = (x @ A.T) @ U.T
```

This minimizes a regularized calibration objective. It does not establish the
best visual or audio result.

## Adding a product

1. Pin a full commit:
   `particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<commit>#subdirectory=packages/particle-sliders-core`
2. Import `winning_formulation` and build the bridge, critic, regularizer, and
   losses from that object.
3. Pass model-surface overrides (learning rates, batch) through `require()`.
   Leave Hub ids, Comfy class names, prompts, and sampling in the product.
4. When this repository changes `stamp.id`, bump the pin. Do not copy
   `formulation.py` across.

The package version identifies the API family. The git revision identifies the
stamp. Existing product pins of `concept-slider-core` at `beaffeb` remain
installable from that historical commit.
