Historical guide: the0.8.3/PG0.9.1 record below remains frozen context.
This modern0.11 target uses officialPG0.9.3; fixed-arm helper behavior remains,
while exact old identities/checkpoints require their original pins.
See MODERN-GRAM.md for the current target.

# Legacy stamp API and the official PG0.9.1 lifecycle line

This source line is core **0.8.3**, depending on exact `particlegan==0.9.1`.
It is a source integration, not a PyPI release or a checkpoint migration. The
reviewed lifecycle ancestors core0.8.1 (`32f6830`) and core0.8.2 (`f6807a9`)
remain immutable; products keep the exact source pins they actually tested.

## Preserved public helpers

`winning_formulation`, `FormulationGame`, `EndpointGame`, `SliderRecipe` and the
other named root exports remain. The frozen historical stamp returned by
`particlegan_locked_shared()['stamp']` again supports attribute access,
`to_dict()`, dataclass fields and immutable values, including `cover_weight=1.5`.
Its 18 fields come from upstream source
[`82b72661c03377f348ccb76de37cd4ca02d45a8d`](https://github.com/255BITS/ParticleGAN/blob/82b72661c03377f348ccb76de37cd4ca02d45a8d/particlegan/locked_shared.py).
`legacy_stamp.json` binds that exact source hash. This metadata object is named
`LegacyLockedShared`; it is not the retired upstream class identity, a new PG
Recipe or permission to load an old checkpoint. No GAN implementation is copied.

The cap and relativistic loss still come from installed upstream ParticleGAN.
The legacy endpoint `locked_shared_recipe()` has cover_weight1.0; the demo
stamp's1.5 does not change that engine. The gmix winning formulation and its
historical overlay are not replaced with ConditionalPG09.

`GradRegularizer is GradientPenalty` remains true within particle-sliders. The
wrapper is an upstream subclass; equality with upstream's class identity is not
promised. The original positional constructor order is restored:
`arm, coeff, kappa, lazy_k, norm, target_anneal, total_steps, method, fd_eps`.
Inactive `total_steps` and positive `fd_eps`, the historical attributes, and
fixed-arm `generator` call arguments remain accepted. Computation delegates to
upstream without changing its formula, optimizer records or extra diagnostics.
The supported fixed L2/autograd arms are `b_cap` and `a_r1r2`; the upstream `k3p`
keyword arm remains available but is not the legacy default.

```python
import torch
from particle_sliders import (
    GradRegularizer, FormulationGame, dummy_features,
    particlegan_locked_shared, winning_formulation,
)

stamp = particlegan_locked_shared()['stamp']
assert stamp.cover_weight == 1.5
cap = GradRegularizer('b_cap', 1., 1., 4, 'l2', 'none', 10, 'autograd', .05)
critic = torch.nn.Linear(3, 1)
x = torch.zeros(2, 3)
penalty, stats = cap.penalty(critic, x, x, step=4)

formulation = winning_formulation()
config = {**formulation.as_dict(), 'g_lr': 2e-5, 'adv_batch': 4}
formulation.require(config)
game = FormulationGame(formulation, config, seed=7, device='cpu')
features = dummy_features(game.rank, 4, 7)
stats = game.step(1, features(1))
```

This numerical example needs no native model or downloads. It is covered by
installed-package regression checks; it makes no pretrained quality claim.

## Retired kernels fail closed

The import `from particle_sliders.grad_regularizers import finite_difference_norm`
works again. PG0.9.1 does **not** supply that kernel: calling it raises
`UnsupportedLegacyRegularizer` before any model forward. Alternate norms (`l1`,
`linf`), finite differences, annealing (`linear`, `delayed`) and removed arms
also raise before upstream state construction. They are not silently mapped to
L2, no annealing, or zero penalties. A finite-difference implementation supplied
by upstream is delegated to; no local approximation is used.

To execute those historical modes or recover exact old class/diagnostic identity,
use the original pinned environment, separately from PG0.9.1:

```sh
python -m venv legacy-core
legacy-core/bin/python -m pip install   'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@17cd484ef00601389256a30f6d52fc79991bcb3d#subdirectory=packages/particle-sliders-core'
```

That source preserves core0.3.0's exact upstream Git dependency82b726. Do not
substitute PG0.9.1 into that environment or resume its results with a new cohort.
The new lifecycle never relaxes source/model/architecture/profile checks; v1
adapter exports remain the CLI default and native-only v2 export is explicit.

## Dependency minimum and independent lanes

Torch `>=2.6,<3` is required by the exact official ParticleGAN0.9.1 distribution
metadata itself. The old core0.3 line allowed Torch>=2.4; lowering only the new
core's bound would not yield a resolvable official-PG0.9.1 environment. The local
CPU qualification uses its recorded Torch build; no cross-version/GPU qualification is implied.
Python>=3.10 remains the package declaration. Optional HNDL has its own Python
markers and its network architecture is not silently substituted for raw models.

Official PG0.9.3 is a separate reviewed source lane; it is not accepted by this
line's exact PG0.9.1 source/profile gate. Unofficial continuous-domain proposals
are neither included nor enabled here. Official wide audio response refusal,
Iris's render refusal, Krea's adapter-only arm, and native qualification gates
remain. Native libraries and model downloads are separate product concerns.
