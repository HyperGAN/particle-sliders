# Historical API compatibility

Current core 0.14 installs shared modules only; it does not install embedded
products or legacy `conceptmod` aliases. See the [distribution boundary](SHARED-CORE-BOUNDARY.md)
and [history](../../docs/history.md) for reproduction rules and archived source.

The retained regularizer compatibility shim refers to legacy source commit
[`17cd484`](https://github.com/HyperGAN/particle-sliders/tree/17cd484ef00601389256a30f6d52fc79991bcb3d).
Use that complete source and its original environment to reproduce retired
penalty modes. Current ParticleGAN kernels are not substituted for unsupported
historical mathematics.

Core 0.13's embedded model/namespace cohort is available at
[`285180e`](https://github.com/HyperGAN/particle-sliders/tree/285180ed9813f12e912bb6bce6bada00496fd19a).
Neither pin makes old checkpoints resumable under a new source identity.

## Retained CPU helper example

The named compatibility helpers remain installed. This complete numerical
example exercises that API without native weights or downloads; new product
training uses the [conditional engine](CONDITIONAL-PG09-CONTRACT.md).

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
